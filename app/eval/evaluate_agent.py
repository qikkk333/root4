"""End-to-end & SQL/Hybrid/Citation evaluator for the Railway Passenger Help Desk.

Supports two strictly separated execution modes:
1. Offline deterministic mode (`--offline`):
   Uses real read-only SQLite execution (`helpdesk.sql.execute_readonly_query`), real indexed chunks
   from `index/chunks.json`, real `RouterAgent` routing/grounding/safety validation, and deterministic
   scripted LLM responses (zero GEMINI_API_KEY required, zero network calls).
2. Live Gemini mode (default when `--offline` is omitted):
   Calls `make_agent(...)` and optional LLM judge (`JUDGE_PROMPT`) against the real Gemini API.

Citation metric conventions (`score_citations`):
- Only citations in `out["sources"]` that are syntactically valid (`{"doc": str, "page": int|None}`),
  exclude SQL database pseudo-documents (`DB_DOC`), have valid pages (`int >= 1` for `.pdf`), exist
  in the indexed `(doc, page)` catalog (when provided), AND appear in `gold_sources` earn `tp` credit
  (deduplicated per unique `(doc, page)` pair).
- Citation Precision = `tp / len(cited_sources)` when `len(cited_sources) > 0`.
- Citation Recall    = `tp / len(gold_set)` when `len(gold_set) > 0`.
- Empty citations when gold docs exist (`len(gold_set) > 0`, `len(cited_sources) == 0`):
  precision = `0.0`, recall = `0.0`, `missing_citation_count = 1`.
- No-gold-source items (`len(gold_set) == 0`, e.g., SQL-only or abstention/refusal):
  precision = `1.0` and recall = `1.0` if `len(cited_sources) == 0`, else `0.0` (penalizes spurious
  citations or SQL `DB_DOC` leakage). Both overall macro-averages and `gold_doc_only` macro-averages
  are reported in `summarize_results`.

Usage:
    python eval/evaluate_agent.py --offline [--suite all|sql|hybrid|refusal|main|hard] [--out eval/results/offline_report.json]
    python eval/evaluate_agent.py [--agent router|rag|multi|tools] [--questions eval/test_set_hybrid.json] [--no-judge] [--out results.json]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpdesk import agent as agent_mod  # noqa: E402
from helpdesk import sql as sql_mod  # noqa: E402
from helpdesk.agent import AGENTS, DB_DOC, LLM, RoutePlan, GroundedAnswer, Evidence, make_agent  # noqa: E402

SUITES: dict[str, Path] = {
    "main": ROOT / "eval" / "test_set.json",
    "hard": ROOT / "eval" / "test_set_hard.json",
    "sql": ROOT / "eval" / "test_set_sql.json",
    "hybrid": ROOT / "eval" / "test_set_hybrid.json",
    "refusal": ROOT / "eval" / "test_set_refusal.json",
}

DEFAULT_OFFLINE_SUITES = ("sql", "hybrid", "refusal")

JUDGE_PROMPT = """You grade a railway help desk. Compare the ANSWER with the REFERENCE for the QUESTION.
Reply "yes" if the answer states the same key facts (numbers, times, yes/no) as the reference and nothing that
contradicts it; extra correct detail is fine. Reply "no" otherwise. If the reference says "I don't know", reply "yes"
only if the answer also declines. Reply with one word: yes or no."""


def judge(llm, q, ref, ans):
    reply = llm.text([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"QUESTION: {q}\nREFERENCE: {ref}\nANSWER: {ans}"},
    ])
    return reply.strip().lower().startswith("yes")


def load_chunks_catalog(chunks_path: Path | None = None) -> tuple[list[dict], set[tuple[str, Any]], set[str]]:
    """Load indexed chunks -> (chunks_list, valid_(doc, page)_set, valid_docs_set)."""
    path = chunks_path or (ROOT / "index" / "chunks.json")
    if not path.is_file():
        return [], set(), set()
    chunks = json.loads(path.read_text(encoding="utf-8"))
    valid_pages = {(c["metadata"]["doc"], c["metadata"].get("page")) for c in chunks}
    valid_docs = {d for d, _ in valid_pages}
    return chunks, valid_pages, valid_docs


def load_dataset(questions: Path | None = None, suite: str = "main") -> tuple[list[dict], list[str]]:
    """Load evaluation items and source file identifiers."""
    if questions is not None:
        items = json.loads(Path(questions).read_text(encoding="utf-8"))
        return items, [Path(questions).name]
    if suite == "all":
        items, files = [], []
        for name in DEFAULT_OFFLINE_SUITES:
            p = SUITES[name]
            loaded = json.loads(p.read_text(encoding="utf-8"))
            for row in loaded:
                row.setdefault("_suite", name)
            items.extend(loaded)
            files.append(p.name)
        return items, files
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}; choose from {list(SUITES) + ['all']}")
    p = SUITES[suite]
    return json.loads(p.read_text(encoding="utf-8")), [p.name]


def _is_syntactic_doc_page(doc: Any, page: Any) -> tuple[bool, bool]:
    """Return (doc_ok, page_syntax_ok) for a citation dict excluding SQL DB_DOC."""
    if not isinstance(doc, str) or not doc.strip() or doc.strip() == DB_DOC:
        return False, False
    if isinstance(page, bool):
        return True, False
    if page is None:
        return True, not doc.strip().lower().endswith(".pdf")
    if not isinstance(page, int) or page < 1:
        return True, False
    return True, True


def score_citations(
    cited_sources: list[dict] | None,
    gold_sources: list[dict] | None,
    valid_pages: set[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute strict per-question Citation Precision, Citation Recall, and error breakdowns."""
    cited = list(cited_sources or [])
    gold = list(gold_sources or [])
    gold_set = {(s["doc"], s.get("page")) for s in gold}
    gold_docs = {d for d, _ in gold_set}
    known_docs = {d for d, _ in valid_pages} if valid_pages is not None else set()

    total_cited = len(cited)
    seen_valid_tp: set[tuple[str, Any]] = set()
    invalid_count = 0
    out_of_range_page_count = 0
    doc_mismatch_count = 0
    sql_leak_count = 0

    for entry in cited:
        if not isinstance(entry, dict):
            invalid_count += 1
            continue
        doc = entry.get("doc")
        page = entry.get("page")
        if doc == DB_DOC:
            sql_leak_count += 1
            invalid_count += 1
            continue

        doc_ok, page_syntax_ok = _is_syntactic_doc_page(doc, page)
        if not doc_ok:
            invalid_count += 1
            doc_mismatch_count += 1
            continue

        pair = (doc.strip(), page)
        in_index = valid_pages is None or pair in valid_pages
        known_doc = (pair[0] in known_docs) or (pair[0] in gold_docs)

        if not page_syntax_ok:
            invalid_count += 1
            out_of_range_page_count += 1
            if gold_docs and pair[0] not in gold_docs:
                doc_mismatch_count += 1
            continue

        if valid_pages is not None and not in_index:
            invalid_count += 1
            if known_doc:
                out_of_range_page_count += 1
            else:
                doc_mismatch_count += 1
            continue

        # Valid citation syntactically and within index catalog; now check relevance to gold
        if not gold_set or pair[0] not in gold_docs:
            doc_mismatch_count += 1
        elif pair in gold_set:
            seen_valid_tp.add(pair)

    tp = len(seen_valid_tp)
    if gold_set:
        recall = tp / len(gold_set)
        precision = (tp / total_cited) if total_cited > 0 else 0.0
        missing_citation = 1 if total_cited == 0 else 0
    else:
        missing_citation = 0
        precision = 1.0 if total_cited == 0 else 0.0
        recall = 1.0 if total_cited == 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    valid_cited_docs = {
        s["doc"].strip()
        for s in cited
        if isinstance(s, dict) and isinstance(s.get("doc"), str) and s.get("doc") != DB_DOC
    }
    doc_identified = bool(gold_docs & valid_cited_docs) if gold_docs else (total_cited == 0)
    page_identified = (tp > 0) if any(p is not None for _, p in gold_set) else doc_identified

    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "tp_citations": tp,
        "total_cited_count": total_cited,
        "gold_doc_count": len(gold_set),
        "doc_identified": doc_identified,
        "page_identified": page_identified,
        "invalid_citation_count": invalid_count,
        "missing_citation_count": missing_citation,
        "out_of_range_page_count": out_of_range_page_count,
        "doc_mismatch_count": doc_mismatch_count,
        "sql_leak_count": sql_leak_count,
    }


def classify_route(out: dict[str, Any]) -> str:
    """Classify runtime contract into 'refuse' | 'abstain' | 'mixed' | 'sql' | 'documents' | 'unknown'."""
    ans = str(out.get("answer") or "")
    dont_know = bool(out.get("dont_know"))
    has_sql = bool(out.get("sql"))
    has_docs = bool(out.get("sources"))
    if dont_know and not has_sql and not has_docs and "read-only" in ans.lower():
        return "refuse"
    if dont_know:
        return "abstain"
    if has_sql and has_docs:
        return "mixed"
    if has_sql:
        return "sql"
    if has_docs:
        return "documents"
    return "unknown"


def _extract_row_values(rows: list[dict[str, Any]]) -> set[str]:
    vals: set[str] = set()
    for row in rows:
        for val in row.values():
            if val is not None:
                vals.add(str(val).strip().lower())
    return vals


def _facts_present(text: str, facts: list[str]) -> bool:
    low = text.lower()
    return all(fact.lower() in low for fact in facts)


def score_sql_case(
    item: dict[str, Any],
    out: dict[str, Any],
    execute_fn: Callable[[str], dict[str, Any]] = sql_mod.execute_readonly_query,
) -> dict[str, Any]:
    """Evaluate SQL execution success and answer/row correctness separately from citation metrics."""
    expected_route = item.get("expected_route") or ("sql" if item.get("sql") else "documents")
    expects_sql = bool(item.get("sql")) or bool(item.get("expected_sql_ran")) or expected_route in ("sql", "mixed")
    expect_rows = bool(item.get("expect_rows", bool(item.get("sql")) and not bool(item.get("dont_know"))))
    is_unsafe = bool(item.get("unsafe", False))
    expected_dk = bool(item.get("dont_know", not item.get("sources") and not item.get("sql")))

    pred_sql = out.get("sql")
    sql_attempted = isinstance(pred_sql, str) and bool(pred_sql.strip())
    exec_res = execute_fn(pred_sql) if sql_attempted else None
    sql_exec_ok = (exec_res is not None and "error" not in exec_res) if sql_attempted else None
    actual_rows = exec_res.get("rows", []) if sql_exec_ok else []

    if is_unsafe:
        unsafe_ok = (
            bool(out.get("dont_know"))
            and pred_sql is None
            and not out.get("sources")
            and "read-only" in str(out.get("answer") or "").lower()
        )
        return {
            "expects_sql": False,
            "sql_attempted": sql_attempted,
            "sql_execution_ok": sql_exec_ok,
            "sql_result_correct": None,
            "unsafe_rejected_ok": unsafe_ok,
            "sql_abstain_ok": None,
        }

    if not expects_sql:
        # Must NOT have run unnecessary/unsafe SQL when expecting pure docs or pure abstention
        clean_no_sql = pred_sql is None
        return {
            "expects_sql": False,
            "sql_attempted": sql_attempted,
            "sql_execution_ok": sql_exec_ok,
            "sql_result_correct": None if clean_no_sql else False,
            "unsafe_rejected_ok": None,
            "sql_abstain_ok": None,
        }

    # Item expected SQL execution
    if not sql_attempted or not sql_exec_ok:
        return {
            "expects_sql": True,
            "sql_attempted": sql_attempted,
            "sql_execution_ok": False,
            "sql_result_correct": False,
            "unsafe_rejected_ok": None,
            "sql_abstain_ok": False if expected_dk else None,
        }

    if expected_dk and not expect_rows:
        # SQL zero-row abstention case (e.g. Vande Bharat in 2016 snapshot)
        abstain_ok = bool(out.get("dont_know")) and len(actual_rows) == 0 and not out.get("sources")
        return {
            "expects_sql": True,
            "sql_attempted": True,
            "sql_execution_ok": True,
            "sql_result_correct": abstain_ok,
            "unsafe_rejected_ok": None,
            "sql_abstain_ok": abstain_ok,
        }

    # Normal SQL-bearing item expecting non-empty rows and factual answer
    has_rows_ok = len(actual_rows) > 0
    rows_blob = json.dumps(actual_rows, ensure_ascii=False)
    facts = item.get("expected_facts") or []
    facts_ok = _facts_present(f"{out.get('answer', '')}\n{rows_blob}", facts) if facts else has_rows_ok

    gold_values_ok = True
    if item.get("sql"):
        gold_exec = execute_fn(item["sql"])
        if "error" not in gold_exec and gold_exec.get("rows"):
            gold_vals = _extract_row_values(gold_exec["rows"])
            pred_vals = _extract_row_values(actual_rows)
            # Every cell value from gold SQL output must appear in predicted SQL rows (alias-agnostic)
            gold_values_ok = gold_vals.issubset(pred_vals) or _facts_present(rows_blob, sorted(gold_vals))

    result_correct = has_rows_ok and facts_ok and gold_values_ok and (out.get("dont_know") is False)
    return {
        "expects_sql": True,
        "sql_attempted": True,
        "sql_execution_ok": True,
        "sql_result_correct": result_correct,
        "unsafe_rejected_ok": None,
        "sql_abstain_ok": None,
    }


def evaluate_item(
    item: dict[str, Any],
    out: dict[str, Any],
    *,
    valid_pages: set[tuple[str, Any]] | None = None,
    judge_llm: Any = None,
    execute_fn: Callable[[str], dict[str, Any]] = sql_mod.execute_readonly_query,
) -> dict[str, Any]:
    """Evaluate one model output against expected route, citations, SQL correctness, and abstention."""
    expected_dk = bool(item.get("dont_know", not item.get("sources") and not item.get("sql")))
    expected_route = item.get("expected_route")
    if not expected_route:
        if item.get("unsafe"):
            expected_route = "refuse"
        elif expected_dk:
            expected_route = "abstain"
        elif item.get("sql") and item.get("sources"):
            expected_route = "mixed"
        elif item.get("sql"):
            expected_route = "sql"
        else:
            expected_route = "documents"

    actual_route = classify_route(out)
    routing_ok = actual_route == expected_route
    dk_ok = bool(out.get("dont_know")) == expected_dk

    cit = score_citations(out.get("sources"), item.get("sources"), valid_pages=valid_pages)
    sql_scores = score_sql_case(item, out, execute_fn=execute_fn)

    facts = item.get("expected_facts") or []
    answer_facts_ok = True if expected_dk else (_facts_present(str(out.get("answer") or ""), facts) if facts else True)

    llm_correct = judge(judge_llm, item["question"], item.get("answer", ""), out.get("answer", "")) if judge_llm else None

    if expected_route == "refuse":
        passed = bool(sql_scores["unsafe_rejected_ok"]) and routing_ok and cit["sql_leak_count"] == 0
    elif expected_route == "abstain":
        sql_ok = True if not sql_scores["expects_sql"] else bool(sql_scores["sql_abstain_ok"])
        passed = dk_ok and routing_ok and sql_ok and cit["total_cited_count"] == 0 and cit["sql_leak_count"] == 0
    elif expected_route == "sql":
        passed = (
            routing_ok
            and dk_ok
            and bool(sql_scores["sql_execution_ok"])
            and bool(sql_scores["sql_result_correct"])
            and answer_facts_ok
            and cit["total_cited_count"] == 0
            and cit["sql_leak_count"] == 0
        )
    elif expected_route == "mixed":
        passed = (
            routing_ok
            and dk_ok
            and bool(sql_scores["sql_execution_ok"])
            and bool(sql_scores["sql_result_correct"])
            and answer_facts_ok
            and cit["citation_recall"] == 1.0
            and cit["invalid_citation_count"] == 0
            and cit["sql_leak_count"] == 0
        )
    else:  # documents
        doc_ok = (cit["tp_citations"] > 0) if item.get("sources") else True
        passed = routing_ok and dk_ok and doc_ok and answer_facts_ok and cit["invalid_citation_count"] == 0 and cit["sql_leak_count"] == 0

    if llm_correct is not None:
        passed = passed and llm_correct

    source_hit = (cit["tp_citations"] > 0) if item.get("sources") else None
    unsupported_answer = (
        expected_route in ("documents", "mixed")
        and (out.get("dont_know") is False)
        and (cit["tp_citations"] == 0 or cit["invalid_citation_count"] > 0)
    )

    return {
        "id": item.get("id", ""),
        "category": item.get("category") or item.get("tests", ""),
        "question": item["question"],
        "reference": item.get("answer", ""),
        "expected_route": expected_route,
        "actual_route": actual_route,
        "routing_ok": routing_ok,
        "expected_dont_know": expected_dk,
        "dont_know": out.get("dont_know"),
        "dk_ok": dk_ok,
        "answer": out.get("answer", ""),
        "sources": out.get("sources", []),
        "sql": out.get("sql"),
        "source_hit": source_hit,
        "unsupported_answer": unsupported_answer,
        "answer_facts_ok": answer_facts_ok,
        "correct": llm_correct,
        **cit,
        **sql_scores,
        "passed": passed,
    }


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question metrics into the capstone evaluation summary dictionary."""
    n = len(rows)
    if n == 0:
        return {"total_questions": 0, "successful_questions": 0, "failed_questions": 0}

    sql_attempted_rows = [r for r in rows if r["sql_attempted"]]
    sql_expected_rows = [r for r in rows if r["expects_sql"]]
    sql_success_rows = [r for r in sql_attempted_rows if r["sql_execution_ok"]]
    sql_fail_rows = [r for r in sql_expected_rows if not r["sql_execution_ok"]]
    sql_result_rows = [r for r in sql_expected_rows if r["sql_result_correct"] is not None]
    doc_rows = [r for r in rows if r["gold_doc_count"] > 0]
    unsafe_rows = [r for r in rows if r["unsafe_rejected_ok"] is not None]
    mixed_rows = [r for r in rows if r["expected_route"] == "mixed"]
    judge_rows = [r for r in rows if r["correct"] is not None]
    with_src_hit = [r for r in rows if r["source_hit"] is not None]

    return {
        "total_questions": n,
        "successful_questions": sum(1 for r in rows if r["passed"]),
        "failed_questions": sum(1 for r in rows if not r["passed"]),
        "abstentions": sum(1 for r in rows if r["dont_know"]),
        "refusals": sum(1 for r in rows if r["actual_route"] == "refuse"),
        "routing_accuracy": sum(1 for r in rows if r["routing_ok"]) / n,
        "dont_know_accuracy": sum(1 for r in rows if r["dk_ok"]) / n,
        "source_hit_rate": (sum(1 for r in with_src_hit if r["source_hit"]) / len(with_src_hit)) if with_src_hit else None,
        "judge_accuracy": (sum(1 for r in judge_rows if r["correct"]) / len(judge_rows)) if judge_rows else None,
        "sql_execution_attempts": len(sql_attempted_rows),
        "sql_expected_questions": len(sql_expected_rows),
        "sql_execution_successes": len(sql_success_rows),
        "sql_execution_failures": len(sql_fail_rows),
        "sql_execution_success_rate": (len(sql_success_rows) / len(sql_expected_rows)) if sql_expected_rows else None,
        "sql_result_correct": sum(1 for r in sql_result_rows if r["sql_result_correct"]),
        "sql_result_accuracy": (sum(1 for r in sql_result_rows if r["sql_result_correct"]) / len(sql_result_rows)) if sql_result_rows else None,
        "citation_precision_macro": sum(r["citation_precision"] for r in rows) / n,
        "citation_recall_macro": sum(r["citation_recall"] for r in rows) / n,
        "citation_f1_macro": sum(r["citation_f1"] for r in rows) / n,
        "citation_precision_doc_only": (sum(r["citation_precision"] for r in doc_rows) / len(doc_rows)) if doc_rows else None,
        "citation_recall_doc_only": (sum(r["citation_recall"] for r in doc_rows) / len(doc_rows)) if doc_rows else None,
        "citation_f1_doc_only": (sum(r["citation_f1"] for r in doc_rows) / len(doc_rows)) if doc_rows else None,
        "correct_doc_identification_rate": (sum(1 for r in doc_rows if r["doc_identified"]) / len(doc_rows)) if doc_rows else None,
        "correct_page_identification_rate": (sum(1 for r in doc_rows if r["page_identified"]) / len(doc_rows)) if doc_rows else None,
        "unsupported_answer_count": sum(1 for r in rows if r.get("unsupported_answer")),
        "invalid_citation_count": sum(r["invalid_citation_count"] for r in rows),
        "missing_citation_count": sum(r["missing_citation_count"] for r in rows),
        "out_of_range_page_count": sum(r["out_of_range_page_count"] for r in rows),
        "doc_mismatch_count": sum(r["doc_mismatch_count"] for r in rows),
        "sql_leak_count": sum(r["sql_leak_count"] for r in rows),
        "unsafe_rejection_rate": (sum(1 for r in unsafe_rows if r["unsafe_rejected_ok"]) / len(unsafe_rows)) if unsafe_rows else None,
        "mixed_success_rate": (sum(1 for r in mixed_rows if r["passed"]) / len(mixed_rows)) if mixed_rows else None,
    }


def build_report(
    rows: list[dict[str, Any]],
    *,
    evaluation_mode: str,
    agent_name: str,
    dataset_files: list[str],
    elapsed_sec: float,
    retrieval_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble structured reproducible evaluation report payload."""
    failures = [
        {
            "id": r["id"],
            "question": r["question"],
            "expected_route": r["expected_route"],
            "actual_route": r["actual_route"],
            "dont_know": r["dont_know"],
            "sql": r["sql"],
            "sources": r["sources"],
            "answer": r["answer"][:200],
        }
        for r in rows
        if not r["passed"]
    ][:10]

    return {
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "m3-v1",
        "datasets": dataset_files,
        "evaluation_mode": evaluation_mode,
        "agent": agent_name,
        "elapsed_seconds": round(elapsed_sec, 3),
        "citation_conventions": {
            "precision": "valid_relevant_unique_cited / total_cited when len(cited)>0; 0.0 if gold exists & cited==0; 1.0 if no gold & cited==0",
            "recall": "valid_relevant_unique_cited / len(gold) when len(gold)>0; 1.0 if no gold & cited==0 else 0.0",
            "f1": "2 * precision * recall / (precision + recall) when (precision + recall)>0 else 0.0",
            "sql_citations": "SQL database passages (DB_DOC) are never counted as valid document citations",
        },
        "known_limitations": [
            "database/rail_timetable.db is an August 2016 snapshot (no Vande Bharat, no live status/fares/PNR/seats/platforms; Mumbai Central uses BCT).",
            "Bilingual Hindi scanned pages omitted during ingest (e.g., rb_cc10_2025_reservation_charts.pdf page 1) reduce max achievable recall on T26.",
            "Offline deterministic mode evaluates real SQL execution, retrieval catalog, and grounding logic using scripted LLM outputs without live Gemini inference.",
        ],
        "summary": summarize_results(rows),
        "failure_examples": failures,
        "retrieval_metrics": retrieval_metrics,
        "per_question": rows,
    }


class _OfflineScriptedLLM:
    """Deterministic scripted LLM executing real RouterAgent flows without Gemini credentials."""

    def __init__(self, items_by_question: dict[str, dict], chunks: list[dict]):
        self.by_q = items_by_question
        self.chunks = chunks
        self._active_q: str | None = None

    def _find_chunk_quote(self, doc: str, page: Any) -> str:
        for c in self.chunks:
            m = c["metadata"]
            if m.get("doc") == doc and m.get("page") == page:
                words = c["text"].strip().split()
                return " ".join(words[: min(len(words), 18)])
        return ""

    def parse(self, messages: list[dict[str, str]], schema: type, **kwargs):
        user_msg = messages[-1]["content"]
        if schema is RoutePlan:
            self._active_q = user_msg
            item = self.by_q[user_msg]
            sql_q = item.get("sql") or ""
            needs_docs = bool(item.get("sources")) or item.get("expected_route") == "abstain"
            if item.get("expected_sql_ran") and not item.get("sources"):
                needs_docs = False
            doc_query = item["question"] if needs_docs else ""
            return RoutePlan(needs_documents=needs_docs, doc_query=doc_query, sql=sql_q)

        # GroundedAnswer synthesis from formatted passages passed in messages[-1]["content"]
        q = self._active_q or user_msg.rsplit("Question:", 1)[-1].strip()
        item = self.by_q[q]
        expected_dk = bool(item.get("dont_know", not item.get("sources") and not item.get("sql")))
        if expected_dk:
            return GroundedAnswer(evidence=[], answer=item["answer"], dont_know=True)

        evidence: list[Evidence] = []
        blocks = user_msg.split("[Source ")
        for block in blocks[1:]:
            header, _, body = block.partition("\n")
            src_num = int(header.split("]", 1)[0])
            is_db = f"doc: {DB_DOC} " in header
            if is_db and item.get("sql"):
                lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith(("Database:", "SQL:", "1 row", "2 row"))]
                quote = next((ln for ln in lines if ":" in ln), body.strip()[:60])
                evidence.append(Evidence(source=src_num, quote=quote))
            else:
                for gold_src in item.get("sources", []):
                    if f"doc: {gold_src['doc']} | page: {gold_src.get('page')} " in header:
                        words = body.strip().split()
                        quote = " ".join(words[: min(len(words), 18)]) or self._find_chunk_quote(gold_src["doc"], gold_src.get("page"))
                        evidence.append(Evidence(source=src_num, quote=quote))
                        break

        return GroundedAnswer(evidence=evidence, answer=item["answer"], dont_know=False)

    def text(self, messages: list[dict[str, str]], **kwargs) -> str:
        return ""


def _offline_search_for_catalog(chunks: list[dict], by_q: dict[str, dict]):
    by_doc_page: dict[tuple[str, Any], list[dict]] = {}
    for c in chunks:
        key = (c["metadata"]["doc"], c["metadata"].get("page"))
        by_doc_page.setdefault(key, []).append({
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "metadata": c["metadata"],
            "score": 1.0,
            "rank": 1,
        })

    def _search(question: str, k: int = 5, **kwargs) -> list[dict]:
        item = by_q.get(question, {})
        found: list[dict] = []
        for s in item.get("sources", []):
            found.extend(by_doc_page.get((s["doc"], s.get("page")), [])[:1])
        if not found:
            # Return a benign policy chunk so abstention items test genuine LLM abstention over unrelated context
            return list(chunks[:1]) and [{
                "chunk_id": chunks[0]["chunk_id"],
                "text": chunks[0]["text"],
                "metadata": chunks[0]["metadata"],
                "score": 0.2,
                "rank": 1,
            }]
        return found[:k]

    return _search


def compute_retrieval_summary(k: int = 5, mode: str = "hybrid_rerank") -> dict[str, Any]:
    """Compute offline retrieval metrics on test_set.json and test_set_hard.json without LLM calls."""
    from eval import evaluate_retrieval as er  # noqa: E402

    summaries: dict[str, Any] = {}
    for label, suite_name in (("main", "main"), ("hard", "hard")):
        path = SUITES[suite_name]
        items = [q for q in json.loads(path.read_text(encoding="utf-8")) if q.get("sources")]
        rows = []
        t0 = time.time()
        for q in items:
            relevant = {(s["doc"], s["page"]) for s in q["sources"]}
            results = er.search(q["question"], k=k * 3, mode=mode)
            m = er.metrics(er.unique_pages(results), relevant, k)
            m["ans"] = er.chunk_level(results, q.get("answer", ""), k)
            rows.append(m)
        n = len(rows)
        ans_vals = [r["ans"] for r in rows if r["ans"] is not None]
        summaries[label] = {
            "questions": n,
            "mode": mode,
            "k": k,
            f"hit_at_{k}": round(sum(r["hit"] for r in rows) / n, 4),
            f"recall_at_{k}": round(sum(r["recall"] for r in rows) / n, 4),
            f"precision_at_{k}": round(sum(r["precision"] for r in rows) / n, 4),
            "mrr": round(sum(r["rr"] for r in rows) / n, 4),
            f"ans_at_{k}": round(sum(ans_vals) / len(ans_vals), 4) if ans_vals else None,
            "ms_per_query": round((time.time() - t0) * 1000 / n, 1),
        }
    return summaries


def run_offline_evaluation(
    *,
    suite: str = "all",
    questions: Path | None = None,
    retrieval_metrics: dict[str, Any] | None = None,
    include_retrieval: bool = False,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run deterministic offline RouterAgent + SQLite + citation evaluation with zero Gemini calls."""
    items, dataset_files = load_dataset(questions=questions, suite=suite)
    chunks, valid_pages, _ = load_chunks_catalog()
    by_q = {item["question"]: item for item in items}

    llm = _OfflineScriptedLLM(by_q, chunks)
    router = agent_mod.RouterAgent(llm=llm)
    t0 = time.time()
    rows = []

    with patch.object(agent_mod, "search", side_effect=_offline_search_for_catalog(chunks, by_q)):
        for item in items:
            out = router.ask(item["question"])
            rows.append(evaluate_item(item, out, valid_pages=valid_pages, judge_llm=None))

    if include_retrieval and retrieval_metrics is None:
        retrieval_metrics = compute_retrieval_summary()

    report = build_report(
        rows,
        evaluation_mode="offline_deterministic",
        agent_name="router (offline scripted LLM + real DB + real index catalog)",
        dataset_files=dataset_files,
        elapsed_sec=time.time() - t0,
        retrieval_metrics=retrieval_metrics,
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_live_evaluation(
    *,
    agent_name: str | None = None,
    suite: str = "main",
    questions: Path | None = None,
    use_judge: bool = True,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run live Gemini end-to-end evaluation (requires GEMINI_API_KEY)."""
    items, dataset_files = load_dataset(questions=questions, suite=suite)
    selected = (agent_name or os.getenv("HELPDESK_AGENT") or "router").strip().lower()
    if selected != "router":
        items = [q for q in items if not q.get("sql") and q.get("expected_route") not in ("sql", "mixed", "refuse")]

    _, valid_pages, _ = load_chunks_catalog()
    agent = make_agent(selected)
    judge_llm = LLM() if use_judge else None
    t0 = time.time()
    rows = []
    for q in items:
        out = agent.ask(q["question"])
        row = evaluate_item(q, out, valid_pages=valid_pages, judge_llm=judge_llm)
        rows.append(row)
        mark = {True: "OK ", False: "BAD", None: " - "}[row["passed"]]
        print(f"{mark} {row['id']} route={row['actual_route']} cit_P={row['citation_precision']:.2f} "
              f"cit_R={row['citation_recall']:.2f} dk_ok={row['dk_ok']} | {row['answer'][:90]}", flush=True)

    report = build_report(
        rows,
        evaluation_mode="live_gemini",
        agent_name=selected,
        dataset_files=dataset_files,
        elapsed_sec=time.time() - t0,
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _print_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    print(f"\n=== Evaluation Summary ({report['evaluation_mode']}) ===")
    print(f"datasets               : {', '.join(report['datasets'])}")
    print(f"agent                  : {report['agent']}")
    print(f"questions              : {s['total_questions']} (passed {s['successful_questions']}, failed {s['failed_questions']})")
    print(f"routing accuracy       : {s['routing_accuracy']:.2%}")
    print(f"dont_know accuracy     : {s['dont_know_accuracy']:.2%} (abstentions={s['abstentions']}, refusals={s['refusals']})")
    if s["source_hit_rate"] is not None:
        print(f"source hit rate        : {s['source_hit_rate']:.2%}")
    if s["judge_accuracy"] is not None:
        print(f"judge accuracy         : {s['judge_accuracy']:.2%}")
    if s["sql_expected_questions"]:
        print(f"SQL execution          : {s['sql_execution_successes']}/{s['sql_expected_questions']} "
              f"(failures={s['sql_execution_failures']})")
        print(f"SQL result accuracy    : {s['sql_result_accuracy']:.2%} ({s['sql_result_correct']}/{s['sql_expected_questions']})")
    print(f"citation macro (all)   : precision={s['citation_precision_macro']:.3f}, "
          f"recall={s['citation_recall_macro']:.3f}, f1={s['citation_f1_macro']:.3f}")
    if s["citation_precision_doc_only"] is not None:
        print(f"citation macro (docs)  : precision={s['citation_precision_doc_only']:.3f}, "
              f"recall={s['citation_recall_doc_only']:.3f}, f1={s['citation_f1_doc_only']:.3f}")
    print(f"citation errors        : invalid={s['invalid_citation_count']}, missing={s['missing_citation_count']}, "
          f"oob_page={s['out_of_range_page_count']}, doc_mismatch={s['doc_mismatch_count']}, sql_leak={s['sql_leak_count']}")
    if s["unsafe_rejection_rate"] is not None:
        print(f"unsafe write rejection : {s['unsafe_rejection_rate']:.2%}")
    if s["mixed_success_rate"] is not None:
        print(f"mixed SQL+doc success  : {s['mixed_success_rate']:.2%}")
    failed = [r for r in report["per_question"] if not r["passed"]]
    if failed:
        print("\nFailed questions:")
        for r in failed:
            print(f"- {r['id']} expected={r['expected_route']} actual={r['actual_route']} "
                  f"sql_ok={r['sql_execution_ok']} sql_res={r['sql_result_correct']} cit=({r['citation_precision']:.2f},{r['citation_recall']:.2f})")


def main():
    ap = argparse.ArgumentParser(description="Evaluate Railway Help Desk (offline deterministic or live Gemini).")
    ap.add_argument("--offline", action="store_true", help="Run deterministic offline evaluation (no Gemini API key)")
    ap.add_argument("--include-retrieval", action="store_true", help="Include local offline retrieval benchmark in report")
    ap.add_argument("--agent", default=None, choices=list(AGENTS))
    ap.add_argument("--suite", default=None, choices=list(SUITES) + ["all"])
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    suite = args.suite or ("all" if args.offline else "main")
    if args.offline:
        report = run_offline_evaluation(
            suite=suite,
            questions=args.questions,
            include_retrieval=args.include_retrieval,
            out_path=args.out,
        )
    else:
        report = run_live_evaluation(
            agent_name=args.agent,
            suite=suite,
            questions=args.questions,
            use_judge=not args.no_judge,
            out_path=args.out,
        )
    _print_summary(report)


if __name__ == "__main__":
    main()
