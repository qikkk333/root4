"""Offline pytest suite for Milestone 3: SQL, Hybrid, Refusal, and Citation Precision/Recall evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from eval import evaluate_agent as ev  # noqa: E402
from helpdesk import agent as agent_mod  # noqa: E402
from helpdesk import sql as sql_mod  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Dataset Integrity (verified against real DB & chunks.json)
# ---------------------------------------------------------------------------
def test_milestone3_datasets_valid_and_grounded_in_real_db_and_index():
    _, valid_pages, _ = ev.load_chunks_catalog()
    assert valid_pages, "index/chunks.json must be present for citation validation"

    all_ids: list[str] = []
    for suite_name in ("main", "hard", "sql", "hybrid", "refusal"):
        items, _ = ev.load_dataset(suite=suite_name)
        assert items
        for item in items:
            all_ids.append(item["id"])
            srcs = item.get("sources", [])
            if suite_name in ("sql", "hybrid", "refusal"):
                for src in srcs:
                    assert (src["doc"], src.get("page")) in valid_pages, (
                        f"{suite_name}:{item['id']} gold source {src} missing from index/chunks.json"
                    )
            elif srcs:
                # Pre-existing main/hard datasets have T26/H26 referencing page 1 of charts PDF (page 2 is indexed)
                indexed_hits = [src for src in srcs if (src["doc"], src.get("page")) in valid_pages]
                assert indexed_hits, f"{suite_name}:{item['id']} has zero indexed sources: {srcs}"
            if item.get("sql"):
                res = sql_mod.execute_readonly_query(item["sql"])
                assert "error" not in res, f"{suite_name}:{item['id']} gold SQL failed: {res}"
                has_rows = res["row_count"] > 0
                assert has_rows == item.get("expect_rows", True)

    assert len(all_ids) == len(set(all_ids)), "Duplicate question IDs across evaluation suites"


# ---------------------------------------------------------------------------
# 2. Citation Precision, Recall, Invalid/OOB/Mismatch & SQL Leak Metrics
# ---------------------------------------------------------------------------
def test_score_citations_precision_recall_and_error_breakdowns():
    _, valid_pages, _ = ev.load_chunks_catalog()
    gold = [
        {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1},
        {"doc": "pib_2026-03-24_reform_express.html", "page": None},
    ]

    # Exact match
    perfect = ev.score_citations(gold, gold, valid_pages=valid_pages)
    assert perfect["citation_precision"] == 1.0
    assert perfect["citation_recall"] == 1.0
    assert perfect["citation_f1"] == 1.0
    assert perfect["doc_identified"] is True
    assert perfect["page_identified"] is True
    assert perfect["invalid_citation_count"] == 0
    assert perfect["missing_citation_count"] == 0
    assert perfect["out_of_range_page_count"] == 0
    assert perfect["doc_mismatch_count"] == 0
    assert perfect["sql_leak_count"] == 0

    # Extra noisy page from valid catalog + duplicate -> lowers precision & F1
    noisy = ev.score_citations(
        [
            {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1},
            {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1},
            {"doc": "rb_cc08_2025_tatkal.pdf", "page": 1},
        ],
        [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
        valid_pages=valid_pages,
    )
    assert noisy["citation_precision"] == pytest.approx(1 / 3)
    assert noisy["citation_recall"] == 1.0
    assert noisy["citation_f1"] == pytest.approx(0.5)
    assert noisy["doc_mismatch_count"] == 1

    # Out-of-range page (irctc PDF only has pages 1..11) + negative page + unknown doc
    errors = ev.score_citations(
        [
            {"doc": "irctc_refund_rules_jan2026.pdf", "page": 99},
            {"doc": "irctc_refund_rules_jan2026.pdf", "page": -1},
            {"doc": "nonexistent.pdf", "page": 1},
        ],
        [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
        valid_pages=valid_pages,
    )
    assert errors["citation_precision"] == 0.0
    assert errors["citation_recall"] == 0.0
    assert errors["invalid_citation_count"] == 3
    assert errors["out_of_range_page_count"] == 2
    assert errors["doc_mismatch_count"] == 1

    # SQL DB_DOC leaked into sources must never receive citation credit
    leaked = ev.score_citations(
        [{"doc": agent_mod.DB_DOC, "page": None}],
        [],
        valid_pages=valid_pages,
    )
    assert leaked["sql_leak_count"] == 1
    assert leaked["invalid_citation_count"] == 1
    assert leaked["citation_precision"] == 0.0
    assert leaked["citation_recall"] == 0.0

    # Empty citation when gold exists -> 0 precision, 0 recall, missing_citation=1
    missing = ev.score_citations([], gold, valid_pages=valid_pages)
    assert missing["citation_precision"] == 0.0
    assert missing["citation_recall"] == 0.0
    assert missing["missing_citation_count"] == 1

    # No-gold-source item with empty citations -> 1.0 / 1.0
    no_gold_ok = ev.score_citations([], [], valid_pages=valid_pages)
    assert no_gold_ok["citation_precision"] == 1.0
    assert no_gold_ok["citation_recall"] == 1.0


# ---------------------------------------------------------------------------
# 3. SQL Execution Success vs Answer Correctness Separation
# ---------------------------------------------------------------------------
def test_score_sql_separates_execution_from_correctness():
    item = {
        "id": "S_TEST",
        "expected_route": "sql",
        "question": "When does train 12002 leave New Delhi?",
        "answer": "Train 12002 departs at 06:15.",
        "sources": [],
        "sql": "SELECT number, name, departure FROM trains WHERE number = '12002'",
        "expected_facts": ["12002", "06:15"],
        "expect_rows": True,
        "dont_know": False,
    }

    # Valid SQLite SELECT that runs cleanly, but queried the WRONG train (12951 instead of 12002)
    wrong_sql_out = {
        "answer": "Train 12951 departs at 16:40.",
        "sources": [],
        "sql": "SELECT number, name, departure FROM trains WHERE number = '12951'",
        "dont_know": False,
    }
    wrong_scores = ev.score_sql_case(item, wrong_sql_out)
    assert wrong_scores["sql_execution_ok"] is True
    assert wrong_scores["sql_result_correct"] is False

    # Syntactically correct query, matching answer facts & DB values
    right_out = {
        "answer": "According to the 2016 timetable snapshot, train 12002 departs at 06:15.",
        "sources": [],
        "sql": "SELECT departure AS dep_time, number, name FROM trains WHERE number = '12002'",
        "dont_know": False,
    }
    right_scores = ev.score_sql_case(item, right_out)
    assert right_scores["sql_execution_ok"] is True
    assert right_scores["sql_result_correct"] is True

    # Failed / invalid SQL
    failed_sql_out = {
        "answer": agent_mod.TIMETABLE_FAILED,
        "sources": [],
        "sql": "SELECT bad_col FROM trains",
        "dont_know": True,
    }
    failed_scores = ev.score_sql_case(item, failed_sql_out)
    assert failed_scores["sql_execution_ok"] is False
    assert failed_scores["sql_result_correct"] is False


# ---------------------------------------------------------------------------
# 4. End-to-End Deterministic Offline Evaluation (SQL, Hybrid, Refusal, Main)
# ---------------------------------------------------------------------------
def test_run_offline_evaluation_all_suites_and_report_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out_file = tmp_path / "offline_report.json"

    report = ev.run_offline_evaluation(suite="all", out_path=out_file)
    assert out_file.is_file()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["evaluation_mode"] == "offline_deterministic"
    assert loaded["datasets"] == ["test_set_sql.json", "test_set_hybrid.json", "test_set_refusal.json"]

    summary = loaded["summary"]
    assert summary["total_questions"] == 39
    assert summary["successful_questions"] == 39
    assert summary["failed_questions"] == 0
    assert summary["routing_accuracy"] == 1.0
    assert summary["dont_know_accuracy"] == 1.0
    assert summary["sql_execution_successes"] == 22
    assert summary["sql_execution_failures"] == 0
    assert summary["sql_result_accuracy"] == 1.0
    assert summary["citation_precision_macro"] == 1.0
    assert summary["citation_recall_macro"] == 1.0
    assert summary["citation_f1_macro"] == 1.0
    assert summary["citation_precision_doc_only"] == 1.0
    assert summary["citation_recall_doc_only"] == 1.0
    assert summary["citation_f1_doc_only"] == 1.0
    assert summary["correct_doc_identification_rate"] == 1.0
    assert summary["correct_page_identification_rate"] == 1.0
    assert summary["unsupported_answer_count"] == 0
    assert summary["invalid_citation_count"] == 0
    assert summary["missing_citation_count"] == 0
    assert summary["out_of_range_page_count"] == 0
    assert summary["doc_mismatch_count"] == 0
    assert summary["sql_leak_count"] == 0
    assert summary["unsafe_rejection_rate"] == 1.0
    assert summary["mixed_success_rate"] == 1.0
    assert loaded["failure_examples"] == []
    assert loaded["known_limitations"]
    assert report["summary"] == summary


def test_run_offline_evaluation_main_document_suite(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    report = ev.run_offline_evaluation(suite="main")
    summary = report["summary"]
    assert summary["total_questions"] == 37
    assert summary["successful_questions"] == 37
    assert summary["routing_accuracy"] == 1.0
    assert summary["dont_know_accuracy"] == 1.0
    assert summary["citation_precision_macro"] == 1.0
    # T26 cites 1 of 2 gold pages because rb_cc10_2025_reservation_charts.pdf page 1 was omitted at ingest
    assert summary["citation_recall_macro"] == pytest.approx(36.5 / 37)
    assert summary["citation_f1_macro"] > 0.99
    assert summary["sql_leak_count"] == 0


def test_smoke_script_missing_key_and_contract_validation(monkeypatch: pytest.MonkeyPatch):
    from eval import smoke_test_live  # noqa: E402

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert smoke_test_live.main() == 2

    def fake_ask(q: str) -> dict:
        if "Delete" in q:
            return {"answer": agent_mod.REFUSAL, "sources": [], "sql": None, "dont_know": True}
        if "live running" in q:
            return {"answer": agent_mod.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}
        if "advance reservation" in q:
            return {
                "answer": "Departs 06:15; ARP 60 days.",
                "sources": [{"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1}],
                "sql": "SELECT departure FROM trains WHERE number='12002'",
                "dont_know": False,
            }
        if "train 12002" in q:
            return {"answer": "Departs 06:15.", "sources": [], "sql": "SELECT departure FROM trains", "dont_know": False}
        return {"answer": "Rs 120.", "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}], "sql": None, "dont_know": False}

    rows = smoke_test_live.run_smoke_tests(ask_fn=fake_ask)
    assert len(rows) == 5
    assert all(r["passed"] and r["no_db_leak"] for r in rows)
