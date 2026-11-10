"""Offline pytest suite for Milestone 2: public ask() API, config, grounding, and error handling.

All tests are deterministic and run without live Gemini credentials or internet access.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import helpdesk  # noqa: E402
import helpdesk.agent as agent  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers & Test Doubles
# ---------------------------------------------------------------------------
def _sample_results():
    return [
        {
            "chunk_id": "doc_0001",
            "text": "Flat cancellation charge for sleeper class is Rs 120 per passenger.",
            "metadata": {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "doc_date": "2026-01-01"},
        },
        {
            "chunk_id": "doc_0002",
            "text": "Ticket Booking/cancellation Hours: 00:20 hours to 23:45 hours",
            "metadata": {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "doc_date": "2026-01-01"},
        },
    ]


def _make_openai_error(cls):
    req = httpx.Request("POST", "https://example.invalid/v1beta/openai/chat/completions")
    resp = httpx.Response(401 if cls is openai.AuthenticationError else 429, request=req)
    if cls in (openai.APIConnectionError, openai.APITimeoutError):
        return cls(request=req)
    return cls(message="simulated api failure", response=resp, body=None)


class _ScriptedLLM:
    def __init__(self, parse_seq=None, text_seq=None, complete_seq=None):
        self.parse_seq = list(parse_seq or [])
        self.text_seq = list(text_seq or [])
        self.complete_seq = list(complete_seq or [])
        self.parse_calls = 0
        self.text_calls = 0
        self.complete_calls = 0

    def parse(self, messages, schema, **kwargs):
        self.parse_calls += 1
        item = self.parse_seq.pop(0) if self.parse_seq else None
        if isinstance(item, Exception):
            raise item
        return item

    def text(self, messages, **kwargs):
        self.text_calls += 1
        item = self.text_seq.pop(0) if self.text_seq else ""
        if isinstance(item, Exception):
            raise item
        return item

    def complete(self, messages, **kwargs):
        self.complete_calls += 1
        item = self.complete_seq.pop(0) if self.complete_seq else SimpleNamespace(content="", tool_calls=None)
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------------------------
# 1. Public ask() API, defaults, CWD independence & .env loading
# ---------------------------------------------------------------------------
def test_public_ask_exported_from_package_and_works_from_root_and_arbitrary_cwd(tmp_path: Path):
    assert helpdesk.ask is not None
    assert callable(agent.ask)

    # Empty question via top-level import from repo root (app.helpdesk) with CWD=tmp_path
    code = (
        "import sys; sys.path.insert(0, r'" + str(REPO_ROOT) + "'); "
        "from app.helpdesk import ask; "
        "out = ask(''); "
        "assert out['dont_know'] is True and out['sql'] is None and out['sources'] == []; "
        "ref = ask('Delete train 12002 from your database'); "
        "assert ref['dont_know'] is True and 'read-only' in ref['answer']"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_make_agent_defaults_to_router_and_honors_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HELPDESK_AGENT", raising=False)
    llm = _ScriptedLLM()
    assert isinstance(agent.make_agent(llm=llm), agent.RouterAgent)

    monkeypatch.setenv("HELPDESK_AGENT", "")
    assert isinstance(agent.make_agent(llm=llm), agent.RouterAgent)

    monkeypatch.setenv("HELPDESK_AGENT", "rag")
    assert isinstance(agent.make_agent(llm=llm), agent.RAGAgent)

    assert isinstance(agent.make_agent("ROUTER", llm=llm), agent.RouterAgent)

    with pytest.raises(ValueError, match="unknown agent"):
        agent.make_agent("nonexistent", llm=llm)


def test_load_dotenv_loads_app_env_regardless_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert agent._APP_DIR.resolve() == APP_DIR.resolve()

    fake_repo = tmp_path / "repo"
    fake_app = fake_repo / "app"
    foreign_cwd = tmp_path / "elsewhere"
    fake_app.mkdir(parents=True)
    foreign_cwd.mkdir()

    (fake_app / ".env").write_text(
        "HELPDESK_TEST_A=from_app\nHELPDESK_TEST_OVERRIDE=from_app\n",
        encoding="utf-8",
    )
    (fake_repo / ".env").write_text(
        "HELPDESK_TEST_A=from_repo\nHELPDESK_TEST_B=from_repo\n",
        encoding="utf-8",
    )
    (foreign_cwd / ".env").write_text("HELPDESK_TEST_C=from_cwd\n", encoding="utf-8")

    for var in ("HELPDESK_TEST_A", "HELPDESK_TEST_B", "HELPDESK_TEST_C"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HELPDESK_TEST_OVERRIDE", "from_process")
    monkeypatch.chdir(foreign_cwd)

    agent._load_env(app_dir=fake_app)

    assert os.environ["HELPDESK_TEST_A"] == "from_app"
    assert os.environ["HELPDESK_TEST_B"] == "from_repo"
    assert os.environ["HELPDESK_TEST_C"] == "from_cwd"
    assert os.environ["HELPDESK_TEST_OVERRIDE"] == "from_process"


def test_public_ask_supports_rag_and_multi_agents_with_contract(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())

    ga = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote="Flat cancellation charge for sleeper class is Rs 120 per passenger.")],
        answer="The sleeper cancellation fee is Rs 120 per passenger [Source 1].",
        dont_know=False,
    )

    rag_out = agent.ask("What is sleeper cancellation fee?", agent="rag", llm=_ScriptedLLM(parse_seq=[ga]))
    assert rag_out == {
        "answer": "The sleeper cancellation fee is Rs 120 per passenger.",
        "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }

    multi_out = agent.ask(
        "What is sleeper cancellation fee?",
        agent="multi",
        llm=_ScriptedLLM(text_seq=["sleeper cancellation charge"], parse_seq=[ga]),
    )
    assert multi_out == rag_out


# ---------------------------------------------------------------------------
# 2. Empty/invalid questions & missing API key handling
# ---------------------------------------------------------------------------
def test_empty_and_invalid_questions_across_all_agents(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: pytest.fail("search must not run on empty input"))
    llm = _ScriptedLLM()

    for name in agent.AGENTS:
        inst = agent.make_agent(name, llm=llm)
        for bad_q in ("", "   ", None, 123):
            out = inst.ask(bad_q)  # type: ignore[arg-type]
            assert out == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    assert llm.parse_calls == 0
    assert llm.text_calls == 0
    assert llm.complete_calls == 0


def test_missing_or_blank_api_key_raises_cleanly_for_real_query(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        agent.ask("What is the cancellation charge?")

    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        agent.LLM()

    # Empty questions and router safety refusals still work without a key
    assert agent.ask("")["dont_know"] is True
    ref = agent.ask("DROP TABLE trains;")
    assert ref == {"answer": agent.REFUSAL, "sources": [], "sql": None, "dont_know": True}


# ---------------------------------------------------------------------------
# 3. parse_json(), clean(), and API error separation vs JSON fallback
# ---------------------------------------------------------------------------
def test_parse_json_and_clean_edge_cases():
    assert agent.parse_json('Note {ignore} then {"answer": "ok", "dont_know": false}') == {
        "answer": "ok",
        "dont_know": False,
    }
    assert agent.parse_json('```json\n{"sql": "SELECT 1"}\n```') == {"sql": "SELECT 1"}
    assert agent.parse_json("no json here {broken") == {}

    assert agent.clean("Refund is Rs 120 [Source 1].") == "Refund is Rs 120."
    assert agent.clean("See rule (Sources 1, 2 and 3) or [Source 1, Source 2] !") == "See rule or!"


@pytest.mark.parametrize(
    "exc_cls",
    [openai.AuthenticationError, openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError],
)
def test_api_errors_not_masked_as_json_fallback(monkeypatch: pytest.MonkeyPatch, exc_cls):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())
    llm = _ScriptedLLM(parse_seq=[_make_openai_error(exc_cls)])

    out = agent.ask("What is the sleeper cancellation fee?", agent="rag", llm=llm)
    assert out == {"answer": agent.SERVICE_ERROR, "sources": [], "sql": None, "dont_know": True}
    assert llm.parse_calls == 1
    assert llm.text_calls == 0  # Must NOT trigger duplicate JSON-mode fallback!


def test_router_api_error_in_plan_not_masked_as_json_fallback():
    llm = _ScriptedLLM(parse_seq=[_make_openai_error(openai.AuthenticationError)])
    out = agent.ask("When does train 12002 leave?", agent="router", llm=llm)
    assert out == {"answer": agent.SERVICE_ERROR, "sources": [], "sql": None, "dont_know": True}
    assert llm.parse_calls == 1
    assert llm.text_calls == 0


def test_structured_output_schema_error_falls_back_to_json_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())
    req = httpx.Request("POST", "https://example.invalid")
    resp = httpx.Response(400, request=req)
    bad_req = openai.BadRequestError(message="response_format unsupported", response=resp, body=None)

    plan_json = json.dumps({"needs_documents": True, "doc_query": "sleeper cancel", "sql": ""})
    ga_json = json.dumps({
        "evidence": [{"source": 1, "quote": "Flat cancellation charge for sleeper class is Rs 120 per passenger."}],
        "answer": "Rs 120 per passenger.",
        "dont_know": False,
    })
    llm = _ScriptedLLM(parse_seq=[bad_req, bad_req], text_seq=[plan_json, ga_json])
    out = agent.ask("What is the sleeper cancellation charge?", agent="router", llm=llm)

    assert llm.parse_calls == 2
    assert llm.text_calls == 2
    assert out["dont_know"] is False
    assert out["answer"] == "Rs 120 per passenger."
    assert out["sources"] == [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}]
    assert out["sql"] is None


# ---------------------------------------------------------------------------
# 4. Strict quote grounding, citation validation & retrieval failure safety
# ---------------------------------------------------------------------------
def test_hallucinated_and_fabricated_quotes_force_abstention(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())

    bad_quote = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote="COMPLETELY HALLUCINATED QUOTE")],
        answer="Rs 999.",
        dont_know=False,
    )
    out = agent.ask("What is the charge?", agent="rag", llm=_ScriptedLLM(parse_seq=[bad_quote, bad_quote]))
    assert out == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    # Partial-word spoofing ("for sleepers" instead of "for sleeper") must fail whole-token matching
    partial_word = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote="charge for sleepers")],
        answer="Rs 120.",
        dont_know=False,
    )
    assert agent.grounding_problems(partial_word, _sample_results())

    # Cross-source misattribution: quoting passage 2 while citing source 1 must fail
    wrong_source = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote="00:20 hours to 23:45 hours")],
        answer="00:20 to 23:45 hours.",
        dont_know=False,
    )
    assert agent.grounding_problems(wrong_source, _sample_results())
    out_wrong_src = agent.ask("Hours?", agent="rag", llm=_ScriptedLLM(parse_seq=[wrong_source, wrong_source]))
    assert out_wrong_src == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}


@pytest.mark.parametrize("trivial_quote", ["", "   ", "...", "---", "is", "the", "cancellation", "a b"])
def test_empty_and_trivial_quotes_rejected_by_grounding(monkeypatch: pytest.MonkeyPatch, trivial_quote: str):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())
    ga = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote=trivial_quote)],
        answer="Rs 999.",
        dont_know=False,
    )
    assert agent.grounding_problems(ga, _sample_results())

    out = agent.ask("What is the charge?", agent="rag", llm=_ScriptedLLM(parse_seq=[ga, ga]))
    assert out == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}


def test_missing_citations_and_empty_answer_force_abstention(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())

    # dont_know=False with evidence=[]
    no_ev = agent.GroundedAnswer(evidence=[], answer="Rs 120.", dont_know=False)
    assert agent.grounding_problems(no_ev, _sample_results())
    assert agent.ask("Fee?", agent="rag", llm=_ScriptedLLM(parse_seq=[no_ev, no_ev]))["dont_know"] is True

    # JSON-mode fallback omitting "evidence" field altogether
    req = httpx.Request("POST", "https://example.invalid")
    bad_req = openai.BadRequestError(message="schema", response=httpx.Response(400, request=req), body=None)
    missing_ev_json = json.dumps({"answer": "Rs 120.", "dont_know": False})
    out_missing_ev = agent.ask(
        "Fee?",
        agent="rag",
        llm=_ScriptedLLM(parse_seq=[bad_req, bad_req], text_seq=[missing_ev_json, missing_ev_json]),
    )
    assert out_missing_ev == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    # Valid quote but answer collapses to empty after stripping "[Source 1]"
    empty_ans = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote="Flat cancellation charge for sleeper class is Rs 120 per passenger.")],
        answer="[Source 1]",
        dont_know=False,
    )
    assert agent.ask("Fee?", agent="rag", llm=_ScriptedLLM(parse_seq=[empty_ans, empty_ans]))["dont_know"] is True


@pytest.mark.parametrize("bad_index", [0, -1, 99, True, False, "bad"])
def test_invalid_document_indices_rejected(monkeypatch: pytest.MonkeyPatch, bad_index):
    assert agent.sources_of([bad_index], _sample_results()) == []

    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())
    if isinstance(bad_index, int):
        ga = agent.GroundedAnswer.model_construct(
            evidence=[agent.Evidence.model_construct(source=bad_index, quote="Flat cancellation charge")],
            answer="Rs 120.",
            dont_know=False,
        )
        assert agent.grounding_problems(ga, _sample_results())
        out = agent.ask("Fee?", agent="rag", llm=_ScriptedLLM(parse_seq=[ga, ga]))
        assert out == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}


@pytest.mark.parametrize(
    "bad_meta",
    [
        {"doc": "irctc_refund_rules_jan2026.pdf", "page": 0},
        {"doc": "irctc_refund_rules_jan2026.pdf", "page": -2},
        {"doc": "irctc_refund_rules_jan2026.pdf", "page": "one"},
        {"doc": "irctc_refund_rules_jan2026.pdf", "page": True},
        {"doc": "irctc_refund_rules_jan2026.pdf", "page": None},
        {"doc": "", "page": 1},
        {"page": 1},
    ],
)
def test_invalid_page_and_doc_metadata_rejected(monkeypatch: pytest.MonkeyPatch, bad_meta: dict):
    bad_results = [{"chunk_id": "bad", "text": "Flat cancellation charge for sleeper class is Rs 120.", "metadata": bad_meta}]
    assert agent.sources_of([1], bad_results) == []

    monkeypatch.setattr(agent, "search", lambda *a, **kw: bad_results)
    ga = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote="Flat cancellation charge for sleeper class is Rs 120.")],
        answer="Rs 120.",
        dont_know=False,
    )
    assert agent.grounding_problems(ga, bad_results)
    out = agent.ask("Fee?", agent="rag", llm=_ScriptedLLM(parse_seq=[ga, ga]))
    assert out == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}


def test_conflicting_raw_page_in_json_evidence_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())
    req = httpx.Request("POST", "https://example.invalid")
    bad_req = openai.BadRequestError(message="schema", response=httpx.Response(400, request=req), body=None)
    fake_page_json = json.dumps({
        "evidence": [{"source": 1, "page": 99, "quote": "Flat cancellation charge for sleeper class is Rs 120 per passenger."}],
        "answer": "Rs 120.",
        "dont_know": False,
    })
    out = agent.ask(
        "Fee?",
        agent="rag",
        llm=_ScriptedLLM(parse_seq=[bad_req, bad_req], text_seq=[fake_page_json, fake_page_json]),
    )
    assert out == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}


def test_sql_results_never_represented_as_document_citations(monkeypatch: pytest.MonkeyPatch):
    db_passage = [{
        "chunk_id": "sql_1",
        "text": "Database: 2016 timetable snapshot (historical, not live)\nSQL: SELECT 1\n1 row(s)\ndeparture: 06:15",
        "metadata": {"doc": agent.DB_DOC, "page": None, "doc_date": "2016 timetable snapshot"},
    }]
    assert agent.sources_of([1], db_passage) == []

    # Direct grounded_answer (allow_db=False) must never accept DB_DOC as a document citation
    ga_db = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote="departure: 06:15")],
        answer="Train departs at 06:15.",
        dont_know=False,
    )
    out_direct = agent.grounded_answer(_ScriptedLLM(parse_seq=[ga_db, ga_db]), "When?", db_passage)
    assert out_direct == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    # With allow_db=True (RouterAgent SQL path), grounded answer succeeds with sources=[] (no DB_DOC leakage)
    out_router_helper = agent.grounded_answer(_ScriptedLLM(parse_seq=[ga_db]), "When?", db_passage, allow_db=True)
    assert out_router_helper == {"answer": "Train departs at 06:15.", "sources": [], "sql": None, "dont_know": False}


def test_tool_agent_enforces_quotes_and_citations(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())

    def _tool_msg():
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="search_documents", arguments=json.dumps({"query": "sleeper cancellation"})),
        )
        return SimpleNamespace(
            tool_calls=[tool_call],
            content=None,
            model_dump=lambda exclude_none=True: {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search_documents", "arguments": tool_call.function.arguments}}],
            },
        )

    # 1. No search_documents call at all (seen == []) -> must abstain even with fabricated evidence
    no_search = SimpleNamespace(
        content=json.dumps({
            "evidence": [{"source": 1, "quote": "Flat cancellation charge for sleeper class is Rs 120 per passenger."}],
            "answer": "Rs 120.",
            "dont_know": False,
        }),
        tool_calls=None,
    )
    assert agent.ask("Sleeper fee?", agent="tools", llm=_ScriptedLLM(complete_seq=[no_search])) == {
        "answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True,
    }

    # 2. Unsupported answer: valid used_sources=[1] but missing evidence quotes -> must abstain
    unsupported = SimpleNamespace(
        content=json.dumps({"answer": "Rs 999 [Source 1].", "used_sources": [1], "dont_know": False}),
        tool_calls=None,
    )
    assert agent.ask("Sleeper fee?", agent="tools", llm=_ScriptedLLM(complete_seq=[_tool_msg(), unsupported])) == {
        "answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True,
    }

    # 3. Empty quote or fabricated quote in ToolAgent -> must abstain after retry
    empty_q = SimpleNamespace(
        content=json.dumps({"evidence": [{"source": 1, "quote": ""}], "answer": "Rs 999.", "dont_know": False}),
        tool_calls=None,
    )
    assert agent.ask("Sleeper fee?", agent="tools", llm=_ScriptedLLM(complete_seq=[_tool_msg(), empty_q, empty_q])) == {
        "answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True,
    }

    # 4. Mixed valid + OOB used_sources [1, 99] even alongside valid evidence -> must abstain
    mixed_oob = SimpleNamespace(
        content=json.dumps({
            "evidence": [{"source": 1, "quote": "Flat cancellation charge for sleeper class is Rs 120 per passenger."}],
            "used_sources": [1, 99],
            "answer": "Rs 120.",
            "dont_know": False,
        }),
        tool_calls=None,
    )
    assert agent.ask("Sleeper fee?", agent="tools", llm=_ScriptedLLM(complete_seq=[_tool_msg(), mixed_oob, mixed_oob])) == {
        "answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True,
    }

    # 5. Corrective retry succeeds when first ToolAgent reply forgot evidence and second provides verbatim quote
    fixed_msg = SimpleNamespace(
        content=json.dumps({
            "evidence": [{"source": 1, "quote": "Flat cancellation charge for sleeper class is Rs 120 per passenger."}],
            "answer": "Rs 120 per passenger [Source 1].",
            "dont_know": False,
        }),
        tool_calls=None,
    )
    out_retry_ok = agent.ask("Sleeper fee?", agent="tools", llm=_ScriptedLLM(complete_seq=[_tool_msg(), unsupported, fixed_msg]))
    assert out_retry_ok == {
        "answer": "Rs 120 per passenger.",
        "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }

    # 6. First-try valid grounded ToolAgent answer works cleanly
    out_ok = agent.ask("Sleeper fee?", agent="tools", llm=_ScriptedLLM(complete_seq=[_tool_msg(), fixed_msg]))
    assert out_ok == {
        "answer": "Rs 120 per passenger.",
        "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }


def test_router_agent_grounding_failures_and_valid_html_concise_quotes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent, "search", lambda *a, **kw: _sample_results())

    # 1. Router document route with fabricated quote must abstain cleanly
    doc_plan = agent.RoutePlan(needs_documents=True, doc_query="sleeper fee", sql="")
    bad_ev = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote="Totally invented policy")],
        answer="Rs 999.",
        dont_know=False,
    )
    out_bad_doc = agent.ask("Sleeper fee?", agent="router", llm=_ScriptedLLM(parse_seq=[doc_plan, bad_ev, bad_ev]))
    assert out_bad_doc == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    # 2. Router SQL route where answer uses empty/hallucinated quote must abstain (never fabricate or cite DB_DOC)
    valid_sql = "SELECT number, name, departure FROM trains WHERE number = '12002'"
    sql_plan = agent.RoutePlan(needs_documents=False, doc_query="", sql=valid_sql)
    empty_ev = agent.GroundedAnswer(evidence=[agent.Evidence(source=1, quote="")], answer="Departs 09:99.", dont_know=False)
    out_bad_sql_ev = agent.ask("When does train 12002 leave?", agent="router", llm=_ScriptedLLM(parse_seq=[sql_plan, empty_ev, empty_ev]))
    assert out_bad_sql_ev == {"answer": agent.DONT_KNOW, "sources": [], "sql": valid_sql, "dont_know": True}

    # 3. Legitimate HTML document (page=None) with concise factual quote ("Rs 120") passes grounding
    html_passages = [{
        "chunk_id": "html_1",
        "text": "Flat sleeper cancellation fee is Rs 120 per passenger.",
        "metadata": {"doc": "indianrail_refund_rules_2015.html", "page": None, "doc_date": "2015-11-04"},
    }]
    monkeypatch.setattr(agent, "search", lambda *a, **kw: html_passages)
    html_ev = agent.GroundedAnswer(
        evidence=[agent.Evidence(source=1, quote="Rs 120")],
        answer="The sleeper cancellation fee is Rs 120.",
        dont_know=False,
    )
    out_html_ok = agent.ask("Fee?", agent="rag", llm=_ScriptedLLM(parse_seq=[html_ev]))
    assert out_html_ok == {
        "answer": "The sleeper cancellation fee is Rs 120.",
        "sources": [{"doc": "indianrail_refund_rules_2015.html", "page": None}],
        "sql": None,
        "dont_know": False,
    }


def test_retrieval_failure_returns_safe_contract(monkeypatch: pytest.MonkeyPatch):
    def broken_search(*args, **kwargs):
        raise FileNotFoundError("No index in index/")

    monkeypatch.setattr(agent, "search", broken_search)

    for name in ("rag", "multi", "router"):
        llm = _ScriptedLLM(
            parse_seq=[agent.RoutePlan(needs_documents=True, doc_query="refund", sql="")] if name == "router" else [],
            text_seq=["refund query"] if name == "multi" else [],
        )
        out = agent.ask("What is the refund rule?", agent=name, llm=llm)
        assert out == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}
