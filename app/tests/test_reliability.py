"""Offline reliability suite for Milestone 4: 20 end-to-end scenario and failure-mode tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from helpdesk import agent as agent_mod  # noqa: E402
from helpdesk import sql as sql_mod  # noqa: E402


class _ScenarioLLM:
    def __init__(self, parse_seq=None, text_seq=None, complete_seq=None):
        self.parse_seq = list(parse_seq or [])
        self.text_seq = list(text_seq or [])
        self.complete_seq = list(complete_seq or [])
        self.calls: list[str] = []

    def parse(self, messages, schema, **kwargs):
        self.calls.append(f"parse:{schema.__name__}")
        item = self.parse_seq.pop(0) if self.parse_seq else None
        if isinstance(item, Exception):
            raise item
        return item

    def text(self, messages, **kwargs):
        self.calls.append("text")
        item = self.text_seq.pop(0) if self.text_seq else ""
        if isinstance(item, Exception):
            raise item
        return item

    def complete(self, messages, **kwargs):
        self.calls.append("complete")
        item = self.complete_seq.pop(0) if self.complete_seq else None
        if isinstance(item, Exception):
            raise item
        return item


def _make_api_err(cls, status=400):
    req = httpx.Request("POST", "https://example.invalid/v1beta/openai/chat/completions")
    resp = httpx.Response(status, request=req)
    if cls in (openai.APIConnectionError, openai.APITimeoutError):
        return cls(request=req)
    return cls(message="simulated api error", response=resp, body=None)


# ---------------------------------------------------------------------------
# Prompt & DB alignment regression
# ---------------------------------------------------------------------------
def test_router_prompt_uses_2016_mumbai_central_bct_code():
    assert "Mumbai Central BCT" in agent_mod.ROUTER_PROMPT
    assert "MMCT" not in agent_mod.ROUTER_PROMPT
    assert sql_mod.execute_readonly_query("SELECT code FROM stations WHERE code = 'BCT'")["row_count"] == 1


# ---------------------------------------------------------------------------
# Scenarios 1 - 3: Document & circular precedence paths
# ---------------------------------------------------------------------------
def test_scenarios_1_to_3_document_rules_and_circular_precedence(monkeypatch: pytest.MonkeyPatch):
    doc_p1 = {
        "chunk_id": "irctc_1",
        "text": "Flat cancellation charge for sleeper class is Rs 120 per passenger.",
        "metadata": {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "doc_date": "2026-01-01"},
    }
    vb_p5 = {
        "chunk_id": "vb_5",
        "text": "25% cancellation charge more than 72 hours before departure; within 8 hours no refund.",
        "metadata": {"doc": "rb_cc08_2026_vande_bharat_sleeper_refund.pdf", "page": 5, "doc_date": "2026-01-16"},
    }
    old_2015 = {
        "chunk_id": "old_2015",
        "text": "Clerkage charge for unreserved second class ticket was Rs 15 before revision and Rs 30 after revision.",
        "metadata": {"doc": "indianrail_refund_rules_2015.html", "page": None, "doc_date": "2015-11-04"},
    }
    new_2026 = {
        "chunk_id": "pib_2026",
        "text": "Railways revised the cancellation windows from 48, 12 and 4 hours to 72, 24 and 8 hours.",
        "metadata": {"doc": "pib_2026-03-24_reform_express.html", "page": None, "doc_date": "2026-03-24"},
    }

    # 1. Simple document question
    monkeypatch.setattr(agent_mod, "search", lambda *a, **kw: [doc_p1])
    out1 = agent_mod.ask(
        "What is sleeper cancellation fee?",
        agent="router",
        llm=_ScenarioLLM(parse_seq=[
            agent_mod.RoutePlan(needs_documents=True, doc_query="sleeper cancellation", sql=""),
            agent_mod.GroundedAnswer(
                evidence=[agent_mod.Evidence(source=1, quote="Flat cancellation charge for sleeper class is Rs 120 per passenger.")],
                answer="Sleeper class flat cancellation charge is Rs 120 per passenger.",
                dont_know=False,
            ),
        ]),
    )
    assert out1 == {"answer": "Sleeper class flat cancellation charge is Rs 120 per passenger.", "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}], "sql": None, "dont_know": False}

    # 2. Exact railway rule question
    monkeypatch.setattr(agent_mod, "search", lambda *a, **kw: [vb_p5])
    out2 = agent_mod.ask(
        "What is Vande Bharat Sleeper rule within 8 hours?",
        agent="router",
        llm=_ScenarioLLM(parse_seq=[
            agent_mod.RoutePlan(needs_documents=True, doc_query="vande bharat sleeper 8 hours", sql=""),
            agent_mod.GroundedAnswer(
                evidence=[agent_mod.Evidence(source=1, quote="within 8 hours no refund")],
                answer="No refund is granted within 8 hours of departure.",
                dont_know=False,
            ),
        ]),
    )
    assert out2["dont_know"] is False
    assert out2["sources"] == [{"doc": "rb_cc08_2026_vande_bharat_sleeper_refund.pdf", "page": 5}]

    # 3. Older vs newer circular comparison / precedence
    monkeypatch.setattr(agent_mod, "search", lambda *a, **kw: [new_2026, old_2015])
    out3 = agent_mod.ask(
        "What are the newer cancellation time windows in March 2026 compared with the older rule?",
        agent="router",
        llm=_ScenarioLLM(parse_seq=[
            agent_mod.RoutePlan(needs_documents=True, doc_query="cancellation windows March 2026", sql=""),
            agent_mod.GroundedAnswer(
                evidence=[agent_mod.Evidence(source=1, quote="from 48, 12 and 4 hours to 72, 24 and 8 hours.")],
                answer="The March 2026 PIB reform revised the 48/12/4-hour windows to 72, 24, and 8 hours.",
                dont_know=False,
            ),
        ]),
    )
    assert out3["sources"] == [{"doc": "pib_2026-03-24_reform_express.html", "page": None}]
    assert out3["dont_know"] is False


# ---------------------------------------------------------------------------
# Scenarios 4 - 9: Real SQL timetable + mixed SQL/document scenarios
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question,sql_query,quote,answer",
    [
        # 4. Train number question
        (
            "Which train is 01101?",
            "SELECT number, name, from_station_code, to_station_code FROM trains WHERE number = '01101'",
            "number: 01101; name: Mumbai LTT - Gwalior (Weekly) Special",
            "According to the 2016 timetable snapshot, train 01101 is Mumbai LTT - Gwalior (Weekly) Special.",
        ),
        # 5. Station lookup (BCT historical code)
        (
            "What is the station code for Mumbai Central?",
            "SELECT code, name FROM stations WHERE code = 'BCT'",
            "code: BCT; name: Mumbai Central",
            "In the 2016 timetable database, Mumbai Central has station code BCT.",
        ),
        # 6. Train route question
        (
            "Where does train 12951 stop?",
            "SELECT st.station_code, s.name FROM stops st JOIN stations s ON s.code = st.station_code WHERE st.train_number = '12951' ORDER BY st.stop_sequence",
            "station_code: BRC; name: VADODARA JN",
            "According to the 2016 timetable snapshot, train 12951 stops at BCT, ST, BRC, RTM, KOTA, and NDLS.",
        ),
        # 7. Intermediate station timing
        (
            "When does train 12951 reach and leave Vadodara Jn?",
            "SELECT arrival, departure, halt_min FROM stops WHERE train_number = '12951' AND station_code = 'BRC'",
            "arrival: 21:06; departure: 21:22; halt_min: 16",
            "According to the 2016 timetable snapshot, train 12951 reaches BRC at 21:06 and leaves at 21:22.",
        ),
        # 8. Overnight train journey
        (
            "How long does 12315 take from JP to UDZ?",
            (
                "SELECT a.departure, b.arrival, "
                "(b.day - a.day)*1440 + (CAST(substr(b.arrival,1,2) AS INTEGER)*60+CAST(substr(b.arrival,4,2) AS INTEGER)) "
                "- (CAST(substr(a.departure,1,2) AS INTEGER)*60+CAST(substr(a.departure,4,2) AS INTEGER)) AS travel_time_min "
                "FROM stops a JOIN stops b ON b.train_number=a.train_number AND b.stop_sequence>a.stop_sequence "
                "WHERE a.train_number='12315' AND a.station_code='JP' AND b.station_code='UDZ'"
            ),
            "departure: 17:20; arrival: 03:00; travel_time_min: 580",
            "According to the 2016 timetable snapshot, train 12315 takes 580 minutes (17:20 to 03:00).",
        ),
    ],
)
def test_scenarios_4_to_8_timetable_sql_paths(question: str, sql_query: str, quote: str, answer: str):
    llm = _ScenarioLLM(parse_seq=[
        agent_mod.RoutePlan(needs_documents=False, doc_query="", sql=sql_query),
        agent_mod.GroundedAnswer(evidence=[agent_mod.Evidence(source=1, quote=quote)], answer=answer, dont_know=False),
    ])
    out = agent_mod.ask(question, agent="router", llm=llm)
    assert out["dont_know"] is False
    assert out["sql"] == sql_query
    assert out["sources"] == []
    assert "2016" in out["answer"]


def test_scenario_9_mixed_document_and_timetable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent_mod, "search", lambda *a, **kw: [{
        "chunk_id": "arp_1",
        "text": "Advance reservation period (ARP) is 60 days excluding the date of journey.",
        "metadata": {"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1, "doc_date": "2024-10-16"},
    }])
    sql_q = "SELECT number, name, departure FROM trains WHERE number = '12002'"
    llm = _ScenarioLLM(parse_seq=[
        agent_mod.RoutePlan(needs_documents=True, doc_query="advance reservation period", sql=sql_q),
        agent_mod.GroundedAnswer(
            evidence=[
                agent_mod.Evidence(source=1, quote="name: New Delhi-Bhopal Shatabdi Express; departure: 06:15"),
                agent_mod.Evidence(source=2, quote="60 days excluding the date of journey."),
            ],
            answer="In the 2016 timetable snapshot, train 12002 departs at 06:15, and the ARP is 60 days.",
            dont_know=False,
        ),
    ])
    out = agent_mod.ask("When does 12002 depart and what is the ARP?", agent="router", llm=llm)
    assert out["dont_know"] is False
    assert out["sql"] == sql_q
    assert out["sources"] == [{"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1}]


# ---------------------------------------------------------------------------
# Scenarios 10 - 16: Empty SQL/retrieval, unsupported live data & unsafe SQL
# ---------------------------------------------------------------------------
def test_scenarios_10_to_16_empty_sources_unsupported_and_unsafe(monkeypatch: pytest.MonkeyPatch):
    # 10. SQL returns 0 rows
    zero_sql = "SELECT number FROM trains WHERE UPPER(name) LIKE '%VANDE BHARAT%'"
    out10 = agent_mod.ask(
        "When does Vande Bharat depart?",
        agent="router",
        llm=_ScenarioLLM(parse_seq=[
            agent_mod.RoutePlan(needs_documents=False, doc_query="", sql=zero_sql),
            agent_mod.GroundedAnswer(evidence=[], answer="Not in 2016 timetable.", dont_know=True),
        ]),
    )
    assert out10["dont_know"] is True
    assert out10["sql"] == zero_sql
    assert out10["sources"] == []

    # 11. Document retrieval returns no evidence
    monkeypatch.setattr(agent_mod, "search", lambda *a, **kw: [])
    out11 = agent_mod.ask(
        "What is the pet rule?",
        agent="router",
        llm=_ScenarioLLM(parse_seq=[agent_mod.RoutePlan(needs_documents=True, doc_query="pet rule", sql="")]),
    )
    assert out11 == {"answer": agent_mod.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    # 12. Both SQL fails AND documents return empty -> abstain safely without fabrication
    bad_sql = "SELECT nope FROM trains"
    out12 = agent_mod.ask(
        "Depart time and refund?",
        agent="router",
        llm=_ScenarioLLM(
            parse_seq=[agent_mod.RoutePlan(needs_documents=True, doc_query="refund", sql=bad_sql)],
            text_seq=[bad_sql],
        ),
    )
    assert out12 == {"answer": agent_mod.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    # 13-15. Unsupported live train status / PNR / seat availability
    monkeypatch.setattr(agent_mod, "search", lambda *a, **kw: [{
        "chunk_id": "c1",
        "text": "Flat cancellation charge for sleeper class is Rs 120 per passenger.",
        "metadata": {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "doc_date": "2026-01-01"},
    }])
    for q in (
        "What is the live running status of 12951?",
        "What is the status of PNR 4512345678?",
        "How many seats are available on train 12002 tomorrow?",
    ):
        llm = _ScenarioLLM(parse_seq=[
            agent_mod.RoutePlan(needs_documents=True, doc_query=q, sql=""),
            agent_mod.GroundedAnswer(evidence=[], answer=agent_mod.DONT_KNOW, dont_know=True),
        ])
        out = agent_mod.ask(q, agent="router", llm=llm)
        assert out["dont_know"] is True and out["sql"] is None and out["sources"] == []

    # 16. Unsafe SQL modification refused without calling LLM
    llm_unused = _ScenarioLLM()
    out16 = agent_mod.ask("DELETE FROM trains WHERE number = '12002';", agent="router", llm=llm_unused)
    assert out16 == {"answer": agent_mod.REFUSAL, "sources": [], "sql": None, "dont_know": True}
    assert llm_unused.calls == []


# ---------------------------------------------------------------------------
# Scenarios 17 - 20: Malformed LLM/JSON responses & Gemini API/network errors
# ---------------------------------------------------------------------------
def test_scenarios_17_to_20_malformed_responses_and_api_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent_mod, "search", lambda *a, **kw: [{
        "chunk_id": "c1",
        "text": "Flat cancellation charge for sleeper class is Rs 120 per passenger.",
        "metadata": {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "doc_date": "2026-01-01"},
    }])

    # 17. Malformed LLM outputs: None message in ToolAgent & None string in MultiQueryAgent.rewrite
    assert agent_mod.ask("Fee?", agent="tools", llm=_ScenarioLLM(complete_seq=[None])) == {
        "answer": agent_mod.DONT_KNOW, "sources": [], "sql": None, "dont_know": True,
    }
    ga_ok = agent_mod.GroundedAnswer(
        evidence=[agent_mod.Evidence(source=1, quote="Flat cancellation charge for sleeper class is Rs 120 per passenger.")],
        answer="Rs 120.",
        dont_know=False,
    )
    multi_out = agent_mod.ask("Fee?", agent="multi", llm=_ScenarioLLM(text_seq=[None], parse_seq=[ga_ok]))
    assert multi_out["dont_know"] is False and multi_out["answer"] == "Rs 120."

    # 18. Invalid JSON response in JSON fallback & ToolAgent
    bad_req = _make_api_err(openai.BadRequestError, 400)
    out18_rag = agent_mod.ask(
        "Fee?",
        agent="rag",
        llm=_ScenarioLLM(parse_seq=[bad_req, bad_req], text_seq=["{not valid json", "still bad"]),
    )
    assert out18_rag == {"answer": agent_mod.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    out18_tools = agent_mod.ask(
        "Fee?",
        agent="tools",
        llm=_ScenarioLLM(complete_seq=[SimpleNamespace(content="plain non-json text", tool_calls=None)]),
    )
    assert out18_tools == {"answer": agent_mod.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    # 19 & 20. Auth failure, rate-limit, network error, AND BadRequestError on fallback/rewrite
    # must return SERVICE_ERROR (not DONT_KNOW) across all agents
    auth_err = _make_api_err(openai.AuthenticationError, 401)
    rate_err = _make_api_err(openai.RateLimitError, 429)
    net_err = _make_api_err(openai.APIConnectionError)
    fallback_bad_req = _make_api_err(openai.BadRequestError, 400)

    assert agent_mod.ask("Fee?", agent="rag", llm=_ScenarioLLM(parse_seq=[auth_err]))["answer"] == agent_mod.SERVICE_ERROR
    assert agent_mod.ask("Fee?", agent="router", llm=_ScenarioLLM(parse_seq=[rate_err]))["answer"] == agent_mod.SERVICE_ERROR
    assert agent_mod.ask("Fee?", agent="multi", llm=_ScenarioLLM(text_seq=[net_err]))["answer"] == agent_mod.SERVICE_ERROR
    assert agent_mod.ask("Fee?", agent="tools", llm=_ScenarioLLM(complete_seq=[net_err]))["answer"] == agent_mod.SERVICE_ERROR
    assert agent_mod.ask(
        "Fee?",
        agent="rag",
        llm=_ScenarioLLM(parse_seq=[bad_req], text_seq=[fallback_bad_req]),
    )["answer"] == agent_mod.SERVICE_ERROR
