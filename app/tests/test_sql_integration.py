"""Automated pytest suite for read-only SQL integration and RouterAgent wiring.

All tests are deterministic and run completely offline without Gemini API keys.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import helpdesk.agent as agent  # noqa: E402
import helpdesk.sql as sql  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def temp_writable_db(tmp_path: Path) -> Path:
    """Create a small writable SQLite DB to prove authorizer/mode=ro protect it."""
    db_file = tmp_path / "temp_rail.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO items (id, name) VALUES (?, ?)",
        [(1, "Shatabdi"), (2, "Rajdhani")],
    )
    conn.commit()
    conn.close()
    return db_file


class _FakeLLM:
    """Minimal offline LLM double for testing RouterAgent SQL flow."""

    def __init__(self, plan: agent.RoutePlan, fixed_sql: str = "", answer: agent.GroundedAnswer | None = None):
        self._plan = plan
        self._fixed_sql = fixed_sql
        self._answer = answer
        self.calls: list[str] = []
        self.messages_log: list[tuple[str, list[dict]]] = []

    def parse(self, messages, schema, **kwargs):
        self.calls.append(f"parse:{schema.__name__}")
        self.messages_log.append((schema.__name__, messages))
        if schema is agent.RoutePlan:
            return self._plan
        if schema is agent.GroundedAnswer:
            return self._answer
        return None

    def text(self, messages, **kwargs):
        self.calls.append("text")
        self.messages_log.append(("text", messages))
        return self._fixed_sql


# ---------------------------------------------------------------------------
# 1 & 2. Path resolution, shim consolidation, and connectivity
# ---------------------------------------------------------------------------
def test_db_path_resolution_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    expected_db = (REPO_ROOT / "database" / "rail_timetable.db").resolve()
    assert expected_db.is_file()
    assert sql.DEFAULT_DB_PATH.resolve() == expected_db
    assert sql.DB_PATH.resolve() == expected_db

    monkeypatch.chdir(tmp_path)
    assert sql.get_db_path() == expected_db
    assert sql.get_schema_path() == expected_db.parent / "schema.md"

    result = sql.find_train("12002")
    assert "error" not in result
    assert result["row_count"] == 1


def test_db_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, temp_writable_db: Path):
    monkeypatch.setenv(sql.DB_PATH_ENV_VAR, str(temp_writable_db))
    assert sql.get_db_path() == temp_writable_db.resolve()

    res = sql.execute_readonly_query("SELECT id, name FROM items ORDER BY id")
    assert res["rows"] == [{"id": 1, "name": "Shatabdi"}, {"id": 2, "name": "Rajdhani"}]


def test_root_helpdesk_sql_shim_delegates_to_authoritative_module():
    # Simulate running from repository root without app/ on sys.path
    code = (
        "import app.helpdesk.sql as auth_sql, helpdesk.sql as shim_sql; "
        "assert shim_sql.execute_readonly_query is auth_sql.execute_readonly_query; "
        "assert shim_sql.find_train is auth_sql.find_train; "
        "assert shim_sql.DB_PATH.resolve() == auth_sql.DB_PATH.resolve(); "
        "res = shim_sql.find_train('12002'); "
        "assert res['row_count'] == 1"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_database_connectivity_and_expected_tables():
    res = sql.execute_readonly_query(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
    )
    assert "error" not in res
    names = {row["name"] for row in res["rows"]}
    assert {"stations", "train_types", "trains", "schedules", "stops"}.issubset(names)


# ---------------------------------------------------------------------------
# 3-6 & 12. Helper queries & overnight calculation against real DB
# ---------------------------------------------------------------------------
def test_find_train_exact_and_leading_zero():
    shatabdi = sql.find_train("12002")
    assert "error" not in shatabdi
    assert shatabdi["row_count"] == 1
    row = shatabdi["rows"][0]
    assert row["number"] == "12002"
    assert row["name"] == "New Delhi-Bhopal Shatabdi Express"
    assert row["from_station_code"] == "NDLS"
    assert row["to_station_code"] == "BPL"
    assert row["departure"] == "06:15"
    assert row["arrival"] == "14:05"

    leading_zero = sql.find_train("04601")
    assert "error" not in leading_zero
    assert leading_zero["row_count"] == 1
    assert leading_zero["rows"][0]["number"] == "04601"

    missing = sql.find_train("99999")
    assert "error" not in missing
    assert missing["row_count"] == 0
    assert missing["rows"] == []


def test_find_station_matches_and_escapes_wildcards():
    ndls = sql.find_station("new delhi")
    assert "error" not in ndls
    assert any(r["code"] == "NDLS" and r["name"] == "NEW DELHI" for r in ndls["rows"])

    bhopal = sql.find_station("BHOPAL")
    assert "error" not in bhopal
    assert {r["code"] for r in bhopal["rows"]} >= {"BPL", "BPKA", "NSZ"}

    # Wildcard characters must be treated literally, not as match-all patterns
    wildcard = sql.find_station("%")
    assert "error" not in wildcard
    assert wildcard["row_count"] == 0

    underscore = sql.find_station("_")
    assert "error" not in underscore
    assert underscore["row_count"] == 0


def test_find_trains_between_stations():
    res = sql.find_trains_between("jp", "udz")
    assert "error" not in res
    assert res["row_count"] == 2
    numbers = {r["number"] for r in res["rows"]}
    assert numbers == {"12315", "12965"}

    empty = sql.find_trains_between("NDLS", "ZZZZZ")
    assert "error" not in empty
    assert empty["row_count"] == 0


def test_get_train_schedule():
    sched_12002 = sql.get_train_schedule("12002")
    assert "error" not in sched_12002
    assert sched_12002["row_count"] == 87
    assert sched_12002["truncated"] is False
    assert sched_12002["rows"][0]["station_code"] == "NDLS"
    assert sched_12002["rows"][-1]["station_code"] == "BPL"

    sched_12951 = sql.get_train_schedule("12951")
    assert "error" not in sched_12951
    assert sched_12951["row_count"] == 202
    assert sched_12951["truncated"] is False


def test_overnight_journey_calculation_real_schema():
    query = """
    SELECT
        a.departure,
        b.arrival,
        a.day AS departure_day,
        b.day AS arrival_day,
        (b.day - a.day) * 1440
        + (CAST(substr(b.arrival, 1, 2) AS INTEGER) * 60 + CAST(substr(b.arrival, 4, 2) AS INTEGER))
        - (CAST(substr(a.departure, 1, 2) AS INTEGER) * 60 + CAST(substr(a.departure, 4, 2) AS INTEGER))
        AS travel_time_min
    FROM stops a
    JOIN stops b
        ON b.train_number = a.train_number
        AND b.stop_sequence > a.stop_sequence
    WHERE a.train_number = ?
      AND a.station_code = ?
      AND b.station_code = ?
    """
    res = sql.execute_readonly_query(query, ("12315", "JP", "UDZ"))
    assert "error" not in res
    assert res["rows"] == [{
        "departure": "17:20",
        "arrival": "03:00",
        "departure_day": 2,
        "arrival_day": 3,
        "travel_time_min": 580,
    }]


# ---------------------------------------------------------------------------
# 7, 8, 9, 10, 11. Parameterization, security, limits, errors, missing DB
# ---------------------------------------------------------------------------
def test_parameterized_queries_and_sqli_resistance():
    res = sql.execute_readonly_query("SELECT number, name FROM trains WHERE number = ?", ("12002",))
    assert res["row_count"] == 1

    for payload in [
        "12002' OR '1'='1",
        "12002'; DROP TABLE trains; --",
        "' UNION SELECT code, name FROM stations --",
    ]:
        inj_train = sql.find_train(payload)
        assert "error" not in inj_train
        assert inj_train["row_count"] == 0

        inj_station = sql.find_station(payload)
        assert "error" not in inj_station
        assert inj_station["row_count"] == 0

    bad_params = sql.execute_readonly_query("SELECT number FROM trains WHERE number = ?", "12002")
    assert "error" in bad_params


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "INSERT INTO items (id, name) VALUES (3, 'Evil')",
        "UPDATE items SET name = 'Evil' WHERE id = 1",
        "DELETE FROM items WHERE id = 1",
        "DROP TABLE items",
        "CREATE TABLE hacked (x INT)",
        "ALTER TABLE items ADD COLUMN evil TEXT",
        "ATTACH DATABASE ':memory:' AS extra",
        "DETACH DATABASE main",
        "SAVEPOINT sp",
        "EXPLAIN SELECT 1",
        "/* comment */ DELETE FROM items",
        "PRAGMA query_only = OFF",
        "PRAGMA table_info(items)",
        "VACUUM",
        "REPLACE INTO items (id, name) VALUES (1, 'Evil')",
        "SELECT load_extension('evil')",
        "SELECT 1; DROP TABLE items",
        "SELECT 1; SELECT 2",
        "WITH cte AS (SELECT 1) INSERT INTO items (id, name) VALUES (9, 'X')",
        "WITH cte AS (SELECT 1) DELETE FROM items",
        "WITH cte AS (SELECT 1) UPDATE items SET name = 'X'",
    ],
)
def test_readonly_security_blocks_writes_on_writable_temp_db(temp_writable_db: Path, malicious_sql: str):
    res = sql.execute_readonly_query(malicious_sql, db_path=temp_writable_db)
    assert "error" in res

    verify = sql.execute_readonly_query("SELECT id, name FROM items ORDER BY id", db_path=temp_writable_db)
    assert verify["rows"] == [{"id": 1, "name": "Shatabdi"}, {"id": 2, "name": "Rajdhani"}]


def test_recursive_cte_timeout_protection(temp_writable_db: Path):
    infinite_cte = (
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt) "
        "SELECT COUNT(*) AS n FROM cnt"
    )
    res = sql.execute_readonly_query(infinite_cte, db_path=temp_writable_db, timeout_sec=0.1)
    assert "error" in res
    assert "timed out" in res["error"].lower()


def test_sql_error_mentioning_interrupted_is_not_misclassified_as_timeout():
    res = sql.execute_readonly_query("SELECT interrupted FROM trains")
    assert "error" in res
    assert "timed out" not in res["error"].lower()
    assert "no such column: interrupted" in res["error"].lower()

    # RouterAgent.run_sql() must still attempt SQL repair on this normal SQL error
    llm = _FakeLLM(
        plan=agent.RoutePlan(needs_documents=False, doc_query="", sql="SELECT interrupted FROM trains"),
        fixed_sql="SELECT number, name, departure FROM trains WHERE number = '12002'",
    )
    router = agent.RouterAgent(llm=llm)
    repaired_sql, repaired_res = router.run_sql("Find train 12002", "SELECT interrupted FROM trains")
    assert "text" in llm.calls
    assert repaired_sql == "SELECT number, name, departure FROM trains WHERE number = '12002'"
    assert "error" not in repaired_res
    assert repaired_res["row_count"] == 1


def test_invalid_queries_and_helper_validation():
    for bad_query in ["", "   ", None, "-- comment only", "/* block */", "SELECT * FROM no_such_table"]:
        res = sql.execute_readonly_query(bad_query)  # type: ignore[arg-type]
        assert "error" in res

    assert "error" in sql.find_train("")
    assert "error" in sql.find_train(None)  # type: ignore[arg-type]
    assert "error" in sql.find_station("")
    assert "error" in sql.find_trains_between("", "UDZ")
    assert "error" in sql.get_train_schedule("")


def test_row_limit_enforcement():
    res = sql.execute_readonly_query("SELECT code FROM stations ORDER BY code", max_rows=5)
    assert "error" not in res
    assert res["row_count"] == 5
    assert res["truncated"] is True

    for invalid_limit in [0, -5, sql.MAX_ROW_LIMIT + 1, True, "10"]:
        err = sql.execute_readonly_query("SELECT 1", max_rows=invalid_limit)  # type: ignore[arg-type]
        assert "error" in err


def test_missing_database_fails_safely_without_creating_file(tmp_path: Path):
    missing_db = tmp_path / "does_not_exist.db"
    assert not missing_db.exists()

    res = sql.execute_readonly_query("SELECT 1", db_path=missing_db)
    assert "error" in res
    assert "not found" in res["error"].lower()
    assert not missing_db.exists()

    empty_res = sql.execute_readonly_query("SELECT 1", db_path="")
    assert "error" in empty_res
    assert "empty" in empty_res["error"].lower()


# ---------------------------------------------------------------------------
# RouterAgent offline integration & historical 2016 snapshot guarantees
# ---------------------------------------------------------------------------
def test_agent_uses_authoritative_sql_without_monkey_patching():
    assert not hasattr(agent, "timetable")
    schema_text = agent.timetable_schema()
    assert "2016 snapshot" in schema_text
    assert "trains" in schema_text and "schedules" in schema_text and "stops" in schema_text


def test_router_agent_executes_sql_and_returns_contract():
    valid_sql = "SELECT number, name, departure, arrival FROM trains WHERE number = '12002'"
    llm = _FakeLLM(
        plan=agent.RoutePlan(needs_documents=False, doc_query="", sql=valid_sql),
        answer=agent.GroundedAnswer(
            evidence=[agent.Evidence(source=1, quote="name: New Delhi-Bhopal Shatabdi Express; departure: 06:15")],
            answer="According to the 2016 timetable snapshot, train 12002 departs at 06:15.",
            dont_know=False,
        ),
    )
    router = agent.RouterAgent(llm=llm)
    out = router.ask("When does train 12002 depart?")

    assert set(out.keys()) == {"answer", "sources", "sql", "dont_know"}
    assert out["sql"] == valid_sql
    assert out["sources"] == []
    assert out["dont_know"] is False
    assert "2016" in out["answer"]


def test_router_agent_repairs_sql_and_revalidates_readonly():
    bad_sql = "SELECT nonexistent_col FROM trains WHERE number = '12002'"
    fixed_sql = "```sql\nSELECT number, name, departure FROM trains WHERE number = '12002'\n```"
    clean_fixed_sql = "SELECT number, name, departure FROM trains WHERE number = '12002'"
    llm = _FakeLLM(
        plan=agent.RoutePlan(needs_documents=False, doc_query="", sql=bad_sql),
        fixed_sql=fixed_sql,
        answer=agent.GroundedAnswer(
            evidence=[agent.Evidence(source=1, quote="number: 12002; name: New Delhi-Bhopal Shatabdi Express")],
            answer="In the 2016 timetable snapshot, train 12002 departs at 06:15.",
            dont_know=False,
        ),
    )
    router = agent.RouterAgent(llm=llm)
    out = router.ask("What time does train 12002 depart?")

    assert out["sql"] == clean_fixed_sql
    assert out["dont_know"] is False


def test_router_agent_blocks_repaired_write_sql_and_never_fabricates():
    bad_sql = "SELECT bad_col FROM trains"
    malicious_fix = "DELETE FROM trains WHERE number = '12002'"
    llm = _FakeLLM(
        plan=agent.RoutePlan(needs_documents=False, doc_query="", sql=bad_sql),
        fixed_sql=malicious_fix,
    )
    router = agent.RouterAgent(llm=llm)
    out = router.ask("Show train 12002 schedule")

    assert out["sql"] is None
    assert out["dont_know"] is True
    assert sql.find_train("12002")["row_count"] == 1


def test_router_agent_preserves_sql_when_zero_rows_and_abstains():
    zero_sql = "SELECT number, name FROM trains WHERE name LIKE '%VANDE BHARAT%'"
    llm = _FakeLLM(
        plan=agent.RoutePlan(needs_documents=False, doc_query="", sql=zero_sql),
        answer=agent.GroundedAnswer(
            evidence=[],
            answer="I don't know. No Vande Bharat train exists in the 2016 timetable snapshot.",
            dont_know=True,
        ),
    )
    router = agent.RouterAgent(llm=llm)
    out = router.ask("When does Vande Bharat Express depart?")

    assert out["sql"] == zero_sql
    assert out["dont_know"] is True
    assert out["sources"] == []


def test_router_agent_failed_sql_never_fabricates_and_handles_empty_input():
    llm = _FakeLLM(
        plan=agent.RoutePlan(needs_documents=False, doc_query="", sql="SELECT bad_col FROM trains"),
        fixed_sql="SELECT still_bad_col FROM trains",
    )
    router = agent.RouterAgent(llm=llm)

    failed = router.ask("When does train 12002 leave?")
    assert failed["sql"] is None
    assert failed["dont_know"] is True
    assert failed["sources"] == []
    assert "parse:GroundedAnswer" not in llm.calls

    for empty_q in ["", "   ", None]:
        out = router.ask(empty_q)  # type: ignore[arg-type]
        assert out == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}


def test_rows_text_truncates_cleanly_at_line_boundary():
    sample = {
        "columns": ["a"],
        "rows": [{"a": "value_" + str(i)} for i in range(10)],
        "truncated": False,
        "row_count": 10,
    }
    text = agent.rows_text(sample, max_chars=45)
    assert "... (output truncated)" in text
    assert "a: value_9" not in text


def test_router_agent_mixed_sql_failure_uses_documents_only(monkeypatch: pytest.MonkeyPatch):
    search_calls: list[tuple[str, int, object]] = []

    def fake_search(question, k=5, extra_queries=None, **kwargs):
        search_calls.append((question, k, extra_queries))
        return [{
            "chunk_id": "doc_0001",
            "text": "Flat cancellation charge for sleeper class is Rs 120 per passenger.",
            "metadata": {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "doc_date": "2026-01-01"},
        }]

    monkeypatch.setattr(agent, "search", fake_search)

    llm = _FakeLLM(
        plan=agent.RoutePlan(
            needs_documents=True,
            doc_query="sleeper class cancellation charge",
            sql="SELECT nonexistent_col FROM trains WHERE number = '12002'",
        ),
        fixed_sql="SELECT still_missing FROM trains",
        answer=agent.GroundedAnswer(
            evidence=[agent.Evidence(source=1, quote="Flat cancellation charge for sleeper class is Rs 120 per passenger.")],
            answer="Sleeper class cancellation charge is Rs 120 per passenger; timetable details could not be fetched.",
            dont_know=False,
        ),
    )
    router = agent.RouterAgent(llm=llm)
    out = router.ask("When does train 12002 depart and what is sleeper cancellation charge?")

    assert len(search_calls) == 1
    assert out["sql"] is None
    assert out["dont_know"] is False
    assert out["sources"] == [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}]

    # Verify GroundedAnswer passages received only the document passage and zero fabricated SQL data
    ga_messages = [m for name, m in llm.messages_log if name == "GroundedAnswer"][0]
    passages_text = ga_messages[1]["content"]
    assert "doc: timetable database" not in passages_text
    assert "SQL:" not in passages_text
    assert "Flat cancellation charge" in passages_text


def test_router_agent_mixed_sql_and_documents_success(monkeypatch: pytest.MonkeyPatch):
    search_calls: list[tuple[str, int, object]] = []

    def fake_search(question, k=5, extra_queries=None, **kwargs):
        search_calls.append((question, k, extra_queries))
        return [{
            "chunk_id": "doc_0002",
            "text": "If confirmed ticket is cancelled more than 48 hours before departure, flat charge applies.",
            "metadata": {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "doc_date": "2026-01-01"},
        }]

    monkeypatch.setattr(agent, "search", fake_search)

    valid_sql = "SELECT number, name, departure FROM trains WHERE number = '12002'"
    llm = _FakeLLM(
        plan=agent.RoutePlan(
            needs_documents=True,
            doc_query="confirmed ticket cancellation 48 hours",
            sql=valid_sql,
        ),
        answer=agent.GroundedAnswer(
            evidence=[
                agent.Evidence(source=1, quote="name: New Delhi-Bhopal Shatabdi Express; departure: 06:15"),
                agent.Evidence(source=2, quote="If confirmed ticket is cancelled more than 48 hours before departure"),
            ],
            answer="According to the 2016 timetable snapshot, train 12002 departs at 06:15, and cancelling more than 48 hours before departure incurs a flat charge.",
            dont_know=False,
        ),
    )
    router = agent.RouterAgent(llm=llm)
    out = router.ask("When does train 12002 leave and what is the rule if cancelled >48 hours before?")

    assert len(search_calls) == 1
    assert search_calls[0][2] == ["confirmed ticket cancellation 48 hours"]
    assert out["sql"] == valid_sql
    assert out["dont_know"] is False
    # Internal timetable DB passage must not leak into document citations
    assert out["sources"] == [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}]


def test_unsafe_write_requests_refused_without_llm_and_no_false_positives():
    llm = _FakeLLM(
        plan=agent.RoutePlan(needs_documents=False, doc_query="", sql="SELECT number, name, departure FROM trains WHERE number = '12002'"),
        answer=agent.GroundedAnswer(
            evidence=[agent.Evidence(source=1, quote="name: New Delhi-Bhopal Shatabdi Express; departure: 06:15")],
            answer="According to the 2016 timetable snapshot, train 12002 departs at 06:15.",
            dont_know=False,
        ),
    )
    router = agent.RouterAgent(llm=llm)

    for attack in [
        "Update the timetable database to add train 99999.",
        "Delete train 12002 from the database.",
        "Modify the records in the trains table.",
        "Insert a fake train into the timetable database.",
        "Delete train 12002 from your database",
        "Update the timetable to add train 99999",
        "Modify the schedule of train 12002",
        "Insert a fake train into trains table",
        "DROP TABLE trains;",
        "PRAGMA writable_schema=ON;",
    ]:
        out = router.ask(attack)
        assert out == {"answer": agent.REFUSAL, "sources": [], "sql": None, "dont_know": True}
    assert llm.calls == []

    # Legitimate passenger questions about schedule/timetable changes must reach normal routing
    legitimate_questions = [
        "Did Indian Railways change the schedule of train 12002?",
        "When does IRCTC update the timetable?",
        "If Railways cancel or alter the timetable, do I get a refund?",
    ]
    for q in legitimate_questions:
        calls_before = len(llm.calls)
        out = router.ask(q)
        assert out["answer"] != agent.REFUSAL
        assert "parse:RoutePlan" in llm.calls[calls_before:]

    # Verify zero false positives on all policy questions in both eval datasets
    for eval_file in (APP_DIR / "eval" / "test_set.json", APP_DIR / "eval" / "test_set_hard.json"):
        for item in json.loads(eval_file.read_text(encoding="utf-8")):
            assert not agent.UNSAFE_RE.search(item["question"]), f"False positive on {item['id']}: {item['question']}"
