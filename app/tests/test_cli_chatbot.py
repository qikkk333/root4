"""Offline tests for the redesigned interactive terminal UI and CLI backward compatibility."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import helpdesk  # noqa: E402
import helpdesk.agent as agent  # noqa: E402
import helpdesk.search as search  # noqa: E402
import helpdesk.terminal_ui as ui  # noqa: E402

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _make_reader(responses):
    """Build a test input reader from strings or Exception instances."""
    it = iter(responses)

    def _read(_prompt=""):
        item = next(it)
        if isinstance(item, BaseException):
            raise item
        return item

    return _read


# ---------------------------------------------------------------------------
# 1. Interactive startup, consecutive questions, commands, and Ctrl+C
# ---------------------------------------------------------------------------
def test_interactive_startup_subprocess_displays_welcome_and_exits_cleanly():
    proc = subprocess.run(
        [sys.executable, "-m", "helpdesk.agent"],
        cwd=str(APP_DIR),
        input="exit\n",
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    out = proc.stdout
    assert "============================================================" in out
    assert "RAILWAY PASSENGER HELP DESK" in out
    assert "AI-POWERED ASSISTANCE" in out
    assert "[1] Train schedules and routes" in out
    assert "[5] Railway rules and circulars" in out
    assert "help     Show commands" in out
    assert "clear    Clear terminal" in out
    assert "exit     Exit application" in out
    assert "You:" in out
    assert "Goodbye" in out
    # Redirected/piped subprocess output must contain zero ANSI escape codes or spinner CRs.
    assert "\x1b[" not in out
    assert "\r" not in out


def test_interactive_consecutive_questions_use_ask_and_reuse_session(monkeypatch: pytest.MonkeyPatch):
    seen_questions = []

    def fake_ask(question):
        seen_questions.append(question)
        if "12002" in question:
            return {
                "answer": "Train 12002 (New Delhi-Bhopal Shatabdi Express) leaves NDLS at 06:15.",
                "sources": [],
                "sql": "SELECT number, name, from_station_code, to_station_code, departure, arrival FROM trains WHERE number = '12002'",
                "dont_know": False,
            }
        return {
            "answer": "Aadhaar OTP authentication became compulsory for Tatkal booking from 15-07-2025.",
            "sources": [{"doc": "rb_cc08_2025_tatkal.pdf", "page": 1}],
            "sql": None,
            "dont_know": False,
        }

    monkeypatch.setattr(agent, "ask", fake_ask)
    buf = io.StringIO()
    rc = agent.interactive_chat(
        input_fn=_make_reader([
            "When does train 12002 leave?",
            "When did Aadhaar OTP become compulsory for Tatkal?",
            "quit",
        ]),
        out=buf,
    )
    output = buf.getvalue()
    assert rc == 0
    assert seen_questions == [
        "When does train 12002 leave?",
        "When did Aadhaar OTP become compulsory for Tatkal?",
    ]
    assert output.count("You:") == 3
    assert "RAILWAY ASSISTANT" in output
    assert "New Delhi-Bhopal Shatabdi Express (12002)" in output
    assert "rb_cc08_2025_tatkal.pdf — Page 1" in output
    assert '"dont_know"' not in output


def test_interactive_exit_quit_empty_help_clear_and_ctrl_c(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fail_if_called(q):
        calls.append(q)
        return agent.contract("unused")

    cleared = []
    buf = io.StringIO()
    rc = agent.interactive_chat(
        ask_fn=fail_if_called,
        input_fn=_make_reader(["", "   ", "\t", "HELP", "Clear", "  EXIT  "]),
        out=buf,
        clear_fn=lambda: cleared.append(True),
    )
    out = buf.getvalue()
    assert rc == 0
    assert calls == []
    assert cleared == [True]
    assert "SUPPORTED COMMANDS" in out
    assert "Example questions:" in out
    assert "Goodbye" in out
    assert "\x1b[" not in out

    # quit alias
    buf2 = io.StringIO()
    assert agent.interactive_chat(ask_fn=fail_if_called, input_fn=_make_reader(["QUIT"]), out=buf2) == 0
    assert calls == []

    # Ctrl+C and EOFError during prompt reading
    buf3 = io.StringIO()
    assert agent.interactive_chat(ask_fn=fail_if_called, input_fn=_make_reader([KeyboardInterrupt()]), out=buf3) == 0
    assert "Goodbye" in buf3.getvalue()

    buf4 = io.StringIO()
    assert agent.interactive_chat(ask_fn=fail_if_called, input_fn=_make_reader([EOFError()]), out=buf4) == 0
    assert "Goodbye" in buf4.getvalue()

    # clear_terminal calls cls/clear on real TTY, and emits zero escapes on redirected plain stream
    ran_cmds = []
    monkeypatch.setattr(os, "system", lambda cmd: ran_cmds.append(cmd) or 0)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    agent.clear_terminal(sys.stdout)
    assert ran_cmds == ["cls" if os.name == "nt" else "clear"]

    plain_buf = io.StringIO()
    agent.clear_terminal(plain_buf)
    assert plain_buf.getvalue() == ""


def test_interactive_handles_missing_api_key_without_crashing():
    calls = []

    def flaky_ask(q):
        calls.append(q)
        if len(calls) == 1:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env (see .env.example).")
        return agent.contract("I can't do that. I only run read-only queries on the timetable.", dont_know=True)

    buf = io.StringIO()
    rc = agent.interactive_chat(
        ask_fn=flaky_ask,
        input_fn=_make_reader(["Tatkal rule?", "Delete train 12002 from your database", "exit"]),
        out=buf,
    )
    out = buf.getvalue()
    assert rc == 0
    assert len(calls) == 2
    assert "Unable to process the request right now: GEMINI_API_KEY is not set" in out
    assert agent.ABSTENTION_MESSAGE in re.sub(r"\s+", " ", out)


def test_interactive_reuses_retrieval_cache_across_questions(monkeypatch: pytest.MonkeyPatch):
    """Repeated ask() calls in one interactive session never reload embedding/reranking models."""
    search._state.clear()
    load_counts = {"embedder": 0, "reranker": 0}

    def fake_search(query, k=5, **kwargs):
        if "embedder" not in search._state:
            load_counts["embedder"] += 1
            search._state["embedder"] = object()
        if "reranker" not in search._state:
            load_counts["reranker"] += 1
            search._state["reranker"] = object()
        return [{
            "chunk_id": "c1",
            "text": "Flat cancellation charge for sleeper class is Rs 120 per passenger.",
            "metadata": {"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "doc_date": "2026-01-01"},
        }]

    class _TwoTurnLLM:
        def __init__(self):
            self.turns = 0

        def parse(self, messages, schema, **kwargs):
            self.turns += 1
            if schema is agent.RoutePlan:
                return agent.RoutePlan(needs_documents=True, doc_query="sleeper fee", sql="")
            return agent.GroundedAnswer(
                evidence=[agent.Evidence(source=1, quote="Flat cancellation charge for sleeper class is Rs 120")],
                answer="The flat cancellation charge for sleeper class is Rs 120 per passenger.",
                dont_know=False,
            )

    llm = _TwoTurnLLM()
    monkeypatch.setattr(agent, "search", fake_search)
    monkeypatch.setattr(agent, "make_agent", lambda *a, **kw: agent.RouterAgent(llm=llm))

    try:
        buf = io.StringIO()
        rc = agent.interactive_chat(
            input_fn=_make_reader(["Sleeper fee?", "Sleeper cancellation charge?", "exit"]),
            out=buf,
        )
        assert rc == 0
        assert load_counts == {"embedder": 1, "reranker": 1}
        assert llm.turns == 4
    finally:
        search._state.clear()


# ---------------------------------------------------------------------------
# 2. Color system, NO_COLOR, and redirected output guarantees
# ---------------------------------------------------------------------------
def test_color_palette_rules_for_policy_timetable_and_abstention(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)

    # Policy answer: GREEN strictly on the main answer text only; Cyan headings; White conditions; Grey sources.
    policy_res = {
        "answer": (
            "Aadhaar OTP authentication becomes compulsory for Tatkal booking from 15-07-2025. "
            "Authorized agents cannot book opening-day Tatkal tickets during the first 30 minutes."
        ),
        "sources": [{"doc": "rb_cc08_2025_tatkal.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }
    colored_policy = agent.format_response(policy_res, width=80, use_color=True)
    lines = colored_policy.splitlines()
    summary_idx = next(i for i, ln in enumerate(lines) if "Policy Summary:" in _strip_ansi(ln))
    answer_zone = lines[:summary_idx]
    metadata_zone = lines[summary_idx:]

    green_lines = [ln for ln in answer_zone if "\x1b[32m" in ln]
    assert green_lines
    assert all("RAILWAY ASSISTANT" not in _strip_ansi(ln) and "---" not in _strip_ansi(ln) for ln in green_lines)
    assert all("\x1b[32m" not in ln for ln in metadata_zone)
    assert any("Policy Summary:" in _strip_ansi(ln) and "\x1b[1;36m" in ln for ln in lines)
    assert any("Effective Date:" in _strip_ansi(ln) and "\x1b[1;36m" in ln for ln in lines)
    assert any("Important Conditions:" in _strip_ansi(ln) and "\x1b[37m" in ln and "\x1b[32m" not in ln for ln in lines)
    assert any("rb_cc08_2025_tatkal.pdf — Page 1" in _strip_ansi(ln) and "\x1b[90m" in ln and "\x1b[32m" not in ln for ln in lines)

    # Timetable answer: GREEN strictly on main answer; labels CYAN; values WHITE; DB source GREY.
    sql_res = {
        "answer": "Train 12002 departs NDLS at 06:15 and arrives BPL at 14:05.",
        "sources": [],
        "sql": "SELECT number, name, from_station_code, to_station_code, departure, arrival, duration_min FROM trains WHERE number = '12002'",
        "dont_know": False,
    }
    colored_sql = agent.format_response(sql_res, width=80, use_color=True)
    sql_lines = colored_sql.splitlines()
    assert any("Train 12002 departs" in _strip_ansi(ln) and "\x1b[32m" in ln for ln in sql_lines)
    assert any("Departure:" in _strip_ansi(ln) and "\x1b[36m" in ln and "\x1b[1;37m" in ln and "\x1b[32m" not in ln for ln in sql_lines)
    assert any("Railway timetable database" in _strip_ansi(ln) and "\x1b[90m" in ln for ln in sql_lines)

    # Abstention: YELLOW message, WHITE details, ZERO GREEN anywhere.
    abst_res = {
        "answer": "The fare of an AC 2 tier ticket is not given in the provided sources.",
        "sources": [],
        "sql": None,
        "dont_know": True,
    }
    colored_abst = agent.format_response(abst_res, width=80, use_color=True)
    assert "\x1b[32m" not in colored_abst
    assert any(agent.ABSTENTION_MESSAGE[:30] in _strip_ansi(ln) and "\x1b[1;33m" in ln for ln in colored_abst.splitlines())
    assert any("not given in the provided sources" in _strip_ansi(ln) and "\x1b[37m" in ln for ln in colored_abst.splitlines())


def test_no_color_env_and_redirected_streams_disable_ansi_escapes(monkeypatch: pytest.MonkeyPatch):
    # Redirected / StringIO output disables colors automatically even without NO_COLOR.
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    stream = io.StringIO()
    assert ui.should_use_color(stream=stream) is False
    rendered = agent.format_response(
        {
            "answer": "Flat cancellation charge is Rs 120.",
            "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
            "sql": None,
            "dont_know": False,
        },
        stream=stream,
    )
    assert "\x1b[" not in rendered

    # Even on an interactive TTY stream, NO_COLOR disables ANSI completely.
    class _FakeTTY(io.StringIO):
        def isatty(self):
            return True

    tty_stream = _FakeTTY()
    monkeypatch.setattr(ui, "enable_windows_ansi", lambda: True)
    assert ui.should_use_color(stream=tty_stream) is True

    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.should_use_color(stream=tty_stream) is False
    assert ui.should_use_color(stream=tty_stream, use_color=True) is False
    welcome_plain = agent.render_welcome(stream=tty_stream)
    card_plain = agent.format_response(
        {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True},
        stream=tty_stream,
    )
    assert "\x1b[" not in welcome_plain
    assert "\x1b[" not in card_plain


# ---------------------------------------------------------------------------
# 3. Dynamic terminal width, word/date wrapping, alignment, and spinner
# ---------------------------------------------------------------------------
def test_dynamic_width_wrapping_preserves_dates_words_and_long_filenames(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)

    # Dynamically respects mocked terminal width and falls back cleanly on OSError.
    with monkeypatch.context() as mp:
        mp.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((58, 24)))
        assert ui.get_terminal_width() == 58

    def _boom(*a, **kw):
        raise OSError("no tty")

    with monkeypatch.context() as mp:
        mp.setattr(shutil, "get_terminal_size", _boom)
        assert ui.get_terminal_width() == 80

    # Date '15-07-2025' and '01.11.2024' must never split across two lines across narrow widths.
    res = {
        "answer": (
            "Starting from 01-07-2025, online Tatkal tickets require Aadhaar verification, and "
            "from 15-07-2025 OTP becomes compulsory after revision on 01.11.2024 and 16 January 2026."
        ),
        "sources": [{"doc": "rb_cc08_2026_vande_bharat_sleeper_refund.pdf", "page": 5}],
        "sql": None,
        "dont_know": False,
    }
    for test_width in (46, 52, 60, 74):
        formatted = agent.format_response(res, width=test_width, use_color=False)
        for line in formatted.splitlines():
            assert len(line) <= test_width, f"Overflow at width={test_width}: {line!r}"
        assert "15-07-2025" in formatted
        assert "01-07-2025" in formatted
        assert "01.11.2024" in formatted
        assert "16 January 2026" in formatted
        if test_width >= 50:
            assert "rb_cc08_2026_vande_bharat_sleeper_refund.pdf" in formatted
        else:
            # Below filename length, splits only on underscore boundaries with zero horizontal overflow.
            assert "rb_cc08_2026_vande_bharat_sleeper_refund.pdf" in re.sub(r"_\s+", "_", formatted)

    # Welcome screen adapts border width dynamically on narrow terminals.
    narrow_welcome = agent.render_welcome(width=48, use_color=False)
    for line in narrow_welcome.splitlines():
        assert len(line) <= 48
    assert "=" * 48 in narrow_welcome

    # Narrow dated filename ('pib_2026-03-24_reform_express.html') wraps without overflowing.
    dated_res = {
        "answer": "Ticket cancellation windows were revised to 72, 24, and 8 hours.",
        "sources": [{"doc": "pib_2026-03-24_reform_express.html", "page": None}],
        "sql": None,
        "dont_know": False,
    }
    dated_fmt = agent.format_response(dated_res, width=44, use_color=False)
    for line in dated_fmt.splitlines():
        assert len(line) <= 44
    assert "pib_2026-03-24_reform_express.html" in dated_fmt


def test_loading_indicator_behaves_cleanly_in_tty_and_redirected_modes():
    # Redirected / disabled spinner writes nothing at all.
    plain_buf = io.StringIO()
    with ui.LoadingIndicator(stream=plain_buf, enabled=False, initial_delay=0.0):
        time.sleep(0.02)
    assert plain_buf.getvalue() == ""

    # Enabled spinner animates and clears cleanly on stop without raising on exception.
    tty_buf = io.StringIO()
    with ui.LoadingIndicator(stream=tty_buf, enabled=True, use_color=False, initial_delay=0.0):
        time.sleep(0.06)
    output = tty_buf.getvalue()
    assert "Searching railway records..." in output
    assert output.endswith("\r")


# ---------------------------------------------------------------------------
# 4. Structured formatting (timetable, policy, mixed, abstention)
# ---------------------------------------------------------------------------
def test_structured_formatting_for_timetable_queries_uses_supported_db_fields():
    single = {
        "answer": "According to the 2016 timetable snapshot, train 12002 departs NDLS at 06:15 and arrives BPL at 14:05.",
        "sources": [],
        "sql": "SELECT number, name, from_station_code, to_station_code, departure, arrival, duration_min, distance_km FROM trains WHERE number = '12002'",
        "dont_know": False,
    }
    text_single = agent.format_response(single, width=84, use_color=False)
    assert "-" * 72 in text_single
    assert "RAILWAY ASSISTANT" in text_single
    assert "Timetable Details:" in text_single
    assert "New Delhi-Bhopal Shatabdi Express (12002)" in text_single
    assert "NDLS -> BPL" in text_single
    assert "06:15" in text_single
    assert "14:05" in text_single
    assert "7h 50m (470 minutes)" in text_single
    assert "Intermediate stops:" not in text_single
    assert "Railway timetable database (2016 snapshot, read-only SQL)" in text_single

    # Station-level join must NOT mistake station name (VADODARA JN) for train name
    station_sql = {
        "answer": "According to the 2016 timetable snapshot, train 12951 reaches BRC at 21:06 and leaves at 21:22.",
        "sources": [],
        "sql": "SELECT s.station_code, st.name, s.arrival, s.departure, s.halt_min, s.day FROM stops s JOIN stations st ON st.code = s.station_code WHERE s.train_number = '12951' AND s.station_code = 'BRC'",
        "dont_know": False,
    }
    text_st = agent.format_response(station_sql, width=84, use_color=False)
    assert "VADODARA JN (BRC)" in text_st
    assert "Train name and number:" not in text_st
    assert "16 minutes" in text_st

    # Multi-stop train route shows origin/dest and intermediate stops cleanly
    route_sql = {
        "answer": "According to the 2016 timetable snapshot, train 12951 stops at BCT, ST, BRC, RTM, KOTA, and NDLS.",
        "sources": [],
        "sql": "SELECT st.stop_sequence, st.station_code, s.name, st.arrival, st.departure, st.day FROM stops st JOIN stations s ON s.code = st.station_code WHERE st.train_number = '12951' ORDER BY st.stop_sequence",
        "dont_know": False,
    }
    text_route = agent.format_response(route_sql, width=88, use_color=False)
    norm_route = re.sub(r"\s+", " ", text_route)
    assert "Mumbai Central (BCT) -> NEW DELHI (NDLS)" in norm_route
    assert "ST (SURAT), BRC (VADODARA JN), RTM (RATLAM JN), KOTA (KOTA JN)" in norm_route

    # Aggregate COUNT(*) does not force empty schedule labels
    count_sql = {
        "answer": "According to the 2016 timetable snapshot, there are 48 Rajdhani Express trains.",
        "sources": [],
        "sql": "SELECT COUNT(*) AS rajdhani_count FROM trains WHERE type_code = 'Raj'",
        "dont_know": False,
    }
    text_count = agent.format_response(count_sql, width=80, use_color=False)
    assert "Timetable Details:" not in text_count
    assert "Sources:" in text_count


def test_structured_formatting_for_policy_mixed_and_short_answers():
    policy_res = {
        "answer": (
            "The advance reservation period was reduced to 60 days excluding the date of journey with effect from 01.11.2024. "
            "However, bookings done up to 31.10.2024 under the 120-day limit remain intact. "
            "No cancellation charge applies beyond 60 days cancellation permission as permitted."
        ),
        "sources": [{"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }
    text_policy = agent.format_response(policy_res, width=90, use_color=False)
    assert "Policy Summary:" in text_policy
    assert "Applicable Rule:" in text_policy
    assert "Important Conditions:" in text_policy
    assert "Effective Date:" in text_policy and "01.11.2024" in text_policy
    assert "Refund / Cancellation:" in text_policy
    assert "rb_cc10_2024_advance_reservation.pdf — Page 1" in text_policy

    # Simple 1-sentence answer without conditions/dates must not force Policy Summary headers
    simple_html = {
        "answer": "The pre-revised clerkage charge for an unreserved second class ticket was Rs 15 per passenger.",
        "sources": [{"doc": "indianrail_refund_rules_2015.html", "page": None}],
        "sql": None,
        "dont_know": False,
    }
    text_simple = agent.format_response(simple_html, width=80, use_color=False)
    assert "Policy Summary:" not in text_simple
    assert "  indianrail_refund_rules_2015.html\n" in text_simple
    assert "Page None" not in text_simple

    # Mixed timetable + policy response separates DB info cleanly without duplicating timetable sentence into Policy Summary
    mixed = {
        "answer": "Train 12002 departs New Delhi at 06:15; Aadhaar OTP authentication became compulsory for Tatkal booking starting from 15-07-2025.",
        "sources": [{"doc": "rb_cc08_2025_tatkal.pdf", "page": 1}],
        "sql": "SELECT number, name, departure, arrival FROM trains WHERE number = '12002'",
        "dont_know": False,
    }
    text_mixed = agent.format_response(mixed, width=86, use_color=False)
    assert "Timetable Details:" in text_mixed
    assert "Policy Summary:" in text_mixed
    policy_block = text_mixed.split("Policy Summary:", 1)[1].split("Sources:", 1)[0]
    assert "departs New Delhi" not in policy_block
    assert "15-07-2025" in policy_block
    assert "rb_cc08_2025_tatkal.pdf — Page 1" in text_mixed
    assert "Railway timetable database (2016 snapshot, read-only SQL)" in text_mixed


def test_abstention_formatting_never_fabricates_sources():
    generic = agent.format_response(
        {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True},
        question="What is the fare of an AC 2 tier ticket today?",
        width=80,
        use_color=False,
    )
    assert agent.ABSTENTION_MESSAGE in re.sub(r"\s+", " ", generic)
    assert "no live train status, current ticket fares" in re.sub(r"\s+", " ", generic)
    assert "Sources:" not in generic

    refusal = agent.format_response(
        {"answer": agent.REFUSAL, "sources": [], "sql": None, "dont_know": True},
        question="Delete train 12002",
        width=80,
        use_color=False,
    )
    assert agent.ABSTENTION_MESSAGE in re.sub(r"\s+", " ", refusal)
    assert agent.REFUSAL in re.sub(r"\s+", " ", refusal)
    assert "Sources:" not in refusal


# ---------------------------------------------------------------------------
# 5. One-shot CLI & public ask(question) API backward compatibility
# ---------------------------------------------------------------------------
def test_oneshot_cli_and_public_ask_api_remain_compatible(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    assert helpdesk.ask("") == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}

    proc = subprocess.run(
        [sys.executable, "-m", "helpdesk.agent", "Delete train 12002 from your database"],
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    parsed = json.loads(proc.stdout)
    assert set(parsed.keys()) == {"answer", "sources", "sql", "dont_know"}
    assert parsed["dont_know"] is True
    assert parsed["sources"] == []
    assert parsed["sql"] is None
    assert "read-only" in parsed["answer"]
    assert "RAILWAY ASSISTANT" not in proc.stdout

    expected = {
        "answer": "Aadhaar-based OTP authentication became compulsory for Tatkal booking from 15-07-2025.",
        "sources": [{"doc": "rb_cc08_2025_tatkal.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }
    monkeypatch.setattr(agent, "ask", lambda q, agent=None: expected)
    monkeypatch.setattr(sys, "argv", ["helpdesk.agent", "What is the Tatkal Aadhaar OTP rule?"])
    agent._main()
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == expected
