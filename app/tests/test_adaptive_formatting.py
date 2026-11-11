"""Offline tests for refined menu centering, streamlined welcome, complete routes, and adaptive Rich formatting."""

from __future__ import annotations

import io
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import helpdesk.agent as agent  # noqa: E402
import helpdesk.cli as cli  # noqa: E402
import helpdesk.sql as sql  # noqa: E402
import helpdesk.terminal_ui as ui  # noqa: E402

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def _make_reader(responses):
    it = iter(responses)

    def _read(_prompt=""):
        item = next(it)
        if isinstance(item, BaseException):
            raise item
        return item

    return _read


# ---------------------------------------------------------------------------
# 1. Dynamic Main Menu Header Centering & Resize Handling
# ---------------------------------------------------------------------------
def test_main_menu_header_centered_across_widths_and_resizes(monkeypatch: pytest.MonkeyPatch):
    for cols in (40, 80, 120, 144):
        buf = io.StringIO()
        con = cli._make_console(stream=buf, use_color=False, width=cols)
        cli._print_main_menu(console=con, stream=buf, use_color=False, width=cols)
        out = buf.getvalue()
        lines = [ln for ln in out.splitlines() if ln.strip()]

        for ln in out.splitlines():
            assert len(_strip_ansi(ln)) <= cols, f"Line overflowed {cols} cols: {ln!r}"

        # Top double-border or Rich panel adapts to exact width cols
        top_border = next((ln for ln in lines if ln.startswith("╔")), "")
        assert len(top_border) == cols

        title_line = next(ln for ln in lines if "RAILWAY PASSENGER HELP DESK" in ln)
        inner_t = title_line.strip("║")
        left_pad = len(inner_t) - len(inner_t.lstrip(" "))
        right_pad = len(inner_t) - len(inner_t.rstrip(" "))
        assert abs(left_pad - right_pad) <= 1, f"Title not centered at width={cols}: {title_line!r}"

        if cols >= 50:
            sub_line = next(ln for ln in lines if "Intelligent Railway Information Assistant" in ln)
            inner_s = sub_line.strip("║")
            s_left = len(inner_s) - len(inner_s.lstrip(" "))
            s_right = len(inner_s) - len(inner_s.rstrip(" "))
            assert abs(s_left - s_right) <= 1, f"Subtitle not centered at width={cols}: {sub_line!r}"

        # Plain-text render_main_menu centers dynamically without hardcoded caps
        plain = cli.render_main_menu(width=cols, use_color=False)
        for ln in plain.splitlines():
            assert len(ln) <= cols
        assert "=" * cols in plain

    # Terminal resize between Main Menu redraws adapts live without stale cached width
    sizes = iter([os.terminal_size((64, 24)), os.terminal_size((118, 30))])
    with monkeypatch.context() as mp:
        mp.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): next(sizes, os.terminal_size((118, 30))))
        buf_resize = io.StringIO()
        con_dyn = cli._make_console(stream=buf_resize, use_color=False)
        assert con_dyn.width == 64
        cli._print_main_menu(console=con_dyn, stream=buf_resize, use_color=False)
    resized_top = next(ln for ln in buf_resize.getvalue().splitlines() if ln.startswith("╔"))
    assert len(resized_top) == 118


# ---------------------------------------------------------------------------
# 2 & 3 & 16. Streamlined Chatbot Welcome, Commands, & Return to Main Menu
# ---------------------------------------------------------------------------
def test_chatbot_welcome_streamlined_and_returns_to_main_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaDummyKeyForChatbotMenuTest")
    cache = tmp_path / "hf_cache"
    for model_id in ("BAAI/bge-small-en-v1.5", "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        snap = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")

    for w in (36, 48, 80, 120):
        banner = agent.render_welcome(width=w, use_color=False)
        assert "RAILWAY PASSENGER HELP DESK" in banner
        assert "Ask your railway-related question" in banner
        assert "I can help you with:" not in banner
        assert "[1]" not in banner
        for ln in banner.splitlines():
            assert len(ln) <= w

    def fake_ask(q):
        return {
            "answer": "Train booking hours are 00:20 to 23:45 daily.",
            "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
            "sql": None,
            "dont_know": False,
        }

    monkeypatch.setattr(agent, "ask", fake_ask)

    # Full round-trip: Main Menu [1] -> Chatbot (help, clear, question, quit) -> Enter -> Main Menu [0]
    buf = io.StringIO()
    cleared = []
    rc = cli.run_main_menu(
        app_dir=APP_DIR,
        input_fn=_make_reader(["1", "help", "clear", "What are booking hours?", "quit", "", "0"]),
        out=buf,
        clear_fn=lambda: cleared.append(True),
        use_color=False,
        width=84,
        cache_root=cache,
    )
    out = buf.getvalue()
    assert rc == 0
    assert cleared == [True]
    assert "Ask your railway-related question below." in out
    assert "SUPPORTED COMMANDS" in out
    assert "Train booking hours are 00:20 to 23:45 daily." in out
    assert "Returning to Main Menu." in out
    assert "Thank you for using Railway Passenger Help Desk. Goodbye!" in out
    assert "I can help you with:" not in out


# ---------------------------------------------------------------------------
# 4, 5, 6, 7. Full Route Preservation (>36 Stops, Missing Fields, No Truncation)
# ---------------------------------------------------------------------------
def test_train_16303_full_route_never_truncated_at_20_or_36():
    sql_16303 = (
        "SELECT s.stop_sequence, s.station_code, st.name AS station_name, t.name AS train_name, "
        "s.day, s.arrival, s.departure, s.halt_min, s.is_halt "
        "FROM schedules s "
        "LEFT JOIN stations st ON st.code = s.station_code "
        "LEFT JOIN trains t ON t.number = s.train_number "
        "WHERE s.train_number = '16303' ORDER BY s.day, s.stop_sequence"
    )
    db_res = sql.execute_readonly_query(sql_16303, max_rows=sql.MAX_ROW_LIMIT)
    assert db_res["row_count"] == 42
    assert db_res["truncated"] is False

    # Verify rows_text and RouterAgent default max_rows preserve ALL 42 stations (not capped at 36 or 4000 chars)
    txt_passage = agent.rows_text(db_res)
    assert "... (output truncated)" not in txt_passage
    assert "station_code: TVC" in txt_passage
    assert agent.RouterAgent(llm=object()).max_rows == sql.MAX_ROW_LIMIT

    # Even if LLM dumps a huge comma list, formatter keeps full table & all 42 rows in exact order
    all_names = ", ".join(r["station_name"] for r in db_res["rows"][:25])
    fake_out = {
        "answer": f"Train 16303 stops at {all_names}. It runs daily in the 2016 snapshot.",
        "sources": [],
        "sql": sql_16303,
        "dont_know": False,
    }
    rendered = agent.format_response(fake_out, "Show me the route of train 16303.", width=92, use_color=False)
    assert ui.classify_response_type(fake_out, "Show me the route of train 16303.", db_res) == "train_route"
    assert "TRAIN ROUTE — 16303" in rendered
    assert "Vanchinad Express (16303)" in rendered
    assert "Origin:       ERNAKULAM JN (ERS)" in rendered
    assert "Destination:  TRIVANDRUM CENTRAL (TVC)" in rendered
    # Previously Ochira (stop 20) falsely appeared as Destination — must show full route to TVC
    assert "│  20 │ OCHIRA (OCR)" in rendered
    assert "│  42 │ TRIVANDRUM CENTRAL (TVC)" in rendered
    assert "│ 09:40" in rendered
    norm_rendered = re.sub(r"\s+", " ", rendered)
    assert "42 (16 halts, 26 pass-through)" in rendered
    assert "It runs daily in the 2016 snapshot." in norm_rendered
    assert "listed stations in the route table below" in norm_rendered
    assert "August 2016" in rendered
    assert "Railway timetable database (2016 snapshot, read-only SQL)" in rendered

    # Color mode must never apply green styling to route table rows/headers/warnings
    colored_route = agent.format_response(fake_out, "Show me the route of train 16303.", width=92, use_color=True)
    header_idx = next(i for i, ln in enumerate(colored_route.splitlines()) if "TRAIN ROUTE — 16303" in _strip_ansi(ln))
    assert all("\x1b[32m" not in ln for ln in colored_route.splitlines()[header_idx:])


def test_long_route_over_100_stops_chunks_cleanly_without_dropping_rows():
    # Train 13049 has 110 halts in the real 2016 stops view
    sql_13049 = (
        "SELECT s.stop_sequence, s.station_code, st.name AS station_name, t.name AS train_name, "
        "s.day, s.arrival, s.departure, s.halt_min "
        "FROM stops s "
        "LEFT JOIN stations st ON st.code = s.station_code "
        "LEFT JOIN trains t ON t.number = s.train_number "
        "WHERE s.train_number = '13049' ORDER BY s.day, s.stop_sequence"
    )
    db_res = sql.execute_readonly_query(sql_13049, max_rows=sql.MAX_ROW_LIMIT)
    assert db_res["row_count"] == 110
    first_code = db_res["rows"][0]["station_code"]
    last_code = db_res["rows"][-1]["station_code"]

    res = {
        "answer": f"Train 13049 runs from {first_code} to {last_code} across 110 stops in the 2016 snapshot.",
        "sources": [],
        "sql": sql_13049,
        "dont_know": False,
    }
    out = agent.format_response(res, "Show complete route of train 13049", width=92, use_color=False)
    assert "Stops 1-40 of 110" in out
    assert "Stops 41-80 of 110" in out
    assert "Stops 81-110 of 110" in out
    assert f"({first_code})" in out
    assert f"({last_code})" in out
    assert "│ 110 │" in out
    assert "Listed Stops" in out and "110" in out

    # Explicit truncation notice if a synthetic query exceeds limit and sets truncated=True
    trunc_rows = [
        {
            "stop_sequence": i,
            "station_code": f"S{i:02d}",
            "station_name": f"Station {i}",
            "arrival": None,
            "departure": None,
        }
        for i in range(1, 5)
    ]
    out_trunc = agent.format_response(
        {
            "answer": "Partial stops listed.",
            "sources": [],
            "sql": "SELECT stop_sequence, station_code, station_name, arrival, departure FROM stops WHERE train_number='99999'",
            "dont_know": False,
        },
        sql_query_fn=lambda q, max_rows=1000: {
            "columns": ["stop_sequence", "station_code", "station_name", "arrival", "departure"],
            "rows": trunc_rows,
            "truncated": True,
            "row_count": 4,
        },
        width=84,
        use_color=False,
    )
    assert "Last Listed Stop (Truncated)" in out_trunc
    assert "Warning:" in out_trunc


def test_route_missing_timings_and_station_only_sequence_never_fabricates():
    # Case A: timings exist as NULL -> display '-'
    rows_nulls = [
        {"stop_sequence": 1, "station_code": "AAA", "station_name": "ALPHA JN", "arrival": None, "departure": "06:00"},
        {"stop_sequence": 2, "station_code": "BBB", "station_name": "BETA TOWN", "arrival": None, "departure": None},
        {
            "stop_sequence": 3,
            "station_code": "CCC",
            "station_name": "GAMMA TERMINAL",
            "arrival": "09:15",
            "departure": None,
        },
    ]
    out_nulls = agent.format_response(
        {
            "answer": "Route runs AAA to CCC.",
            "sources": [],
            "sql": "SELECT stop_sequence, station_code, station_name, arrival, departure FROM stops",
            "dont_know": False,
        },
        sql_query_fn=lambda q, max_rows=1000: {
            "columns": list(rows_nulls[0].keys()),
            "rows": rows_nulls,
            "truncated": False,
            "row_count": 3,
        },
        width=80,
        use_color=False,
    )
    assert "Arrival" in out_nulls and "Departure" in out_nulls
    assert "│   2 │ BETA TOWN (BBB)      │ -       │ -         │" in out_nulls

    # Case B: SQL returns ONLY station names without arrival/departure -> station-sequence table ONLY (no invented timings)
    rows_names_only = [
        {"stop_sequence": 1, "station_code": "NDLS", "station_name": "NEW DELHI"},
        {"stop_sequence": 2, "station_code": "AGC", "station_name": "AGRA CANTT"},
        {"stop_sequence": 3, "station_code": "BPL", "station_name": "BHOPAL JN"},
    ]
    out_seq_only = agent.format_response(
        {
            "answer": "Train 12002 passes NEW DELHI, AGRA CANTT, and BHOPAL JN.",
            "sources": [],
            "sql": "SELECT stop_sequence, station_code, station_name FROM stops",
            "dont_know": False,
        },
        sql_query_fn=lambda q, max_rows=1000: {
            "columns": list(rows_names_only[0].keys()),
            "rows": rows_names_only,
            "truncated": False,
            "row_count": 3,
        },
        width=80,
        use_color=False,
    )
    assert "ROUTE DETAILS" in out_seq_only
    assert "Arrival" not in out_seq_only
    assert "Departure" not in out_seq_only
    assert "NEW DELHI (NDLS)" in out_seq_only
    assert "BHOPAL JN (BPL)" in out_seq_only


# ---------------------------------------------------------------------------
# 8, 15. Long Station Names & Narrow Terminal Wrapping Without Overflow
# ---------------------------------------------------------------------------
def test_long_station_names_wrap_cleanly_without_overflow_or_ellipsis_at_narrow_widths():
    rows_long = [
        {
            "stop_sequence": 1,
            "station_code": "TVC",
            "station_name": "THIRUVANANTHAPURAM CENTRAL RAILWAY TERMINUS",
            "arrival": None,
            "departure": "05:00",
        },
        {
            "stop_sequence": 2,
            "station_code": "MAS",
            "station_name": "PURATCHI THALAIVAR DR M G RAMACHANDRAN CENTRAL",
            "arrival": "18:45",
            "departure": None,
        },
    ]
    res = {
        "answer": "Route runs between two major terminus stations.",
        "sources": [],
        "sql": "SELECT stop_sequence, station_code, station_name, arrival, departure FROM stops",
        "dont_know": False,
    }

    def runner(q, max_rows=1000):
        return {"columns": list(rows_long[0].keys()), "rows": rows_long, "truncated": False, "row_count": 2}

    for narrow_w in (34, 40, 46, 52):
        out = agent.format_response(res, sql_query_fn=runner, width=narrow_w, use_color=False)
        for ln in out.splitlines():
            assert len(ln) <= narrow_w, f"Overflow at width={narrow_w}: {ln!r}"
        # Full name characters must be preserved via wrapping/folding rather than '...' ellipsis clipping
        compact = re.sub(r"\s+", "", out.replace("│", ""))
        assert "THIRUVANANTHAPURAM" in compact
        assert "RAMACHANDRAN" in compact
        assert "…" not in out


# ---------------------------------------------------------------------------
# 9, 10, 11, 12, 13, 14. Intelligent Multi-Type Formatting (SQL, Compare, Refund, Tatkal, Steps, General, Sources, Abstain)
# ---------------------------------------------------------------------------
def test_train_comparison_and_multicolumn_sql_tables():
    # Comparison between stations (real DB query)
    cmp_sql = (
        "SELECT t.number, t.name, a.departure AS departs_from_a, b.arrival AS arrives_at_b "
        "FROM stops a JOIN stops b ON b.train_number = a.train_number AND b.stop_sequence > a.stop_sequence "
        "JOIN trains t ON t.number = a.train_number "
        "WHERE a.station_code = 'NDLS' AND b.station_code = 'BPL' ORDER BY a.departure LIMIT 5"
    )
    cmp_res = {
        "answer": "Several trains connect NDLS and BPL in the 2016 snapshot.",
        "sources": [],
        "sql": cmp_sql,
        "dont_know": False,
    }
    cmp_out = agent.format_response(cmp_res, "Compare trains from NDLS to BPL", width=88, use_color=False)
    assert "TRAIN SCHEDULE COMPARISON" in cmp_out
    assert "Train No." in cmp_out
    assert "12002" in cmp_out
    assert "New Delhi-Bhopal Shatabdi Express" in cmp_out
    assert "06:15" in cmp_out

    # General multi-column stations SQL query
    st_sql = (
        "SELECT code, name, state, zone FROM stations WHERE zone = 'WR' AND state IS NOT NULL ORDER BY code LIMIT 4"
    )
    st_res = {
        "answer": "Here are four Western Railway stations from the 2016 snapshot.",
        "sources": [],
        "sql": st_sql,
        "dont_know": False,
    }
    st_out = agent.format_response(st_res, "List Western Railway stations", width=86, use_color=False)
    assert "TIMETABLE RESULTS" in st_out
    assert "Code" in st_out and "Station Name" in st_out and "State" in st_out and "Zone" in st_out
    assert "WR" in st_out


def test_refund_tatkal_markdown_steps_general_and_sources_and_abstention():
    # 1. Refund slabs table (D)
    refund_slabs = {
        "answer": (
            "For Vande Bharat Sleeper tickets: cancellation more than 72 hours before departure "
            "attracts a charge of 25% of fare; between 72 and 8 hours before departure, 50% of fare "
            "is deducted; within 8 hours before departure, no refund is granted."
        ),
        "sources": [{"doc": "rb_cc08_2026_vande_bharat_sleeper_refund.pdf", "page": 5}],
        "sql": None,
        "dont_know": False,
    }
    out_slabs = agent.format_response(refund_slabs, width=84, use_color=False)
    assert "REFUND & CANCELLATION SUMMARY" in out_slabs
    assert "Cancellation Timing" in out_slabs and "Refund / Charge Rule" in out_slabs
    assert "25% of fare" in out_slabs and "50% of fare" in out_slabs and "No refund" in out_slabs
    assert "rb_cc08_2026_vande_bharat_sleeper_refund.pdf — Page 5" in out_slabs

    # 2. Refund class flat charges table (D)
    refund_classes = {
        "answer": (
            "Flat cancellation charges per passenger are Rs 240 + GST for AC First Class, "
            "Rs 200 + GST for AC 2-Tier, and Rs 120 for Sleeper Class."
        ),
        "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }
    out_classes = agent.format_response(refund_classes, width=84, use_color=False)
    assert "Class / Category" in out_classes and "Cancellation Charge" in out_classes
    assert "AC First Class" in out_classes and "Rs 240 + GST" in out_classes

    # 3. Tatkal booking rules table (E) — exact 10:00 AC and 11:00 non-AC distinction
    tatkal_res = {
        "answer": (
            "Tatkal booking opens at 10:00 hrs for AC classes and at 11:00 hrs for non-AC classes "
            "one day before journey. Authorized agents cannot book during the first 30 minutes, "
            "and Aadhaar OTP authentication is compulsory."
        ),
        "sources": [{"doc": "rb_cc08_2025_tatkal.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }
    out_tatkal = agent.format_response(tatkal_res, width=84, use_color=False)
    assert "TATKAL BOOKING RULES" in out_tatkal
    assert re.search(r"AC Classes\s*│\s*Opens at 10:00 hrs", out_tatkal)
    assert re.search(r"Non-AC Classes\s*│\s*Opens at 11:00 hrs", out_tatkal)

    # 4. Multi-step procedure renders as numbered steps (H/C) and NOT forced into a table
    steps_res = {
        "answer": (
            "Follow these steps to book online:\n"
            "1. Log in between 00:20 and 23:45.\n"
            "2. Enter journey and passenger details.\n"
            "3. Pay and save the ticket."
        ),
        "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }
    out_steps = agent.format_response(steps_res, "How do I book an online ticket?", width=80, use_color=False)
    assert "  1. Log in between 00:20 and 23:45." in out_steps
    assert "  2. Enter journey and passenger details." in out_steps
    assert "╭" not in out_steps

    # 5. Embedded Markdown table converts cleanly into Rich table
    md_res = {
        "answer": "Summary:\n| Rule | Status |\n| --- | --- |\n| Aadhaar OTP | Compulsory |",
        "sources": [{"doc": "rb_cc08_2025_tatkal.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }
    out_md = agent.format_response(md_res, width=80, use_color=False)
    assert "╭─" in out_md
    assert "│ Aadhaar OTP │ Compulsory │" in out_md

    # 6. General concise explanation stays plain paragraph without forced tables (G)
    general_res = {
        "answer": "Booking and cancellation hours on IRCTC are 00:20 to 23:45.",
        "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }
    out_gen = agent.format_response(general_res, width=80, use_color=False)
    assert "╭" not in out_gen
    assert "Policy Summary:" not in out_gen
    assert "irctc_refund_rules_jan2026.pdf — Page 1" in out_gen

    # 7. Abstention preserves yellow warning and zero green or sources
    abst_out = agent.format_response(
        {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True},
        question="What is the live running status of train 16303?",
        width=80,
        use_color=True,
    )
    assert "\x1b[32m" not in abst_out
    assert "\x1b[1;33m" in abst_out
    assert "Sources:" not in _strip_ansi(abst_out)
