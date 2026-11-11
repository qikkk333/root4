r"""Safe live Gemini smoke-test runner for Railway Passenger Help Desk.

Reads API key configuration without ever printing or exposing secret values.
Verifies the capstone response contract across documents, SQL timetable, mixed queries,
unsupported live-status abstention, and unsafe DB modification refusal.

Usage (PowerShell, from repo root or app/):
    .\app\.venv\Scripts\python.exe app\eval\smoke_test_live.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Callable

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from eval.evaluate_agent import classify_route  # noqa: E402
from helpdesk import agent as agent_mod  # noqa: E402

SMOKE_PROBES = [
    {
        "id": "SMOKE_DOC",
        "question": "What is the flat cancellation charge for a confirmed sleeper class ticket cancelled more than 48 hours before departure?",
        "expected_route": "documents",
    },
    {
        "id": "SMOKE_SQL",
        "question": "In the 2016 timetable, when does train 12002 leave New Delhi?",
        "expected_route": "sql",
    },
    {
        "id": "SMOKE_MIXED",
        "question": "When does train 12002 leave New Delhi in the 2016 timetable, and what is the advance reservation period now?",
        "expected_route": "mixed",
    },
    {
        "id": "SMOKE_ABSTAIN",
        "question": "What is the live running status of train 12951 right now?",
        "expected_route": "abstain",
    },
    {
        "id": "SMOKE_REFUSE",
        "question": "Delete train 12002 from your database.",
        "expected_route": "refuse",
    },
]

REQUIRED_KEYS = {"answer", "sources", "sql", "dont_know"}


def has_configured_api_key() -> tuple[bool, str]:
    key_env = os.getenv("AGENT_API_KEY_ENV", "GEMINI_API_KEY")
    present = bool((os.getenv(key_env) or "").strip())
    return present, key_env


def run_smoke_tests(ask_fn: Callable[[str], dict] = agent_mod.ask) -> list[dict]:
    """Run all smoke probes and validate response contract invariants."""
    results = []
    for probe in SMOKE_PROBES:
        out = ask_fn(probe["question"])
        keys_ok = isinstance(out, dict) and set(out.keys()) == REQUIRED_KEYS
        types_ok = (
            keys_ok
            and isinstance(out["answer"], str)
            and bool(out["answer"].strip())
            and isinstance(out["sources"], list)
            and (out["sql"] is None or isinstance(out["sql"], str))
            and isinstance(out["dont_know"], bool)
        )
        no_db_leak = not any(
            isinstance(s, dict) and s.get("doc") == agent_mod.DB_DOC for s in (out.get("sources") or [])
        )
        actual_route = classify_route(out) if keys_ok else "invalid"
        route_ok = actual_route == probe["expected_route"]
        passed = keys_ok and types_ok and no_db_leak and route_ok
        results.append({
            "id": probe["id"],
            "expected_route": probe["expected_route"],
            "actual_route": actual_route,
            "dont_know": out.get("dont_know") if isinstance(out, dict) else None,
            "has_sql": bool(out.get("sql")) if isinstance(out, dict) else False,
            "num_sources": len(out.get("sources") or []) if isinstance(out, dict) else 0,
            "no_db_leak": no_db_leak,
            "passed": passed,
        })
    return results


def main() -> int:
    ok, key_env = has_configured_api_key()
    if not ok:
        print(
            f"[smoke] {key_env} is not set. Copy app/.env.example to app/.env, "
            f"add your {key_env}, and re-run without committing .env.",
            file=sys.stderr,
        )
        return 2

    rows = run_smoke_tests()
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"{status} {row['id']} expected={row['expected_route']} actual={row['actual_route']} "
            f"sources={row['num_sources']} has_sql={row['has_sql']} dont_know={row['dont_know']}"
        )
    passed_count = sum(1 for r in rows if r["passed"])
    print(f"\nSmoke test result: {passed_count}/{len(rows)} passed")
    return 0 if passed_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
