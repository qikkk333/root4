"""Read-only SQLite access layer for the 2016 Indian Railways timetable database."""

import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


# Resolve repository-root database/rail_timetable.db regardless of current working directory
PACKAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PACKAGE_DIR.parent
REPO_ROOT = APP_DIR.parent
DEFAULT_DB_PATH = REPO_ROOT / "database" / "rail_timetable.db"
DB_PATH = DEFAULT_DB_PATH

DB_PATH_ENV_VAR = "RAIL_TIMETABLE_DB_PATH"
DEFAULT_ROW_LIMIT = 100
MAX_ROW_LIMIT = 1000
DEFAULT_QUERY_TIMEOUT = 5.0

_BLOCKED_FUNCTIONS = {"load_extension", "fts3_tokenizer"}
_PROGRESS_HANDLER_STEPS = 10000


def get_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve the timetable database path from an explicit override, env var, or repo default."""

    if db_path is not None:
        if isinstance(db_path, str) and not db_path.strip():
            raise ValueError("Database path cannot be empty.")
        return Path(db_path).expanduser().resolve()

    env_path = os.getenv(DB_PATH_ENV_VAR, "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()

    return Path(DB_PATH).expanduser().resolve()


def get_schema_path(db_path: str | Path | None = None) -> Path:
    """Return the expected schema.md path adjacent to the timetable database."""

    return get_db_path(db_path).parent / "schema.md"


def is_select_query(query: str) -> bool:
    """Check whether the SQL statement starts with SELECT or WITH."""

    if not isinstance(query, str):
        return False

    pattern = r"\A(?:\s|--[^\r\n]*(?:\r?\n|\Z)|/\*.*?\*/)*(SELECT|WITH)\b"

    return bool(re.match(pattern, query, re.IGNORECASE | re.DOTALL))


def readonly_authorizer(action, arg1, arg2, database_name, trigger_name):
    """Allow only read-related SQLite operations."""

    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }

    # Block dangerous extension-loading / tokenizer functions
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (arg2 or "").lower()

        if function_name in _BLOCKED_FUNCTIONS:
            return sqlite3.SQLITE_DENY

    if action in allowed_actions:
        return sqlite3.SQLITE_OK

    return sqlite3.SQLITE_DENY


def execute_readonly_query(
    query: str,
    params: Sequence[Any] | Mapping[str, Any] = (),
    max_rows: int = DEFAULT_ROW_LIMIT,
    *,
    db_path: str | Path | None = None,
    timeout_sec: float = DEFAULT_QUERY_TIMEOUT,
) -> dict:
    """
    Execute a safe, read-only SQL query.

    Returns columns, rows, truncation status, and row_count, or {"error": ...}.
    """

    if not isinstance(query, str) or not query.strip():
        return {"error": "SQL query cannot be empty."}

    if not is_select_query(query):
        return {"error": "Only SELECT queries are allowed."}

    if (
        not isinstance(params, (Sequence, Mapping))
        or isinstance(params, (str, bytes, bytearray))
    ):
        return {"error": "Query parameters must be a sequence or mapping."}

    if not isinstance(max_rows, int) or isinstance(max_rows, bool):
        return {"error": "Row limit must be an integer."}

    if max_rows < 1 or max_rows > MAX_ROW_LIMIT:
        return {
            "error": f"Row limit must be between 1 and {MAX_ROW_LIMIT}."
        }

    if (
        not isinstance(timeout_sec, (int, float))
        or isinstance(timeout_sec, bool)
        or timeout_sec <= 0
    ):
        return {"error": "Query timeout must be a positive number."}

    try:
        resolved_db = get_db_path(db_path)
    except ValueError as error:
        return {"error": str(error)}

    if not resolved_db.is_file():
        return {"error": f"Database not found: {resolved_db}"}

    connection = None
    timed_out = False

    try:
        # Open the existing database strictly in read-only mode (never creates a new file)
        database_uri = resolved_db.as_uri() + "?mode=ro"

        connection = sqlite3.connect(database_uri, uri=True)
        connection.row_factory = sqlite3.Row

        # Enable SQLite's query-only protection
        connection.execute("PRAGMA query_only = ON")

        # Abort runaway queries or infinite recursive CTEs after timeout_sec
        deadline = time.monotonic() + float(timeout_sec)

        def _check_timeout() -> int:
            nonlocal timed_out
            if time.monotonic() >= deadline:
                timed_out = True
                return 1
            return 0

        connection.set_progress_handler(_check_timeout, _PROGRESS_HANDLER_STEPS)

        # Block database modifications and other restricted operations
        connection.set_authorizer(readonly_authorizer)

        cursor = connection.cursor()

        # Execute the query using safe parameter binding
        cursor.execute(query, params)

        columns = [column[0] for column in cursor.description or []]

        # Fetch one extra row to detect whether results were truncated
        fetched_rows = cursor.fetchmany(max_rows + 1)

        truncated = len(fetched_rows) > max_rows
        fetched_rows = fetched_rows[:max_rows]

        rows = [dict(row) for row in fetched_rows]

        return {
            "columns": columns,
            "rows": rows,
            "truncated": truncated,
            "row_count": len(rows),
        }

    except sqlite3.Error as error:
        if timed_out:
            return {"error": f"Database query timed out after {timeout_sec}s."}
        return {"error": f"Database query failed: {error}"}

    finally:
        if connection is not None:
            try:
                connection.set_progress_handler(None, 0)
            except sqlite3.Error:
                pass
            connection.close()


def find_train(
    train_number: str,
    *,
    db_path: str | Path | None = None,
) -> dict:
    """Find a train using its train number."""

    if not isinstance(train_number, str) or not train_number.strip():
        return {"error": "Train number must be a non-empty string."}

    query = """
    SELECT
        number,
        name,
        from_station_code,
        to_station_code,
        departure,
        arrival,
        duration_min,
        distance_km
    FROM trains
    WHERE number = ?
    """

    return execute_readonly_query(
        query,
        (train_number.strip(),),
        db_path=db_path,
    )


def find_trains_between(
    from_station_code: str,
    to_station_code: str,
    max_rows: int = DEFAULT_ROW_LIMIT,
    *,
    db_path: str | Path | None = None,
) -> dict:
    """Find trains travelling between two stations."""

    if (
        not isinstance(from_station_code, str)
        or not from_station_code.strip()
        or not isinstance(to_station_code, str)
        or not to_station_code.strip()
    ):
        return {"error": "Origin and destination station codes must be non-empty strings."}

    query = """
    SELECT
        t.number,
        t.name,
        a.departure,
        b.arrival
    FROM stops a
    JOIN stops b
        ON b.train_number = a.train_number
        AND b.stop_sequence > a.stop_sequence
    JOIN trains t
        ON t.number = a.train_number
    WHERE a.station_code = ?
      AND b.station_code = ?
    ORDER BY a.departure
    """

    return execute_readonly_query(
        query,
        (from_station_code.strip().upper(),
         to_station_code.strip().upper()),
        max_rows,
        db_path=db_path,
    )


def get_train_schedule(
    train_number: str,
    max_rows: int = MAX_ROW_LIMIT,
    *,
    db_path: str | Path | None = None,
) -> dict:
    """Get the complete timetable of a train."""

    if not isinstance(train_number, str) or not train_number.strip():
        return {"error": "Train number must be a non-empty string."}

    query = """
    SELECT
        s.train_number,
        s.stop_sequence,
        s.station_code,
        st.name AS station_name,
        s.day,
        s.arrival,
        s.departure,
        s.halt_min,
        s.is_halt
    FROM schedules s
    LEFT JOIN stations st
        ON st.code = s.station_code
    WHERE s.train_number = ?
    ORDER BY s.day, s.stop_sequence
    """

    return execute_readonly_query(
        query,
        (train_number.strip(),),
        max_rows,
        db_path=db_path,
    )


def _escape_like(value: str) -> str:
    """Escape SQL LIKE wildcards so literal %, _, and \\ match safely."""

    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def find_station(
    station_name: str,
    max_rows: int = 20,
    *,
    db_path: str | Path | None = None,
) -> dict:
    """Find stations by name."""

    if not isinstance(station_name, str) or not station_name.strip():
        return {"error": "Station name must be a non-empty string."}

    query = """
    SELECT code, name, state, zone
    FROM stations
    WHERE name LIKE ? ESCAPE '\\'
    ORDER BY name
    """

    escaped = _escape_like(station_name.strip())

    return execute_readonly_query(
        query,
        (f"%{escaped}%",),
        max_rows,
        db_path=db_path,
    )

