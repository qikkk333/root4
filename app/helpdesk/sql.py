
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence


# Find the database relative to this Python file
DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "database"
    / "rail_timetable.db"
)

DEFAULT_ROW_LIMIT = 100
MAX_ROW_LIMIT = 1000


def is_select_query(query: str) -> bool:
    """Check whether the SQL statement starts with SELECT or WITH."""

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

    # Block the load_extension function
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (arg2 or "").lower()

        if function_name == "load_extension":
            return sqlite3.SQLITE_DENY

    if action in allowed_actions:
        return sqlite3.SQLITE_OK

    return sqlite3.SQLITE_DENY


def execute_readonly_query(
    query: str,
    params: Sequence[Any] | Mapping[str, Any] = (),
    max_rows: int = DEFAULT_ROW_LIMIT
) -> dict:
    """
    Execute a safe, read-only SQL query.

    Returns columns, rows, and truncation status.
    """

    if not isinstance(query, str) or not query.strip():
        return {"error": "SQL query cannot be empty."}

    if not is_select_query(query):
        return {"error": "Only SELECT queries are allowed."}

    if not isinstance(max_rows, int) or isinstance(max_rows, bool):
        return {"error": "Row limit must be an integer."}

    if max_rows < 1 or max_rows > MAX_ROW_LIMIT:
        return {
            "error": f"Row limit must be between 1 and {MAX_ROW_LIMIT}."
        }

    if not DB_PATH.is_file():
        return {"error": f"Database not found: {DB_PATH}"}

    connection = None

    try:
        # Open the database in read-only mode
        database_uri = DB_PATH.as_uri() + "?mode=ro"

        connection = sqlite3.connect(database_uri, uri=True)
        connection.row_factory = sqlite3.Row

        # Enable SQLite's query-only protection
        connection.execute("PRAGMA query_only = ON")

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
            "row_count": len(rows)
        }

    except sqlite3.Error as error:
        return {"error": f"Database query failed: {error}"}

    finally:
        if connection is not None:
            connection.close()

def find_train(train_number: str) -> dict:
    """Find a train using its train number."""

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
        (train_number.strip(),)
    )


def find_trains_between(
    from_station_code: str,
    to_station_code: str,
    max_rows: int = DEFAULT_ROW_LIMIT
) -> dict:
    """Find trains travelling between two stations."""

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
        max_rows
    )



def get_train_schedule(
    train_number: str,
    max_rows: int = MAX_ROW_LIMIT
) -> dict:
    """Get the complete timetable of a train."""

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
        max_rows
    )

def find_station(station_name: str, max_rows: int = 20) -> dict:
    """Find stations by name."""

    query = """
    SELECT code, name, state, zone
    FROM stations
    WHERE name LIKE ?
    ORDER BY name
    """

    return execute_readonly_query(
        query,
        (f"%{station_name.strip()}%",),
        max_rows
    )

