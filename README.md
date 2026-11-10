# Railway Passenger Help Desk (`root4`)

Hybrid RAG + SQL passenger help desk built for the GenAI Nexus Fellowship Capstone Project.

## Repository Layout

```text
root4/
├── database/
│   ├── rail_timetable.db          # 2016 SQLite Indian Railways timetable snapshot
│   └── schema.md                  # Schema reference and snapshot limitations
├── helpdesk/
│   └── sql.py                     # Compatibility shim re-exporting app.helpdesk.sql
├── sql_tests/
│   ├── inspect_db.py              # Sample route inspection script (JP -> UDZ)
│   └── test_sample_sql.py         # Sample SQL smoke checks
└── app/
    ├── helpdesk/
    │   ├── sql.py                 # Authoritative read-only SQLite access layer
    │   ├── ingest.py              # Policy document ingestion & chunking
    │   ├── search.py              # Hybrid dense + BM25 + cross-encoder retrieval
    │   └── agent.py               # Switchable agents including RouterAgent (RAG + SQL)
    ├── eval/                      # Retrieval & end-to-end evaluation scripts/datasets
    ├── index/                     # Pre-built ChromaDB + JSON index (103 chunks)
    └── tests/
        └── test_sql_integration.py # Automated offline pytest SQL & RouterAgent suite
```

## SQL Architecture & Path Resolution

- **Authoritative SQL module:** `app/helpdesk/sql.py` is the single source of truth for all database access (`execute_readonly_query`, `find_train`, `find_trains_between`, `get_train_schedule`, `find_station`, `get_db_path`, `get_schema_path`).
- **Root compatibility shim:** `helpdesk/sql.py` at the repository root re-exports symbols directly from `app.helpdesk.sql` with no duplicate implementation logic so imports from either `app/` or the repository root share one implementation.
- **Database path resolution:** `app/helpdesk/sql.py` resolves `DEFAULT_DB_PATH` relative to `Path(__file__).resolve()` (`<repo_root>/database/rail_timetable.db`), completely independent of the current working directory and without runtime monkey-patching.
- **Configuration & missing-DB behavior:** Set `RAIL_TIMETABLE_DB_PATH` (or pass `db_path=...` in Python) to point to a different SQLite file. If the target file does not exist, queries return `{"error": "Database not found: ..."}` and never create an empty replacement database.

## Read-Only SQL Security Model

Every query executed through `execute_readonly_query()` passes five defense-in-depth layers:

1. **Statement validation (`is_select_query`):** Rejects anything that does not start with `SELECT` or `WITH` (ignoring leading comments/whitespace).
2. **Read-only URI connection:** Opens the database via `Path.as_uri() + "?mode=ro"` (`uri=True`).
3. **`PRAGMA query_only = ON`:** Enforces SQLite read-only mode at the connection level.
4. **SQLite authorizer (`readonly_authorizer`):** Allows only `SQLITE_SELECT`, `SQLITE_READ`, `SQLITE_FUNCTION`, and `SQLITE_RECURSIVE`, explicitly denying `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `ATTACH`, `PRAGMA`, `VACUUM`, `REPLACE`, `load_extension`, and `fts3_tokenizer`.
5. **Execution bounds & parameterization:** Rejects multi-statement strings (`sqlite3.Cursor.execute`), checks `params` type and uses `?`/named parameter binding, enforces `1 <= max_rows <= 1000`, and aborts runaway queries/recursive CTEs via `set_progress_handler` after `timeout_sec` (default `5.0`s).

`RouterAgent` in `app/helpdesk/agent.py` adds an early pre-LLM refusal guard (`UNSAFE_RE`) for write attempts, runs generated SQL through `sql.execute_readonly_query()`, revalidates any 1-shot repaired query through the same `execute_readonly_query()` gate, preserves the executed query in `out["sql"]`, and abstains safely if SQL execution fails.

## Historical Timetable Limitations (2016 Snapshot)

- `database/rail_timetable.db` is a **historical 2016 snapshot** (see [database/schema.md](database/schema.md)).
- Train schedules, numbers, and routes may differ from current operations, and newer services (such as **Vande Bharat Express** trains) are absent.
- The database contains **no fares, seat availability, waiting-list predictions, PNR status, or live running status**.

## Running SQL & Integration Tests

All SQL tests are deterministic, offline, and require **no `GEMINI_API_KEY`**.

```powershell
# Automated pytest suite (requires pytest, e.g. system Python 3.12 with pytest)
python -m pytest app/tests/test_sql_integration.py -v

# Standalone SQL smoke scripts (run cleanly from repo root or app/)
.\app\.venv\Scripts\python.exe sql_tests/inspect_db.py
.\app\.venv\Scripts\python.exe sql_tests/test_sample_sql.py
```

*(Note: `app/.venv` contains all runtime dependencies from `app/requirements.txt`, while `pytest` is used from the environment where `pytest` is installed.)*
