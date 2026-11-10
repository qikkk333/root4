"""Backward-compatible re-export shim for ``app.helpdesk.sql``.

The authoritative SQL implementation lives in ``app/helpdesk/sql.py``.
This module exists only so imports of ``helpdesk.sql`` from the repository
root resolve to that single implementation without duplicating SQL code.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.helpdesk.sql import (  # noqa: E402, F401
    APP_DIR,
    DB_PATH,
    DB_PATH_ENV_VAR,
    DEFAULT_DB_PATH,
    DEFAULT_QUERY_TIMEOUT,
    DEFAULT_ROW_LIMIT,
    MAX_ROW_LIMIT,
    PACKAGE_DIR,
    REPO_ROOT,
    execute_readonly_query,
    find_station,
    find_train,
    find_trains_between,
    get_db_path,
    get_schema_path,
    get_train_schedule,
    is_select_query,
    readonly_authorizer,
)

__all__ = [
    "APP_DIR",
    "DB_PATH",
    "DB_PATH_ENV_VAR",
    "DEFAULT_DB_PATH",
    "DEFAULT_QUERY_TIMEOUT",
    "DEFAULT_ROW_LIMIT",
    "MAX_ROW_LIMIT",
    "PACKAGE_DIR",
    "REPO_ROOT",
    "execute_readonly_query",
    "find_station",
    "find_train",
    "find_trains_between",
    "get_db_path",
    "get_schema_path",
    "get_train_schedule",
    "is_select_query",
    "readonly_authorizer",
]

