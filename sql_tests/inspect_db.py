"""Inspect sample timetable routes from the read-only 2016 SQLite database."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.helpdesk.sql import find_trains_between, get_db_path  # noqa: E402


def main():
    db_path = get_db_path()
    print(f"Using database: {db_path}")

    result = find_trains_between("JP", "UDZ")
    if "error" in result:
        raise RuntimeError(result["error"])

    print("Trains from Jaipur (JP) to Udaipur City (UDZ):")
    for train in result["rows"]:
        print((train["number"], train["name"], train["departure"], train["arrival"]))


if __name__ == "__main__":
    main()
