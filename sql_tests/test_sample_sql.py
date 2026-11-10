"""Sample smoke checks against the authoritative app.helpdesk.sql module."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.helpdesk.sql import execute_readonly_query, find_station  # noqa: E402


def main():
    # S07: Find the departure time of train 12002 at New Delhi
    query1 = """
    SELECT departure
    FROM schedules
    WHERE train_number = ?
      AND station_code = ?
    """
    result1 = execute_readonly_query(query1, ("12002", "NDLS"))
    assert "error" not in result1, result1
    print("S07 - Train 12002 departure:")
    print(result1)

    # S09: Count the Rajdhani trains
    query2 = """
    SELECT COUNT(*) AS rajdhani_count
    FROM trains
    WHERE type_code = ?
    """
    result2 = execute_readonly_query(query2, ("Raj",))
    assert "error" not in result2, result2
    print("\nS09 - Rajdhani train count:")
    print(result2)

    # Test overnight journey duration: train 12315 (JP -> UDZ)
    query3 = """
    SELECT
        a.departure,
        b.arrival,
        a.day AS departure_day,
        b.day AS arrival_day,
        (b.day - a.day) * 1440
        + (
            CAST(substr(b.arrival, 1, 2) AS INTEGER) * 60
            + CAST(substr(b.arrival, 4, 2) AS INTEGER)
          )
        - (
            CAST(substr(a.departure, 1, 2) AS INTEGER) * 60
            + CAST(substr(a.departure, 4, 2) AS INTEGER)
          ) AS travel_time_min
    FROM stops a
    JOIN stops b
        ON b.train_number = a.train_number
        AND b.stop_sequence > a.stop_sequence
    WHERE a.train_number = ?
      AND a.station_code = ?
      AND b.station_code = ?
    """
    result3 = execute_readonly_query(query3, ("12315", "JP", "UDZ"))
    assert "error" not in result3, result3
    print("\nOvernight journey test:")
    print(result3)

    result4 = find_station("NEW DELHI")
    assert "error" not in result4, result4
    print("\nFind station NEW DELHI:")
    print(result4)


if __name__ == "__main__":
    main()
