
from rail_capstone_pack.helpdesk.sql import execute_readonly_query

# S07: Find the departure time of train 12002 at New Delhi
query1 = """
SELECT departure
FROM schedules
WHERE train_number = ?
AND station_code = ?
"""

result1 = execute_readonly_query(query1, ("12002", "NDLS"))

print("S07 - Train 12002 departure:")
print(result1)


# S09: Count the Rajdhani trains
query2 = """
SELECT COUNT(*)
FROM trains
WHERE type_code = ?
"""

result2 = execute_readonly_query(query2, ("Raj",))

print("\nS09 - Rajdhani train count:")
print(result2)

# Test overnight journey duration: train 12315
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

result3 = execute_readonly_query(
    query3, ("12315", "JP", "UDZ")
)

print("\nOvernight journey test:")
print(result3)
from rail_capstone_pack.helpdesk.sql import find_station

result = find_station("NEW DELHI")
print(result)