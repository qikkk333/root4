
import sqlite3

connection = sqlite3.connect(
    "file:rail_capstone_pack/database/rail_timetable.db?mode=ro",
    uri=True
)

cursor = connection.cursor()

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

cursor.execute(query, ("JP", "UDZ"))

results = cursor.fetchall()

print("Trains from Jaipur to Udaipur City:")

for train in results:
    print(train)

connection.close()
