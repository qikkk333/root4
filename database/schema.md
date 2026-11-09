# rail_timetable.db: schema

SQLite, about 28 MB. Four tables and one view. Built by `build_pack.py` from the datameet railways data (CC0).

## Snapshot date: read this first

The timetable is a **2016 snapshot** (the source repository was last changed on 8 Aug 2016, after a few months of collecting). Today is 30 Oct 2026, so it is ten years old:

- Many trains have changed times, numbers or routes since then, and newer trains (for example every Vande Bharat train) are not in it.
- Treat answers as "according to the 2016 timetable in this pack", not as today's live timetable.
- There are no fares, seat availability, waiting lists, PNRs or live running status in the database.

## Tables

### stations (8,990 rows)

| Column | Type | Meaning |
|---|---|---|
| code | TEXT, primary key | Station code, for example `NDLS` (New Delhi), `SBC` (Bangalore City Jn) |
| name | TEXT | Station name as in the source, for example `NEW DELHI`, `Kalka`. Mixed upper and lower case |
| state | TEXT | State, for example `Goa`. NULL for 4,593 stations (about half), so counts by state are incomplete |
| zone | TEXT | Railway zone code, for example `NR`, `WR`, `KR`. NULL for 4,638 stations |
| address | TEXT | Free-text address from the source, often NULL |
| latitude | REAL | Latitude, NULL for 293 stations |
| longitude | REAL | Longitude, NULL for 293 stations |

### train_types (16 rows)

| Column | Type | Meaning |
|---|---|---|
| type_code | TEXT, primary key | Short code used in `trains.type_code`, for example `Raj` |
| meaning | TEXT | What it stands for, for example `Rajdhani Express`, `Shatabdi Express`, `Jan Shatabdi Express`, `Duronto Express`, `Garib Rath Express`, `Superfast Express`, `Passenger` |

The type comes from the source and is not always what the name suggests: for example 12004 (a Swarna Shatabdi) has type `SF`, and some Garib Rath trains have type `Exp`.

### trains (5,208 rows)

| Column | Type | Meaning |
|---|---|---|
| number | TEXT, primary key | Train number, for example `12951`. Text, not a number: keep the leading zero (`04601`). 69 slip services look like `11039-Slip` |
| name | TEXT | Train name, for example `Mumbai Central-New Delhi Rajdhani Express` |
| type_code | TEXT | Train type, see `train_types`. NULL for 15 trains |
| zone | TEXT | Zone that runs the train, NULL for 290 |
| from_station_code | TEXT | Code of the starting station |
| to_station_code | TEXT | Code of the last station |
| departure | TEXT | Departure time from the starting station, `HH:MM` (24-hour) |
| arrival | TEXT | Arrival time at the last station, `HH:MM` |
| duration_min | INTEGER | Total journey time in minutes |
| distance_km | INTEGER | Total distance in km |
| return_train | TEXT | Number of the train running the opposite way, if known |
| has_1a | INTEGER | 1 if the train has AC First class (1A), else 0 |
| has_2a | INTEGER | 1 if it has AC 2-tier (2A) |
| has_3a | INTEGER | 1 if it has AC 3-tier (3A) |
| has_sl | INTEGER | 1 if it has Sleeper class (SL) |
| has_cc | INTEGER | 1 if it has AC Chair Car (CC) |
| has_fc | INTEGER | 1 if it has First Class (FC, non-AC) |

The class flags come from the source and are not complete (for example many unreserved passenger trains have every flag at 0).

### schedules (417,080 rows)

One row per station on a train's route, **including stations the train passes without stopping**.

| Column | Type | Meaning |
|---|---|---|
| train_number | TEXT | Train number (`trains.number`) |
| stop_sequence | INTEGER | Order along the route: 1 is the starting station. Primary key is (train_number, stop_sequence) |
| station_code | TEXT | Station code (`stations.code`) |
| day | INTEGER | Day of the journey at this station: 1 is the day the train leaves its starting station, 2 the next day, and so on. NULL for 641 trains that have no times in the source |
| arrival | TEXT | Arrival time `HH:MM`. NULL at the starting station |
| departure | TEXT | Departure time `HH:MM`. NULL at the last station |
| halt_min | INTEGER | Minutes the train stands at the station (departure minus arrival). NULL at the first and last station |
| is_halt | INTEGER | 1 if the train stops here (first station, last station, or departure later than arrival). 0 if it only passes through (arrival equals departure in the source). NULL if the source has no times for this row |

### stops (view)

`SELECT * FROM schedules WHERE is_halt = 1`: only the stations where the train actually stops. Use this for questions like "how many stops", "which trains go from A to B" or "which trains stop at X". For example, the Mumbai Rajdhani (12951) has 202 rows in `schedules` but only 6 rows in `stops`.

## Useful patterns

- Trains from A to B: join `stops` to itself on `train_number`, with A's `stop_sequence` smaller than B's.
- Travel time between two stops: `(b.day - a.day) * 1440` plus B's arrival minus A's departure, in minutes. Without `day`, overnight trips come out negative.
- Station names: match with `LIKE` and check the result, since several stations can share a word (for example `VADODARA JN` and `VADODARA C CABIN`).
- Times are text `HH:MM`, so `ORDER BY departure` sorts correctly within one day.

## Cleaning done by build_pack.py

- Times shortened from `HH:MM:SS` to `HH:MM`; the text `"None"` turned into NULL.
- `stop_sequence` taken from the source row order (the source has no sequence column).
- `is_halt` and `halt_min` worked out from arrival and departure; the `stops` view added.
- State `Orissa` renamed to `Odisha`; empty states and the zone `?` set to NULL.
- Six class columns renamed (`first_ac` to `has_1a` and so on) and stored as 0 or 1; `duration_h` and `duration_m` combined into `duration_min`.
- The `train_types` table was added by hand from the codes used in the source.
