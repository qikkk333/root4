# Retrieval experiments

Retrieval only, no LLM. Each change is measured on two question sets and kept only if it helps without hurting the other set.

The scores are page-level: several chunks from the same page count once. K = 5.

- **main**: `eval/test_set.json`, 32 questions with gold pages.
- **hard**: `eval/test_set_hard.json`, 25 rewordings of the main questions with abbreviations and chat-style wording ("SL", "2AC CNF", "hrs", "dep", "WL", "ARP", "6 pm"). The gold pages are the same as in the main set.

Run: `python eval/evaluate_retrieval.py [--questions eval/test_set_hard.json] [--verbose]`

## Part A: ingest and metadata (`helpdesk/ingest.py` replaces `dev/dev_chunks.py`)

| Step | Chunks | main hybrid_rerank Recall / MRR |
|---|---|---|
| dev chunks (baseline) | 128 | 0.94 / 0.88 |
| ingest.py: tables kept whole, list scope inheritance, 300 dpi OCR, Hindi/junk pages dropped, OLD/REVISED rule split | 98 | 0.94 / 0.90 |
| + PIB footer and duplicate page copy removed | 95 | 0.94 / 0.90 |

## Part B: search (`helpdesk/search.py`)

The hard-set baseline uses the Part A index.

| Change | main hybrid R / MRR | main rerank R / MRR | hard hybrid R / MRR | hard rerank R / MRR | Kept? |
|---|---|---|---|---|---|
| baseline (Part A index) | 0.90 / 0.84 | 0.94 / 0.90 | 0.83 / 0.73 | 0.95 / 0.82 | |
| B2: abbreviation expansion, days→hours, am/pm→24h | 0.88 / 0.84 | 0.94 / 0.90 | **0.95** / 0.74 | 0.95 / 0.82 | yes |
| cross-encoder reads the original instead of the expanded question | 0.88 / 0.84 | 0.94 / 0.90 | 0.95 / 0.74 | 0.95 / 0.82 | no (same) |
| cross-encoder reads `embedding_text` (title/date prefix) | 0.88 / 0.84 | 0.94 / **0.88** | 0.95 / 0.74 | 0.96 / 0.83 | no (hurts main) |
| B4: recency, old rule −2 and doc 2+ years older than newest on same topic −2 | 0.88 / 0.84 | 0.94 / 0.90 | 0.95 / 0.74 | 0.95 / **0.85** | yes |
| (penalty 1 / 3 instead of 2) | | 0.94 / 0.90 | | 0.95 / 0.84 / 0.85 | 2 chosen |
| B6: RERANK_POOL 10 | | 0.91 / 0.88 | | 0.95 / 0.88 | no (loses main recall) |
| B6: RERANK_POOL 30 | | 0.94 / 0.90 | | 0.95 / 0.85 | no (same, slower) |
| B6: RRF_K 20 | | 0.94 / 0.90 | | 0.95 / 0.85 | no (same) |
| B6: CANDIDATE_K 20 → 30 | **0.91 / 0.85** | 0.94 / **0.91** | 0.95 / 0.75 | 0.95 / 0.85 | yes |

### Extra tests after the full end-to-end run (trying to fix T01)

| Change | main hybrid R / MRR | main rerank R / MRR | hard hybrid R / MRR | hard rerank R / MRR | Kept? |
|---|---|---|---|---|---|
| BM25 stemming (`STEM`: booking→book, cancellation→cancel) | 0.96 / 0.93 | 0.94 / 0.91 | 0.91 / 0.79 | 0.95 / 0.85 | no (mixed: loses hard recall; rerank same) |
| synonym "times of day / timings" → "hours" | 0.88 / 0.84 | **0.91** / 0.90 | 0.95 / 0.74 | 0.95 / 0.88 | no (loses main recall) |
| `DOC_EXPAND`: short forms expanded inside chunks for BM25 too | 0.91 / 0.85 | 0.91 / 0.90 | 0.93 / 0.77 | 0.95 / 0.88 | no (measured together with the synonym above) |

None of these brought the T01 chunk into the top 10. `STEM` and `DOC_EXPAND` stay in `search.py` as switches, set to False.

### T01 ingest fix (preamble sections + normal times)

`ingest.py` changes:

- The document header (title / w.e.f. / Authority) becomes its own section, "Document authority and effective date".
- A short "Label: value" line before the first heading gets its own section named after its label. This applies to the "Ticket Booking/cancellation Hours" line.
- "00.20 HRS" is written as "00:20 hours".
- New topic: `booking_hours`.

Only the IRCTC document changed (40 → 41 chunks, 96 in total).

`evaluate_retrieval.py` has a new column, **Ans@5**: the share of the reference answer's numbers found in the top-5 **chunks**, which is what the agent reads. Page-level scores hid T01: its page counted at rank 5, but its chunk was not in the top 5.

| | main rerank R / MRR / Ans@5 | hard rerank R / MRR / Ans@5 |
|---|---|---|
| before fix | 0.94 / 0.91 / 0.74 | 0.95 / 0.85 / (n/a) |
| after fix | 0.94 / 0.91 / 0.74 | 0.95 / 0.85 / 0.63 |

- No regressions. T31 (the Authority letter) is still rank 1.
- The booking-hours chunk is rank 1 for "booking hours" or "ticket booking and cancellation hours".
- T01's own wording ("what times of day can I book or cancel online") still does not reach it. Re-tested on the new index: STEM and a "times of day → hours" rule do not bring it into the top 10, and the rule costs main recall (0.94 → 0.91). Both stay off.
- Next idea: let the agent's rewrite (multi / tools agent) supply "booking hours". Retrieval cannot guess that wording from rules alone.

### Document overview chunks + OVERVIEW_PENALTY

- `ingest.py` adds one `<doc>_overview` chunk per document, listing its sections and topics (originally 103 chunks in total; later audited/cleaned to 100 chunks after removing Hindi Gazette header stubs and uniting Rule 11 clauses).
- Problem found by the end-to-end test: on the plain index the IRCTC overview chunk took rank 2 for T23 and pushed
  the answer chunk to rank 6, so rag fell from 35/37 to 34/37.
- Fix in `search.py`: overview chunks lose 3 rerank points unless the question asks about topics, sections,
  contents or what a document covers (`OVERVIEW_RE`).

| | main rerank R / MRR / Ans@5 | hard rerank R / MRR / Ans@5 | rag end-to-end |
|---|---|---|---|
| overview chunks, no penalty | 0.94 / 0.92 / 0.75 | 0.95 / 0.85 / 0.65 | 34/37 (T23 lost) |
| + OVERVIEW_PENALTY 3 | 0.94 / 0.91 / 0.76 | 0.95 / 0.85 / 0.63 | **35/37** |

"All the topics in the IRCTC refund rules document" still returns the overview chunk first.

### Final (current defaults)

| Set | Mode | Hit@5 | Recall@5 | MRR |
|---|---|---|---|---|
| main | hybrid_rerank | 1.00 | 0.94 | 0.91 (was 0.88) |
| main | hybrid | 0.97 | 0.91 | 0.85 (was 0.84) |
| hard | hybrid_rerank | 1.00 | 0.95 | 0.85 (was 0.82) |
| hard | hybrid | 1.00 | 0.95 | 0.75 (was 0.73) |

### Not done, and why

- **B3, automatic soft `prefer` from question words** (scope/topic): every Vande Bharat and Amrit Bharat question is already rank 1, so there was nothing left for it to fix. The `prefer=` argument is still there for the agent to use.
- **B5, neighbour chunks**: added as `search(..., neighbors=True)`, which puts `context_text` (the chunk plus the chunks before and after it in the same section) on each result. It does not change ranking, so it has no metric. It is useful when an answer crosses a chunk border.
- **B7, bigger models** (bge-base, a larger cross-encoder): not tried yet. They are slower, and the hard set is already at Hit@5 1.00.

### Still weak (rank 3 to 5)

- T01 "times of day to book/cancel online": the gold page is IRCTC p1, but the answer is a single line in a long chunk.
- T30 / H30 "is a TDR needed when the train is cancelled": the answer is spread over the 2015 HTML and IRCTC p8.
- H18 "counter ticket online cancellation": the documents say "i-ticket" / "PRS counter ticket".
