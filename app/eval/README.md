# Evaluation Framework & Metric Definitions

All evaluation scripts and datasets live in `app/eval/`.

## 1. Datasets

| Dataset | Purpose | Questions | Key Required |
|---|---|---:|---|
| [`test_set.json`](test_set.json) | Policy document QA & unanswerable rules (`T01`-`T32`, `D01`-`D05`) | 37 | Retrieval: No / Agent: Yes (live) or Offline |
| [`test_set_hard.json`](test_set_hard.json) | Hard paraphrased / abbreviation queries (`H03`-`H32`) | 25 | Retrieval: No |
| [`test_set_sql.json`](test_set_sql.json) | 2016 SQLite timetable queries, overnight math, aggregations, ambiguous stations, zero-row & live refusals (`S01`-`S17`) | 17 | `--offline`: No / Live: Yes |
| [`test_set_hybrid.json`](test_set_hybrid.json) | Docs-only, SQL-only, mixed SQL+docs, timetable follow-ups, unsupported abstentions (`M01`-`M10`) | 10 | `--offline`: No / Live: Yes |
| [`test_set_refusal.json`](test_set_refusal.json) | Live status, PNR, seats, fares, unsupported policies, ambiguous questions, SQL write attacks (`R01`-`R12`) | 12 | `--offline`: No / Live: Yes |

Every train number, station code, halt time, duration, count, and `(doc, page)` pair in `test_set_sql.json`, `test_set_hybrid.json`, and `test_set_refusal.json` was independently verified against `database/rail_timetable.db` and `app/index/chunks.json`.

---

## 2. Retrieval Metric Formulas (`evaluate_retrieval.py`)

For each question with non-empty gold sources $G = \{(\text{doc}, \text{page})\}$ retrieved top-$K$ unique pages $R_K = [p_1, \dots, p_m]$ ($m \le K$):

- **$\text{Hit}@K$:** $1.0$ if $R_K \cap G \neq \emptyset$, else $0.0$.
- **$\text{Recall}@K$:** $\frac{|R_K \cap G|}{|G|}$ (note: pages within the same document deduplicated).
- **$\text{Precision}@K$:** $\frac{|R_K \cap G|}{K}$.
- **$\text{MRR}$ (Mean Reciprocal Rank):** Mean of $\frac{1}{\text{rank}_1}$ across questions where $\text{rank}_1$ is the 1-based rank of the first element of $R_K$ in $G$ ($0.0$ if none).
- **$\text{Ans}@K$:** Fraction of normalized numbers/times in the reference answer that appear in the top-$K$ retrieved chunk texts (`None` when the reference has no numbers).

---

## 3. Citation Precision, Citation Recall & Error Counters (`evaluate_agent.py`)

Let $C$ be the list of dicts in `out["sources"]`, $G = \{(\text{doc}, \text{page})\}$ be the gold document citations for the question, and $V$ be the set of valid indexed `(doc, page)` pairs loaded from `app/index/chunks.json`.

A cited entry $c = (\text{doc}, \text{page}) \in C$ counts as a valid relevant true positive ($\text{tp}$) **only once** per unique `(doc, page)` when **all** of the following hold:
1. $\text{doc}$ is a non-empty string and $\text{doc} \neq \text{"timetable database"}$ (`DB_DOC` SQL leakage is never credited as a document citation).
2. $\text{page}$ is syntactically valid (an integer $\ge 1$ for `.pdf` files; `None` or integer $\ge 1$ for `.html`/non-PDF documents; never `bool` or $\le 0$).
3. $(\text{doc}, \text{page}) \in V$ when $V$ is loaded.
4. $(\text{doc}, \text{page}) \in G$.

Formulas and boundary conventions:

- **When $|G| > 0$ (gold documents exist):**
  - $\text{Citation Precision} = \frac{\text{tp}}{|C|}$ if $|C| > 0$, else $0.0$ (`missing_citation_count += 1`).
  - $\text{Citation Recall} = \frac{\text{tp}}{|G|}$.
- **When $|G| = 0$ (SQL-only, refusal, or abstention items expecting zero document citations):**
  - $\text{Citation Precision} = 1.0$ and $\text{Citation Recall} = 1.0$ when $|C| = 0$.
  - If $|C| > 0$ (spurious citation or SQL leakage), both score $0.0$.
- `summarize_results()` reports both **overall macro-averages** (`citation_precision_macro`, `citation_recall_macro`) across all items and **gold-doc-only macro-averages** (`citation_precision_doc_only`, `citation_recall_doc_only`), plus total `invalid_citation_count`, `missing_citation_count`, `out_of_range_page_count`, `doc_mismatch_count`, and `sql_leak_count`.

---

## 4. SQL & Routing Metrics (`evaluate_agent.py`)

- **Routing Accuracy:** Fraction of items whose classified output route (`sql`, `documents`, `mixed`, `abstain`, `refuse`) matches `expected_route`.
- **SQL Execution Success vs. SQL Result Correctness (kept strictly separate):**
  - **Execution success:** `out["sql"]` executes cleanly through `helpdesk.sql.execute_readonly_query()` without database or policy errors.
  - **Result correctness:** Row presence (`expect_rows`) matches expectations, required `expected_facts` appear in output rows/answer, and every scalar value returned by the gold SQL appears in the predicted SQL result rows (column-alias agnostic). For zero-row queries (`S15`), verifies clean `dont_know=True` abstention without fabricated timetable facts.
- **Unsafe Rejection Rate:** Share of explicit write/DDL/DML attacks cleanly blocked pre-LLM with `REFUSAL` and `sql=None`.

---

## 5. Commands (PowerShell, run from `app/`)

```powershell
# 1. Offline retrieval metrics (no GEMINI_API_KEY)
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe eval\evaluate_retrieval.py --verbose
.\.venv\Scripts\python.exe eval\evaluate_retrieval.py --questions eval\test_set_hard.json

# 2. Deterministic offline SQL + Hybrid + Refusal evaluation (no GEMINI_API_KEY)
.\.venv\Scripts\python.exe eval\evaluate_agent.py --offline --suite all --out eval\results\offline_report.json

# 3. Include live local retrieval benchmark summary inside the offline report
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe eval\evaluate_agent.py --offline --suite all --include-retrieval --out eval\results\offline_report.json

# 4. Live Gemini end-to-end evaluation (requires GEMINI_API_KEY in app/.env)
.\.venv\Scripts\python.exe eval\evaluate_agent.py --agent router --suite all --out eval\results\live_report.json
```
