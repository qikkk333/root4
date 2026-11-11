# Manual Actions Required

All locally automatable engineering, SQL integration, public API, grounding, offline evaluation, reliability, retrieval audit, and documentation tasks have been completed and verified on branch `gemini-integration`.

The following items genuinely require your manual involvement because they depend on external credentials, remote permissions, or source assets intentionally excluded from Git:

---

## 1. Create untracked `app/.env` and Add Your Real `GEMINI_API_KEY`

`.env` is explicitly ignored by Git at both repository root (`.gitignore`) and `app/` (`app/.gitignore`). Do **not** paste your key in chat and do **not** commit `.env`.

```powershell
# 1. Verify Git ignores .env while keeping app/.env.example tracked
git check-ignore -v .env app/.env

# 2. Create app/.env from template
Copy-Item app\.env.example app\.env

# 3. Edit app/.env in Notepad (replace GEMINI_API_KEY= with your real key; uncomment HELPDESK_AGENT=router or leave commented since router is default)
notepad app\.env
```

Or write `app/.env` securely without storing your API key in PowerShell command history:

```powershell
Copy-Item app\.env.example app\.env
$k = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR(
        (Read-Host "Enter GEMINI_API_KEY" -AsSecureString)
    )
).Trim()
@("GEMINI_API_KEY=$k", "HELPDESK_AGENT=router") | Set-Content app\.env -Encoding utf8
Remove-Variable k
```

Confirm Git still treats `app/.env` as ignored:

```powershell
git status --short --branch
```

---

## 2. Run the Live Gemini Smoke Test (`ask()` + `RouterAgent`)

Once `app/.env` contains a valid `GEMINI_API_KEY`, run the smoke-test runner (which invokes the public `ask()` API via `RouterAgent` across all five routes, never prints secret values, and runs Chroma queries against an isolated temporary copy so `database/rail_timetable.db` and `app/index/chroma/` stay byte-identical):

```powershell
$env:HF_HUB_OFFLINE="1"
.\app\.venv\Scripts\python.exe app\eval\smoke_test_live.py
```

Expected successful output (exit code `0`):

```text
[smoke] agent=router key_env=GEMINI_API_KEY configured=yes (value hidden)
PASS SMOKE_DOC expected=documents actual=documents sources=1 has_sql=False dont_know=False
PASS SMOKE_SQL expected=sql actual=sql sources=0 has_sql=True dont_know=False
PASS SMOKE_MIXED expected=mixed actual=mixed sources=1 has_sql=True dont_know=False
PASS SMOKE_ABSTAIN expected=abstain actual=abstain sources=0 has_sql=False dont_know=True
PASS SMOKE_REFUSE expected=refuse actual=refuse sources=0 has_sql=False dont_know=True
Integrity check: database/rail_timetable.db and app/index/chroma unchanged.

Smoke test result: 5/5 passed
```

Or invoke individual `ask()` probes directly:

```powershell
$env:HF_HUB_OFFLINE="1"
.\app\.venv\Scripts\python.exe -c "from app.helpdesk import ask; print(ask('When does train 12002 leave New Delhi?'))"
.\app\.venv\Scripts\python.exe -c "from app.helpdesk import ask; print(ask('When does train 12002 depart New Delhi and what is the advance reservation period?'))"
```

Expected checks:
- Verify `gemini-3.5-flash-lite` (default in `AGENT_MODEL`) works cleanly on your Gemini key, or set `AGENT_MODEL=<your-model>` in `app/.env` if your account uses a different Gemini model string.
- Confirm mixed/SQL answers mention the `"2016"` timetable snapshot caveat and never include `"timetable database"` in `out["sources"]`.
- Note: Running direct CLI commands (`python -m helpdesk.agent` or `evaluate_agent.py`) opens `app/index/chroma/` directly on Windows (`PersistentClient` updates SQLite/HNSW headers without modifying the `103` chunks); run `git checkout -- app/index/chroma` afterward if needed.

---

## 3. Run Live Gemini End-to-End Evaluation

Run live end-to-end scoring (outputs saved inside gitignored `app/eval/results/`):

```powershell
$env:HF_HUB_OFFLINE="1"
.\app\.venv\Scripts\python.exe app\eval\evaluate_agent.py --agent router --suite main --out app\eval\results\live_main.json
.\app\.venv\Scripts\python.exe app\eval\evaluate_agent.py --agent router --suite all --out app\eval\results\live_all.json
```

---

## 4. Only If Index Rebuilding Is Needed: Supply Raw Document Pack

`app/index/` (`103` chunks in `chunks.json` and `chroma/`) is pre-built and intact, so rebuilding is **not** needed for normal execution. If you ever modify `app/helpdesk/ingest.py` or want to rebuild from raw PDFs/HTMLs (e.g., if investigating `rb_cc10_2025_reservation_charts.pdf` page 1), extract the original capstone document pack into `app/rail_capstone_pack/` (gitignored) and run:

```powershell
.\app\.venv\Scripts\python.exe app\build_index.py
```

---

## 5. Review Status & Approve Final Commit/Push When Ready

Milestone commits `5abb2d0..829dc34` are already on `gemini-integration` and `origin/gemini-integration` (`main` and `origin/main` remain untouched). Review any follow-up working-tree changes, run final live acceptance checks, and commit/push only when you approve:

```powershell
git log --oneline --decorate -10
git status --short --branch
```

*(Note: Optional local convenience step — `pytest` runs cleanly via system `python -m pytest app/tests -v`. If you want `pytest` inside `app\.venv` too, run `.\app\.venv\Scripts\python.exe -m pip install pytest`.)*
