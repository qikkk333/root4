# Manual Actions Required

All locally automatable engineering, SQL integration, public API, grounding, offline evaluation, reliability, retrieval audit, and documentation tasks have been completed and verified on branch `gemini-integration`.

The following items genuinely require your manual involvement because they depend on external credentials, remote permissions, or source assets intentionally excluded from Git:

---

## 1. Add Your Real `GEMINI_API_KEY` to `app/.env`

Do **not** commit `.env` (it is already listed in `.gitignore` and `app/.gitignore`).

```powershell
Copy-Item app\.env.example app\.env
notepad app\.env
```

Set inside `app/.env`:

```env
GEMINI_API_KEY=your-real-gemini-api-key
HELPDESK_AGENT=router
```

---

## 2. Run a Live Gemini Smoke Test (`ask()` and CLI)

Once `app/.env` contains `GEMINI_API_KEY`, verify live structured outputs, tool calling, SQL planning, and grounding against Gemini:

```powershell
$env:HF_HUB_OFFLINE="1"
.\app\.venv\Scripts\python.exe -c "from app.helpdesk import ask; print(ask('When does train 12002 leave New Delhi?'))"
.\app\.venv\Scripts\python.exe -c "from app.helpdesk import ask; print(ask('What is the flat cancellation charge for a sleeper class ticket cancelled more than 48 hours before departure?'))"
.\app\.venv\Scripts\python.exe -c "from app.helpdesk import ask; print(ask('When does train 12002 depart and what is the advance reservation period?'))"
.\app\.venv\Scripts\python.exe -c "from app.helpdesk import ask; print(ask('What is the live status of train 12951?'))"
```

Expected checks:
- Verify `gemini-3.5-flash-lite` (default in `AGENT_MODEL`) works cleanly on your Gemini key, or override `AGENT_MODEL` in `app/.env` if your account uses a different Gemini model string.
- Confirm mixed/SQL answers cite `"2016"` snapshot caveats and never include `"timetable database"` in `out["sources"]`.

---

## 3. Run Live Gemini End-to-End Evaluation

Run live end-to-end scoring (and save local reports inside `app/eval/results/`, which is gitignored):

```powershell
$env:HF_HUB_OFFLINE="1"
.\app\.venv\Scripts\python.exe app\eval\evaluate_agent.py --agent router --suite main --out app\eval\results\live_main.json
.\app\.venv\Scripts\python.exe app\eval\evaluate_agent.py --agent router --suite all --out app\eval\results\live_all.json
```

---

## 4. Only If Index Rebuilding Is Needed: Supply Raw Document Pack

`app/index/` (`103` chunks in `chunks.json` and `chroma/`) is pre-built and intact, so rebuilding is **not** needed for normal execution. If you ever modify `app/helpdesk/ingest.py` or want to inspect/rebuild from raw PDFs/HTMLs (e.g., if investigating `rb_cc10_2025_reservation_charts.pdf` page 1), extract the original capstone document pack into `app/rail_capstone_pack/` (gitignored) and run:

```powershell
.\app\.venv\Scripts\python.exe app\build_index.py
```

---

## 5. Review Local Commits & Push When Ready

All milestone commits on `gemini-integration` were kept local per instructions (`main` and `origin/main` remain untouched):

```powershell
git log --oneline --decorate -10
git status --short --branch
git push origin gemini-integration
```

*(Note: Optional local convenience step — `pytest` runs cleanly via system `python -m pytest app/tests -v`. If you want `pytest` inside `app\.venv` too, run `.\app\.venv\Scripts\python.exe -m pip install pytest`.)*
