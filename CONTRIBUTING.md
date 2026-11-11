# Contributing to Railway Passenger Help Desk

Thank you for contributing to the Railway Passenger Help Desk project! This guide covers our team workflow, coding standards, testing requirements, and security practices.

---

## 1. Repository & Branch Workflow

- **Repository:** `https://github.com/qikkk333/root4`
- **`main`:** Stable integration branch. **Do not push unfinished changes directly to `main`.**
- **`gemini-integration`:** Primary shared development branch for active milestones.
- All new features, bug fixes, tests, and documentation updates should be developed in dedicated branches and merged via Pull Request.

### Recommended Branch Naming

- `feature/short-description` (new capabilities or UI enhancements)
- `fix/short-description` (bug fixes or regression repairs)
- `docs/short-description` (README, guides, or inline documentation)
- `test/short-description` (new automated tests or evaluation cases)

---

## 2. Cloning & Creating a Branch

1. Fork the repository (external contributors) or clone directly (team members):

```bash
git clone https://github.com/qikkk333/root4.git
cd root4
```

2. Sync locally and create a topic branch from `main` (or `gemini-integration` if extending active development):

```bash
git fetch origin
git checkout -b feature/your-topic-name origin/main
```

---

## 3. Local Development Setup

**Requires Python 3.12.** Never install project dependencies globally.

### Option A — Windows PowerShell Project Launcher (Recommended on Windows)

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\setup-helpdesk.ps1
.\helpdesk.ps1 --setup
.\app\.venv\Scripts\python.exe -m pip install -r app\requirements-dev.txt
```

### Option B — Cross-Platform Manual Virtual Environment (Linux / macOS / Windows)

```bash
python3.12 -m venv app/.venv
# Linux / macOS:
./app/.venv/bin/python -m pip install --upgrade pip
./app/.venv/bin/python -m pip install -r app/requirements.txt -r app/requirements-dev.txt
```

### Environment Configuration (Live Gemini Queries Only)

All automated tests and `--offline` evaluations run without an API key. For live interactive chat:

```powershell
Copy-Item app\.env.example app\.env
# Open app/.env locally and set GEMINI_API_KEY=<your-real-key>
```

---

## 4. Running Tests, Linting & Offline Evaluation

Before opening a Pull Request, verify every CI check locally inside your virtual environment:

```bash
# 1. Bytecode compilation check
./app/.venv/bin/python -m compileall -q app

# 2. Complete automated pytest suite
./app/.venv/bin/python -m pytest app/tests -v

# 3. Ruff linting & formatting checks
./app/.venv/bin/ruff check app
./app/.venv/bin/ruff format --check app

# 4. Deterministic offline evaluations (requires no Gemini API key)
./app/.venv/bin/python app/eval/evaluate_agent.py --offline --suite main
./app/.venv/bin/python app/eval/evaluate_agent.py --offline --suite all
```

*(On Windows PowerShell, substitute `.\app\.venv\Scripts\python.exe` or `.\app\.venv\Scripts\ruff.exe`.)*

---

## 5. Commit Message Conventions

Write clear, focused commits using conventional prefixes:

- `feat:` new feature or user-facing capability
- `fix:` bug fix or incorrect output/edge-case repair
- `docs:` README, `CONTRIBUTING.md`, or comment/doc updates
- `test:` unit, integration, or regression tests
- `refactor:` code structure improvement without behavior change
- `ci:` GitHub Actions workflows, templates, or tooling configuration
- `security:` secret hygiene, read-only SQL hardening, or dependency audits

Keep unrelated changes in separate commits and run `git diff --check` before committing.

---

## 6. Updating Your Feature Branch with `main` & Resolving Merge Conflicts

Keep your branch current with `main` before opening or updating a PR:

```bash
git fetch origin
git merge origin/main
```

If a merge conflict occurs:
1. Inspect conflicting files with `git status`.
2. Open each conflicted file, resolve `<<<<<<<` / `=======` / `>>>>>>>` markers carefully without discarding teammates' changes.
3. Re-run `compileall`, `pytest app/tests -v`, `ruff check app`, and `ruff format --check app`.
4. Stage resolved files (`git add <file>`) and run `git commit` to finish the merge.

---

## 7. Pull Request & Code Review Process

1. Push your branch (`git push -u origin feature/your-topic-name`).
2. Open a Pull Request targeting `main` (or `gemini-integration`) and fill out `.github/pull_request_template.md`.
3. Ensure both GitHub Actions workflows (**CI** and **Security**) pass cleanly.
4. Reviewers verify:
   - Read-only SQLite safeguards remain intact (`SELECT`/`WITH` only, authorizer, timeout, no write leaks).
   - Citations stay verified against `app/index/chunks.json` and unsupported queries abstain (`dont_know=True`).
   - Full route/table outputs do not introduce silent row truncation or horizontal terminal overflow.
   - No unintended mutations to `database/rail_timetable.db` or `app/index/`.
5. Address all review comments before merging.

---

## 8. Reporting Bugs

Use our GitHub Issue templates (`.github/ISSUE_TEMPLATE/bug_report.yml` or `feature_request.yml`). Always include Python version, OS, exact steps to reproduce, and sanitized logs with API keys removed.

---

## 9. Protecting API Keys & Tracked Assets

- **Never commit** `app/.env`, real `GEMINI_API_KEY` values, private keys, tokens, or `app/.venv/`.
- Do not paste API keys into terminal logs, screenshots, PR descriptions, or bug reports.
- Note that running live semantic queries invokes `chromadb.PersistentClient` on `app/index/chroma/`, which may touch internal Chroma binary timestamps in the local working tree. Always check `git status` before committing and restore unmodified index binaries with `git checkout -- app/index/chroma` unless you intentionally rebuilt the index.

