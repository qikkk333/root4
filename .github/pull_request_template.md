## Summary of Changes

<!-- Describe clearly what this PR adds, fixes, or improves. -->

## Reason for Change

<!-- Explain why this change is needed and any context or trade-offs. -->

## Related Issue

<!-- Reference any related issue (e.g., Fixes #12, Closes #15) or write N/A. -->
Closes #

## Type of Change

- [ ] Bug fix (`fix:`)
- [ ] New feature (`feat:`)
- [ ] Documentation (`docs:`)
- [ ] Tests / evaluation (`test:`)
- [ ] Refactor (`refactor:`)
- [ ] CI / Security / Repository tooling (`ci:` / `security:`)

## Testing Performed

- [ ] `python -m compileall -q app`
- [ ] `python -m pytest app/tests -v`
- [ ] `ruff check app` and `ruff format --check app`
- [ ] `python app/eval/evaluate_agent.py --offline --suite main` (37 cases)
- [ ] `python app/eval/evaluate_agent.py --offline --suite all` (39 cases)
- [ ] Manual interactive smoke check via `helpdesk` / `python -m helpdesk.agent` (if UI/agent behavior changed)

## Screenshots or Terminal Output (if UI-facing)

<!-- Paste redacted terminal output or screenshots for CLI/menu/table layout changes. -->

## Security & Integrity Checklist

- [ ] I confirm no API keys, `.env` files, tokens, or credentials are included in code, logs, or screenshots.
- [ ] I confirm existing functionality (`ask()` contract, read-only SQLite protections, citations, abstention) is preserved.
- [ ] I confirmed `git status` contains no unintended runtime changes to `database/rail_timetable.db` or `app/index/chroma/`.

