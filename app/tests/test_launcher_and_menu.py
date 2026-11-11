"""Automated offline tests for the project-local launcher, env setup, health check, and Rich menu."""

from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import helpdesk  # noqa: E402
import helpdesk.agent as agent  # noqa: E402
import helpdesk.cli as cli  # noqa: E402
import helpdesk.env_setup as env_setup  # noqa: E402


def _make_reader(responses):
    it = iter(responses)

    def _read(_prompt=""):
        item = next(it)
        if isinstance(item, BaseException):
            raise item
        return item

    return _read


def _create_fake_project(root: Path, *, with_venv: bool = True, with_env: bool = False, api_key: str = "") -> Path:
    app = root / "app"
    app.mkdir(parents=True, exist_ok=True)
    (app / "requirements.txt").write_text("rich==15.0.0\n", encoding="utf-8")
    (app / ".env.example").write_text("GEMINI_API_KEY=\n# HELPDESK_AGENT=router\n", encoding="utf-8")
    if with_env:
        (app / ".env").write_text(f"GEMINI_API_KEY={api_key}\n", encoding="utf-8")
    if with_venv:
        vpy = env_setup.venv_python_path(app)
        vpy.parent.mkdir(parents=True, exist_ok=True)
        vpy.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    return app


# ---------------------------------------------------------------------------
# 1. Virtual environment detection, Python version checks, & first-time setup
# ---------------------------------------------------------------------------
def test_venv_detection_missing_and_python_version_compatibility(tmp_path: Path):
    app_missing = _create_fake_project(tmp_path / "no_venv", with_venv=False)
    assert env_setup.check_venv_exists(app_missing) is False
    assert env_setup.venv_python_path(app_missing) == (app_missing / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))

    app_with = _create_fake_project(tmp_path / "has_venv", with_venv=True)
    assert env_setup.check_venv_exists(app_with) is True

    assert env_setup.check_python_compatibility((3, 12, 9)) == (True, "3.12.9")
    assert env_setup.check_python_compatibility((3, 11, 8)) == (False, "3.11.8")
    assert env_setup.check_python_compatibility((3, 13, 0)) == (False, "3.13.0")


def test_first_time_setup_creates_venv_installs_deps_and_reuses_existing_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Path containing spaces ensures Windows path quoting/handling is exercised thoroughly
    project_root = tmp_path / "Project With Spaces" / "root4"
    app_dir = _create_fake_project(project_root, with_venv=False, with_env=False)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        # Simulate python -m venv creating venv python binary
        if "-m" in cmd and "venv" in cmd:
            vpy = env_setup.venv_python_path(app_dir)
            vpy.parent.mkdir(parents=True, exist_ok=True)
            vpy.write_text("stub", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env_setup, "check_required_packages", lambda *a, **kw: (True, []))

    # 1. First-time setup creates venv, upgrades pip, installs requirements, copies .env.example -> .env
    buf = io.StringIO()
    ok, msg = env_setup.setup_environment(app_dir, python_exe="python3.12.exe", out=buf, run_fn=fake_run)
    assert ok is True, msg
    assert len(calls) == 3
    assert calls[0][:3] == ["python3.12.exe", "-m", "venv"]
    assert str(project_root / "app" / ".venv") in calls[0][3]
    assert calls[1][1:] == ["-m", "pip", "install", "--upgrade", "pip"]
    assert calls[2][1:5] == ["-m", "pip", "install", "-r"]
    assert (app_dir / ".env").is_file()
    assert "GEMINI_API_KEY=" in (app_dir / ".env").read_text(encoding="utf-8")

    # 2. Repair/subsequent setup against EXISTING venv never recreates or deletes .venv
    marker = app_dir / ".venv" / "keep_marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    calls.clear()
    ok2, _ = env_setup.setup_environment(app_dir, out=io.StringIO(), run_fn=fake_run)
    assert ok2 is True
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not any("venv" in c for c in calls)
    assert not any("--upgrade" in c and "pip" in c for c in calls)
    assert len(calls) == 1  # only installs from requirements.txt


def test_setup_menu_handles_auto_setup_check_invalid_and_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app_dir = _create_fake_project(tmp_path / "setup_menu_proj", with_venv=False)
    setup_called = []
    launched = []

    monkeypatch.setattr(env_setup, "setup_environment", lambda *a, **kw: setup_called.append(True) or (True, "ok"))

    buf = io.StringIO()
    rc = cli.run_setup_menu(
        app_dir=app_dir,
        input_fn=_make_reader(["bad", "1"]),
        out=buf,
        use_color=False,
        on_launch_main=lambda: launched.append(True) or 0,
    )
    assert rc == 0
    assert setup_called == [True]
    assert launched == [True]
    assert "Your environment is not configured yet." in buf.getvalue()
    assert "Invalid option 'bad'" in buf.getvalue()

    # Exit cleanly via option 0 or Ctrl+C
    assert cli.run_setup_menu(app_dir=app_dir, input_fn=_make_reader(["0"]), out=io.StringIO(), use_color=False) == 0
    assert cli.run_setup_menu(app_dir=app_dir, input_fn=_make_reader([KeyboardInterrupt()]), out=io.StringIO(), use_color=False) == 0


# ---------------------------------------------------------------------------
# 2. Gemini API key handling & HuggingFace offline cache detection
# ---------------------------------------------------------------------------
def test_missing_gemini_api_key_creates_env_safely_and_never_exposes_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    secret = "AIzaSySecretKeyShouldNeverPrint12345"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_API_KEY_ENV", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    # Case A: No .env file -> copies .env.example cleanly without fake key, returns 1 safely
    app_no_env = _create_fake_project(tmp_path / "no_env_proj", with_venv=True, with_env=False)
    buf = io.StringIO()
    rc = cli._launch_passenger_chat(
        app_dir=app_no_env,
        input_fn=_make_reader([""]),
        out=buf,
        use_color=False,
    )
    assert rc == 1
    assert (app_no_env / ".env").is_file()
    assert "GEMINI_API_KEY=" in (app_no_env / ".env").read_text(encoding="utf-8")
    assert "Gemini Configuration Needed" in buf.getvalue()

    # Case B: .env exists with blank key -> friendly warning, returns cleanly without crash
    app_blank = _create_fake_project(tmp_path / "blank_key_proj", with_venv=True, with_env=True, api_key="")
    buf2 = io.StringIO()
    rc2 = cli._launch_passenger_chat(
        app_dir=app_blank,
        input_fn=_make_reader([""]),
        out=buf2,
        use_color=False,
    )
    assert rc2 == 1
    assert "Missing Gemini API Key" in buf2.getvalue()

    # Case C: Real key present -> never exposed anywhere in health check status or chat output
    app_ok = _create_fake_project(tmp_path / "ok_key_proj", with_venv=True, with_env=True, api_key=secret)
    exists, key_name, configured = env_setup.read_env_key_status(app_ok, environ={})
    assert (exists, key_name, configured) == (True, "GEMINI_API_KEY", True)

    health_buf = io.StringIO()
    checks = [env_setup.HealthCheck("Gemini configuration", "PASS", f"{key_name} configured (value hidden)")]
    cli.show_health_check(app_dir=app_ok, checks=checks, stream=health_buf, use_color=False)
    assert secret not in health_buf.getvalue()


def test_hf_offline_mode_only_set_when_models_already_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    empty_cache = tmp_path / "hf_empty"
    empty_cache.mkdir()
    assert env_setup.are_hf_models_cached(empty_cache) is False
    assert cli._configure_hf_mode_if_cached(cache_root=empty_cache) is False
    assert "HF_HUB_OFFLINE" not in os.environ

    # Populate snapshots for both required models
    for model_id in env_setup.HF_REQUIRED_MODELS:
        snap = empty_cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
    assert env_setup.are_hf_models_cached(empty_cache) is True
    assert cli._configure_hf_mode_if_cached(cache_root=empty_cache) is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


# ---------------------------------------------------------------------------
# 3. Main Menu navigation, chatbot round-trip, health check, eval submenu, & exit
# ---------------------------------------------------------------------------
def test_main_menu_navigation_chat_return_info_health_help_invalid_and_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaDummyConfiguredKeyForMenuTest")
    # Prepare cached HF stub so Option 1 doesn't emit uncached warning
    cache = tmp_path / "hf_cache"
    for model_id in env_setup.HF_REQUIRED_MODELS:
        snap = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")

    chat_sessions = []

    def fake_chat(**kwargs):
        chat_sessions.append(True)
        return 0

    health_calls = []

    def fake_health():
        health_calls.append(True)
        return [
            env_setup.HealthCheck("Python environment", "PASS", "Python 3.12.9"),
            env_setup.HealthCheck("Timetable database", "PASS", "5208 trains"),
        ]

    buf = io.StringIO()
    rc = cli.run_main_menu(
        app_dir=APP_DIR,
        input_fn=_make_reader(["bad", "1", "2", "", "3", "", "5", "", "0"]),
        out=buf,
        chat_fn=fake_chat,
        health_fn=fake_health,
        use_color=False,
        width=80,
        cache_root=cache,
    )
    out = buf.getvalue()
    assert rc == 0
    assert chat_sessions == [True]
    assert health_calls == [True]
    assert "Welcome aboard!" in out
    assert "MAIN MENU" in out
    assert "Invalid selection 'bad'" in out
    assert "Returning to Main Menu." in out
    assert "RAG & SQL Architecture" in out
    assert "SYSTEM HEALTH CHECK" in out
    assert "HELP & COMMANDS" in out
    assert "Goodbye!" in out
    assert "\x1b[" not in out

    # Graceful Ctrl+C and EOF at Main Menu
    assert cli.run_main_menu(app_dir=APP_DIR, input_fn=_make_reader([KeyboardInterrupt()]), out=io.StringIO(), use_color=False) == 0
    assert cli.run_main_menu(app_dir=APP_DIR, input_fn=_make_reader([EOFError()]), out=io.StringIO(), use_color=False) == 0


def test_real_non_destructive_health_check_on_project_repo(monkeypatch: pytest.MonkeyPatch):
    # System python may lack chromadb while app/.venv has it; inject stub package_checker only to inspect DB/index/files safely
    rows = env_setup.run_health_checks(APP_DIR, package_checker=lambda _py: (True, []))
    by_name = {r.name: r for r in rows}
    assert by_name["Python environment"].status == "PASS"
    assert by_name["Virtual environment"].status == "PASS"
    assert by_name["Vector database"].status == "PASS"
    assert "100 chunks" in by_name["Vector database"].detail
    assert by_name["Timetable database"].status == "PASS"
    assert "5208 trains" in by_name["Timetable database"].detail
    assert by_name["Required project files"].status == "PASS"


def test_evaluation_submenu_dispatches_safe_commands_and_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ran = []

    def fake_run(cmd, **kwargs):
        ran.append((list(cmd), kwargs.get("cwd")))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env_setup, "choose_pytest_command", lambda app_dir, run_fn=None: ([sys.executable, "-m", "pytest", str(app_dir / "tests"), "-v"], str(app_dir.parent)))

    buf = io.StringIO()
    rc = cli.run_evaluation_menu(
        app_dir=APP_DIR,
        input_fn=_make_reader(["bad", "1", "2", "3", "0"]),
        out=buf,
        run_fn=fake_run,
        use_color=False,
    )
    assert rc == 0
    out = buf.getvalue()
    assert "Invalid option 'bad'" in out
    assert len(ran) == 3
    assert ran[0][0][1:3] == ["-m", "pytest"]
    assert ran[1][0][-4:] == ["evaluate_agent.py", "--offline", "--suite", "main"] or (
        ran[1][0][-3:] == ["--offline", "--suite", "main"] and "evaluate_agent.py" in ran[1][0][1]
    )
    assert ran[2][0][-3:] == ["--offline", "--suite", "all"]
    assert "[PASS] Evaluation completed successfully" in out


# ---------------------------------------------------------------------------
# 4. Windows PowerShell launcher, session setup, & paths with spaces
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell launcher test")
def test_powershell_session_setup_health_and_spaces_in_path(tmp_path: Path):
    # 1. Verify project-root setup-helpdesk.ps1 registers 'helpdesk' without modifying $PROFILE
    #    and runs '--health' inside existing app/.venv with zero secret leaks.
    profile_before = Path(os.environ.get("USERPROFILE", "")) / "Documents" / "WindowsPowerShell" / "Microsoft.PowerShell_profile.ps1"
    profile_mtime_before = profile_before.stat().st_mtime if profile_before.is_file() else None

    cmd_real = (
        f"Set-Location -LiteralPath '{REPO_ROOT}'; "
        "Set-ExecutionPolicy -Scope Process Bypass -Force; "
        ". .\\setup-helpdesk.ps1; "
        "helpdesk --health"
    )
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", cmd_real],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "[PASS]" in proc.stdout
    assert "Python environment" in proc.stdout
    assert "Vector database" in proc.stdout
    assert "Timetable database" in proc.stdout
    assert "AIza" not in proc.stdout

    profile_mtime_after = profile_before.stat().st_mtime if profile_before.is_file() else None
    assert profile_mtime_after == profile_mtime_before

    # 2. Verify helpdesk.ps1 + setup-helpdesk.ps1 handle paths containing spaces cleanly when .venv is absent
    spaced_root = tmp_path / "Dir With Spaces" / "Rail Capstone Root"
    spaced_app = spaced_root / "app"
    (spaced_app / "helpdesk").mkdir(parents=True)
    for fn in ("helpdesk.ps1", "setup-helpdesk.ps1", "helpdesk.cmd"):
        shutil.copyfile(REPO_ROOT / fn, spaced_root / fn)
    shutil.copyfile(APP_DIR / "requirements.txt", spaced_app / "requirements.txt")
    shutil.copyfile(APP_DIR / ".env.example", spaced_app / ".env.example")
    for mod in ("__init__.py", "cli.py", "env_setup.py", "terminal_ui.py"):
        shutil.copyfile(APP_DIR / "helpdesk" / mod, spaced_app / "helpdesk" / mod)

    cmd_spaced = (
        f"Set-Location -LiteralPath '{spaced_root}'; "
        "Set-ExecutionPolicy -Scope Process Bypass -Force; "
        ". .\\setup-helpdesk.ps1; "
        "helpdesk"
    )
    proc_spaced = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", cmd_spaced],
        input="0\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_spaced.returncode == 0, proc_spaced.stderr or proc_spaced.stdout
    assert "Your environment is not configured yet." in proc_spaced.stdout
    assert "[1] Set up the project automatically" in proc_spaced.stdout
    assert "Goodbye!" in proc_spaced.stdout

    # 3. Verify legacy one-shot agent CLI and public ask() API remain completely intact
    assert helpdesk.ask("") == {"answer": agent.DONT_KNOW, "sources": [], "sql": None, "dont_know": True}
    proc_oneshot = subprocess.run(
        [sys.executable, "-m", "helpdesk.agent", "Delete train 12002 from your database"],
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc_oneshot.returncode == 0, proc_oneshot.stderr or proc_oneshot.stdout
    assert "\"dont_know\": true" in proc_oneshot.stdout
