"""Interactive Rich main menu, setup wizard, health check, and evaluation runner for Railway Help Desk."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from . import env_setup
from . import terminal_ui

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    HAS_RICH = True
except ImportError:  # pragma: no cover - stdlib fallback only before dependencies install
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    box = None  # type: ignore[assignment]
    HAS_RICH = False


def _make_console(stream=None, use_color=None, width=None):
    target = sys.stdout if stream is None else stream
    if not HAS_RICH:
        return None
    color_on = terminal_ui.should_use_color(stream=target, use_color=use_color)
    cols = terminal_ui.get_raw_terminal_width(width=width, stream=target)
    return Console(
        file=target,
        no_color=not color_on,
        force_terminal=color_on,
        color_system="standard" if color_on else None,
        legacy_windows=False,
        highlight=False,
        soft_wrap=False,
        width=cols,
    )


def _emit(text="", *, style="", console=None, stream=None) -> None:
    target = sys.stdout if stream is None else stream
    if console is not None:
        console.print(text, style=style or None)
    else:
        print(text, file=target, flush=True)


def _render_panel(body: str, *, title: str = "", border_style: str = "cyan", console=None, stream=None, width=None) -> None:
    target = sys.stdout if stream is None else stream
    actual_w = terminal_ui.get_raw_terminal_width(width=width, stream=target)
    if console is not None and HAS_RICH:
        if width is None and hasattr(console, "width"):
            console.width = actual_w
        w = min(actual_w, 84)
        pad_x = 1 if w < 50 else 2
        console.print(Panel(body, title=title or None, border_style=border_style, box=box.ROUNDED, width=w, padding=(0, pad_x)))
    else:
        w = min(actual_w, 72)
        bar = "=" * w
        lines = [bar]
        if title:
            lines.extend(terminal_ui.center_text_lines([title], w))
            lines.append(bar)
        lines.append(body)
        lines.append(bar)
        print("\n".join(lines), file=target, flush=True)


def _prompt(prompt_text: str, *, reader=None, console=None, stream=None, use_color=None) -> str:
    target = sys.stdout if stream is None else stream
    pal = terminal_ui.get_palette(stream=target, use_color=use_color)
    styled = f"{pal.bold_blue}{prompt_text}{pal.reset}" if pal.bold_blue else prompt_text
    if reader is None:
        print(f"{styled} ", end="", file=target, flush=True)
        return input("")
    print(f"{styled} ", end="", file=target, flush=True)
    return reader("")


def _pause_continue(reader=None, stream=None, use_color=None) -> None:
    target = sys.stdout if stream is None else stream
    try:
        _prompt("\nPress Enter to continue...", reader=reader, stream=stream, use_color=use_color)
    except (EOFError, KeyboardInterrupt):
        print(file=target, flush=True)


def render_main_menu(stream=None, use_color=None, width=None) -> str:
    """Render main menu header (dynamically centered to terminal width) and options as plain text."""
    w = terminal_ui.get_raw_terminal_width(width=width, stream=stream)
    border = "=" * w
    heading_lines = terminal_ui.center_text_lines(
        ["RAILWAY PASSENGER HELP DESK", "Intelligent Railway Information Assistant"],
        w,
    )
    lines = [
        border,
        *heading_lines,
        border,
        "",
        "Welcome aboard!",
        "",
        "MAIN MENU",
        "",
        "[1] Start Passenger Chat",
        "[2] Project Information",
        "[3] System Health Check",
        "[4] Run Evaluation Tests",
        "[5] Help & Commands",
        "[0] Exit",
    ]
    return "\n".join(lines)


def _print_main_menu(console=None, stream=None, use_color=None, width=None) -> None:
    target = sys.stdout if stream is None else stream
    actual_w = terminal_ui.get_raw_terminal_width(width=width, stream=target)
    if console is not None and HAS_RICH:
        if width is None and hasattr(console, "width"):
            console.width = actual_w
        pad_x = 1 if actual_w < 50 else 2
        heading = Text.assemble(
            ("RAILWAY PASSENGER HELP DESK\n", "bold cyan"),
            ("Intelligent Railway Information Assistant", "cyan"),
            justify="center",
        )
        header = Panel(
            heading,
            box=box.DOUBLE,
            border_style="cyan",
            width=actual_w,
            padding=(0, pad_x),
        )
        console.print()
        console.print(header)
        console.print("[bold]Welcome aboard![/bold]\n")
        table = Table(title="MAIN MENU", title_justify="left", title_style="bold cyan", box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Option", style="bold blue", no_wrap=True)
        table.add_column("Description", style="white", overflow="fold")
        for opt, desc in (
            ("[1]", "Start Passenger Chat"),
            ("[2]", "Project Information"),
            ("[3]", "System Health Check"),
            ("[4]", "Run Evaluation Tests"),
            ("[5]", "Help & Commands"),
            ("[0]", "Exit"),
        ):
            table.add_row(opt, desc)
        console.print(table)
    else:
        print("\n" + render_main_menu(stream=target, use_color=use_color, width=actual_w), file=target, flush=True)


def _configure_hf_mode_if_cached(cache_root=None) -> bool:
    """Enable HF_HUB_OFFLINE=1 only when both HF models exist locally; never set on first uncached run."""
    cached = env_setup.are_hf_models_cached(cache_root=cache_root)
    if cached and "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return cached


def _launch_passenger_chat(
    *,
    app_dir=None,
    chat_fn=None,
    input_fn=None,
    out=None,
    clear_fn=None,
    use_color=None,
    width=None,
    cache_root=None,
    pause_on_return=True,
) -> int:
    stream = sys.stdout if out is None else out
    console = _make_console(stream=stream, use_color=use_color, width=width)
    app_path = env_setup.resolve_app_dir(app_dir)

    created_env, env_path = env_setup.ensure_env_file(app_path)
    _, key_name, has_key = env_setup.read_env_key_status(app_path)
    if created_env:
        _render_panel(
            f"Created configuration file from template:\n  {env_path}\n\n"
            f"Please open it and set your API key:\n  {key_name}=<your-gemini-api-key>\n\n"
            "Live chatbot queries require a valid Gemini API key.\n"
            "Offline health checks, documentation, and pytest/offline evaluations work without a key.",
            title="Gemini Configuration Needed",
            border_style="yellow",
            console=console,
            stream=stream,
            width=width,
        )
        if pause_on_return:
            _pause_continue(reader=input_fn, stream=stream, use_color=use_color)
        return 1

    if not has_key:
        _render_panel(
            f"{key_name} is not configured in:\n  {env_path}\n\n"
            f"Add {key_name}=<your-gemini-api-key> to app\\.env (never share or commit .env).\n"
            "Returning cleanly so you can review Project Info, Health Check, or Offline Tests.",
            title="Missing Gemini API Key",
            border_style="yellow",
            console=console,
            stream=stream,
            width=width,
        )
        if pause_on_return:
            _pause_continue(reader=input_fn, stream=stream, use_color=use_color)
        return 1

    if not _configure_hf_mode_if_cached(cache_root=cache_root):
        _emit(
            "Note: Embedding/reranking models (BAAI/bge-small-en-v1.5, cross-encoder/ms-marco-MiniLM-L-6-v2) "
            "are not cached locally yet. First query requires an internet connection.",
            style="yellow",
            console=console,
            stream=stream,
        )

    from . import agent

    agent._load_env(app_path)
    runner = agent.interactive_chat if chat_fn is None else chat_fn
    try:
        try:
            runner(input_fn=input_fn, out=stream, clear_fn=clear_fn, width=width, use_color=use_color)
        except TypeError:
            runner()
    except (KeyboardInterrupt, EOFError):
        pass
    if pause_on_return:
        _emit("\nReturning to Main Menu.", style="cyan", console=console, stream=stream)
    return 0


def show_project_info(*, app_dir=None, console=None, stream=None, use_color=None, width=None) -> None:
    target = sys.stdout if stream is None else stream
    con = console if console is not None else _make_console(stream=target, use_color=use_color, width=width)
    app_path = env_setup.resolve_app_dir(app_dir)
    repo_root = app_path.parent

    overview = (
        "Project: Railway Passenger Help Desk (root4)\n"
        f"Location: {repo_root}\n\n"
        "A hybrid Retrieval-Augmented Generation (RAG) and Read-Only SQL assistant for Indian Railways "
        "passenger queries spanning official rules/circulars and historical timetable schedules."
    )
    _render_panel(overview, title="Project Overview", border_style="cyan", console=con, stream=target, width=width)

    arch_text = (
        "1. RouterAgent: Plans document search, read-only SQL, mixed routes, or safe refusals.\n"
        "2. Dense Retrieval: ChromaDB vector collection ('rail_chunks', BAAI/bge-small-en-v1.5).\n"
        "3. Lexical Retrieval: Code- and number-aware BM25Okapi + railway synonym expansion.\n"
        "4. Reranking: Reciprocal Rank Fusion (RRF_K=60) + cross-encoder/ms-marco-MiniLM-L-6-v2 + recency safety.\n"
        "5. SQLite Layer: Read-only 2016 snapshot via 5 defense-in-depth protections (helpdesk/sql.py).\n"
        "6. Gemini Grounding: Pydantic quote-verified citations and strict abstention when unsupported."
    )
    _render_panel(arch_text, title="RAG & SQL Architecture", border_style="cyan", console=con, stream=target, width=width)

    tech_limits = (
        "Technologies: Python 3.12, OpenAI SDK (Gemini endpoint), Pydantic, ChromaDB, Sentence-Transformers, "
        "PyTorch, rank-bm25, PyMuPDF, RapidOCR, BeautifulSoup4, SQLite, Rich.\n\n"
        "Current Limitations:\n"
        "- Timetable DB reflects an August 2016 snapshot (5,208 trains, 8,990 stations; historical codes like BCT).\n"
        "- No live running status, current fares, PNR status, seat availability, or platform numbers.\n"
        "- Policy corpus covers 7 indexed documents (100 chunks: 93 content + 7 overview).\n"
        "- Unsupported queries abstain rather than inventing policy or timetable data."
    )
    _render_panel(tech_limits, title="Technologies & Limitations", border_style="white", console=con, stream=target, width=width)


def show_health_check(
    *,
    app_dir=None,
    checks=None,
    console=None,
    stream=None,
    use_color=None,
    width=None,
) -> list[env_setup.HealthCheck]:
    target = sys.stdout if stream is None else stream
    con = console if console is not None else _make_console(stream=target, use_color=use_color, width=width)
    rows = env_setup.run_health_checks(app_dir) if checks is None else list(checks)

    if con is not None and HAS_RICH:
        table = Table(title="SYSTEM HEALTH CHECK", title_justify="left", title_style="bold cyan", box=box.ROUNDED)
        table.add_column("Status", no_wrap=True)
        table.add_column("Component", style="bold white")
        table.add_column("Details")
        for row in rows:
            tag = {
                "PASS": "[bold green][PASS][/bold green]",
                "FAIL": "[bold red][FAIL][/bold red]",
                "INFO": "[bold yellow][INFO][/bold yellow]",
            }.get(row.status, f"[{row.status}]")
            detail = row.detail + (f"\n[yellow]Fix: {row.remediation}[/yellow]" if row.remediation else "")
            table.add_row(tag, row.name, detail)
        con.print()
        con.print(table)
    else:
        print("\nSYSTEM HEALTH CHECK", file=target, flush=True)
        for row in rows:
            print(f"[{row.status}] {row.name}: {row.detail}", file=target, flush=True)
            if row.remediation:
                print(f"       Fix: {row.remediation}", file=target, flush=True)
    return rows


def run_evaluation_menu(
    *,
    app_dir=None,
    input_fn=None,
    out=None,
    run_fn=subprocess.run,
    use_color=None,
    width=None,
    cache_root=None,
) -> int:
    """Interactive submenu for running pytest and offline deterministic agent evaluation suites."""
    stream = sys.stdout if out is None else out
    con = _make_console(stream=stream, use_color=use_color, width=width)
    app_path = env_setup.resolve_app_dir(app_dir)
    repo_root = app_path.parent
    vpy = env_setup.venv_python_path(app_path)
    py_exe = str(vpy) if vpy.is_file() else sys.executable

    while True:
        body = (
            "[1] Run complete offline pytest suite\n"
            "[2] Run main evaluation suite (37 cases, offline deterministic)\n"
            "[3] Run full evaluation suite (39 SQL/hybrid/refusal cases, offline)\n"
            "[0] Back"
        )
        _render_panel(body, title="EVALUATION & TEST SUITE", border_style="cyan", console=con, stream=stream, width=width)
        try:
            choice = (_prompt("Select evaluation option:", reader=input_fn, stream=stream, use_color=use_color) or "").strip()
        except (EOFError, KeyboardInterrupt):
            return 0

        if choice == "0":
            return 0

        cmd = None
        cwd = str(repo_root)
        env = dict(os.environ)
        if env_setup.are_hf_models_cached(cache_root=cache_root) and "HF_HUB_OFFLINE" not in env:
            env["HF_HUB_OFFLINE"] = "1"

        if choice == "1":
            cmd, cwd = env_setup.choose_pytest_command(app_path, run_fn=run_fn)
            if cmd is None:
                _emit(
                    "pytest is not installed in app\\.venv or on PATH. Install with:\n"
                    f"  \"{py_exe}\" -m pip install pytest",
                    style="yellow",
                    console=con,
                    stream=stream,
                )
                continue
        elif choice == "2":
            cmd = [py_exe, str(app_path / "eval" / "evaluate_agent.py"), "--offline", "--suite", "main"]
        elif choice == "3":
            cmd = [py_exe, str(app_path / "eval" / "evaluate_agent.py"), "--offline", "--suite", "all"]
        else:
            _emit(f"Invalid option '{choice}'. Choose 1, 2, 3, or 0.", style="yellow", console=con, stream=stream)
            continue

        _emit(f"\nRunning: {' '.join(cmd)}", style="cyan", console=con, stream=stream)
        try:
            res = run_fn(cmd, cwd=cwd, env=env, check=False)
            if res.returncode == 0:
                _emit("\n[PASS] Evaluation completed successfully (exit code 0).", style="bold green", console=con, stream=stream)
            else:
                _emit(f"\n[FAIL] Evaluation finished with exit code {res.returncode}.", style="bold red", console=con, stream=stream)
        except Exception as error:
            _emit(f"\n[FAIL] Could not run test command: {error}", style="bold red", console=con, stream=stream)


def show_help_and_commands(*, app_dir=None, console=None, stream=None, use_color=None, width=None) -> None:
    target = sys.stdout if stream is None else stream
    con = console if console is not None else _make_console(stream=target, use_color=use_color, width=width)
    app_path = env_setup.resolve_app_dir(app_dir)
    repo_root = app_path.parent

    help_text = (
        "Session Launch (Windows PowerShell from project root):\n"
        "  Set-ExecutionPolicy -Scope Process Bypass -Force; . .\\setup-helpdesk.ps1\n"
        "  helpdesk\n\n"
        "Main Menu Options:\n"
        "  [1] Start Passenger Chat   Launch chatbot (type 'help', 'clear', 'exit', or Ctrl+C)\n"
        "  [2] Project Information    View architecture, models, and data snapshot limits\n"
        "  [3] System Health Check    Non-destructive check of Python, .env, Chroma, and SQLite\n"
        "  [4] Run Evaluation Tests   Run pytest and offline evaluation suites (no API key needed)\n"
        "  [5] Help & Commands        Display this guide\n"
        "  [0] Exit                   Quit application cleanly\n\n"
        "Project Locations:\n"
        f"  Repository root : {repo_root}\n"
        f"  App directory   : {app_path}\n"
        f"  Virtual env     : {env_setup.venv_dir_path(app_path)}\n"
        f"  Env config      : {app_path / '.env'}\n"
        f"  Timetable DB    : {repo_root / 'database' / 'rail_timetable.db'}\n\n"
        "Configuration & Recovery:\n"
        "  - Add GEMINI_API_KEY=<your-key> inside app\\.env (copied from app\\.env.example).\n"
        "  - Repair environment: .\\helpdesk.ps1 --setup\n"
        "  - Direct one-shot JSON: .\\app\\.venv\\Scripts\\python.exe -m helpdesk.agent \"<question>\"\n"
        "  - Pytest suite: python -m pytest app/tests -v"
    )
    _render_panel(help_text, title="HELP & COMMANDS", border_style="cyan", console=con, stream=target, width=width)


def run_setup_menu(
    *,
    app_dir=None,
    input_fn=None,
    out=None,
    run_fn=subprocess.run,
    use_color=None,
    width=None,
    auto_setup=False,
    on_launch_main=None,
) -> int:
    """Interactive Environment Setup Menu when venv or dependencies are missing/broken."""
    stream = sys.stdout if out is None else out
    con = _make_console(stream=stream, use_color=use_color, width=width)
    app_path = env_setup.resolve_app_dir(app_dir)

    if auto_setup:
        ok, msg = env_setup.setup_environment(app_path, out=stream, run_fn=run_fn)
        _emit(msg, style="green" if ok else "red", console=con, stream=stream)
        if not ok:
            return 1
        return _handoff_to_main_menu(app_path, run_fn=run_fn, on_launch_main=on_launch_main, input_fn=input_fn, out=stream, use_color=use_color, width=width)

    while True:
        body = (
            "Your environment is not configured yet.\n\n"
            "[1] Set up the project automatically\n"
            "[2] Check existing installation\n"
            "[0] Exit"
        )
        _render_panel(body, title="RAILWAY PASSENGER HELP DESK", border_style="yellow", console=con, stream=stream, width=width)
        try:
            choice = (_prompt("Select an option:", reader=input_fn, stream=stream, use_color=use_color) or "").strip()
        except (EOFError, KeyboardInterrupt):
            _emit("\nExiting setup. Goodbye!", style="cyan", console=con, stream=stream)
            return 0

        if choice == "0":
            _emit("Goodbye!", style="cyan", console=con, stream=stream)
            return 0
        if choice == "1":
            _emit("\nStarting automatic project setup...\n", style="cyan", console=con, stream=stream)
            ok, msg = env_setup.setup_environment(app_path, out=stream, run_fn=run_fn)
            _emit(f"\n[{'PASS' if ok else 'FAIL'}] {msg}", style="bold green" if ok else "bold red", console=con, stream=stream)
            if ok:
                return _handoff_to_main_menu(
                    app_path,
                    run_fn=run_fn,
                    on_launch_main=on_launch_main,
                    input_fn=input_fn,
                    out=stream,
                    use_color=use_color,
                    width=width,
                )
        elif choice == "2":
            show_health_check(app_dir=app_path, console=con, stream=stream, use_color=use_color, width=width)
            _pause_continue(reader=input_fn, stream=stream, use_color=use_color)
        else:
            _emit(f"Invalid option '{choice}'. Please choose 1, 2, or 0.", style="yellow", console=con, stream=stream)


def _handoff_to_main_menu(
    app_path: Path,
    *,
    run_fn=subprocess.run,
    on_launch_main=None,
    input_fn=None,
    out=None,
    use_color=None,
    width=None,
) -> int:
    if callable(on_launch_main):
        return int(on_launch_main())
    vpy = env_setup.venv_python_path(app_path)
    if vpy.is_file() and Path(sys.executable).resolve() != vpy.resolve():
        proc = run_fn([str(vpy), "-m", "helpdesk.cli"], cwd=str(app_path.parent), check=False)
        return int(proc.returncode)
    return run_main_menu(app_dir=app_path, input_fn=input_fn, out=out, run_fn=run_fn, use_color=use_color, width=width)


def run_main_menu(
    *,
    app_dir=None,
    input_fn=None,
    out=None,
    clear_fn=None,
    chat_fn=None,
    run_fn=subprocess.run,
    health_fn=None,
    use_color=None,
    width=None,
    cache_root=None,
    pause_after_views=True,
) -> int:
    """Interactive Railway Passenger Help Desk Main Menu loop."""
    stream = sys.stdout if out is None else out
    con = _make_console(stream=stream, use_color=use_color, width=width)
    app_path = env_setup.resolve_app_dir(app_dir)

    while True:
        if width is None and con is not None and hasattr(con, "width"):
            con.width = terminal_ui.get_raw_terminal_width(stream=stream)
        _print_main_menu(console=con, stream=stream, use_color=use_color, width=width)
        try:
            choice = (_prompt("\nSelect an option:", reader=input_fn, stream=stream, use_color=use_color) or "").strip()
        except (EOFError, KeyboardInterrupt):
            _emit("\nThank you for using Railway Passenger Help Desk. Goodbye!", style="cyan", console=con, stream=stream)
            return 0

        if choice == "0":
            _emit("Thank you for using Railway Passenger Help Desk. Goodbye!", style="cyan", console=con, stream=stream)
            return 0
        if choice == "1":
            _launch_passenger_chat(
                app_dir=app_path,
                chat_fn=chat_fn,
                input_fn=input_fn,
                out=stream,
                clear_fn=clear_fn,
                use_color=use_color,
                width=width,
                cache_root=cache_root,
                pause_on_return=pause_after_views,
            )
        elif choice == "2":
            show_project_info(app_dir=app_path, console=con, stream=stream, use_color=use_color, width=width)
            if pause_after_views:
                _pause_continue(reader=input_fn, stream=stream, use_color=use_color)
        elif choice == "3":
            rows = health_fn() if callable(health_fn) else None
            show_health_check(app_dir=app_path, checks=rows, console=con, stream=stream, use_color=use_color, width=width)
            if pause_after_views:
                _pause_continue(reader=input_fn, stream=stream, use_color=use_color)
        elif choice == "4":
            run_evaluation_menu(
                app_dir=app_path,
                input_fn=input_fn,
                out=stream,
                run_fn=run_fn,
                use_color=use_color,
                width=width,
                cache_root=cache_root,
            )
        elif choice == "5":
            show_help_and_commands(app_dir=app_path, console=con, stream=stream, use_color=use_color, width=width)
            if pause_after_views:
                _pause_continue(reader=input_fn, stream=stream, use_color=use_color)
        else:
            _emit(f"Invalid selection '{choice}'. Please choose an option from 0 to 5.", style="yellow", console=con, stream=stream)


def _main(argv=None) -> int:
    terminal_ui.configure_terminal_io()
    ap = argparse.ArgumentParser(description="Railway Passenger Help Desk Interactive Menu & Setup Launcher.")
    ap.add_argument("--setup-menu", action="store_true", help="launch interactive environment setup menu")
    ap.add_argument("--setup", action="store_true", help="set up or repair app/.venv automatically and launch")
    ap.add_argument("--health", action="store_true", help="run non-destructive system health checks and exit")
    ap.add_argument("--chat", action="store_true", help="skip main menu and start passenger chat directly")
    args = ap.parse_args(argv)

    if args.health:
        rows = show_health_check()
        return 1 if any(r.status == "FAIL" for r in rows) else 0
    if args.setup or args.setup_menu:
        return run_setup_menu(auto_setup=args.setup)
    if args.chat:
        return _launch_passenger_chat(pause_on_return=False)
    return run_main_menu()


if __name__ == "__main__":
    raise SystemExit(_main())
