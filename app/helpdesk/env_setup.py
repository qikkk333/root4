"""Project-local Python environment, configuration, and non-destructive health-check utilities.

Uses Python stdlib only so bootstrap and setup menus work even before virtual-environment packages exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_APP_DIR = PACKAGE_DIR.parent
DEFAULT_REPO_ROOT = DEFAULT_APP_DIR.parent

COMPATIBLE_PYTHON = (3, 12)
REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("rank_bm25", "rank-bm25"),
    ("numpy", "numpy"),
    ("openai", "openai"),
    ("pydantic", "pydantic"),
    ("dotenv", "python-dotenv"),
    ("pymupdf", "pymupdf"),
    ("bs4", "beautifulsoup4"),
    ("rapidocr_onnxruntime", "rapidocr-onnxruntime"),
    ("rich", "rich"),
)
HF_REQUIRED_MODELS: tuple[str, ...] = (
    "BAAI/bge-small-en-v1.5",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
REQUIRED_PROJECT_FILES: tuple[str, ...] = (
    "app/requirements.txt",
    "app/helpdesk/__init__.py",
    "app/helpdesk/agent.py",
    "app/helpdesk/search.py",
    "app/helpdesk/sql.py",
    "app/helpdesk/ingest.py",
    "app/helpdesk/terminal_ui.py",
    "app/helpdesk/cli.py",
    "app/index/chunks.json",
    "app/index/manifest.json",
    "app/index/chroma/chroma.sqlite3",
    "database/rail_timetable.db",
    "database/schema.md",
)


@dataclass(frozen=True)
class HealthCheck:
    """Result row returned by non-destructive system health inspection."""

    name: str
    status: str  # "PASS" | "FAIL" | "INFO"
    detail: str
    remediation: str = ""


def resolve_app_dir(app_dir=None) -> Path:
    return Path(app_dir).resolve() if app_dir is not None else DEFAULT_APP_DIR


def venv_dir_path(app_dir=None) -> Path:
    return resolve_app_dir(app_dir) / ".venv"


def venv_python_path(app_dir=None) -> Path:
    vdir = venv_dir_path(app_dir)
    if os.name == "nt":
        return vdir / "Scripts" / "python.exe"
    return vdir / "bin" / "python"


def check_venv_exists(app_dir=None) -> bool:
    return venv_python_path(app_dir).is_file()


def check_python_compatibility(version_info=None) -> tuple[bool, str]:
    info = sys.version_info if version_info is None else tuple(version_info)
    major, minor = int(info[0]), int(info[1])
    patch = int(info[2]) if len(info) > 2 else 0
    return (major, minor) == COMPATIBLE_PYTHON, f"{major}.{minor}.{patch}"


def _probe_python_candidate(cmd_prefix, run_fn=subprocess.run) -> tuple[bool, str, list[str]]:
    try:
        proc = run_fn(
            list(cmd_prefix)
            + [
                "-c",
                "import sys; print(sys.executable); print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False, "", []
    if proc.returncode != 0:
        return False, "", []
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return False, "", []
    exe_path, ver = lines[0], lines[1]
    parts = ver.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        if (int(parts[0]), int(parts[1])) == COMPATIBLE_PYTHON:
            return True, ver, [exe_path]
    return False, ver, []


def find_compatible_python(run_fn=subprocess.run) -> tuple[list[str] | None, str]:
    """Locate a compatible Python 3.12 interpreter command without accepting mismatched versions."""
    ok_cur, cur_ver = check_python_compatibility()
    if ok_cur and Path(sys.executable).is_file():
        return [sys.executable], cur_ver
    for candidate in (["py", "-3.12"], ["python3.12"], ["python"]):
        ok, ver, resolved = _probe_python_candidate(candidate, run_fn=run_fn)
        if ok:
            return resolved, ver
    return None, cur_ver


def parse_env_file(env_path: Path) -> dict[str, str]:
    """Read key=value entries from a dotenv file safely without mutating os.environ."""
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def read_env_key_status(app_dir=None, environ=None) -> tuple[bool, str, bool]:
    """Return (env_file_exists, key_env_name, has_nonempty_key) without ever returning the key value."""
    app_path = resolve_app_dir(app_dir)
    repo_root = app_path.parent
    env_file = app_path / ".env"
    proc_env = os.environ if environ is None else environ

    merged: dict[str, str] = {}
    for candidate in (repo_root / ".env", env_file):
        merged.update(parse_env_file(candidate))
    for k, v in proc_env.items():
        if v:
            merged[k] = v

    key_name = (merged.get("AGENT_API_KEY_ENV") or "GEMINI_API_KEY").strip() or "GEMINI_API_KEY"
    key_val = (merged.get(key_name) or "").strip()
    return env_file.is_file(), key_name, bool(key_val)


def ensure_env_file(app_dir=None) -> tuple[bool, Path]:
    """Create app/.env from app/.env.example if missing; never inserts placeholder or fake secrets."""
    app_path = resolve_app_dir(app_dir)
    env_path = app_path / ".env"
    example_path = app_path / ".env.example"
    if env_path.is_file():
        return False, env_path
    if example_path.is_file():
        shutil.copyfile(example_path, env_path)
        return True, env_path
    return False, env_path


def check_required_packages(python_exe=None, run_fn=subprocess.run) -> tuple[bool, list[str]]:
    """Verify all required packages from requirements.txt can be discovered cleanly."""
    if python_exe is None or Path(python_exe).resolve() == Path(sys.executable).resolve():
        missing = [pkg_name for mod_name, pkg_name in REQUIRED_IMPORTS if importlib.util.find_spec(mod_name) is None]
        return len(missing) == 0, missing

    target = Path(python_exe)
    if not target.is_file():
        return False, [pkg_name for _, pkg_name in REQUIRED_IMPORTS]

    mods_csv = ",".join(mod for mod, _ in REQUIRED_IMPORTS)
    probe_code = (
        "import importlib.util, json; "
        f"mods = {mods_csv!r}.split(','); "
        "missing = [m for m in mods if importlib.util.find_spec(m) is None]; "
        "print(json.dumps(missing))"
    )
    try:
        proc = run_fn([str(target), "-c", probe_code], capture_output=True, text=True, timeout=20, check=False)
    except Exception:
        return False, [pkg_name for _, pkg_name in REQUIRED_IMPORTS]
    if proc.returncode != 0:
        return False, [pkg_name for _, pkg_name in REQUIRED_IMPORTS]
    try:
        missing_mods = set(json.loads((proc.stdout or "").strip() or "[]"))
    except json.JSONDecodeError:
        return False, [pkg_name for _, pkg_name in REQUIRED_IMPORTS]
    mod_to_pkg = dict(REQUIRED_IMPORTS)
    missing = [mod_to_pkg[m] for m in missing_mods if m in mod_to_pkg]
    return len(missing) == 0, missing


def hf_cache_hub_dir(cache_root=None) -> Path:
    if cache_root is not None:
        return Path(cache_root)
    explicit = (os.environ.get("HF_HUB_CACHE") or "").strip()
    if explicit:
        return Path(explicit)
    hf_home = (os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def are_hf_models_cached(cache_root=None) -> bool:
    """Return True only if both required embedding and reranker HF snapshots already exist locally."""
    hub = hf_cache_hub_dir(cache_root=cache_root)
    if not hub.is_dir():
        return False
    for model_id in HF_REQUIRED_MODELS:
        repo_dir = hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
        if not repo_dir.is_dir():
            return False
        snapshots = [p for p in repo_dir.iterdir() if p.is_dir() and any(p.iterdir())]
        if not snapshots:
            return False
    return True


def setup_environment(
    app_dir=None,
    *,
    python_exe=None,
    upgrade_pip=True,
    out=None,
    run_fn=subprocess.run,
) -> tuple[bool, str]:
    """Create app/.venv if absent, upgrade pip cleanly, install app/requirements.txt, and ensure app/.env."""
    stream = sys.stdout if out is None else out
    app_path = resolve_app_dir(app_dir)
    req_file = app_path / "requirements.txt"
    vdir = venv_dir_path(app_path)
    vpy = venv_python_path(app_path)

    def _log(msg: str) -> None:
        print(msg, file=stream, flush=True)

    if not req_file.is_file():
        return False, f"Requirements file not found: {req_file}"

    created_venv = False
    if not vpy.is_file():
        base_cmd = [str(python_exe)] if python_exe else None
        ver_str = ""
        if base_cmd is None:
            base_cmd, ver_str = find_compatible_python(run_fn=run_fn)
        if not base_cmd:
            return False, (
                f"Compatible Python {COMPATIBLE_PYTHON[0]}.{COMPATIBLE_PYTHON[1]} not found "
                f"(detected: {ver_str or 'none'}). Install Python 3.12 and retry."
            )
        _log(f"Creating virtual environment at {vdir} ...")
        res_venv = run_fn(list(base_cmd) + ["-m", "venv", str(vdir)], check=False)
        if res_venv.returncode != 0 or not vpy.is_file():
            return False, f"Failed to create virtual environment in {vdir}."
        created_venv = True

    if upgrade_pip and created_venv:
        _log("Upgrading pip inside app/.venv ...")
        res_pip = run_fn([str(vpy), "-m", "pip", "install", "--upgrade", "pip"], check=False)
        if res_pip.returncode != 0:
            return False, "Failed to upgrade pip inside app/.venv."

    _log(f"Installing dependencies from {req_file.name} ...")
    res_req = run_fn([str(vpy), "-m", "pip", "install", "-r", str(req_file)], check=False)
    if res_req.returncode != 0:
        return False, f"Dependency installation failed from {req_file}."

    ok_pkgs, missing = check_required_packages(python_exe=vpy, run_fn=run_fn)
    if not ok_pkgs:
        return False, f"Dependencies incomplete after installation: {', '.join(missing)}"

    created_env, env_path = ensure_env_file(app_path)
    _, _, has_key = read_env_key_status(app_path)
    if created_env:
        _log(f"Created {env_path} from .env.example (add your Gemini API key there for live chat).")
    elif not has_key:
        _log(f"Note: Gemini API credentials are not configured in {env_path} yet (required only for live Gemini chat).")

    return True, "Environment setup and verification succeeded."


def run_health_checks(
    app_dir=None,
    *,
    package_checker=None,
    cache_root=None,
) -> list[HealthCheck]:
    """Run non-destructive health checks across Python, venv, deps, config, Chroma, and SQLite DB."""
    app_path = resolve_app_dir(app_dir)
    repo_root = app_path.parent
    results: list[HealthCheck] = []

    # 1. Python compatibility
    py_ok, py_ver = check_python_compatibility()
    results.append(
        HealthCheck(
            name="Python environment",
            status="PASS" if py_ok else "FAIL",
            detail=f"Python {py_ver} ({'compatible 3.12' if py_ok else 'expected Python 3.12'})",
            remediation="" if py_ok else "Run using Python 3.12 (for example via py -3.12 or app\\.venv).",
        )
    )

    # 2. Virtual environment
    vpy = venv_python_path(app_path)
    venv_ok = vpy.is_file()
    in_venv = venv_ok and Path(sys.executable).resolve() == vpy.resolve()
    results.append(
        HealthCheck(
            name="Virtual environment",
            status="PASS" if venv_ok else "FAIL",
            detail=(
                "Active at app\\.venv" if in_venv else ("Found at app\\.venv" if venv_ok else "Missing app\\.venv")
            ),
            remediation="" if venv_ok else "Choose Set up project automatically or run .\\helpdesk.ps1 --setup.",
        )
    )

    # 3. Required Python packages
    checker = package_checker if callable(package_checker) else check_required_packages
    target_py = vpy if venv_ok else None
    pkgs_ok, missing = checker(target_py)
    results.append(
        HealthCheck(
            name="Python dependencies",
            status="PASS" if pkgs_ok else "FAIL",
            detail="All required packages installed" if pkgs_ok else f"Missing: {', '.join(missing)}",
            remediation="" if pkgs_ok else "Run setup/repair to install packages from app\\requirements.txt.",
        )
    )

    # 4. Gemini configuration (presence only; never reads or exposes secret value)
    env_exists, _, has_key = read_env_key_status(app_path)
    if has_key:
        results.append(HealthCheck("Gemini configuration", "PASS", "Gemini API credentials configured (value hidden)"))
    elif env_exists:
        results.append(
            HealthCheck(
                "Gemini configuration",
                "FAIL",
                "Gemini API credentials are empty in app\\.env",
                remediation="Open app\\.env and set GEMINI_API_KEY=<your-gemini-api-key>.",
            )
        )
    else:
        results.append(
            HealthCheck(
                "Gemini configuration",
                "FAIL",
                "app\\.env not found",
                remediation="Copy app\\.env.example to app\\.env and set GEMINI_API_KEY.",
            )
        )

    # 5. Vector database + manifest + chunk consistency (read-only URI on chroma.sqlite3)
    chunks_path = app_path / "index" / "chunks.json"
    manifest_path = app_path / "index" / "manifest.json"
    chroma_db = app_path / "index" / "chroma" / "chroma.sqlite3"
    if not (chunks_path.is_file() and manifest_path.is_file() and chroma_db.is_file()):
        results.append(
            HealthCheck(
                "Vector database",
                "FAIL",
                "Missing chunks.json, manifest.json, or chroma.sqlite3 under app\\index",
                remediation="Restore app\\index\\ or rebuild from the document pack via build_index.py.",
            )
        )
    else:
        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            from .search import COLLECTION, EMBED_MODEL, validate_chunks

            validate_chunks(chunks)
            con = sqlite3.connect(chroma_db.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                collections = [r[0] for r in con.execute("SELECT name FROM collections").fetchall()]
                chroma_ids = {r[0] for r in con.execute("SELECT embedding_id FROM embeddings").fetchall()}
            finally:
                con.close()
            chunk_ids = {c["chunk_id"] for c in chunks}
            sync_ok = (
                manifest.get("embedding_model") == EMBED_MODEL
                and manifest.get("chunks") == len(chunks)
                and COLLECTION in collections
                and chroma_ids == chunk_ids
            )
            if sync_ok:
                results.append(
                    HealthCheck(
                        "Vector database", "PASS", f"Chroma, manifest, and chunks.json synced ({len(chunks)} chunks)"
                    )
                )
            else:
                results.append(
                    HealthCheck(
                        "Vector database",
                        "FAIL",
                        f"Index mismatch (manifest={manifest.get('chunks')}, chunks={len(chunks)}, chroma={len(chroma_ids)})",
                        remediation="Rebuild index with .\\app\\.venv\\Scripts\\python.exe app\\build_index.py.",
                    )
                )
        except Exception as error:
            results.append(
                HealthCheck(
                    "Vector database",
                    "FAIL",
                    f"Validation failed: {error}",
                    remediation="Inspect app\\index\\ consistency or run pytest tests\\test_retrieval_audit.py.",
                )
            )

    # 6. Timetable SQLite database + read-only integrity check
    db_path = repo_root / "database" / "rail_timetable.db"
    if not db_path.is_file():
        results.append(
            HealthCheck(
                "Timetable database",
                "FAIL",
                "database\\rail_timetable.db not found",
                remediation="Restore database\\rail_timetable.db from repository.",
            )
        )
    else:
        try:
            from .sql import execute_readonly_query

            qres = execute_readonly_query("SELECT COUNT(*) AS cnt FROM trains", db_path=db_path)
            con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                pragma_row = con.execute("PRAGMA quick_check").fetchone()
            finally:
                con.close()
            if "error" in qres or not qres.get("rows") or not pragma_row or pragma_row[0] != "ok":
                results.append(
                    HealthCheck(
                        "Timetable database",
                        "FAIL",
                        f"Integrity or query check failed ({qres.get('error') or pragma_row})",
                        remediation="Restore uncorrupted database\\rail_timetable.db.",
                    )
                )
            else:
                cnt = qres["rows"][0]["cnt"]
                results.append(
                    HealthCheck("Timetable database", "PASS", f"Read-only check ok ({cnt} trains, quick_check=ok)")
                )
        except Exception as error:
            results.append(HealthCheck("Timetable database", "FAIL", f"Database check error: {error}"))

    # 7. Required project files
    missing_files = [rel for rel in REQUIRED_PROJECT_FILES if not (repo_root / rel).is_file()]
    results.append(
        HealthCheck(
            "Required project files",
            "PASS" if not missing_files else "FAIL",
            f"All {len(REQUIRED_PROJECT_FILES)} core files present"
            if not missing_files
            else f"Missing: {', '.join(missing_files)}",
            remediation="" if not missing_files else "Restore missing repository files.",
        )
    )

    # 8. HuggingFace cache readiness (informational; never fails offline evaluation before first download)
    hf_cached = are_hf_models_cached(cache_root=cache_root)
    results.append(
        HealthCheck(
            "HuggingFace model cache",
            "PASS" if hf_cached else "INFO",
            (
                "Cached locally (offline retrieval supported)"
                if hf_cached
                else "Not cached yet (first query will download models over internet)"
            ),
        )
    )
    return results


def choose_pytest_command(app_dir=None, run_fn=subprocess.run) -> tuple[list[str] | None, str]:
    """Pick an existing compatible Python interpreter with pytest installed without forcing global installs."""
    app_path = resolve_app_dir(app_dir)
    repo_root = app_path.parent
    candidates: list[Path] = []
    vpy = venv_python_path(app_path)
    if vpy.is_file():
        candidates.append(vpy)
    if Path(sys.executable) not in candidates:
        candidates.append(Path(sys.executable))
    for alt in (shutil.which("python"),):
        if alt and Path(alt) not in candidates:
            candidates.append(Path(alt))

    for exe in candidates:
        if not exe.is_file():
            continue
        try:
            proc = run_fn(
                [
                    str(exe),
                    "-c",
                    "import importlib.util, sys; "
                    "sys.exit(0 if (sys.version_info[:2] == (3, 12) and importlib.util.find_spec('pytest')) else 1)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if proc.returncode == 0:
                return [str(exe), "-m", "pytest", str(app_path / "tests"), "-v"], str(repo_root)
        except Exception:
            continue
    return None, str(repo_root)
