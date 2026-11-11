"""Interactive terminal presentation layer for the Railway Passenger Help Desk."""

from __future__ import annotations

from dataclasses import dataclass
import io
import os
import re
import shutil
import sys
import textwrap
import threading

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    HAS_RICH = True
except ImportError:  # pragma: no cover - stdlib fallback only before dependencies install
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    box = None  # type: ignore[assignment]
    HAS_RICH = False

MIN_TERMINAL_WIDTH = 44
ABS_MIN_TERMINAL_WIDTH = 24
DEFAULT_TERMINAL_WIDTH = 80
MAX_TERMINAL_WIDTH = 96
MAX_UI_SQL_ROWS = 1000
ROUTE_CHUNK_THRESHOLD = 45
ROUTE_CHUNK_SIZE = 40

ABSTENTION_MESSAGE = (
    "I'm sorry, but I couldn't find sufficient information in the available railway records "
    "to answer this question reliably."
)

DIVIDER = "-" * 60
WELCOME_SCREEN = """========================================================================
                      RAILWAY PASSENGER HELP DESK
                Ask your railway-related question below.
========================================================================"""

_MONTHS_LONG = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
_DATE_VAL_RE = (
    rf"(?:\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}"
    rf"|\d{{1,2}}\s+(?:{_MONTHS_LONG})\s+\d{{4}}"
    rf"|(?:{_MONTHS_LONG})\s+\d{{4}})"
)
_MULTIWORD_DATE_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTHS_LONG})\s+(\d{{4}})\b")
_EFFECTIVE_DATE_RE = re.compile(
    rf"\b(?:with\s+effect\s+from|w\.e\.f\.?|effective(?:\s+from)?|starting(?:\s+from)?|"
    rf"came\s+into\s+force(?:\s+on|\s+from)?|comes?\s+into\s+force(?:\s+on|\s+from)?|"
    rf"applies?(?:\s+\w+){{0,4}}\s+(?:on|from)|applicable\s+from|"
    rf"compulsory(?:\s+[\w-]+){{0,6}}\s+from|only\s+be\s+booked(?:\s+[\w-]+){{0,6}}\s+from|from)\s+({_DATE_VAL_RE})\b",
    re.I,
)
_REFUND_RE = re.compile(
    r"\b(refund|cancel(?:lation|led)?\s+charges?|deduct(?:ed|ion)?|forfeit(?:ed)?|"
    r"clerkage|no\s+refund|tdr|ticket\s+deposit\s+receipt|entire\s+fare|admissible)\b",
    re.I,
)
_POLICY_SENTENCE_RE = re.compile(
    r"\b(refund|cancel(?:lation|led)?|tatkal|aadhaar|otp|advance\s+reservation|arp|"
    r"chart(?:ing|s)?|tdr|clerkage|booking\s+window|circular|rule|vande\s+bharat|"
    r"amrit\s+bharat|gst|fare\s+deduct|foreign\s+tourists?)\b",
    re.I,
)
_TIMETABLE_SENTENCE_RE = re.compile(
    r"\b(2016\s+timetable|timetable\s+snapshot|departs?|leaves?|arrives?|stops?\s+at|"
    r"station\s+code|halt(?:s|\s+duration)?|takes?\s+\d+\s+minutes?)\b",
    re.I,
)
_ABBREV_END_RE = re.compile(
    r"(?:\b(?:No|Rs|etc|Dr|Mr|Ms|St|Jn|Ref|Sec|Dept|Fig|Approx|Max|Min|v|vs)\.$"
    r"|(?:\bw\.e\.f|\be\.g|\bi\.e|\bp\.m|\ba\.m|\bG\.S\.R)\.$|\b[A-Z]\.$)",
    re.I,
)
_OUT_OF_SCOPE_HINT_RE = re.compile(
    r"\b(live|running\s+status|delay(?:ed)?|pnr|fare|ticket\s+price|cost|seat|availability|"
    r"waitlist|waiting\s+list|confirm(?:ation)?|platform)\b",
    re.I,
)
_TIMETABLE_NULLS = {"", "none", "null", "na", "n/a"}


@dataclass(frozen=True)
class Palette:
    """ANSI SGR palette enforcing consistent semantic terminal colors."""

    green: str = ""
    cyan: str = ""
    bold_cyan: str = ""
    blue: str = ""
    bold_blue: str = ""
    yellow: str = ""
    bold_yellow: str = ""
    grey: str = ""
    white: str = ""
    bold_white: str = ""
    reset: str = ""


_ANSI_PALETTE = Palette(
    green="\x1b[32m",
    cyan="\x1b[36m",
    bold_cyan="\x1b[1;36m",
    blue="\x1b[94m",
    bold_blue="\x1b[1;94m",
    yellow="\x1b[33m",
    bold_yellow="\x1b[1;33m",
    grey="\x1b[90m",
    white="\x1b[37m",
    bold_white="\x1b[1;37m",
    reset="\x1b[0m",
)
_PLAIN_PALETTE = Palette()


def _clean_text(text):
    text = re.sub(
        r"\s*[\[(]\s*Sources?\s*\d+(?:\s*(?:,|and|&)\s*(?:Sources?\s*)?\d+)*\s*[\])]",
        "",
        text or "",
        flags=re.I,
    )
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _norm(text):
    return re.sub(r"[^a-z0-9%]+", " ", (text or "").lower()).strip()


def enable_windows_ansi() -> bool:
    """Enable Virtual Terminal ANSI escape support on Windows consoles without third-party dependencies."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        vt_flag = 0x0004
        for std_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(std_id)
            if handle in (0, -1):
                continue
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                if not (mode.value & vt_flag):
                    kernel32.SetConsoleMode(handle, mode.value | vt_flag)
        return True
    except Exception:
        return False


def configure_terminal_io() -> None:
    """Configure UTF-8 output and Windows ANSI VT mode without disrupting non-console streams."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    enable_windows_ansi()


def should_use_color(stream=None, use_color=None) -> bool:
    """Return True only when ANSI styling should be emitted to an interactive terminal."""
    if os.environ.get("NO_COLOR", "") != "":
        return False
    if use_color is not None:
        return bool(use_color)
    if (os.environ.get("TERM") or "").strip().lower() == "dumb":
        return False
    target = sys.stdout if stream is None else stream
    try:
        is_tty = bool(target is not None and hasattr(target, "isatty") and target.isatty())
    except Exception:
        return False
    if not is_tty:
        return False
    if os.name == "nt":
        vt_ok = enable_windows_ansi()
        if not vt_ok and not (os.environ.get("WT_SESSION") or os.environ.get("TERM") or os.environ.get("ANSICON")):
            return False
    return True


def get_palette(stream=None, use_color=None) -> Palette:
    """Return the active ANSI color palette or empty strings when color is disabled."""
    return _ANSI_PALETTE if should_use_color(stream=stream, use_color=use_color) else _PLAIN_PALETTE


def get_raw_terminal_width(width=None, stream=None, min_width=ABS_MIN_TERMINAL_WIDTH) -> int:
    """Detect actual current terminal width dynamically with zero artificial upper clamp."""
    if width is not None:
        try:
            cols = int(width)
        except (TypeError, ValueError):
            cols = DEFAULT_TERMINAL_WIDTH
        return max(min_width, cols) if cols > 0 else DEFAULT_TERMINAL_WIDTH
    try:
        cols = shutil.get_terminal_size((DEFAULT_TERMINAL_WIDTH, 24)).columns
    except Exception:
        cols = DEFAULT_TERMINAL_WIDTH
    if cols <= 0:
        cols = DEFAULT_TERMINAL_WIDTH
    return max(min_width, cols)


def get_terminal_width(width=None, stream=None) -> int:
    """Detect current terminal width for prose layout, allowing genuinely narrow terminals without overflow."""
    cols = get_raw_terminal_width(width=width, stream=stream, min_width=ABS_MIN_TERMINAL_WIDTH)
    return min(cols, MAX_TERMINAL_WIDTH)


def center_text_lines(items, width: int) -> list[str]:
    """Wrap (if narrower than width) and horizontally center each line against exact width."""
    w = max(ABS_MIN_TERMINAL_WIDTH, int(width))
    out: list[str] = []
    for raw in items:
        text = (raw or "").strip()
        if not text:
            out.append("")
            continue
        if len(text) <= w:
            out.append(text.center(w).rstrip())
        else:
            wrapped = wrap_paragraph(text, w)
            for piece in wrapped:
                out.append(piece.center(w).rstrip())
    return out


def _style(text, color, reset):
    return f"{color}{text}{reset}" if color else text


def _split_long_token(token, limit):
    """Break only oversized filenames/paths cleanly at '_' or '/' boundaries when wider than the terminal."""
    if len(token) <= limit:
        return [token]
    segments = re.split(r"(?<=[_/])", token)
    if len(segments) == 1:
        return [token]
    out, cur = [], ""
    for seg in segments:
        if cur and len(cur) + len(seg) > limit:
            out.append(cur)
            cur = seg
        else:
            cur += seg
    if cur:
        out.append(cur)
    return out


def wrap_paragraph(text, width, indent="", subsequent_indent=None) -> list[str]:
    """Wrap one paragraph without splitting words or dates such as 15-07-2025 across lines."""
    sub = indent if subsequent_indent is None else subsequent_indent
    clean_p = re.sub(r"\s+", " ", (text or "").strip())
    if not clean_p:
        return []

    first_limit = max(16, width - len(indent))
    cont_limit = max(16, width - len(sub))
    # Keep multi-word dates (e.g. '16 January 2026') together on one line whenever they fit.
    protected = _MULTIWORD_DATE_RE.sub(lambda m: f"{m.group(1)}\u00a0{m.group(2)}\u00a0{m.group(3)}", clean_p)

    # Pre-split oversized filenames only if a single unbroken non-date token exceeds available width.
    tokens = []
    for tok in protected.split(" "):
        is_date_tok = bool(re.fullmatch(r"\d{1,2}[-./]\d{1,2}[-./]\d{2,4}[.,;:)]*", tok))
        if len(tok) > min(first_limit, cont_limit) and ("_" in tok or "/" in tok) and not is_date_tok:
            tokens.extend(_split_long_token(tok, min(first_limit, cont_limit)))
        else:
            tokens.append(tok)
    prepared = " ".join(tokens)

    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=indent,
        subsequent_indent=sub,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=True,
    )
    return [line.replace("\u00a0", " ") for line in wrapper.wrap(prepared)]


def wrap_paragraphs(text, width, indent="") -> list[str]:
    """Wrap multi-paragraph text while preserving blank paragraph separators."""
    paragraphs = [p.strip() for p in re.split(r"\r?\n\s*\r?\n", text or "") if p.strip()]
    lines = []
    for idx, para in enumerate(paragraphs):
        if idx > 0:
            lines.append("")
        lines.extend(wrap_paragraph(para, width, indent=indent))
    return lines


def format_kv_lines(
    pairs,
    width,
    *,
    key_color="",
    val_color="",
    reset="",
    indent="  ",
    highlight_keys=None,
    highlight_color="",
) -> list[str]:
    """Format label/value rows with aligned columns (or clean stacked lines on narrow terminals)."""
    if not pairs:
        return []
    highlights = set(highlight_keys or ())
    max_label_w = max(len(label) + 1 for label, _ in pairs)
    # Use 2-column alignment when value area retains >= 24 readable characters; otherwise stack cleanly.
    aligned_col = len(indent) + max_label_w + 2
    use_aligned = (width - aligned_col) >= 24
    lines = []
    for label, val in pairs:
        v_col = highlight_color if (label in highlights and highlight_color) else val_color
        key_text = f"{label}:"
        if use_aligned:
            pad = " " * (max_label_w - len(key_text) + 2)
            first_prefix_plain = f"{indent}{key_text}{pad}"
            cont_prefix_plain = " " * len(first_prefix_plain)
            wrapped = wrap_paragraph(val, width, indent=first_prefix_plain, subsequent_indent=cont_prefix_plain)
            for i, wline in enumerate(wrapped):
                if i == 0:
                    val_part = wline[len(first_prefix_plain):]
                    lines.append(f"{indent}{_style(key_text, key_color, reset)}{pad}{_style(val_part, v_col, reset)}")
                else:
                    val_part = wline[len(cont_prefix_plain):]
                    lines.append(f"{cont_prefix_plain}{_style(val_part, v_col, reset)}")
        else:
            lines.append(f"{indent}{_style(key_text, key_color, reset)}")
            cont_indent = indent + "  "
            for wline in wrap_paragraph(val, width, indent=cont_indent):
                val_part = wline[len(cont_indent):]
                lines.append(f"{cont_indent}{_style(val_part, v_col, reset)}")
    return lines


def render_welcome(width=None, stream=None, use_color=None) -> str:
    """Render compact, dynamically centered interactive chatbot heading (no category menu)."""
    raw_w = get_raw_terminal_width(width=width, stream=stream)
    border_w = min(raw_w, MAX_TERMINAL_WIDTH)
    pal = get_palette(stream=stream, use_color=use_color)
    border = _style("=" * border_w, pal.cyan, pal.reset)
    centered = center_text_lines(
        ["RAILWAY PASSENGER HELP DESK", "Ask your railway-related question below."],
        border_w,
    )
    styled_lines = [border]
    for idx, line in enumerate(centered):
        col = pal.bold_cyan if idx == 0 else pal.cyan
        styled_lines.append(_style(line, col, pal.reset))
    styled_lines.append(border)
    return "\n".join(styled_lines)


def render_help(width=None, stream=None, use_color=None) -> str:
    """Render supported chatbot commands, usage guidance, and example railway queries."""
    w = get_terminal_width(width=width, stream=stream)
    pal = get_palette(stream=stream, use_color=use_color)
    border = _style("-" * min(w, 72), pal.grey, pal.reset)
    header = _style("SUPPORTED COMMANDS", pal.bold_cyan, pal.reset)

    lines = [border, header, border]
    for hline in wrap_paragraph(
        "Ask any railway question in plain English (2016 timetable schedules, train routes, refunds, Tatkal rules, or circulars).",
        w,
    ):
        lines.append(_style(hline, pal.white, pal.reset))
    lines.append("")
    for cmd, desc in (
        ("help", "Display supported commands and usage guide"),
        ("clear", "Clear the terminal screen"),
        ("exit / quit", "Exit chatbot and return (or quit standalone session)"),
    ):
        pad = " " * max(2, 14 - len(cmd))
        lines.append(f"  {_style(cmd, pal.blue, pal.reset)}{pad}{_style(desc, pal.white, pal.reset)}")

    lines.extend(["", _style("Example questions:", pal.cyan, pal.reset)])
    examples = (
        "Show me the route of train 16303.",
        "When does train 12002 leave New Delhi?",
        "What is the Tatkal Aadhaar OTP rule?",
        "Cancellation charge for Vande Bharat Sleeper 10 hours before departure",
        "What is the advance reservation period now?",
    )
    for ex in examples:
        for j, wline in enumerate(wrap_paragraph(ex, w, indent="  - ", subsequent_indent="    ")):
            text = wline[4:] if j == 0 else wline[4:]
            prefix = "  - " if j == 0 else "    "
            lines.append(f"{_style(prefix, pal.grey, pal.reset)}{_style(text, pal.white, pal.reset)}")

    lines.append(border)
    return "\n".join(lines)


def clear_terminal(stream=None, use_color=None) -> None:
    """Clear the terminal safely without emitting ANSI escape bytes to redirected files."""
    target = sys.stdout if stream is None else stream
    is_tty = False
    try:
        is_tty = bool(target is sys.stdout and hasattr(target, "isatty") and target.isatty())
    except Exception:
        is_tty = False
    if is_tty:
        os.system("cls" if os.name == "nt" else "clear")
    elif should_use_color(stream=target, use_color=use_color):
        print("\x1b[2J\x1b[H", end="", file=target, flush=True)


class LoadingIndicator:
    """Lightweight non-misleading progress spinner erased cleanly before output rendering."""

    def __init__(self, stream=None, message="Searching railway records...", *, enabled=None, use_color=None, initial_delay=0.05):
        self.stream = sys.stdout if stream is None else stream
        self.message = message
        self.initial_delay = initial_delay
        self._use_color = should_use_color(stream=self.stream, use_color=use_color)
        if enabled is None:
            try:
                self.enabled = bool(hasattr(self.stream, "isatty") and self.stream.isatty())
            except Exception:
                self.enabled = False
        else:
            self.enabled = bool(enabled)
        self._stop = threading.Event()
        self._thread = None
        self._drawn_len = 0

    def _draw_loop(self):
        if self._stop.wait(self.initial_delay):
            return
        frames = ["|", "/", "-", "\\"]
        idx = 0
        pal = _ANSI_PALETTE if self._use_color else _PLAIN_PALETTE
        while not self._stop.is_set():
            text = f"[{frames[idx % len(frames)]}] {self.message}"
            self._drawn_len = max(self._drawn_len, len(text))
            styled = f"\r{_style(text, pal.cyan, pal.reset)}"
            try:
                self.stream.write(styled)
                self.stream.flush()
            except Exception:
                break
            idx += 1
            if self._stop.wait(0.1):
                break

    def start(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._draw_loop, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self.enabled and self._drawn_len > 0:
            try:
                self.stream.write("\r" + (" " * self._drawn_len) + "\r")
                self.stream.flush()
            except Exception:
                pass
            self._drawn_len = 0

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


def _clean_val(val):
    if val is None:
        return None
    text = str(val).strip()
    return None if text.lower() in _TIMETABLE_NULLS else text


def _format_duration(minutes):
    cleaned = _clean_val(minutes)
    if cleaned is None:
        return None
    try:
        total = int(float(cleaned))
    except (ValueError, TypeError):
        return cleaned
    if total < 0:
        return cleaned
    hours, mins = divmod(total, 60)
    return f"{hours}h {mins}m ({total} minutes)" if hours else f"{total} minutes"


def _sql_name_kind(sql_query):
    """Identify whether an unaliased 'name' column in SELECT came from trains or stations."""
    q = sql_query or ""
    select_part = re.split(r"\bFROM\b", q, maxsplit=1, flags=re.I)[0]
    from_part = q[len(select_part):]
    aliases = {}
    for table, alias in re.findall(r"\b(?:FROM|JOIN)\s+(trains|stations|train_types)\s+(?:AS\s+)?([A-Za-z_]\w*)\b", from_part, re.I):
        if alias.upper() not in {"WHERE", "ON", "JOIN", "ORDER", "GROUP", "LIMIT", "LEFT", "INNER"}:
            aliases[alias.lower()] = table.lower()
    for expr in select_part.split(","):
        if re.search(r"\bAS\s+name\b", expr, re.I) or re.search(r"(?:^|[.\s])name\s*$", expr.strip(), re.I):
            m = re.search(r"([A-Za-z_]\w*)\.name\b", expr, re.I)
            if m:
                tbl = aliases.get(m.group(1).lower())
                if tbl in ("trains", "stations"):
                    return tbl
    has_trains = bool(re.search(r"\bFROM\s+trains\b|\bJOIN\s+trains\b", from_part, re.I))
    has_stations = bool(re.search(r"\bFROM\s+stations\b|\bJOIN\s+stations\b", from_part, re.I))
    if has_stations and not has_trains:
        return "stations"
    if has_trains and not has_stations:
        return "trains"
    return "unknown"


def _get_cell(row, *candidates):
    for cand in candidates:
        val = _clean_val(row.get(cand))
        if val is not None:
            return val
    return None


def _extract_train_number_from_sql(sql_query: str) -> str | None:
    m = re.search(r"\b(?:train_number|number)\s*=\s*'([^']+)'", sql_query or "", re.I)
    return m.group(1).strip() if m else None


def _format_duration_compact(minutes):
    cleaned = _clean_val(minutes)
    if cleaned is None:
        return None
    try:
        total = int(float(cleaned))
    except (ValueError, TypeError):
        return cleaned
    if total < 0:
        return cleaned
    hours, mins = divmod(total, 60)
    return f"{hours}h {mins}m" if hours else f"{total}m"


def _format_station_label(row: dict, name_kind: str) -> str:
    code = _get_cell(row, "station_code", "code")
    st_name = _get_cell(row, "station_name") or (_get_cell(row, "name") if name_kind in ("stations", "unknown") else None)
    if st_name and code and st_name.upper() != code.upper():
        return f"{st_name} ({code})"
    return st_name or code or "-"


def _fetch_sql_result(sql_query, query_runner=None, max_rows=MAX_UI_SQL_ROWS):
    """Execute read-only SQL safely without artificial 20-row cutoff."""
    if not isinstance(sql_query, str) or not sql_query.strip():
        return None
    if callable(query_runner):
        exec_fn = query_runner
    else:
        from . import sql as sql_mod

        exec_fn = sql_mod.execute_readonly_query
    try:
        try:
            res = exec_fn(sql_query, max_rows=max_rows)
        except TypeError:
            res = exec_fn(sql_query)
    except Exception:
        return None
    if not isinstance(res, dict) or "error" in res or not res.get("rows"):
        return None
    return res


def _extract_timetable_fields(sql_query, query_runner=None):
    """Derive grounded timetable fields and complete multi-row listings from executed read-only SQL."""
    res = _fetch_sql_result(sql_query, query_runner=query_runner, max_rows=MAX_UI_SQL_ROWS)
    if res is None:
        return [], []

    rows = res["rows"]
    name_kind = _sql_name_kind(sql_query)
    first = rows[0]
    keys = set(first.keys())

    if len(rows) == 1 and len(keys) == 1 and any(k.lower().startswith(("count", "total", "halt_count", "max", "min")) for k in keys):
        return [], []

    if len(rows) > 1:
        has_stop_seq = "stop_sequence" in keys or ("station_code" in keys and "number" not in keys)
        distinct_trains = {_get_cell(r, "number", "train_number") for r in rows} - {None}
        if has_stop_seq and len(distinct_trains) <= 1 and ("station_code" in keys or "station_name" in keys):
            fields = []
            train_no = next(iter(distinct_trains), None) or _extract_train_number_from_sql(sql_query)
            train_nm = _get_cell(rows[0], "train_name")
            if train_no and train_nm:
                fields.append(("Train name and number", f"{train_nm} ({train_no})"))
            elif train_no:
                fields.append(("Train name and number", train_no))
            first_code = _get_cell(rows[0], "station_code")
            last_code = _get_cell(rows[-1], "station_code")
            first_st = _get_cell(rows[0], "station_name") or (_get_cell(rows[0], "name") if name_kind == "stations" else None)
            last_st = _get_cell(rows[-1], "station_name") or (_get_cell(rows[-1], "name") if name_kind == "stations" else None)
            if first_code and last_code:
                orig = f"{first_st} ({first_code})" if first_st and first_st != first_code else first_code
                dest = f"{last_st} ({last_code})" if last_st and last_st != last_code else last_code
                fields.append(("Origin and destination", f"{orig} -> {dest}"))
            dep = _get_cell(rows[0], "departure")
            arr = _get_cell(rows[-1], "arrival")
            if dep:
                fields.append(("Departure", dep))
            if arr:
                fields.append(("Arrival", arr))
            mid_rows = rows[1:-1] if len(rows) > 2 else []
            if mid_rows:
                stops = []
                for r in mid_rows:
                    code = _get_cell(r, "station_code") or ""
                    st_name = _get_cell(r, "station_name") or (_get_cell(r, "name") if name_kind == "stations" else None)
                    label = f"{code} ({st_name})" if code and st_name else (code or st_name or "")
                    if label:
                        stops.append(label)
                if stops:
                    fields.append(("Intermediate stops", ", ".join(stops)))
            return fields, []

        if "number" in keys or "train_number" in keys:
            listings = []
            for r in rows:
                t_no = _get_cell(r, "number", "train_number") or ""
                t_name = _get_cell(r, "train_name") or (_get_cell(r, "name") if name_kind in ("trains", "unknown") else None)
                dep = _get_cell(r, "departure", "departs_from_a")
                arr = _get_cell(r, "arrival", "arrives_at_b")
                head = f"{t_no} — {t_name}" if t_no and t_name else (t_no or t_name or "")
                parts = [p for p in (f"Dep: {dep}" if dep else None, f"Arr: {arr}" if arr else None) if p]
                listings.append(f"{head} ({', '.join(parts)})" if head and parts else (head or ", ".join(parts)))
            return [], [x for x in listings if x]

        if "code" in keys or "station_code" in keys:
            listings = []
            for r in rows:
                code = _get_cell(r, "code", "station_code")
                st_name = _get_cell(r, "station_name", "name")
                if code and st_name:
                    listings.append(f"{code} — {st_name}")
                elif code or st_name:
                    listings.append(code or st_name)
            return [], listings
        return [], []

    fields = []
    t_no = _get_cell(first, "number", "train_number")
    t_name = _get_cell(first, "train_name") or (
        _get_cell(first, "name")
        if name_kind == "trains" or (name_kind == "unknown" and "station_code" not in keys and "code" not in keys)
        else None
    )
    if t_no and t_name:
        fields.append(("Train name and number", f"{t_name} ({t_no})"))
    elif t_no:
        fields.append(("Train name and number", t_no))
    elif t_name:
        fields.append(("Train name and number", t_name))

    orig = _get_cell(first, "from_station_code", "origin")
    dest = _get_cell(first, "to_station_code", "destination")
    if orig and dest:
        fields.append(("Origin and destination", f"{orig} -> {dest}"))
    elif _get_cell(first, "station_code", "code"):
        st_code = _get_cell(first, "station_code", "code")
        st_name = _get_cell(first, "station_name") or (_get_cell(first, "name") if name_kind == "stations" else None)
        fields.append(("Station", f"{st_name} ({st_code})" if st_name else st_code))

    dep = _get_cell(first, "departure", "departs_from_a")
    if dep:
        dep_day = _get_cell(first, "departure_day", "day") if ("arrival" not in keys or _get_cell(first, "departure_day")) else None
        fields.append(("Departure", f"{dep} (Day {dep_day})" if dep_day else dep))

    arr = _get_cell(first, "arrival", "arrives_at_b")
    if arr:
        arr_day = _get_cell(first, "arrival_day", "day")
        fields.append(("Arrival", f"{arr} (Day {arr_day})" if arr_day else arr))

    dur = _format_duration(_get_cell(first, "duration_min", "travel_time_min"))
    if dur:
        fields.append(("Journey duration", dur))
    halt = _get_cell(first, "halt_min")
    if halt is not None:
        fields.append(("Halt duration", f"{halt} minutes"))
    dist = _get_cell(first, "distance_km")
    if dist is not None:
        fields.append(("Distance", f"{dist} km"))
    return fields, []


def _split_sentences(text):
    """Split text into sentences without breaking Indian dates (01.11.2024) or abbreviations (No., Rs., w.e.f.)."""
    s = (text or "").strip()
    if not s:
        return []
    parts, start = [], 0
    for m in re.finditer(r"([.!?])\s+(?=[A-Z0-9\"'(])", s):
        end_pos = m.start(1) + 1
        prefix = s[start:end_pos]
        if m.group(1) == "." and (_ABBREV_END_RE.search(prefix) or re.search(r"(?:^|\s)\d+\.$", prefix)):
            continue
        if len(prefix.strip()) >= 2:
            parts.append(prefix.strip())
            start = m.end()
    tail = s[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _policy_focus_text(answer, is_mixed=False):
    """For mixed SQL+policy replies, drop pure timetable clauses before extracting Policy Summary."""
    if not is_mixed:
        return answer
    clauses = []
    for sent in _split_sentences(answer):
        if ";" in sent:
            for sub in sent.split(";"):
                sub = sub.strip()
                if sub and (_POLICY_SENTENCE_RE.search(sub) or not _TIMETABLE_SENTENCE_RE.search(sub)):
                    clauses.append(sub if sub.endswith((".", "!", "?")) else sub + ".")
        elif _POLICY_SENTENCE_RE.search(sent) or not _TIMETABLE_SENTENCE_RE.search(sent):
            clauses.append(sent)
    return " ".join(clauses) if clauses else answer


def _extract_policy_sections(answer, *, is_mixed=False):
    """Organize multi-aspect railway policy answers into relevant labelled fields without forcing empty headings."""
    focused = _policy_focus_text(answer, is_mixed=is_mixed)
    sentences = _split_sentences(focused)
    if not sentences:
        return []

    effective_dates = []
    for m in _EFFECTIVE_DATE_RE.finditer(focused):
        val = _clean_text(m.group(1))
        if val and val not in effective_dates:
            effective_dates.append(val)

    rule = sentences[0]
    conditions = []
    refund_notes = []

    if len(sentences) == 1:
        split_m = re.search(
            r"^(.{20,}?)[,;]\s+((?:provided|unless|subject to|only if|except|however)\b.{12,})$",
            rule,
            re.I,
        )
        if split_m:
            rule = split_m.group(1).strip().rstrip(",;") + "."
            conditions.append(split_m.group(2).strip())

    for sent in sentences[1:]:
        starts_cond = bool(re.match(r"^(?:provided|unless|subject to|except|however|additionally|note\b|further|all\s+bookings)\b", sent, re.I))
        if _REFUND_RE.search(sent) and not starts_cond:
            refund_notes.append(sent)
        else:
            conditions.append(sent)

    refund_notes = [r for r in refund_notes if _norm(r) != _norm(rule) and r not in conditions]
    if not (conditions or effective_dates or refund_notes):
        return []

    sections = [("Applicable Rule", rule)]
    if conditions:
        sections.append(("Important Conditions", " ".join(conditions)))
    if effective_dates:
        sections.append(("Effective Date", "; ".join(effective_dates)))
    if refund_notes:
        sections.append(("Refund / Cancellation", " ".join(refund_notes)))
    return sections


_ANSI_STRIP_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SQL_COL_LABELS = {
    "number": "Train No.",
    "train_number": "Train No.",
    "train_name": "Train Name",
    "station_code": "Code",
    "code": "Code",
    "station_name": "Station",
    "from_station_code": "Origin",
    "to_station_code": "Dest",
    "departs_from_a": "Departs",
    "arrives_at_b": "Arrives",
    "departure": "Departure",
    "arrival": "Arrival",
    "duration_min": "Duration",
    "travel_time_min": "Duration",
    "distance_km": "Distance",
    "halt_min": "Halt",
    "stop_sequence": "Seq",
    "day": "Day",
    "is_halt": "Halt?",
    "type_code": "Type",
    "meaning": "Train Type",
    "state": "State",
    "zone": "Zone",
}
_RAIL_CLASSES_RE = re.compile(
    r"\b(AC\s+First(?:\s+Class|\s*/\s*Executive)?|Executive\s+Class|AC\s+2[- ]?Tier|2A|"
    r"AC\s+3[- ]?Tier|3A|AC\s+Chair\s+Car|CC|Sleeper\s+Class|SL|Second\s+Class|2S|First\s+Class|FC)\b",
    re.I,
)
_TIME_SLAB_RE = re.compile(
    r"((?:more\s+than|beyond|over|up\s+to|within|less\s+than|between)\s+\d+(?:\s+(?:and|to|-)\s+\d+)?\s+"
    r"(?:hours?|hrs?|minutes?|mins?|days?)(?:\s+(?:before|of|prior\s+to)\s+(?:the\s+)?(?:scheduled\s+)?departure)?|"
    r"after\s+(?:chart(?:ing)?|scheduled\s+departure)|during\s+(?:the\s+)?first\s+\d+\s+minutes?)",
    re.I,
)


def _strip_ansi(text: str) -> str:
    return _ANSI_STRIP_RE.sub("", text or "")


def _render_table_lines(
    headers: list[str],
    rows: list[list[str]],
    *,
    width: int,
    use_color: bool,
    col_justify: list[str] | None = None,
    col_styles: list[str] | None = None,
    no_wrap_cols: set[int] | None = None,
) -> list[str]:
    """Render structured rows as a width-adaptive Rich table (or stacked fallback on extreme narrow screens)."""
    if not headers or not rows:
        return []

    w = max(ABS_MIN_TERMINAL_WIDTH, int(width))
    # Stack rows vertically only when terminal is far too narrow for multi-column alignment.
    if len(headers) >= 3 and w < max(34, len(headers) * 7 + 4):
        pal = _ANSI_PALETTE if use_color else _PLAIN_PALETTE
        out: list[str] = []
        for r_idx, row in enumerate(rows):
            if r_idx > 0:
                out.append(_style("-" * min(w, 28), pal.grey, pal.reset))
            pairs = [(headers[i], row[i] if i < len(row) else "-") for i in range(len(headers))]
            out.extend(
                format_kv_lines(
                    pairs,
                    w,
                    key_color=pal.cyan,
                    val_color=pal.white,
                    reset=pal.reset,
                    indent="  ",
                )
            )
        return out

    if HAS_RICH:
        narrow = w < 52
        table = Table(
            box=box.SIMPLE_HEAD if narrow else box.ROUNDED,
            border_style="bright_black",
            header_style="bold cyan",
            padding=(0, 1),
            show_lines=False,
            expand=False,
        )
        no_wraps = no_wrap_cols or set()
        for idx, head in enumerate(headers):
            just = col_justify[idx] if col_justify and idx < len(col_justify) else "left"
            c_style = col_styles[idx] if col_styles and idx < len(col_styles) else "white"
            keep_no_wrap = (idx in no_wraps) and not narrow
            table.add_column(
                head,
                justify=just,
                style=c_style,
                no_wrap=keep_no_wrap,
                overflow="fold",
            )
        for row in rows:
            cells = [str(row[i] if i < len(row) else "-") for i in range(len(headers))]
            table.add_row(*cells)

        buf = io.StringIO()
        con = Console(
            file=buf,
            width=w,
            no_color=not use_color,
            force_terminal=bool(use_color),
            color_system="standard" if use_color else None,
            legacy_windows=False,
            highlight=False,
            soft_wrap=False,
        )
        con.print(table)
        lines = [ln.rstrip() for ln in buf.getvalue().splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        if lines and max((len(_strip_ansi(ln)) for ln in lines), default=0) <= w:
            return lines

    # Stdlib plain ASCII fallback (only when Rich absent or extreme edge wrap)
    pal = _ANSI_PALETTE if use_color else _PLAIN_PALETTE
    widths = [max([len(headers[i])] + [len(str(r[i] if i < len(r) else "-")) for r in rows]) for i in range(len(headers))]
    total_w = sum(widths) + (3 * (len(headers) - 1)) + 4
    if total_w > w:
        avail = max(12, w - (3 * (len(headers) - 1)) - 4)
        scale = avail / max(1, sum(widths))
        widths = [max(4, int(col_w * scale)) for col_w in widths]
    sep = "+" + "+".join("-" * (cw + 2) for cw in widths) + "+"
    hdr_line = "| " + " | ".join(headers[i][: widths[i]].ljust(widths[i]) for i in range(len(headers))) + " |"
    lines = [
        _style(sep, pal.grey, pal.reset),
        _style(hdr_line, pal.bold_cyan, pal.reset),
        _style(sep, pal.grey, pal.reset),
    ]
    for row in rows:
        vals = [str(row[i] if i < len(row) else "-") for i in range(len(headers))]
        wrapped_cols = [
            wrap_paragraph(vals[i], widths[i]) or ["-"] for i in range(len(headers))
        ]
        height = max(len(c) for c in wrapped_cols)
        for r_ln in range(height):
            cells = [
                (wrapped_cols[c][r_ln] if r_ln < len(wrapped_cols[c]) else "").ljust(widths[c])
                for c in range(len(headers))
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append(_style(sep, pal.grey, pal.reset))
    return lines


def _extract_markdown_table(text: str):
    """Extract an embedded Markdown pipe table into (lead_text, headers, rows, tail_text) if present."""
    raw_lines = (text or "").splitlines()
    for i in range(len(raw_lines) - 1):
        if "|" in raw_lines[i] and re.match(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$", raw_lines[i + 1]):
            headers = [c.strip() for c in raw_lines[i].strip().strip("|").split("|")]
            if len([h for h in headers if h]) < 2:
                continue
            rows: list[list[str]] = []
            end_idx = i + 2
            while end_idx < len(raw_lines) and "|" in raw_lines[end_idx]:
                cells = [c.strip() for c in raw_lines[end_idx].strip().strip("|").split("|")]
                if any(cells):
                    padded = (cells + ["-"] * len(headers))[: len(headers)]
                    rows.append([c or "-" for c in padded])
                end_idx += 1
            if rows:
                lead = "\n".join(raw_lines[:i]).strip()
                tail = "\n".join(raw_lines[end_idx:]).strip()
                return lead, headers, rows, tail
    return None


def _extract_refund_table_rows(text: str):
    """Extract structured cancellation timing or class charge rows strictly from existing answer text."""
    if not text or not _REFUND_RE.search(text):
        return None

    # 1. Class-wise flat charges mentioned in one policy sentence (e.g. Rs 240 for AC First, Rs 120 for Sleeper)
    class_pairs: list[tuple[str, str]] = []
    seen_cls: set[str] = set()
    for m in re.finditer(
        r"(Rs\.?\s*\d+(?:\s*\+\s*GST)?(?:\s*per\s+passenger)?)\s+(?:for|on)\s+([^,.;]+?)(?=(?:,|;|\band\b|\.(?:\s|$)|$))",
        text,
        re.I,
    ):
        charge = m.group(1).strip()
        cls_raw = _clean_text(m.group(2))
        if _RAIL_CLASSES_RE.search(cls_raw) and cls_raw.lower() not in seen_cls:
            seen_cls.add(cls_raw.lower())
            class_pairs.append((cls_raw, charge))
    if len(class_pairs) >= 2:
        return ["Class / Category", "Cancellation Charge"], [[c, r] for c, r in class_pairs]

    # 2. Time-window cancellation rules across sentences or semicolon clauses
    slab_rows: list[list[str]] = []
    seen_windows: set[str] = set()
    clauses: list[str] = []
    for sent in _split_sentences(text):
        if ";" in sent:
            clauses.extend(p.strip() for p in sent.split(";") if p.strip())
        else:
            clauses.append(sent)

    outcome_re = re.compile(
        r"(?:no\s+refund(?:\s+is\s+(?:admissible|granted|given))?|flat(?:\s+cancellation)?\s+charge(?:\s+is|\s+of)?\s+[^.;]+|"
        r"\d+%\s+(?:of\s+(?:the\s+)?)?(?:base\s+)?fare(?:\s+(?:is\s+)?deducted)?|Rs\.?\s*\d+(?:\s*\+\s*GST)?[^.;]*|"
        r"entire\s+fare[^.;]*|full\s+refund[^.;]*)",
        re.I,
    )
    for clause in clauses:
        tm = _TIME_SLAB_RE.search(clause)
        om = outcome_re.search(clause)
        if tm and om:
            win = _clean_text(tm.group(1)).rstrip(",:;")
            rule = _clean_text(om.group(0)).rstrip(",.;")
            win_norm = win.lower()
            if win_norm not in seen_windows and win.lower() != rule.lower():
                seen_windows.add(win_norm)
                slab_rows.append([win[:1].upper() + win[1:], rule[:1].upper() + rule[1:]])
    if len(slab_rows) >= 2:
        return ["Cancellation Timing", "Refund / Charge Rule"], slab_rows
    return None


def _extract_tatkal_table_rows(text: str):
    """Extract structured Tatkal class/timing/restriction rows strictly when >= 2 facts exist."""
    if not text or not re.search(r"\btatkal\b", text, re.I):
        return None

    time_re = re.compile(r"(\d{1,2}[:.]\d{2}\s*(?:hrs?|hours?|a\.?m\.?|p\.?m\.?)?)", re.I)
    clauses = [
        _clean_text(c)
        for c in re.split(r"[.;]+|,\s*(?=(?:and|while|whereas|for\b))|\band\s+(?=(?:at\b|authorised|authorized|aadhaar))", text, flags=re.I)
        if c and c.strip()
    ]

    ac_time = None
    non_ac_time = None
    agent_rule = None
    auth_rule = None

    for clause in clauses:
        tm = time_re.search(clause)
        if tm and re.search(r"\bnon[- ]AC\b", clause, re.I) and non_ac_time is None:
            non_ac_time = tm.group(1).strip()
        elif tm and re.search(r"\bAC\b", clause, re.I) and not re.search(r"\bnon[- ]AC\b", clause, re.I) and ac_time is None:
            ac_time = tm.group(1).strip()
        elif re.search(r"\bagents?\b", clause, re.I) and re.search(r"\b(?:first\s+\d+\s+minutes?|restrict|cannot\s+book)\b", clause, re.I):
            agent_rule = clause
        elif re.search(r"\bAadhaar\b", clause, re.I) and re.search(r"\b(?:OTP|compulsory|mandatory|required)\b", clause, re.I):
            auth_rule = clause

    if not (ac_time and non_ac_time):
        return None

    rows: list[list[str]] = [
        ["AC Classes", f"Opens at {ac_time}"],
        ["Non-AC Classes", f"Opens at {non_ac_time}"],
    ]
    if agent_rule:
        rows.append(["Authorized Agents", agent_rule])
    if auth_rule:
        rows.append(["Authentication", auth_rule])
    return ["Category", "Booking Time / Rule"], rows


def _extract_numbered_steps(text: str, question: str = ""):
    """Extract ordered procedural steps (rendered cleanly as numbered steps, never forced into tables)."""
    raw = (text or "").strip()
    if not raw:
        return None

    # 1. Explicit numbered items (multi-line or inline '1. ... 2. ...') or 'Step N:'
    step_matches = list(re.finditer(r"(?:^|\s)(?:Step\s*)?(\d+)[.):]\s+", raw, re.I | re.M))
    if len(step_matches) >= 2:
        nums = [int(m.group(1)) for m in step_matches]
        if nums[0] in (1, 0) and all(nums[i] == nums[i - 1] + 1 for i in range(1, len(nums))):
            lead = raw[: step_matches[0].start()].strip()
            steps: list[str] = []
            for idx, m in enumerate(step_matches):
                start = m.end()
                end = step_matches[idx + 1].start() if idx + 1 < len(step_matches) else len(raw)
                body = re.sub(r"\s+", " ", raw[start:end].strip())
                if body:
                    steps.append(body)
            if len(steps) >= 2:
                return lead, steps

    # 2. Multi-line bullet list ('- ' or '* ')
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    bullets = [re.sub(r"^[-*•]\s+", "", ln) for ln in lines if re.match(r"^[-*•]\s+\S+", ln)]
    if len(bullets) >= 2 and len(bullets) >= len(lines) - 2:
        lead_lines = [ln for ln in lines if not re.match(r"^[-*•]\s+\S+", ln)]
        return " ".join(lead_lines), bullets

    # 3. Clear procedural question with First/Next/Then/Finally cues
    if re.search(r"\b(how\s+to|steps?\s+to|procedure\s+(?:for|to))\b", question or "", re.I):
        sents = _split_sentences(raw)
        if len(sents) >= 3 and re.search(r"\b(first|then|next|afterwards|finally|step)\b", raw, re.I):
            return "", sents
    return None


def classify_text_response_type(answer: str, question: str = "", *, is_mixed: bool = False) -> str:
    """Deterministically select formatting strategy for document/prose responses."""
    focused = _policy_focus_text(answer, is_mixed=is_mixed)
    if _extract_markdown_table(focused) is not None:
        return "markdown_table"
    if _extract_refund_table_rows(focused) is not None:
        return "refund_policy"
    if _extract_tatkal_table_rows(focused) is not None:
        return "tatkal_rules"
    if _extract_numbered_steps(focused, question=question) is not None:
        return "procedural_steps"
    if _extract_policy_sections(focused, is_mixed=is_mixed):
        return "railway_circular"
    return "general_factual"


def classify_response_type(result, question: str = "", sql_data: dict | None = None) -> str:
    """Identify answer category deterministically from response contract, SQL shape, and prose structure."""
    res = result if isinstance(result, dict) else {}
    if bool(res.get("dont_know")):
        return "abstention"

    if sql_data and sql_data.get("rows"):
        rows = sql_data["rows"]
        keys = set(rows[0].keys())
        if len(rows) == 1 and len(keys) == 1 and any(
            k.lower().startswith(("count", "total", "halt_count", "max", "min", "avg")) for k in keys
        ):
            return "sql_summary"
        if len(rows) > 1:
            distinct_trains = {_get_cell(r, "number", "train_number") for r in rows} - {None}
            has_stop_shape = (
                "stop_sequence" in keys
                or (("station_code" in keys or "station_name" in keys) and "number" not in keys)
            )
            if has_stop_shape and len(distinct_trains) <= 1 and ("station_code" in keys or "station_name" in keys or "name" in keys):
                return "train_route"
            if (
                "number" in keys
                or "train_number" in keys
                or ("departure" in keys and "arrival" in keys)
                or ("departs_from_a" in keys and "arrives_at_b" in keys)
            ):
                return "train_comparison"
            return "sql_table"
        return "timetable_single"

    return classify_text_response_type(res.get("answer") or "", question=question, is_mixed=False)


def _prepare_route_lead(answer: str, rows: list[dict], orig_label: str, dest_label: str) -> str:
    """Keep factual overview sentences intact while compacting huge inline comma dumps duplicated in route table."""
    if len(rows) <= 8 or not answer:
        return answer
    station_tokens = set()
    for r in rows:
        for k in ("station_code", "station_name", "name"):
            v = _clean_val(r.get(k))
            if v:
                station_tokens.add(v.lower())
    out_sents = []
    for sent in _split_sentences(answer):
        comma_parts = [p.strip() for p in sent.split(",") if p.strip()]
        if len(comma_parts) >= 8:
            hits = sum(
                1 for p in comma_parts if any(tok in p.lower() for tok in station_tokens if len(tok) >= 2)
            )
            if hits >= 5:
                prefix_m = re.match(
                    r"^(.{10,}?(?:\bstops\s+at\b|\bpasses\s+through\b|\broute\s+(?:is|includes?)\b|\bstations?\s+include\b))",
                    sent,
                    re.I,
                )
                lead_prefix = prefix_m.group(1).strip() if prefix_m else f"Route from {orig_label} to {dest_label} covers"
                # Preserve any trailing factual sentence merged across a station abbreviation ending in '.' (e.g. 'JN.')
                tail_sent = ""
                last_comma_pos = sent.rfind(",")
                if last_comma_pos != -1:
                    after_last_comma = sent[last_comma_pos + 1 :]
                    period_m = re.search(r"\.\s+([A-Z].+)$", after_last_comma)
                    if period_m:
                        tail_sent = period_m.group(1).strip()
                compacted = f"{lead_prefix} all {len(rows)} listed stations in the route table below."
                out_sents.append(f"{compacted} {tail_sent}".strip())
                continue
        out_sents.append(sent)
    return " ".join(out_sents) if out_sents else answer


def _format_train_route_sections(
    sql_query: str,
    sql_data: dict,
    answer: str,
    *,
    prose_w: int,
    table_w: int,
    pal: Palette,
    use_color: bool,
) -> tuple[str, list[str]]:
    """Build complete non-truncated train route tables, summary table, and 2016 snapshot notices."""
    rows = sql_data["rows"]
    name_kind = _sql_name_kind(sql_query)
    if name_kind == "unknown" and ("station_code" in rows[0] or "station_name" in rows[0]):
        name_kind = "stations"
    keys = set(rows[0].keys())
    train_no = next(iter({_get_cell(r, "train_number", "number") for r in rows} - {None}), None) or _extract_train_number_from_sql(sql_query)
    train_name = _get_cell(rows[0], "train_name") or (
        _get_cell(rows[0], "name") if name_kind == "trains" else None
    )
    orig_label = _format_station_label(rows[0], name_kind)
    dest_label = _format_station_label(rows[-1], name_kind)
    dep_time = _get_cell(rows[0], "departure")
    arr_time = _get_cell(rows[-1], "arrival")
    truncated = bool(sql_data.get("truncated"))

    lead_text = _prepare_route_lead(answer, rows, orig_label, dest_label)

    sections: list[str] = []
    heading = f"TRAIN ROUTE — {train_no}" if train_no else "TRAIN ROUTE"
    sections.append(_style(heading, pal.bold_cyan, pal.reset))

    meta_pairs: list[tuple[str, str]] = []
    if train_no and train_name:
        meta_pairs.append(("Train", f"{train_name} ({train_no})"))
    elif train_name:
        meta_pairs.append(("Train", train_name))
    elif train_no:
        meta_pairs.append(("Train", train_no))
    meta_pairs.append(("Origin", orig_label))
    meta_pairs.append(("Last Listed Stop (Truncated)" if truncated else "Destination", dest_label))
    sections.extend(
        format_kv_lines(
            meta_pairs,
            prose_w,
            key_color=pal.blue,
            val_color=pal.bold_white,
            reset=pal.reset,
            indent="  ",
        )
    )

    # Determine which timetable columns are actually present in SQL output (never fabricate missing ones)
    has_arr = "arrival" in keys
    has_dep = "departure" in keys
    has_halt = "halt_min" in keys
    days_present = {_get_cell(r, "day") for r in rows} - {None}
    show_day = "day" in keys and (len(days_present) > 1 or any(d not in ("1", None) for d in days_present))
    has_pass_flags = "is_halt" in keys and any(str(_clean_val(r.get("is_halt")) or "") == "0" for r in rows)

    narrow = table_w < 52
    headers = ["No.", "Station"]
    justify = ["right", "left"]
    styles = ["blue", "white"]
    no_wraps = {0}

    if show_day and table_w >= 48:
        headers.append("Day")
        justify.append("right")
        styles.append("white")
        no_wraps.add(len(headers) - 1)
    if has_arr:
        headers.append("Arr" if narrow else "Arrival")
        justify.append("left")
        styles.append("bold white")
        no_wraps.add(len(headers) - 1)
    if has_dep:
        headers.append("Dep" if narrow else "Departure")
        justify.append("left")
        styles.append("bold white")
        no_wraps.add(len(headers) - 1)
    if has_halt and table_w >= 62:
        headers.append("Halt")
        justify.append("right")
        styles.append("white")
        no_wraps.add(len(headers) - 1)
    if has_pass_flags and table_w >= 68:
        headers.append("Stop")
        justify.append("left")
        styles.append("bright_black")
        no_wraps.add(len(headers) - 1)

    table_rows: list[list[str]] = []
    for idx, row in enumerate(rows, start=1):
        rec = [str(idx), _format_station_label(row, name_kind)]
        if show_day and table_w >= 48:
            rec.append(_get_cell(row, "day") or "-")
        if has_arr:
            rec.append(_get_cell(row, "arrival") or "-")
        if has_dep:
            rec.append(_get_cell(row, "departure") or "-")
        if has_halt and table_w >= 62:
            hval = _get_cell(row, "halt_min")
            rec.append(f"{hval}m" if hval is not None else "-")
        if has_pass_flags and table_w >= 68:
            ish = _clean_val(row.get("is_halt"))
            rec.append("Halt" if ish == "1" else ("Pass" if ish == "0" else "-"))
        table_rows.append(rec)

    total = len(table_rows)
    if total > ROUTE_CHUNK_THRESHOLD:
        for start in range(0, total, ROUTE_CHUNK_SIZE):
            end = min(total, start + ROUTE_CHUNK_SIZE)
            sections.extend(["", _style(f"ROUTE DETAILS (Stops {start + 1}-{end} of {total})", pal.bold_cyan, pal.reset)])
            sections.extend(
                _render_table_lines(
                    headers,
                    table_rows[start:end],
                    width=table_w,
                    use_color=use_color,
                    col_justify=justify,
                    col_styles=styles,
                    no_wrap_cols=no_wraps,
                )
            )
    else:
        sections.extend(["", _style("ROUTE DETAILS", pal.bold_cyan, pal.reset)])
        sections.extend(
            _render_table_lines(
                headers,
                table_rows,
                width=table_w,
                use_color=use_color,
                col_justify=justify,
                col_styles=styles,
                no_wrap_cols=no_wraps,
            )
        )

    summary_rows: list[list[str]] = []
    if has_dep:
        summary_rows.append(["Departure", dep_time or "-"])
    if has_arr:
        summary_rows.append(["Arrival", arr_time or "-"])
    summary_rows.append(["Origin", orig_label])
    summary_rows.append(["Last Listed Stop" if truncated else "Destination", dest_label])
    if has_pass_flags:
        halts_n = sum(1 for r in rows if str(_clean_val(r.get("is_halt")) or "") == "1")
        summary_rows.append(["Listed Stations", f"{total} ({halts_n} halts, {total - halts_n} pass-through)"])
    else:
        summary_rows.append(["Listed Stops", str(total)])
    summary_rows.append(["Timetable Snapshot", "August 2016 (historical, not live)"])

    sections.extend(["", _style("TIMETABLE SUMMARY", pal.bold_cyan, pal.reset)])
    sections.extend(
        _render_table_lines(
            ["Detail", "Value"],
            summary_rows,
            width=min(table_w, 72),
            use_color=use_color,
            col_justify=["left", "left"],
            col_styles=["blue", "bold white"],
            no_wrap_cols={0},
        )
    )

    if truncated:
        for warn in wrap_paragraph(
            f"Warning: Displayed rows reached the query row limit ({total} rows); additional stops may exist in the database.",
            prose_w,
        ):
            sections.append(_style(warn, pal.yellow, pal.reset))

    sections.append("")
    for note_ln in wrap_paragraph(
        "Historical Snapshot: Timetable figures reflect the August 2016 database, not live running status.",
        prose_w,
    ):
        sections.append(_style(note_ln, pal.yellow, pal.reset))

    return lead_text, sections


def _format_sql_cell(col: str, val: object, name_kind: str) -> str:
    cleaned = _clean_val(val)
    if cleaned is None:
        return "-"
    cl = col.lower()
    if cl in ("duration_min", "duration_minutes", "travel_time_min", "travel_time_minutes"):
        return _format_duration_compact(cleaned) or cleaned
    if cl == "distance_km":
        return f"{cleaned} km"
    if cl == "halt_min":
        return f"{cleaned}m"
    if cl == "is_halt":
        return "Halt" if cleaned == "1" else ("Pass" if cleaned == "0" else cleaned)
    return cleaned


def _format_sql_table_sections(
    sql_query: str,
    sql_data: dict,
    *,
    prose_w: int,
    table_w: int,
    pal: Palette,
    use_color: bool,
    is_comparison: bool,
) -> list[str]:
    """Render any multi-row SQL result into readable width-adapted Rich tables without truncation."""
    rows = sql_data["rows"]
    cols = list(sql_data.get("columns") or rows[0].keys())
    name_kind = _sql_name_kind(sql_query)
    narrow = table_w < 54

    headers: list[str] = []
    justify: list[str] = []
    styles: list[str] = []
    no_wraps: set[int] = set()
    for idx, col in enumerate(cols):
        cl = col.lower()
        if cl == "name":
            label = "Station Name" if name_kind == "stations" else ("Train Name" if name_kind == "trains" else "Name")
        elif cl in ("duration_min", "duration_minutes", "travel_time_min", "travel_time_minutes"):
            label = "Dur" if narrow else "Duration"
        elif cl.startswith("departs_from_") and cl != "departs_from_a":
            label = f"Dep ({cl.removeprefix('departs_from_').upper()})"
        elif cl.startswith("arrives_at_") and cl != "arrives_at_b":
            label = f"Arr ({cl.removeprefix('arrives_at_').upper()})"
        else:
            label = _SQL_COL_LABELS.get(cl, col.replace("_", " ").title())
        if narrow and label == "Departure":
            label = "Dep"
        elif narrow and label == "Arrival":
            label = "Arr"
        headers.append(label)

        if cl in ("stop_sequence", "day", "halt_min", "distance_km"):
            justify.append("right")
        else:
            justify.append("left")

        if cl in ("number", "train_number", "code", "station_code"):
            styles.append("blue")
            no_wraps.add(idx)
        elif cl in ("departure", "arrival", "departs_from_a", "arrives_at_b", "duration_min", "duration_minutes", "travel_time_min", "travel_time_minutes") or cl.startswith(("departs_from_", "arrives_at_")):
            styles.append("bold white")
            no_wraps.add(idx)
        else:
            styles.append("white")

    table_rows = [[_format_sql_cell(col, row.get(col), name_kind) for col in cols] for row in rows]
    title = "TRAIN SCHEDULE COMPARISON" if is_comparison else "TIMETABLE RESULTS"

    out: list[str] = []
    total = len(table_rows)
    if total > ROUTE_CHUNK_THRESHOLD:
        for start in range(0, total, ROUTE_CHUNK_SIZE):
            end = min(total, start + ROUTE_CHUNK_SIZE)
            out.extend(["", _style(f"{title} (Rows {start + 1}-{end} of {total})", pal.bold_cyan, pal.reset)])
            out.extend(
                _render_table_lines(
                    headers,
                    table_rows[start:end],
                    width=table_w,
                    use_color=use_color,
                    col_justify=justify,
                    col_styles=styles,
                    no_wrap_cols=no_wraps,
                )
            )
    else:
        out.extend(["", _style(title, pal.bold_cyan, pal.reset)])
        out.extend(
            _render_table_lines(
                headers,
                table_rows,
                width=table_w,
                use_color=use_color,
                col_justify=justify,
                col_styles=styles,
                no_wrap_cols=no_wraps,
            )
        )

    if sql_data.get("truncated"):
        out.append("")
        for warn in wrap_paragraph(
            f"Warning: Showing the first {total} rows returned by SQL; more matching records exist in the database.",
            prose_w,
        ):
            out.append(_style(warn, pal.yellow, pal.reset))

    out.append("")
    for note_ln in wrap_paragraph(
        "Historical Snapshot: Timetable results reflect the August 2016 database, not live status.",
        prose_w,
    ):
        out.append(_style(note_ln, pal.yellow, pal.reset))
    return out


def format_sources(result):
    """Format document citations and optional database reference for terminal display."""
    lines, seen = [], set()
    for src in result.get("sources") or []:
        if not isinstance(src, dict):
            continue
        doc = str(src.get("doc") or "").strip()
        if not doc:
            continue
        page = src.get("page")
        label = f"{doc} — Page {page}" if isinstance(page, int) and not isinstance(page, bool) else doc
        if label not in seen:
            seen.add(label)
            lines.append(label)
    if result.get("sql") and not result.get("dont_know"):
        lines.append("Railway timetable database (2016 snapshot, read-only SQL)")
    return lines


def format_response(
    result,
    question="",
    *,
    sql_query_fn=None,
    width=None,
    stream=None,
    use_color=None,
    dont_know_default="I don't know. The documents in this help desk don't cover that.",
) -> str:
    """Convert a capstone contract dict into an intelligent, width-adaptive terminal assistant card."""
    w = get_terminal_width(width=width, stream=stream)
    raw_w = get_raw_terminal_width(width=width, stream=stream)
    table_w = min(raw_w, 120) if width is None else raw_w
    pal = get_palette(stream=stream, use_color=use_color)
    color_on = bool(pal.green)
    res = result if isinstance(result, dict) else {"answer": dont_know_default, "sources": [], "sql": None, "dont_know": True}
    answer = (res.get("answer") or "").strip()
    dont_know = bool(res.get("dont_know"))

    border = _style("-" * min(w, 72), pal.grey, pal.reset)
    header = _style("RAILWAY ASSISTANT", pal.bold_cyan, pal.reset)
    out = [border, header, border, ""]

    if dont_know:
        # CRITICAL: Never use GREEN on abstentions or unavailable answers; abstention in YELLOW, details in WHITE.
        for line in wrap_paragraphs(ABSTENTION_MESSAGE, w):
            out.append(_style(line, pal.bold_yellow, pal.reset) if line else "")
        if answer and _norm(answer) != _norm(dont_know_default):
            out.append("")
            for line in wrap_paragraphs(answer, w):
                out.append(_style(line, pal.white, pal.reset) if line else "")
        elif _OUT_OF_SCOPE_HINT_RE.search(question or ""):
            out.append("")
            note = (
                "Available records cover the 2016 timetable snapshot and official policy circulars only "
                "(no live train status, current ticket fares, PNR status, seat availability, "
                "waiting-list predictions, or platform numbers)."
            )
            for line in wrap_paragraphs(note, w):
                out.append(_style(line, pal.white, pal.reset) if line else "")
        out.extend(["", border])
        return "\n".join(out)

    main_text = answer or dont_know_default
    sql_query = res.get("sql")
    sql_data = _fetch_sql_result(sql_query, query_runner=sql_query_fn, max_rows=MAX_UI_SQL_ROWS) if sql_query else None
    resp_type = classify_response_type(res, question=question, sql_data=sql_data)
    is_mixed = bool(sql_query and res.get("sources"))

    # Render Answer Zone cleanly (GREEN strictly reserved for actual grounded answer text)
    rendered_extra_sections: list[str] = []

    if resp_type == "train_route" and sql_data:
        lead_text, route_sections = _format_train_route_sections(
            sql_query,
            sql_data,
            main_text,
            prose_w=w,
            table_w=table_w,
            pal=pal,
            use_color=color_on,
        )
        for line in wrap_paragraphs(lead_text, w):
            out.append(_style(line, pal.green, pal.reset) if line else "")
        out.append("")
        out.extend(route_sections)
    else:
        # Check if prose itself is a markdown table or numbered procedure (non-SQL or policy focus)
        text_type = classify_text_response_type(main_text, question=question, is_mixed=False) if not sql_data else "general_factual"
        if text_type == "markdown_table":
            parsed_md = _extract_markdown_table(main_text)
            if parsed_md is not None:
                lead, md_headers, md_rows, tail = parsed_md
                if lead:
                    for line in wrap_paragraphs(lead, w):
                        out.append(_style(line, pal.green, pal.reset) if line else "")
                    out.append("")
                out.extend(
                    _render_table_lines(
                        md_headers,
                        md_rows,
                        width=min(table_w, 96),
                        use_color=color_on,
                    )
                )
                if tail:
                    out.append("")
                    for line in wrap_paragraphs(tail, w):
                        out.append(_style(line, pal.green, pal.reset) if line else "")
            else:
                for line in wrap_paragraphs(main_text, w):
                    out.append(_style(line, pal.green, pal.reset) if line else "")
        elif text_type == "procedural_steps":
            parsed_steps = _extract_numbered_steps(main_text, question=question)
            if parsed_steps is not None:
                lead, steps = parsed_steps
                if lead:
                    for line in wrap_paragraphs(lead, w):
                        out.append(_style(line, pal.green, pal.reset) if line else "")
                    out.append("")
                for s_idx, step in enumerate(steps, start=1):
                    prefix = f"  {s_idx}. "
                    sub = " " * len(prefix)
                    for j, wline in enumerate(wrap_paragraph(step, w, indent=prefix, subsequent_indent=sub)):
                        body = wline[len(prefix) if j == 0 else len(sub) :]
                        if j == 0:
                            out.append(f"  {_style(f'{s_idx}.', pal.bold_cyan, pal.reset)} {_style(body, pal.green, pal.reset)}")
                        else:
                            out.append(f"{sub}{_style(body, pal.green, pal.reset)}")
            else:
                for line in wrap_paragraphs(main_text, w):
                    out.append(_style(line, pal.green, pal.reset) if line else "")
        else:
            for line in wrap_paragraphs(main_text, w):
                out.append(_style(line, pal.green, pal.reset) if line else "")

        if resp_type in ("train_comparison", "sql_table") and sql_data:
            rendered_extra_sections.extend(
                _format_sql_table_sections(
                    sql_query,
                    sql_data,
                    prose_w=w,
                    table_w=table_w,
                    pal=pal,
                    use_color=color_on,
                    is_comparison=(resp_type == "train_comparison"),
                )
            )
        elif sql_query and resp_type == "timetable_single":
            fields, listings = _extract_timetable_fields(sql_query, query_runner=sql_query_fn)
            if fields or listings:
                rendered_extra_sections.extend(["", _style("Timetable Details:", pal.bold_cyan, pal.reset)])
                if fields:
                    rendered_extra_sections.extend(
                        format_kv_lines(
                            fields,
                            w,
                            key_color=pal.cyan,
                            val_color=pal.bold_white,
                            reset=pal.reset,
                        )
                    )
                for item in listings:
                    for j, line in enumerate(wrap_paragraph(item, w, indent="  ", subsequent_indent="    ")):
                        lead = "  " if j == 0 else "    "
                        rendered_extra_sections.append(f"{lead}{_style(line[len(lead):], pal.white, pal.reset)}")

        out.extend(rendered_extra_sections)

    # Policy / Document structured tables & summaries (works cleanly for both docs-only & mixed SQL+docs!)
    if res.get("sources") or (not sql_data and resp_type != "markdown_table"):
        doc_type = classify_text_response_type(main_text, question=question, is_mixed=is_mixed)
        focused_policy = _policy_focus_text(main_text, is_mixed=is_mixed)
        if doc_type == "refund_policy":
            extracted_ref = _extract_refund_table_rows(focused_policy)
            if extracted_ref:
                r_headers, r_rows = extracted_ref
                out.extend(["", _style("REFUND & CANCELLATION SUMMARY", pal.bold_cyan, pal.reset)])
                out.extend(
                    _render_table_lines(
                        r_headers,
                        r_rows,
                        width=min(table_w, 90),
                        use_color=color_on,
                        col_styles=["blue", "white"],
                    )
                )
        elif doc_type == "tatkal_rules":
            extracted_tatkal = _extract_tatkal_table_rows(focused_policy)
            if extracted_tatkal:
                t_headers, t_rows = extracted_tatkal
                out.extend(["", _style("TATKAL BOOKING RULES", pal.bold_cyan, pal.reset)])
                out.extend(
                    _render_table_lines(
                        t_headers,
                        t_rows,
                        width=min(table_w, 90),
                        use_color=color_on,
                        col_styles=["blue", "white"],
                    )
                )

        if res.get("sources"):
            policy_sections = _extract_policy_sections(main_text, is_mixed=is_mixed)
            if doc_type in ("refund_policy", "tatkal_rules", "procedural_steps"):
                policy_sections = [p for p in policy_sections if p[0] == "Effective Date"]
            if policy_sections:
                out.extend(["", _style("Policy Summary:", pal.bold_cyan, pal.reset)])
                out.extend(
                    format_kv_lines(
                        policy_sections,
                        w,
                        key_color=pal.bold_white,
                        val_color=pal.white,
                        reset=pal.reset,
                        highlight_keys={"Effective Date"},
                        highlight_color=pal.bold_cyan,
                    )
                )

    source_lines = format_sources(res)
    if source_lines:
        out.extend(["", _style("Sources:", pal.cyan, pal.reset)])
        for src in source_lines:
            for j, line in enumerate(wrap_paragraph(src, w, indent="  ", subsequent_indent="    ")):
                prefix = "  " if j == 0 else "    "
                out.append(f"{prefix}{_style(line[len(prefix):], pal.grey, pal.reset)}")

    out.extend(["", border])
    return "\n".join(out)


def _invoke_ask(ask_callable, question, agent_name=None):
    """Call ask_fn(question) cleanly while forwarding --agent only when specified and accepted."""
    if agent_name is None:
        return ask_callable(question)
    try:
        return ask_callable(question, agent=agent_name)
    except TypeError:
        return ask_callable(question)


def _is_interactive_tty(stream) -> bool:
    try:
        return bool(stream is sys.stdout and hasattr(stream, "isatty") and stream.isatty())
    except Exception:
        return False


def interactive_chat(
    *,
    agent_name=None,
    initial_question=None,
    ask_fn=None,
    sql_query_fn=None,
    input_fn=None,
    out=None,
    clear_fn=None,
    width=None,
    use_color=None,
    show_spinner=None,
    error_redactor=str,
    dont_know_default="I don't know. The documents in this help desk don't cover that.",
) -> int:
    """Launch continuous interactive passenger chatbot loop over the existing ask() API."""
    stream = sys.stdout if out is None else out
    if ask_fn is None:
        from .agent import ask as ask_callable
    else:
        ask_callable = ask_fn
    reader = input if input_fn is None else input_fn
    clearer = (lambda: clear_terminal(stream=stream, use_color=use_color)) if clear_fn is None else clear_fn
    pal = get_palette(stream=stream, use_color=use_color)
    interactive_tty = _is_interactive_tty(stream) and input_fn is None

    print(render_welcome(width=width, stream=stream, use_color=use_color), file=stream, flush=True)

    pending = (initial_question or "").strip() or None
    if pending:
        prompt_text = _style("You:", pal.bold_blue, pal.reset)
        q_text = _style(pending, pal.blue, pal.reset)
        print(f"\n{prompt_text} {q_text}", file=stream, flush=True)
    while True:
        from_reader = False
        if pending is None:
            prompt_text = _style("You:", pal.bold_blue, pal.reset)
            print(f"\n{prompt_text} ", end="", file=stream, flush=True)
            try:
                raw = reader("")
            except (EOFError, KeyboardInterrupt):
                bye = _style("Exiting Railway Passenger Help Desk. Goodbye!", pal.cyan, pal.reset)
                print(f"\n{bye}", file=stream, flush=True)
                return 0
            pending = (raw or "").strip()
            from_reader = True

        if from_reader and not interactive_tty:
            print(file=stream, flush=True)

        if not pending:
            pending = None
            continue

        cmd = pending.lower()
        if cmd in ("exit", "quit", "/exit", "/quit"):
            bye = _style("Thank you for using the Railway Passenger Help Desk. Goodbye!", pal.cyan, pal.reset)
            print(bye, file=stream, flush=True)
            return 0
        if cmd in ("help", "/help"):
            print(render_help(width=width, stream=stream, use_color=use_color), file=stream, flush=True)
            pending = None
            continue
        if cmd in ("clear", "/clear"):
            clearer()
            rehead = _style("RAILWAY PASSENGER HELP DESK", pal.bold_cyan, pal.reset)
            hint = _style("Type 'help' for commands or 'exit' to quit.", pal.grey, pal.reset)
            print(f"{rehead}\n{hint}", file=stream, flush=True)
            pending = None
            continue

        try:
            with LoadingIndicator(stream=stream, enabled=show_spinner, use_color=use_color):
                result = _invoke_ask(ask_callable, pending, agent_name=agent_name)
            print(
                format_response(
                    result,
                    pending,
                    sql_query_fn=sql_query_fn,
                    width=width,
                    stream=stream,
                    use_color=use_color,
                    dont_know_default=dont_know_default,
                ),
                file=stream,
                flush=True,
            )
        except (KeyboardInterrupt, EOFError):
            bye = _style("Exiting Railway Passenger Help Desk. Goodbye!", pal.cyan, pal.reset)
            print(f"\n{bye}", file=stream, flush=True)
            return 0
        except Exception as error:
            w = get_terminal_width(width=width, stream=stream)
            border = _style("-" * min(w, 72), pal.grey, pal.reset)
            title = _style("RAILWAY ASSISTANT", pal.bold_cyan, pal.reset)
            err_lines = [
                _style(line, pal.yellow, pal.reset)
                for line in wrap_paragraphs(f"Unable to process the request right now: {error_redactor(error)}", w)
            ]
            print("\n".join([border, title, border, "", *err_lines, "", border]), file=stream, flush=True)
        pending = None
