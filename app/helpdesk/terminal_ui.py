"""Interactive terminal presentation layer for the Railway Passenger Help Desk."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import sys
import textwrap
import threading
import time

MIN_TERMINAL_WIDTH = 44
DEFAULT_TERMINAL_WIDTH = 80
MAX_TERMINAL_WIDTH = 96

ABSTENTION_MESSAGE = (
    "I'm sorry, but I couldn't find sufficient information in the available railway records "
    "to answer this question reliably."
)

DIVIDER = "-" * 60
WELCOME_SCREEN = """============================================================
                 RAILWAY PASSENGER HELP DESK
                  AI-POWERED ASSISTANCE
============================================================

Welcome!

I can help you with:
  [1] Train schedules and routes
  [2] Ticket reservations
  [3] Cancellation and refund policies
  [4] Tatkal booking
  [5] Railway rules and circulars

Commands:
  help     Show commands
  clear    Clear terminal
  exit     Exit application

============================================================"""

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
    r"\b(refund|cancel(?:lation|led)?\s+charge|deduct(?:ed|ion)?|forfeit(?:ed)?|"
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


def get_terminal_width(width=None, stream=None) -> int:
    """Detect current terminal width dynamically with sensible minimum and maximum bounds."""
    if width is not None:
        try:
            cols = int(width)
        except (TypeError, ValueError):
            cols = DEFAULT_TERMINAL_WIDTH
        return max(MIN_TERMINAL_WIDTH, min(cols, MAX_TERMINAL_WIDTH)) if cols > 0 else DEFAULT_TERMINAL_WIDTH
    try:
        cols = shutil.get_terminal_size((DEFAULT_TERMINAL_WIDTH, 24)).columns
    except Exception:
        cols = DEFAULT_TERMINAL_WIDTH
    if cols <= 0:
        cols = DEFAULT_TERMINAL_WIDTH
    return max(MIN_TERMINAL_WIDTH, min(cols, MAX_TERMINAL_WIDTH))


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
    """Render the dynamically sized, balanced interactive welcome banner."""
    w = get_terminal_width(width=width, stream=stream)
    pal = get_palette(stream=stream, use_color=use_color)
    border_w = min(w, 72)
    border = _style("=" * border_w, pal.cyan, pal.reset)
    title = _style("RAILWAY PASSENGER HELP DESK".center(border_w).rstrip(), pal.bold_cyan, pal.reset)
    subtitle = _style("AI-POWERED ASSISTANCE".center(border_w).rstrip(), pal.cyan, pal.reset)

    topics = [
        ("1", "Train schedules and routes"),
        ("2", "Ticket reservations"),
        ("3", "Cancellation and refund policies"),
        ("4", "Tatkal booking"),
        ("5", "Railway rules and circulars"),
    ]
    cmds = [
        ("help", "Show commands"),
        ("clear", "Clear terminal"),
        ("exit", "Exit application"),
    ]

    lines = [border, title, subtitle, border, ""]
    lines.append(_style("Welcome!", pal.bold_cyan, pal.reset))
    lines.append("")
    lines.append(_style("I can help you with:", pal.cyan, pal.reset))
    for num, desc in topics:
        tag = _style(f"[{num}]", pal.cyan, pal.reset)
        for j, dline in enumerate(wrap_paragraph(desc, w, indent="  [0] ", subsequent_indent="      ")):
            body = dline[6:]
            if j == 0:
                lines.append(f"  {tag} {_style(body, pal.white, pal.reset)}")
            else:
                lines.append(f"      {_style(body, pal.white, pal.reset)}")

    lines.append("")
    lines.append(_style("Commands:", pal.cyan, pal.reset))
    for cmd, desc in cmds:
        pad = " " * (9 - len(cmd))
        lines.append(f"  {_style(cmd, pal.blue, pal.reset)}{pad}{_style(desc, pal.white, pal.reset)}")

    lines.extend(["", border])
    return "\n".join(lines)


def render_help(width=None, stream=None, use_color=None) -> str:
    """Render supported chatbot commands and example railway queries."""
    w = get_terminal_width(width=width, stream=stream)
    pal = get_palette(stream=stream, use_color=use_color)
    border = _style("-" * min(w, 72), pal.grey, pal.reset)
    header = _style("SUPPORTED COMMANDS", pal.bold_cyan, pal.reset)

    lines = [border, header, border]
    for cmd, desc in (
        ("help", "Display supported commands and examples"),
        ("clear", "Clear the terminal screen"),
        ("exit / quit", "Exit application gracefully"),
    ):
        pad = " " * max(2, 14 - len(cmd))
        lines.append(f"  {_style(cmd, pal.blue, pal.reset)}{pad}{_style(desc, pal.white, pal.reset)}")

    lines.extend(["", _style("Example questions:", pal.cyan, pal.reset)])
    examples = (
        "When does train 12002 leave New Delhi?",
        "Where does train 12951 stop?",
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


def _extract_timetable_fields(sql_query, query_runner=None):
    """Derive grounded timetable fields and multi-row listings from the executed read-only SQL."""
    if not isinstance(sql_query, str) or not sql_query.strip():
        return [], []
    if callable(query_runner):
        exec_fn = query_runner
    else:
        from . import sql as sql_mod

        exec_fn = sql_mod.execute_readonly_query
    try:
        res = exec_fn(sql_query, max_rows=20)
    except Exception:
        return [], []
    if not isinstance(res, dict) or "error" in res or not res.get("rows"):
        return [], []

    rows = res["rows"]
    name_kind = _sql_name_kind(sql_query)
    first = rows[0]
    keys = set(first.keys())

    if len(rows) == 1 and len(keys) == 1 and any(k.lower().startswith(("count", "total", "halt_count", "max", "min")) for k in keys):
        return [], []

    def _get(row, *candidates):
        for cand in candidates:
            val = _clean_val(row.get(cand))
            if val is not None:
                return val
        return None

    if len(rows) > 1:
        has_stop_seq = "stop_sequence" in keys or ("station_code" in keys and "number" not in keys)
        distinct_trains = {_get(r, "number", "train_number") for r in rows} - {None}
        if has_stop_seq and len(distinct_trains) <= 1 and ("station_code" in keys or "station_name" in keys):
            fields = []
            train_no = next(iter(distinct_trains), None)
            train_nm = _get(rows[0], "train_name")
            if train_no and train_nm:
                fields.append(("Train name and number", f"{train_nm} ({train_no})"))
            elif train_no:
                fields.append(("Train name and number", train_no))
            first_code = _get(rows[0], "station_code")
            last_code = _get(rows[-1], "station_code")
            first_st = _get(rows[0], "station_name") or (_get(rows[0], "name") if name_kind == "stations" else None)
            last_st = _get(rows[-1], "station_name") or (_get(rows[-1], "name") if name_kind == "stations" else None)
            if first_code and last_code:
                orig = f"{first_st} ({first_code})" if first_st and first_st != first_code else first_code
                dest = f"{last_st} ({last_code})" if last_st and last_st != last_code else last_code
                fields.append(("Origin and destination", f"{orig} -> {dest}"))
            dep = _get(rows[0], "departure")
            arr = _get(rows[-1], "arrival")
            if dep:
                fields.append(("Departure", dep))
            if arr:
                fields.append(("Arrival", arr))
            mid_rows = rows[1:-1] if len(rows) > 2 else []
            if mid_rows:
                stops = []
                for r in mid_rows[:12]:
                    code = _get(r, "station_code") or ""
                    st_name = _get(r, "station_name") or (_get(r, "name") if name_kind == "stations" else None)
                    label = f"{code} ({st_name})" if code and st_name else (code or st_name or "")
                    if label:
                        stops.append(label)
                if len(mid_rows) > 12:
                    stops.append(f"... (+{len(mid_rows) - 12} more stops)")
                if stops:
                    fields.append(("Intermediate stops", ", ".join(stops)))
            return fields, []

        if "number" in keys or "train_number" in keys:
            listings = []
            for r in rows[:10]:
                t_no = _get(r, "number", "train_number") or ""
                t_name = _get(r, "train_name") or (_get(r, "name") if name_kind in ("trains", "unknown") else None)
                dep = _get(r, "departure", "departs_from_a")
                arr = _get(r, "arrival", "arrives_at_b")
                head = f"{t_no} — {t_name}" if t_no and t_name else (t_no or t_name or "")
                parts = [p for p in (f"Dep: {dep}" if dep else None, f"Arr: {arr}" if arr else None) if p]
                listings.append(f"{head} ({', '.join(parts)})" if head and parts else (head or ", ".join(parts)))
            if len(rows) > 10:
                listings.append(f"... (+{len(rows) - 10} more matching trains)")
            return [], [x for x in listings if x]

        if "code" in keys or "station_code" in keys:
            listings = []
            for r in rows[:10]:
                code = _get(r, "code", "station_code")
                st_name = _get(r, "station_name", "name")
                if code and st_name:
                    listings.append(f"{code} — {st_name}")
                elif code or st_name:
                    listings.append(code or st_name)
            return [], listings
        return [], []

    fields = []
    t_no = _get(first, "number", "train_number")
    t_name = _get(first, "train_name") or (
        _get(first, "name")
        if name_kind == "trains" or (name_kind == "unknown" and "station_code" not in keys and "code" not in keys)
        else None
    )
    if t_no and t_name:
        fields.append(("Train name and number", f"{t_name} ({t_no})"))
    elif t_no:
        fields.append(("Train name and number", t_no))
    elif t_name:
        fields.append(("Train name and number", t_name))

    orig = _get(first, "from_station_code", "origin")
    dest = _get(first, "to_station_code", "destination")
    if orig and dest:
        fields.append(("Origin and destination", f"{orig} -> {dest}"))
    elif _get(first, "station_code", "code"):
        st_code = _get(first, "station_code", "code")
        st_name = _get(first, "station_name") or (_get(first, "name") if name_kind == "stations" else None)
        fields.append(("Station", f"{st_name} ({st_code})" if st_name else st_code))

    dep = _get(first, "departure", "departs_from_a")
    if dep:
        dep_day = _get(first, "departure_day", "day") if ("arrival" not in keys or _get(first, "departure_day")) else None
        fields.append(("Departure", f"{dep} (Day {dep_day})" if dep_day else dep))

    arr = _get(first, "arrival", "arrives_at_b")
    if arr:
        arr_day = _get(first, "arrival_day", "day")
        fields.append(("Arrival", f"{arr} (Day {arr_day})" if arr_day else arr))

    dur = _format_duration(_get(first, "duration_min", "travel_time_min"))
    if dur:
        fields.append(("Journey duration", dur))
    halt = _get(first, "halt_min")
    if halt is not None:
        fields.append(("Halt duration", f"{halt} minutes"))
    dist = _get(first, "distance_km")
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
        if m.group(1) == "." and _ABBREV_END_RE.search(prefix):
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
    """Convert a capstone contract dict into a wrapped, color-coded terminal assistant card."""
    w = get_terminal_width(width=width, stream=stream)
    pal = get_palette(stream=stream, use_color=use_color)
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

    # Successful answer: GREEN is strictly reserved for the actual grounded answer text.
    main_text = answer or dont_know_default
    for line in wrap_paragraphs(main_text, w):
        out.append(_style(line, pal.green, pal.reset) if line else "")

    is_mixed = bool(res.get("sql") and res.get("sources"))

    if res.get("sql"):
        fields, listings = _extract_timetable_fields(res.get("sql"), query_runner=sql_query_fn)
        if fields or listings:
            out.extend(["", _style("Timetable Details:", pal.bold_cyan, pal.reset)])
            if fields:
                out.extend(
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
                    out.append(f"{lead}{_style(line[len(lead):], pal.white, pal.reset)}")

    if res.get("sources"):
        policy_sections = _extract_policy_sections(main_text, is_mixed=is_mixed)
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
