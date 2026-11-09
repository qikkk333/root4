"""Ingest: data pack documents -> clean, labelled chunks for search.py (Module 2, Session 1).

Steps
  1. Extract   PDFs: PyMuPDF text layer, lines grouped into paragraphs; tables read cell by cell (clip text keeps
               spaces); scanned pages -> RapidOCR at 300 dpi (250 merged two lines and lost a rule).
               HTML: BeautifulSoup; the 2015 rule table -> one item per rule with OLD and NEW clearly labelled.
  2. Clean     drop Hindi pages/paragraphs and OCR garbage (the bge model is English-only; the English pages carry
               the same rules), normalise spacing.
  3. Sections  numbered headings ("2. Cancellation Charges for RAC or Waitlisted Tickets:-"), "X:-" sub-headings,
               circular "Sub:" lines, PIB bold sub-headings. A new page opens a new section; the heading path carries on.
  4. Metadata  three levels (Session 1): document (title, issue date, effective date, circular/gazette no., type)
               -> page (ocr) -> paragraph/section (topic, scope = which trains a rule is for, rule_status).
  5. Chunks    paragraphs packed up to CHUNK_SIZE without breaking a paragraph, a new chunk whenever the train scope
               changes; tables kept whole (split by rows only when too big, header repeated, the sentence that
               introduces the table and any notes under it attached); embedding_text = context prefix + text.
  6. Validate  required keys, doc/page present, no empty text, unique ids, exact duplicates removed.

Public API:
    build_chunks(pack_dir=None) -> list of {"chunk_id", "text", "embedding_text", "metadata"}

CLI:
    python -m helpdesk.ingest                       build and save to index/chunks_preview.json, print stats
    python -m helpdesk.ingest --show DOC PAGE       print the chunks of one page (to inspect extraction)
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZE = 900          # characters of text per chunk (rules are short paragraphs; ~200 tokens)
TABLE_CHUNK_SIZE = 1400   # small tables stay whole even when a bit bigger
LONG_SPLIT_OVERLAP = 120  # overlap used only when a single paragraph is longer than CHUNK_SIZE
OCR_DPI = 300
OCR_MIN_CHARS = 50        # a PDF page with less text than this is treated as a scan
OCR_MIN_WORDS = 80        # an OCR'd page with fewer English words is a Hindi scan read as junk (real pages: ~245)

MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
MARKER_RE = re.compile(r"^\s*(\(\s*[a-z]{1,4}\s*\)|\(\s*\d{1,2}\s*\)|\d{1,2}\s*\.\s|[ivx]{1,4}\s*[.)]\s|[a-h]\s*\.\s)", re.I)
HEADING_NUM_RE = re.compile(r"^\s*(\d{1,2})\s*\.\s+([A-Z].{2,100})$")
SUBHEADING_RE = re.compile(r"^([A-Z][A-Za-z/&()\- ]{2,45}?)\s*:\s*[-–]\s*(.{0,45})$")
VERB_RE = re.compile(r"\b(shall|has been|have been|may be|will be|decided|is to be)\b", re.I)
LIST_INTRO_RE = re.compile(r"(as follows|namely|as under|given below)\s*[:;]?\s*[-–]*\s*$|:\s*[-–]\s*$", re.I)


def marker_kind(text):
    """Kind of list marker a paragraph starts with: roman '(ii)', letter '(b)', pnum '(4)', num '2.', letdot 'a.'."""
    t = text.lstrip()
    if re.match(r"\(\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\s*\)", t, re.I):
        return "roman"
    if re.match(r"\(\s*[a-h]\s*\)", t, re.I):
        return "letter"
    if re.match(r"\(\s*\d{1,2}\s*\)", t):
        return "pnum"
    if re.match(r"\d{1,2}\s*\.\s", t):
        return "num"
    if re.match(r"[a-h]\s*\.\s", t, re.I):
        return "letdot"
    return None
NOTE_RE = re.compile(r"^\s*(\*|note\b|notes\b|n\.?b\.?\b|explanation\b)", re.I)
COMMON_WORDS = set("""the of to and in for be by is shall will on with as from are this that all train trains ticket
tickets railway railways hours refund fare passenger passengers before after date time no not which or at it under
board india ministry new delhi circular case any per up upto other these may given""".split())


# ====================================================================== small helpers
def clean_space(text):
    text = re.sub(r"[  -​ 　\t]", " ", text)   # PDFs use em/en/thin spaces
    text = re.sub(r"[ ]{2,}", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def devanagari_share(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum("ऀ" <= ch <= "ॿ" for ch in letters) / len(letters)


def english_quality(text):
    """Share of common English words among word tokens: real English text ~0.3+, OCR'd Hindi junk ~0.05."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(words) < 8:
        return 1.0 if words else 0.0
    return sum(w in COMMON_WORDS for w in words) / len(words)


def to_iso(day, month, year):
    try:
        d, m, y = int(day), int(month), int(year)
        if 1 <= d <= 31 and 1 <= m <= 12 and 1990 <= y <= 2100:
            return f"{y:04d}-{m:02d}-{d:02d}"
    except ValueError:
        pass
    return None


# ====================================================================== 1. extraction
_ocr_engine = None


def _ocr_lines(page):
    """RapidOCR on a rendered page -> [(y, x, height, text)] in reading order."""
    global _ocr_engine
    import numpy as np
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    result, _ = _ocr_engine(img)
    lines = [(float(b[0][1]), float(b[0][0]), float(b[3][1] - b[0][1]), t.strip()) for b, t, _s in (result or []) if t.strip()]
    # same visual line = similar y; sort by line then by x
    lines.sort(key=lambda l: (l[0], l[1]))
    rows, current = [], []
    for line in lines:
        if current and abs(line[0] - current[0][0]) > 0.6 * max(current[0][2], 10):
            rows.append(current)
            current = []
        current.append(line)
    if current:
        rows.append(current)
    out = []
    for row in rows:
        row.sort(key=lambda l: l[1])
        out.append((row[0][0], row[0][1], max(l[2] for l in row), " ".join(l[3] for l in row)))
    return out


def _lines_to_paragraphs(lines):
    """[(y, x, height, text)] -> paragraphs. New paragraph on a big vertical gap, a list marker or a heading."""
    paras, current, prev = [], [], None
    for y, x, h, text in lines:
        starts_new = (prev is None or y - prev[0] > 1.75 * max(prev[2], 8) or MARKER_RE.match(text)
                      or is_heading(text) or text.lower().startswith(("sub:", "ref:")))
        if starts_new and current:
            paras.append(" ".join(current))
            current = []
        current.append(text)
        prev = (y, x, h)
    if current:
        paras.append(" ".join(current))
    return [clean_space(p) for p in paras if p.strip()]


def is_heading(text):
    """Level 1: '2. Cancellation Charges for RAC or Waitlisted Tickets:-' (short, no rule verbs, no full stop)."""
    t = text.strip()
    m = HEADING_NUM_RE.match(t)
    if not m or len(t.split()) > 13 or t.endswith((".", ",", ";")) or VERB_RE.search(t):
        return None
    return clean_heading(t)


def is_subheading(text):
    """Level 2: 'Clerkage Charge:-', 'Reservation Failed Transaction:- (i-ticket & e-tickets)'."""
    t = text.strip()
    m = SUBHEADING_RE.match(t)
    if not m or VERB_RE.search(t):
        return None
    return clean_heading(t)


def clean_heading(t):
    t = re.sub(r"\s*:\s*[-–]+\s*", ": ", t.strip())
    t = re.sub(r"^(\d{1,2})\s+\.", r"\1.", t)
    return t.rstrip(": -–").strip()


def _cell_text(page, rect):
    return clean_space(page.get_text("text", clip=rect)).replace("\n", " ") if rect else ""


def _table_item(page, table, last_header):
    """A PyMuPDF table -> {"type": "table", "header", "rows"} or a 'layout' paragraph for 2-column rule boxes."""
    grid = [[_cell_text(page, c) for c in row.cells] for row in table.rows]
    grid = [r for r in grid if any(c.strip() for c in r)]
    if not grid:
        return None, last_header
    first = grid[0]
    looks_like_header = all(len(c) <= 60 for c in first) and not any(re.fullmatch(r"\d{1,3}", c) for c in first[:1])
    # a 2-column "rule box" (condition | charge): one row, or the left text continues in lower case on the next row
    layout_box = len(first) == 2 and (len(grid) == 1 or (len(grid) > 1 and grid[1][0][:1].islower()))
    if layout_box:
        # a two-column box: left = condition, right = charge. Join each column as one sentence.
        left = " ".join(r[0] for r in grid).strip()
        right = " ".join(r[1] for r in grid if len(r) > 1).strip()
        return {"type": "paragraph", "text": clean_space(f"{left}: {right}")}, last_header
    if looks_like_header and len(grid) > 1:
        header, rows = first, grid[1:]
    elif last_header and len(last_header) == len(first):
        header, rows = last_header, grid          # table continued from the previous page: reuse its header
    else:
        header, rows = [f"Column {i + 1}" for i in range(len(first))], grid
    return {"type": "table", "header": header, "rows": rows}, header


def pdf_pages(path):
    """Yield (page_number, items, was_ocr) for a PDF. Items: heading / subheading / paragraph / table."""
    import pymupdf
    last_header = None
    with pymupdf.open(path) as pdf:
        for page in pdf:
            raw = page.get_text("text")
            if len(raw.strip()) < OCR_MIN_CHARS:
                paragraphs = _lines_to_paragraphs(_ocr_lines(page))
                yield page.number + 1, [_classify(p) for p in paragraphs], True
                continue
            try:
                tables = page.find_tables().tables
            except Exception:
                tables = []
            boxes = [pymupdf.Rect(t.bbox) for t in tables]
            elements = []        # (y, item)
            for t, box in zip(tables, boxes):
                item, last_header = _table_item(page, t, last_header)
                if item:
                    elements.append((box.y0, item))
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    r = pymupdf.Rect(line["bbox"])
                    if not text or any(b.contains(pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)) for b in boxes):
                        continue
                    lines.append((r.y0, r.x0, r.height, text))
            lines.sort(key=lambda l: (round(l[0]), l[1]))
            # paragraphs between tables, positioned by their first line
            chunk_lines, para_elements = [], []
            for line in lines:
                chunk_lines.append(line)
            for para_y, para in _paragraphs_with_y(chunk_lines):
                para_elements.append((para_y, _classify(para)))
            elements += para_elements
            elements.sort(key=lambda e: e[0])
            yield page.number + 1, [e[1] for e in elements], False


def _paragraphs_with_y(lines):
    out, current, start_y, prev, prev_heading = [], [], None, None, False
    for y, x, h, text in lines:
        heading = bool(is_heading(text) or is_subheading(text))
        starts_new = (prev is None or prev_heading or y - prev[0] > 1.75 * max(prev[2], 8) or MARKER_RE.match(text)
                      or heading or text.lower().startswith(("sub:", "ref:")))
        if starts_new and current:
            out.append((start_y, clean_space(" ".join(current))))
            current = []
        if not current:
            start_y = y
        current.append(text)
        prev, prev_heading = (y, x, h), heading      # a heading line always ends its own paragraph
    if current:
        out.append((start_y, clean_space(" ".join(current))))
    return out


def _classify(paragraph):
    if is_heading(paragraph):
        return {"type": "heading", "text": is_heading(paragraph)}
    if is_subheading(paragraph):
        return {"type": "subheading", "text": is_subheading(paragraph)}
    return {"type": "paragraph", "text": paragraph}


def html_pages(path):
    """Yield (None, items, False) once: HTML has no pages (cite with page null)."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    items = []
    rule_table = None
    for table in soup.find_all("table"):
        head = [c.get_text(" ", strip=True) for c in (table.find("tr").find_all(["td", "th"]) if table.find("tr") else [])]
        if len(head) == 3 and "pre revised" in " ".join(head).lower():
            rule_table = table
            break
    if rule_table is not None:
        # 2015 rule comparison: one item per rule, OLD and NEW labelled so the model never mixes them up
        header = [clean_space(c.get_text(" ", strip=True)) for c in rule_table.find("tr").find_all(["td", "th"])]
        intro = []
        for el in rule_table.find_all_previous(["p", "td", "strong"]):
            t = clean_space(el.get_text(" ", strip=True))
            if t and t not in intro:
                intro.insert(0, t)
        for t in intro:
            for part in re.split(r"(?=\b\d\.\s)", t):
                if len(part.strip()) > 20:
                    items.append(_classify(part.strip()))
        items.append({"type": "heading", "text": "3. Major changes in the refund rules (old vs revised)"})
        for tr in rule_table.find_all("tr")[1:]:
            cells = [clean_space(c.get_text("\n", strip=True)) for c in tr.find_all(["td", "th"])]
            if len(cells) != 3 or not (cells[1] or cells[2]):
                continue
            # two chunks per rule, each fully labelled, so a split can never hide which version a number belongs to
            n = cells[0].rstrip(".")
            if cells[1]:
                items.append({"type": "paragraph", "rule_status": "old",
                              "text": f"Rule {n} - OLD rule, {header[1]} (no longer applies):\n{cells[1]}"})
            if cells[2]:
                items.append({"type": "paragraph", "rule_status": "revised",
                              "text": f"Rule {n} - REVISED rule, {header[2]}:\n{cells[2]}"})
        return [(None, items, False)]
    # generic page (PIB press release): bold/heading lines become sections
    heads = {clean_space(t.get_text(" ", strip=True)) for t in soup.find_all(["h1", "h2", "h3", "strong", "b"])}
    heads = {h for h in heads if 3 <= len(h) <= 90}
    for line in soup.get_text("\n").split("\n"):
        line = clean_space(line)
        if not line:
            continue
        if re.match(r"\(?Release ID", line):
            # end of the release: the rest is the site footer and a second copy of the page
            if items and items[-1]["type"] == "heading":   # the byline (officer names) just before it
                items.pop()
            break
        if line in heads and len(line) <= 90:
            items.append({"type": "heading", "text": line})
        else:
            items.append({"type": "paragraph", "text": line})
    return [(None, _merge_short_lines(items), False)]


def _merge_short_lines(items):
    """HTML text often breaks sentences over lines: join consecutive paragraph lines into real paragraphs."""
    out = []
    for it in items:
        if (it["type"] == "paragraph" and out and out[-1]["type"] == "paragraph"
                and not re.search(r"[.!?:]$", out[-1]["text"]) and len(out[-1]["text"]) < 1200):
            out[-1] = {"type": "paragraph", "text": out[-1]["text"] + " " + it["text"]}
        else:
            out.append(dict(it))
    return out


# ====================================================================== 2-4. metadata
def document_metadata(path, first_text, all_text):
    """Level 1: true for the whole document."""
    name = path.name
    meta = {"doc": name, "doc_type": ("railway_board_circular" if name.startswith("rb_cc") else
                                      "press_release" if name.startswith("pib") else "refund_rules")}
    # title
    title = None
    m = re.search(r"Sub\s*[:;]\s*(.{5,120}?)(?:\n|Ref\s*:|$)", first_text, re.I)
    if m:
        title = m.group(1)
    if not title:
        title = next((l.strip() for l in first_text.split("\n") if len(l.strip()) > 15), name)
        title = re.split(r"\s+(?:w\.?e\.?f\.?|Authority\b)|(?<=[a-z])\.\s", title, flags=re.I)[0]
    meta["doc_title"] = clean_space(title)[:100].rstrip(" .:-")
    # issue date: file name full date > "New Delhi ... dated dd.mm.yyyy" > file name month > w.e.f. date > year
    fm = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    tm = re.search(r"dated\s*[;:,]?\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{4})", first_text, re.I)
    mm = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{4})", name)
    wm = re.search(r"w\.?\s*e\.?\s*f\.?\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{4})", all_text, re.I)
    if fm:
        meta["doc_date"], meta["date_source"] = fm.group(0), "file name"
    elif tm and to_iso(*tm.groups()):
        meta["doc_date"], meta["date_source"] = to_iso(*tm.groups()), "text: dated"
    elif mm:
        meta["doc_date"], meta["date_source"] = f"{mm.group(2)}-{MONTHS[mm.group(1)]:02d}-01", "file name month"
    elif wm and to_iso(*wm.groups()):
        meta["doc_date"], meta["date_source"] = to_iso(*wm.groups()), "text: w.e.f."
    else:
        y = re.search(r"(19|20)\d{2}", name)
        meta["doc_date"], meta["date_source"] = (f"{y.group(0)}-01-01", "file name year") if y else (None, None)
    # effective date ("w.e.f." / "with effect from")
    em = re.search(r"(?:w\.?\s*e\.?\s*f\.?|with effect from)\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{4})", all_text, re.I)
    if em and to_iso(*em.groups()):
        meta["effective_date"] = to_iso(*em.groups())
    # circular / gazette numbers (first mention = this document's own number)
    cm = re.search(r"Commercial\s+Circular\s*(?:No\.?|no\.?)?\s*(\d{1,3})\s*of\s*(\d{4})", all_text, re.I)
    if cm:
        meta["circular_no"] = f"{int(cm.group(1)):02d}/{cm.group(2)}"
    gm = re.search(r"G\.\s*S\.\s*R\.\s*(\d{1,4})\s*\(\s*E\s*\)", all_text)
    if gm:
        meta["gazette_no"] = f"G.S.R. {gm.group(1)}(E)"
    return meta


TOPIC_RULES = [  # first match wins (Session 1 if/elif order: most specific first)
    ("booking_hours", r"booking/cancellation hours|booking hours|cancellation hours"),
    ("document_authority", r"^document authority"),
    ("tdr", r"\btdr\b|ticket deposit"),
    ("tatkal", r"tatkal"),
    ("reservation_chart", r"chart"),
    ("advance_reservation", r"advance reservation|\barp\b"),
    ("boarding_change", r"boarding"),
    ("train_cancelled", r"cancellation of train|train cancel"),
    ("late_train", r"late"),
    ("rac_waitlist", r"rac\b|wait ?list|wait-list"),
    ("partially_confirmed", r"partially confirmed"),
    ("i_ticket", r"i-ticket"),
    ("e_ticket", r"e-ticket"),
    ("failed_transaction", r"failed transaction"),
    ("refund_cancellation", r"refund|cancellation|cancel"),
]


def find_topic(text):
    t = (text or "").lower()
    for topic, pattern in TOPIC_RULES:
        if re.search(pattern, t):
            return topic
    return None


def find_scope(text):
    t = text.lower()
    found = []
    if re.search(r"vande\s*bharat\s*sleeper", t):
        found.append("vande_bharat_sleeper")
    if re.search(r"amrit\s*bharat\s*(ii|2)\b", t):
        found.append("amrit_bharat_ii")
    return found


# ====================================================================== 5. chunking
def table_to_lines(item):
    """Each row as 'Header: value | Header: value' so every line carries its column names."""
    lines = []
    for row in item["rows"]:
        pairs = [f"{h}: {v}" for h, v in zip(item["header"], row) if v.strip()]
        if pairs:
            lines.append(" | ".join(pairs))
    return lines


def split_long(text, size=CHUNK_SIZE, overlap=LONG_SPLIT_OVERLAP):
    """A single paragraph longer than size: split at sentence ends, with a small overlap (Session 1 recursive split)."""
    sentences = re.split(r"(?<=[.;])\s+", text)
    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            out.append(cur.strip())
            cur = cur[-overlap:].split(" ", 1)[-1] + " "
        cur += s + " "
    if cur.strip():
        out.append(cur.strip())
    return out


def section_units(items):
    """Items of one section -> units: (text, kind, extra). Tables carry their intro sentence and notes."""
    units, i = [], 0
    while i < len(items):
        it = items[i]
        if it["type"] == "table":
            intro = ""
            if units and units[-1][1] == "paragraph" and len(units[-1][0]) < 400:
                intro = units.pop()[0]
            notes, j = [], i + 1
            while j < len(items) and items[j]["type"] == "paragraph" and NOTE_RE.match(items[j]["text"]):
                notes.append(items[j]["text"])
                j += 1
            units.append(((intro, table_to_lines(it), notes), "table", {}))
            i = j
            continue
        if it["type"] == "paragraph":
            text = it["text"]
            # a lone marker like "(b)" belongs to what follows it
            if re.fullmatch(r"\(?[a-z0-9]{1,4}\)?\.?", text.strip(), re.I) and i + 1 < len(items):
                nxt = dict(items[i + 1])
                if nxt["type"] == "paragraph":
                    nxt["text"] = text.strip() + " " + nxt["text"]
                    items[i + 1] = nxt
                    i += 1
                    continue
            units.append((text, "paragraph", {k: v for k, v in it.items() if k not in ("type", "text")}))
        i += 1
    return units


def chunk_units(units, heading_line):
    """Units -> [(text, split_method, scope_list, extra)]. Never splits a paragraph unless it alone is too long;
    starts a new chunk when the train scope changes, so a Vande Bharat rule never shares a chunk with general rules."""
    chunks, cur, cur_scope, cur_extra = [], [], None, {}
    intro_scope, intro_kind = [], None   # a sentence that opens a list ("... as follows: -") and its marker kind

    def flush():
        nonlocal cur, cur_scope, cur_extra
        if cur:
            chunks.append(("\n".join(([heading_line] if heading_line else []) + cur), "paragraphs", cur_scope or [], cur_extra))
        cur, cur_scope, cur_extra = [], None, {}

    for unit, kind, extra in units:
        if kind == "table":
            flush()
            intro, rows, notes = unit
            scope = find_scope(" ".join([intro] + rows + notes))
            head = "\n".join(x for x in [heading_line, intro] if x)
            tail = "\n".join(notes)
            body = "\n".join(rows)
            if len(head) + len(body) + len(tail) <= TABLE_CHUNK_SIZE:
                chunks.append(("\n".join(x for x in [head, body, tail] if x), "table", scope, {}))
            else:
                group = []
                for row in rows:
                    if group and len(head) + len("\n".join(group + [row])) + len(tail) > TABLE_CHUNK_SIZE:
                        chunks.append(("\n".join(x for x in [head, "\n".join(group), tail] if x), "table_rows", scope, {}))
                        group = []
                    group.append(row)
                if group:
                    chunks.append(("\n".join(x for x in [head, "\n".join(group), tail] if x), "table_rows", scope, {}))
            continue
        scope = find_scope(unit)
        kind = marker_kind(unit)
        in_list = kind is not None and intro_kind != kind and intro_kind != "done"
        if in_list and not scope:
            scope = intro_scope        # "(ii) ... fifty per cent" inherits "(f) In the case of Vande Bharat Sleeper ... as follows"
        if LIST_INTRO_RE.search(unit):
            intro_scope, intro_kind = scope, kind     # this sentence opens a list for the items after it
        elif not in_list:
            intro_scope, intro_kind = [], "done"      # same-level item or plain text: the list is over
        limit = TABLE_CHUNK_SIZE if (in_list and scope == cur_scope) else CHUNK_SIZE   # keep a list together
        if extra.get("rule_status"):
            # an OLD or REVISED rule version is always a chunk of its own, so its label can't be mixed up
            flush()
            chunks.append(("\n".join(([heading_line] if heading_line else []) + [unit]), "rule_version", scope, extra))
            continue
        if cur and (scope != cur_scope or len("\n".join(cur)) + len(unit) + 1 > limit):
            flush()
        if len(unit) > CHUNK_SIZE:
            for piece in split_long(unit):
                chunks.append(("\n".join(([heading_line] if heading_line else []) + [piece]), "long_split", scope, extra))
            continue
        cur.append(unit)
        cur_scope = scope
        cur_extra = {**cur_extra, **extra}
    flush()
    return chunks


# ====================================================================== 6. putting it together
def find_pack():
    for p in [ROOT / "data", ROOT / "data" / "rail_capstone_pack", ROOT / "rail_capstone_pack"]:
        if (p / "documents").is_dir():
            return p
    raise FileNotFoundError("Data pack not found. Unzip it into data/ (see README).")


TIME_RE = re.compile(r"\b([01]\d|2[0-3])\.([0-5]\d)\s*HRS\b", re.I)
LABEL_RE = re.compile(r"^([A-Z][A-Za-z/&()\- ]{3,50}):\s*(\S.{0,120})$")


def normal_times(text):
    """'00.20 HRS' -> '00:20 hours' so questions with 'time', 'hours' or '00:20' match."""
    return TIME_RE.sub(lambda m: f"{m.group(1)}:{m.group(2)} hours", text)


def _label_preamble(items):
    """Lines before the first heading of a document: the title/w.e.f./Authority header gets its own section
    ('Document authority and effective date'), and a short 'Label: value' fact (e.g. 'Ticket Booking/cancellation
    Hours: 00.20 HRS to 23.45 HRS') gets a section named after its label, so it is not buried in the header."""
    i = 0
    while i < len(items) and items[i]["type"] == "paragraph":
        text = items[i]["text"]
        m = LABEL_RE.match(text)
        if re.search(r"\bauthority\b|w\.e\.f", text, re.I) and i == 0:
            items.insert(i, {"type": "heading", "text": "Document authority and effective date", "preamble": True})
            i += 2
        elif m and not re.match(r"(sub|authority|note)\b", m.group(1), re.I):
            items.insert(i, {"type": "heading", "text": m.group(1).strip(), "preamble": True})
            i += 2
        else:
            i += 1


def ingest_document(path):
    pages = list(pdf_pages(path)) if path.suffix.lower() == ".pdf" else html_pages(path)
    # drop Hindi / OCR-garbage paragraphs; remember what was dropped
    kept_pages, dropped = [], 0
    for page, items, ocr in pages:
        page_text = " ".join(it.get("text", "") for it in items)
        if ocr and len(re.findall(r"[A-Za-z]{3,}", page_text)) < OCR_MIN_WORDS:
            dropped += len(items)            # a scanned Hindi page: OCR gives only a few junk words
            continue
        kept = []
        for it in items:
            sample = it.get("text") or " ".join(" ".join(r) for r in it.get("rows", []))
            if devanagari_share(sample) > 0.3 or (ocr and english_quality(sample) < 0.12 and len(sample) > 40):
                dropped += 1
                continue
            kept.append(it)
        kept_pages.append((page, kept, ocr))
    kept_pages = [p for p in kept_pages if p[1]]
    first_text = "\n".join(it.get("text", "") for it in (kept_pages[0][1] if kept_pages else []))
    all_text = "\n".join(it.get("text", "") for _, its, _ in kept_pages for it in its)
    doc_meta = document_metadata(path, first_text, all_text)
    if doc_meta["doc_type"] == "railway_board_circular":
        # circulars and gazettes number their PARAGRAPHS ("4. In the said rules ..."); their section is the "Sub:" line
        for _, items, _ in kept_pages:
            for it in items:
                if it["type"] in ("heading", "subheading"):
                    it["type"] = "paragraph"

    if doc_meta["doc_type"] != "railway_board_circular" and kept_pages:
        _label_preamble(kept_pages[0][1])
    for _, items, _ in kept_pages:
        for it in items:
            if it.get("text"):
                it["text"] = normal_times(it["text"])

    chunks, level1, level2, preamble_l1 = [], None, None, False
    for page, items, ocr in kept_pages:
        # split the page into sections at headings; the current heading path carries over from the previous page
        sections, current = [], {"l1": level1, "l2": level2, "items": []}
        for it in items:
            if it["type"] == "heading":
                sections.append(current)
                level1, level2, preamble_l1 = it["text"], None, bool(it.get("preamble"))
                current = {"l1": level1, "l2": None, "items": []}
            elif it["type"] == "subheading":
                sections.append(current)
                if preamble_l1:                     # a preamble label is not the parent of the real sections
                    level1, preamble_l1 = None, False
                level2 = it["text"]
                current = {"l1": level1, "l2": level2, "items": []}
            else:
                if it["type"] == "paragraph" and re.match(r"^sub\s*[:;]", it["text"], re.I) and level1 is None:
                    level1 = clean_space(it["text"])[:120]
                    current["l1"] = level1
                current["items"].append(it)
        sections.append(current)
        for sec in sections:
            if not sec["items"]:
                continue
            path_list = [p for p in (sec["l1"], sec["l2"]) if p]
            heading_line = " > ".join(path_list)
            for text, method, scope, extra in chunk_units(section_units(list(sec["items"])), heading_line):
                topic = find_topic(heading_line) or find_topic(text)
                meta = {**doc_meta, "page": page, "section_path": path_list, "section": heading_line or None,
                        "topic": topic, "scope": scope[0] if len(scope) == 1 else ("+".join(scope) if scope else "general"),
                        "split_method": method, "ocr": ocr, **extra}
                chunks.append({"text": text, "metadata": meta})
    return chunks, dropped


def embedding_prefix(meta):
    parts = [f"Document: {meta['doc_title']} ({meta['doc']})", f"Date: {meta.get('doc_date') or 'unknown'}"]
    if meta.get("circular_no"):
        parts.append(f"Commercial Circular {meta['circular_no']}")
    if meta.get("gazette_no"):
        parts.append(meta["gazette_no"])
    if meta.get("section"):
        parts.append(f"Section: {meta['section']}")
    if meta.get("scope") and meta["scope"] != "general":
        parts.append(f"Applies to: {meta['scope'].replace('_', ' ').replace('+', ' and ')}")
    return " | ".join(parts)


def overview_chunk(path, doc_chunks):
    """One extra chunk per document that answers 'what does this document cover / list its sections / topics'.
    Built from the document's own metadata and section names (no new facts). Cited as the document's first page."""
    first = doc_chunks[0]["metadata"]
    sections, topics = [], []
    for c in doc_chunks:
        sec, top = c["metadata"].get("section"), c["metadata"].get("topic")
        if sec and sec not in sections:
            sections.append(sec)
        if top and top not in topics and top != "document_authority":
            topics.append(top)
    lines = [f"Document overview: {first['doc_title']} ({path.name})",
             f"Type: {first['doc_type'].replace('_', ' ')} | Date: {first.get('doc_date') or 'unknown'}"
             + (f" | Commercial Circular No. {first['circular_no']}" if first.get("circular_no") else "")
             + (f" | {first['gazette_no']}" if first.get("gazette_no") else ""),
             "Sections and topics covered in this document:"]
    lines += [f"- {s}" for s in sections] or ["- (no section headings)"]
    lines.append("Topics: " + ", ".join(t.replace("_", " ") for t in topics))
    text = "\n".join(lines)
    meta = {k: v for k, v in first.items() if k not in ("section_path", "rule_status", "position")}
    meta.update({"section": "Document overview", "section_path": ["Document overview"], "topic": "document_overview",
                 "scope": "general", "split_method": "doc_overview", "position": None})
    return {"chunk_id": f"{path.stem}_overview", "text": text,
            "embedding_text": embedding_prefix(meta) + "\n" + text, "metadata": meta}


def build_chunks(pack_dir=None, verbose=True):
    pack = Path(pack_dir) if pack_dir else find_pack()
    out, seen = [], set()
    for path in sorted((pack / "documents").iterdir()):
        if path.suffix.lower() not in (".pdf", ".html", ".htm"):
            continue
        chunks, dropped = ingest_document(path)
        n = 0
        for c in chunks:
            key = re.sub(r"\W+", " ", c["text"].lower()).strip()
            if len(c["text"]) < 40 or key in seen:          # tiny fragments and exact repeats
                continue
            seen.add(key)
            n += 1
            c["metadata"]["position"] = n                    # order inside the document (neighbour lookups)
            out.append({"chunk_id": f"{path.stem}_{n:04d}", "text": c["text"],
                        "embedding_text": embedding_prefix(c["metadata"]) + "\n" + c["text"], "metadata": c["metadata"]})
        if chunks:
            out.append(overview_chunk(path, [c for c in out if c["metadata"]["doc"] == path.name]))
        if verbose:
            m = chunks[0]["metadata"] if chunks else {}
            print(f"  {n:4} chunks | dropped {dropped:3} hi/ocr-junk items | date {m.get('doc_date')} "
                  f"({m.get('date_source')}) | {path.name}", file=sys.stderr)
    from .search import validate_chunks
    validate_chunks(out)
    return out


def _main():
    ap = argparse.ArgumentParser(description="Build chunks from the data pack.")
    ap.add_argument("--pack", default=None)
    ap.add_argument("--out", default=str(ROOT / "index" / "chunks_preview.json"))
    ap.add_argument("--show", nargs=2, metavar=("DOC", "PAGE"), help="print the chunks of one page")
    args = ap.parse_args()
    chunks = build_chunks(args.pack)
    if args.show:
        doc, page = args.show
        page = None if page.lower() in ("none", "null") else int(page)
        for c in chunks:
            if c["metadata"]["doc"].startswith(doc) and c["metadata"]["page"] == page:
                m = c["metadata"]
                print("=" * 90)
                print(c["chunk_id"], "|", m["split_method"], "| topic:", m["topic"], "| scope:", m["scope"])
                print("EMBED PREFIX:", c["embedding_text"].split("\n", 1)[0])
                print("-" * 90)
                print(c["text"])
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    _main()
