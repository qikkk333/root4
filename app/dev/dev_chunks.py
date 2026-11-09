"""TEMPORARY stand-in for helpdesk/ingest.py (owned by a teammate), only so retrieval can be built and tested.
Delete this folder once ingest.py exists.

- PDFs: PyMuPDF text layer; pages with almost no text are OCR'd with RapidOCR (scans).
- HTML: BeautifulSoup; table rows become "cell | cell | cell" lines.
- Chunks: paragraphs packed to ~800 characters with 100 characters of overlap, never across pages.

Usage:  python dev/dev_chunks.py [pack_dir] [out.json]
        defaults: data/ or rail_capstone_pack/  ->  dev/chunks.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZE, OVERLAP, OCR_MIN_CHARS = 800, 100, 50
_ocr = None


def find_pack():
    for p in [ROOT / "data", ROOT / "data" / "rail_capstone_pack", ROOT / "rail_capstone_pack"]:
        if (p / "documents").is_dir():
            return p
    raise FileNotFoundError("No pack found in data/ or rail_capstone_pack/")


def ocr_page(page):
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    pix = page.get_pixmap(dpi=250)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    result, _ = _ocr(img)
    if not result:
        return ""
    lines = sorted(((box[0][1], box[0][0], box[3][1] - box[0][1], text) for box, text, score in result), key=lambda x: (round(x[0] / 12), x[1]))
    out, prev_y, prev_h = [], None, 0
    for y, x, h, text in lines:
        if prev_y is not None and y - prev_y > 1.8 * max(prev_h, 10):
            out.append("")                      # a big vertical gap starts a new paragraph
        out.append(text)
        prev_y, prev_h = y, h
    return "\n".join(out)


def doc_date(name):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if m:
        return m.group(0)
    months = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{4})", name)
    if m:
        return f"{m.group(2)}-{months[m.group(1)]:02d}-01"
    m = re.search(r"(20\d{2})", name)
    return f"{m.group(1)}-01-01" if m else None


def doc_type(name):
    if name.startswith("rb_cc"):
        return "railway_board_circular"
    if name.startswith("pib"):
        return "press_release"
    return "refund_rules"


def pages(path):
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as pdf:
            for page in pdf:
                text = page.get_text("text", sort=True)
                ocr = len(text.strip()) < OCR_MIN_CHARS
                yield page.number + 1, (ocr_page(page) if ocr else text), ocr
    else:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        for table in soup.find_all("table"):
            rows = [" | ".join(c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])) for tr in table.find_all("tr")]
            table.replace_with("\n\n" + "\n".join(r for r in rows if r.strip()) + "\n\n")
        yield None, soup.get_text("\n"), False


def pack_text(text):
    paras = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    chunks, cur = [], ""
    for p in filter(None, paras):
        while len(p) > CHUNK_SIZE:
            cut = p.rfind(" ", 0, CHUNK_SIZE) if " " in p[:CHUNK_SIZE] else CHUNK_SIZE
            if cur:
                chunks.append(cur.strip()); cur = ""
            chunks.append(p[:cut].strip())
            p = p[max(cut - OVERLAP, 1):]
        if cur and len(cur) + len(p) + 1 > CHUNK_SIZE:
            chunks.append(cur.strip())
            tail = cur[-OVERLAP:]
            cur = tail[tail.find(" ") + 1:] + "\n" if " " in tail else ""
        cur += p + "\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def main():
    pack = Path(sys.argv[1]) if len(sys.argv) > 1 else find_pack()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "dev" / "chunks.json"
    chunks, seen = [], set()
    for path in sorted((pack / "documents").iterdir()):
        n = 0
        for page, text, ocr in pages(path):
            for piece in pack_text(text):
                key = re.sub(r"\W+", " ", piece.lower()).strip()
                if len(piece) < 30 or key in seen:         # skip tiny fragments and exact repeats
                    continue
                seen.add(key)
                n += 1
                meta = {"doc": path.name, "page": page, "section_path": [], "doc_date": doc_date(path.name),
                        "doc_type": doc_type(path.name), "ocr": ocr}
                prefix = f"Document: {path.name} | Type: {meta['doc_type']} | Date: {meta['doc_date']}\n"
                chunks.append({"chunk_id": f"{path.stem}_{n:04d}", "text": piece,
                               "embedding_text": prefix + piece, "metadata": meta})
    out.write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
    per_doc = {}
    for c in chunks:
        per_doc[c["metadata"]["doc"]] = per_doc.get(c["metadata"]["doc"], 0) + 1
    print(f"{len(chunks)} chunks -> {out}")
    for d, n in per_doc.items():
        print(f"  {n:4} {d}")


if __name__ == "__main__":
    main()
