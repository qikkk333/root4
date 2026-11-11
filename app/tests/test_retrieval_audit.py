"""Offline audit & regression tests for Milestone 5: ingestion, chunk quality, metadata, and hybrid retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from eval import evaluate_retrieval as er  # noqa: E402
from helpdesk import ingest, search  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Indexed Corpus & Manifest Integrity Audit
# ---------------------------------------------------------------------------
def test_index_manifest_and_chunks_pass_validation_and_metadata_audit():
    manifest = json.loads(search.MANIFEST_FILE.read_text(encoding="utf-8"))
    chunks = json.loads(search.CHUNKS_FILE.read_text(encoding="utf-8"))

    assert manifest["embedding_model"] == search.EMBED_MODEL
    assert manifest["query_instruction"] == search.QUERY_INSTRUCTION
    assert manifest["rerank_model"] == search.RERANK_MODEL
    assert manifest["chunks"] == len(chunks) == 100

    # validate_chunks enforces schema, doc+page, non-empty text, unique chunk_ids
    search.validate_chunks(chunks)

    # Zero normalized-text duplicates
    norms = [re.sub(r"\W+", " ", c["text"].lower()).strip() for c in chunks]
    assert len(norms) == len(set(norms))

    # Bounds and expected document coverage (7 source docs, 1 overview chunk each)
    overviews = [c for c in chunks if c["metadata"].get("split_method") == "doc_overview"]
    content_chunks = [c for c in chunks if c["metadata"].get("split_method") != "doc_overview"]
    assert len(overviews) == 7
    assert len(content_chunks) == 93
    docs = {c["metadata"]["doc"] for c in chunks}
    assert docs == {
        "indianrail_refund_rules_2015.html",
        "irctc_refund_rules_jan2026.pdf",
        "pib_2026-03-24_reform_express.html",
        "rb_cc08_2025_tatkal.pdf",
        "rb_cc08_2026_vande_bharat_sleeper_refund.pdf",
        "rb_cc10_2024_advance_reservation.pdf",
        "rb_cc10_2025_reservation_charts.pdf",
    }

    for c in chunks:
        m = c["metadata"]
        sec = m.get("section")
        body = c["text"].split("\n", 1)[1] if (sec and c["text"].startswith(sec + "\n")) else c["text"]
        assert len(body.strip()) >= 40, f"Stub body chunk {c['chunk_id']}: {c['text']!r}"
        assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", c["text"]), f"Control char in {c['chunk_id']}"
        assert len(c["text"]) <= ingest.TABLE_CHUNK_SIZE
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", m["doc_date"]), f"Bad doc_date on {c['chunk_id']}: {m}"
        assert isinstance(m["ocr"], bool)
        if m["doc"].endswith(".pdf"):
            assert isinstance(m["page"], int) and m["page"] >= 1
        else:
            assert m["page"] is None

    # Audit page coverage and OCR cleanup across the circulars and gazette PDFs
    vb_pages = {c["metadata"]["page"] for c in content_chunks if c["metadata"]["doc"] == "rb_cc08_2026_vande_bharat_sleeper_refund.pdf"}
    assert vb_pages == {1, 2, 5}
    assert not any("xxxGID" in c["text"] for c in chunks)

    tatkal_pages = {c["metadata"]["page"] for c in content_chunks if c["metadata"]["doc"] == "rb_cc08_2025_tatkal.pdf"}
    assert tatkal_pages == {1, 2}

    arp_chunks = [c for c in chunks if c["metadata"]["doc"] == "rb_cc10_2024_advance_reservation.pdf"]
    arp_blob = "\n".join(c["text"] for c in arp_chunks)
    assert {c["metadata"]["page"] for c in arp_chunks} == {1}
    assert all(c["metadata"]["ocr"] is True for c in arp_chunks)
    assert "Advance Reservations" in arp_blob and "Advanee" not in arp_blob
    assert "under the ARP of 120 days" in arp_blob and "uncler" not in arp_blob
    assert "\nii. Cancellations" in arp_blob and "\niii. There will be no change" in arp_blob

    chart_chunks = [c for c in chunks if c["metadata"]["doc"] == "rb_cc10_2025_reservation_charts.pdf"]
    chart_blob = "\n".join(c["text"] for c in chart_chunks)
    assert {c["metadata"]["page"] for c in chart_chunks} == {2}
    assert all(c["metadata"]["ocr"] is True for c in chart_chunks)
    assert "between 0500 hours and 1400 hours" in chart_blob and "between.0500" not in chart_blob
    assert "21:00 of previous day" in chart_blob and "21.:00" not in chart_blob
    assert "pos?" not in chart_blob

    # Chroma SQLite store is strictly synchronized with chunks.json (read-only URI never mutates disk)
    chroma_db = (search.CHROMA_DIR / "chroma.sqlite3").resolve()
    assert chroma_db.is_file()
    con = sqlite3.connect(chroma_db.as_uri() + "?mode=ro", uri=True)
    try:
        collections = [r[0] for r in con.execute("SELECT name FROM collections").fetchall()]
        chroma_ids = {r[0] for r in con.execute("SELECT embedding_id FROM embeddings").fetchall()}
    finally:
        con.close()
    assert search.COLLECTION in collections
    assert chroma_ids == {c["chunk_id"] for c in chunks}

    # 2015 comparison table preserves explicit OLD / REVISED rule versions
    statuses = Counter(c["metadata"].get("rule_status") for c in chunks)
    assert statuses["old"] == 5 and statuses["revised"] == 5


# ---------------------------------------------------------------------------
# 2. Ingest Helpers & Chunk Construction Sanity
# ---------------------------------------------------------------------------
def test_ingest_helpers_dates_scopes_times_and_cleaning():
    assert ingest.clean_space("Rs\u00a0120\tper  passenger") == "Rs 120 per passenger"
    assert ingest.to_iso("16", "01", "2026") == "2026-01-16"
    assert ingest.to_iso("32", "13", "2026") is None
    assert ingest.normal_times("Booking Hours: 00.20 HRS to 23.45 HRS") == "Booking Hours: 00:20 hours to 23:45 hours"
    assert ingest.clean_space("BOARD\x0b\x11  test") == "BOARD test"
    assert ingest._clean_ocr_text("between.0500 hours by 21.:00 on dated 16 .10.2024") == "between 0500 hours by 21:00 on dated 16.10.2024"
    assert ingest._clean_ocr_text("Time limit for Advanee Reservations yide No. 1 uncler ARP") == "Time limit for Advance Reservations vide No. 1 under ARP"
    assert ingest._clean_ocr_text("Ph0ne Ro0m 4th Flo0r g0v.in G20 train 12002 12.50") == "Phone Room 4th Floor gov.in G20 train 12002 12.50"
    assert ingest._fix_ocr_roman_sequence(["i. First", "i. Second", "i. Third", "iv. Fourth"]) == [
        "i. First", "ii. Second", "iii. Third", "iv. Fourth",
    ]
    assert "vande_bharat_sleeper" in ingest.find_scope("Vande Bharat Sleeper Train")
    assert "amrit_bharat_ii" in ingest.find_scope("Amrit Bharat II Express")
    assert ingest.devanagari_share("Cancellation charge Rs 120") == 0.0
    assert ingest.devanagari_share("रेलवे नियम") > 0.8


def test_ingest_skips_hindi_gazette_pages_and_attaches_comma_rule_leads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_pdf = tmp_path / "rb_cc08_2026_test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    hindi_paras = [
        {"type": "paragraph", "text": "xxxGIDHxxx 349 GI/2026 [PART II—SEC. 3(i)] THE GAZETTE OF INDIA"},
        {"type": "paragraph", "text": "रेल मंत्रालय अधिसूचना वंदे भारत स्लीपर एक्सप्रेस के रद्दीकरण प्रभार लागू होंगे।"},
    ]
    eng_paras = [
        {"type": "paragraph", "text": "Sub: Test rule updates."},
        {"type": "paragraph", "text": "1. New Delhi, dated 19.01.2026. These rules apply from 16.01.2026."},
        {"type": "paragraph", "text": "4. In the said rules, in rule 11,"},
        {"type": "paragraph", "text": "(a) Vande Bharat Sleeper Express seventy-two hours cancellation rule."},
        {"type": "paragraph", "text": "(b) Amrit Bharat II Express seventy-two hours cancellation rule."},
    ]
    monkeypatch.setattr(ingest, "pdf_pages", lambda _p: [(1, hindi_paras, False), (2, eng_paras, False)])
    chunks, _ = ingest.ingest_document(fake_pdf)
    assert all(c["metadata"]["page"] == 2 for c in chunks)
    assert any("rule 11,\n(a) Vande Bharat Sleeper Express" in c["text"] for c in chunks)
    assert not any(c["text"].rstrip().endswith("in rule 11,") for c in chunks)


# ---------------------------------------------------------------------------
# 3. Search Tokenizer, Query Expansion, RRF, Recency & Context Audit
# ---------------------------------------------------------------------------
def test_search_tokenizer_expansion_and_recency_penalties():
    toks = search.tokenize("Circular 10/2025 G.S.R. 41(E) dep 06:15")
    assert "10/2025" in toks and "2025" in toks
    assert "g.s.r" in toks and "41(e)" in toks
    assert "06:15" in toks and "06" in toks and "15" in toks

    expanded = search.expand_query("SL cancel 3 days before dep 6 pm")
    assert "sleeper class" in expanded
    assert "72 hours" in expanded
    assert "departure" in expanded
    assert "18:00" in expanded

    # Recency safety penalizes old/stale chunks EXCEPT for historical past queries
    old_chunk = {"chunk_id": "old", "text": "Old rule", "metadata": {"doc": "old.html", "page": None, "topic": "refund", "doc_date": "2015-11-12", "rule_status": "old"}}
    new_chunk = {"chunk_id": "new", "text": "New rule", "metadata": {"doc": "new.pdf", "page": 1, "topic": "refund", "doc_date": "2026-01-01"}}
    assert search._recency_penalties("What is sleeper cancellation fee now?", [old_chunk, new_chunk]) == [4.0, 0.0]
    assert search._recency_penalties("cancel 4 hours before departure", [old_chunk, new_chunk]) == [4.0, 0.0]
    assert search._recency_penalties("What was the clerkage before 2015?", [old_chunk, new_chunk]) == [0.0, 0.0]

    # RRF fusion rewards items ranked high in multiple lists & dedupe drops normalized duplicates
    a = {"chunk_id": "c1", "text": "Sleeper fee Rs 120.", "metadata": {"doc": "d1", "page": 1}}
    b = {"chunk_id": "c2", "text": "Tatkal fee rule.", "metadata": {"doc": "d2", "page": 1}}
    dup_a = {"chunk_id": "c1_dup", "text": "Sleeper fee: Rs 120", "metadata": {"doc": "d3", "page": 2}}
    fused = search.rrf([[a, b], [a]])
    assert fused[0]["chunk_id"] == "c1"
    assert len(search.dedupe([a, dup_a, b])) == 2


def test_bm25_and_neighbor_context_work_without_touching_chroma(monkeypatch: pytest.MonkeyPatch):
    # Prevent PersistentClient disk mutation during offline unit tests while loading real chunks.json BM25
    search._state.pop("chunks", None)
    fake_client = MagicMock()
    fake_client.get_collection.return_value = MagicMock()
    monkeypatch.setattr(search, "_chroma", lambda: fake_client)

    try:
        res_gsr = search.search("G.S.R. 41(E)", k=3, mode="bm25")
        assert res_gsr and res_gsr[0]["metadata"]["doc"] == "rb_cc08_2026_vande_bharat_sleeper_refund.pdf"

        res_sl = search.search("SL cancel flat cancellation charge sleeper class", k=5, mode="bm25", neighbors=True)
        assert any(r["metadata"]["doc"] == "irctc_refund_rules_jan2026.pdf" and "context_text" in r for r in res_sl)
    finally:
        search._state.pop("chunks", None)


def test_retrieval_eval_metrics_formulas():
    retrieved = [("docA.pdf", 2), ("docA.pdf", 1), ("docB.pdf", 3)]
    relevant = {("docA.pdf", 1), ("docC.html", None)}
    m = er.metrics(retrieved, relevant, k=3)
    assert m["hit"] == 1.0
    assert m["recall"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["rr"] == pytest.approx(0.5)
    assert er.chunk_level([{"text": "Hours: 00.20 HRS to 23:45"}], "Ticket booking 00:20 to 23:45.", k=1) == 1.0
