"""Offline audit & regression tests for Milestone 5: ingestion, chunk quality, metadata, and hybrid retrieval."""

from __future__ import annotations

import json
import re
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
    assert manifest["chunks"] == len(chunks) == 103

    # validate_chunks enforces schema, doc+page, non-empty text, unique chunk_ids
    search.validate_chunks(chunks)

    # Zero normalized-text duplicates
    norms = [re.sub(r"\W+", " ", c["text"].lower()).strip() for c in chunks]
    assert len(norms) == len(set(norms))

    # Bounds and expected document coverage (7 source docs, 1 overview chunk each)
    overviews = [c for c in chunks if c["metadata"].get("split_method") == "doc_overview"]
    assert len(overviews) == 7
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
        assert len(c["text"]) <= ingest.TABLE_CHUNK_SIZE
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", m["doc_date"]), f"Bad doc_date on {c['chunk_id']}: {m}"
        assert isinstance(m["ocr"], bool)
        if m["doc"].endswith(".pdf"):
            assert isinstance(m["page"], int) and m["page"] >= 1
        else:
            assert m["page"] is None

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
    assert "vande_bharat_sleeper" in ingest.find_scope("Vande Bharat Sleeper Train")
    assert "amrit_bharat_ii" in ingest.find_scope("Amrit Bharat II Express")
    assert ingest.devanagari_share("Cancellation charge Rs 120") == 0.0
    assert ingest.devanagari_share("रेलवे नियम") > 0.8


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
