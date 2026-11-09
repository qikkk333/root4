"""Retrieval for the Railway Passenger Help Desk. Makes NO LLM calls: swapping the LLM never changes these results.

Pipeline (Module 2, Sessions 2-4):
    question ─┬─► dense: bge-small-en-v1.5 (+ query instruction) in ChromaDB (cosine) ─┐
              └─► BM25 with a code/number-aware tokenizer ─────────────────────────────┴─► RRF (k=60)
              ─► remove duplicate texts ─► cross-encoder rerank of the top 20 ─► top k

Optional extras the caller can pass (the caller may use an LLM to make them; search itself never does):
    extra_queries : rephrasings or sub-questions   (multi-query / decomposition, Session 4 Ex 4-5)
    hyde_passage  : a hypothetical answer passage  (HyDE, Session 3 Part J; embedded WITHOUT the instruction)
    filters       : hard metadata filter, e.g. {"doc_type": "railway_board_circular"}
    prefer        : soft filter; filtered results are fused WITH unfiltered ones (Session 4 challenge, fix B)

Public API:
    search(question, k=5, mode="hybrid_rerank", ...) -> [{"chunk_id","text","metadata","score","rank"}, ...]
    build_index(chunks)                               -> writes index/ (chunks.json, chroma/, manifest.json)

Expected chunk format (from ingest.py):
    {"chunk_id": str, "text": str, "embedding_text": str, "metadata": {"doc": str, "page": int|None, ...}}

CLI:
    python -m helpdesk.search "your question" [--mode dense|bm25|hybrid|hybrid_rerank] [--k 5]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
CHROMA_DIR = INDEX_DIR / "chroma"
COLLECTION = "rail_chunks"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "   # queries only, never chunks
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60           # Session 3/4: flat curve, rewards chunks that BOTH methods find
CANDIDATE_K = 30     # each retriever's list length before fusion (tuned: 20 -> 30, eval/experiments.md)
RERANK_POOL = 20     # fused candidates the cross-encoder reads (Session 4 Ex 3: ~20 is the sweet spot)
MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

# ------------------------------------------------------------------ BM25 tokenizer
STOP = set("""a an the of to in on for and or is are was were be been by with at from as it its this that
these those i my me we our you your can do does did will would should how what which when where who why if""".split())
# codes such as g.s.r, 41(e), 12:30, 10/2024, 16.01.2026, then plain words/numbers
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.:/][a-z0-9]+)+\.?|[a-z0-9]+\([a-z0-9]+\)|[a-z0-9]+")


# Both tested and left OFF (mixed results, see eval/experiments.md). Set True to try again.
DOC_EXPAND = False  # BM25 index also expands short forms inside chunks (HRS, SL, WL ...)
STEM = False        # BM25 matches word forms: book/booking, cancel/cancelled/cancellation, hour/hours
_SUFFIXES = ("ations", "ation", "ings", "ing", "ed", "s")


def stem(word):
    """Tiny suffix stripper (no extra library). Only plain words of 5+ letters; codes and numbers stay as they are.
    cancellation -> cancel, cancelled -> cancel, booking -> book, hours -> hour, charges -> charge."""
    if not STEM or len(word) < 5 or not word.isalpha() or word.endswith("ss"):
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[:-len(suf)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeious":   # cancell -> cancel
                word = word[:-1]
            break
    return word


def tokenize(text):
    """Lowercase tokens. A code stays whole AND its parts are added: 'G.S.R. 41(E)' -> g.s.r g s r 41(e) 41 e.
    The class tokenizer split '7.2' into '7' and '2', so 7.1 / 7.2 / 7.3 looked the same.
    Plain words are stemmed when STEM is True (see stem)."""
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.rstrip(".")
        if not tok or tok in STOP:
            continue
        tokens.append(stem(tok))
        parts = re.findall(r"[a-z0-9]+", tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p not in STOP)
    return tokens


# ------------------------------------------------------------------ query expansion (no LLM)
EXPAND_QUERIES = True     # add the full words for railway abbreviations, hours for days, 24h clock for am/pm
RERANK_EXPANDED = True       # the cross-encoder reads the expanded question too
RERANK_FIELD = "text"        # what the cross-encoder reads: "text" or "embedding_text" (with title/date prefix)
OLD_RULE_PENALTY = 2.0       # rerank points taken from rules marked "no longer applies", unless the question asks about the past
STALE_DOC_PENALTY = 2.0      # rerank points taken from a document much older than the newest one on the same topic
STALE_YEARS = 2
OVERVIEW_PENALTY = 3.0       # document-overview chunks only win when the question asks what a document covers
OVERVIEW_RE = re.compile(r"\b(topics?|sections?|overview|contents?|summar(y|ise|ize)|covers?|covered|list (all|the)"
                         r"|what (is|does) (the|this|that) (document|circular|notification|press release|pdf)"
                         r"|(documents?|circulars?|notifications?) (do you have|are there|available))\b", re.I)
# "before" alone is NOT the past ("cancel 4 hours before departure"); only clear references to old rules are
PAST_RE = re.compile(r"\b(before (the )?(19|20)\d\d|before the (revision|change|amendment)|old (rule|rules|charge)"
                     r"|earlier (rule|rules|charge)|previous (rule|rules|charge)|pre[- ]?revised|used to be"
                     r"|19\d\d|20(0\d|1[0-5]))\b", re.I)
# whole-word, case-insensitive; the original words are kept and the meaning is appended
SYNONYMS = [
    (r"\bsl\b", "sleeper class"), (r"\b1a(c)?\b", "AC first class"), (r"\b2a(c)?\b", "AC 2 tier"),
    (r"\b3a(c)?\b", "AC 3 tier"), (r"\b3e\b", "AC 3 economy"), (r"\bcc\b", "AC chair car"),
    (r"\bec\b", "executive chair car"), (r"\b2s\b", "second sitting"), (r"\bur\b", "unreserved"),
    (r"\b2nd\b", "second"), (r"\bcnf\b", "confirmed"), (r"\bwl\b", "waitlisted wait list"),
    (r"\brac\b", "reservation against cancellation"), (r"\be-?tkt\b|\betkt\b", "e-ticket"),
    (r"\btkt\b", "ticket"), (r"\bhrs?\b", "hours"), (r"\bdep\b|\bdept\b", "departure"),
    (r"\bvb\b", "Vande Bharat"), (r"\barp\b", "advance reservation period"),
    (r"\btdr\b", "ticket deposit receipt"), (r"\bpnr\b", "passenger name record"),
    (r"\bcharting\b", "chart preparation chart prepared"), (r"\bmoney back\b", "refund"),
    (r"\bamrit bharat (2|ii)\b", "Amrit Bharat II"), (r"\bgc\b", "gauge conversion"),
    (r"\beft\b", "excess fare ticket"), (r"\bprs\b", "passenger reservation system counter"),
]


def _days_to_hours(m):
    return f"{int(m.group(1)) * 24} hours"


def _clock(m):
    h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
    return f"{h:02d}:{m.group(2) or '00'}"


def expand_query(question):
    """'SL cancel 3 days before, train at 6 pm' -> same text + 'sleeper class 72 hours 18:00'. Rules only."""
    return _expand(question) if EXPAND_QUERIES else question


def _expand(question):
    low, extra = question.lower(), []
    for pattern, meaning in SYNONYMS:
        if re.search(pattern, low) and meaning.lower() not in low:
            extra.append(meaning)
    extra += [_days_to_hours(m) for m in re.finditer(r"\b(\d{1,2})\s*days?\b", low) if int(m.group(1)) <= 10]
    extra += [_clock(m) for m in re.finditer(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)\b", low)]
    return question if not extra else f"{question} ({'; '.join(extra)})"


# ------------------------------------------------------------------ lazy state (models + index load once)
_state = {}


def _embedder():
    if "embedder" not in _state:
        from sentence_transformers import SentenceTransformer
        _state["embedder"] = SentenceTransformer(EMBED_MODEL)
    return _state["embedder"]


def _reranker():
    if "reranker" not in _state:
        from sentence_transformers import CrossEncoder
        _state["reranker"] = CrossEncoder(RERANK_MODEL)
    return _state["reranker"]


def _chroma():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def _load():
    if "chunks" in _state:
        return
    if not CHUNKS_FILE.exists() or not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"No index in {INDEX_DIR}. Build it first: python build_index.py")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("embedding_model") != EMBED_MODEL:
        raise RuntimeError(f"index/ was built with {manifest.get('embedding_model')} but search uses {EMBED_MODEL}. "
                           "Rebuild it: python build_index.py")
    from rank_bm25 import BM25Okapi
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    _state["chunks"] = chunks
    _state["by_id"] = {c["chunk_id"]: c for c in chunks}
    _state["collection"] = _chroma().get_collection(COLLECTION)
    # DOC_EXPAND: chunks get the same short-form meanings as questions ("HRS" also counts as "hours")
    _state["bm25"] = BM25Okapi([tokenize(_expand(c["embedding_text"]) if DOC_EXPAND else c["embedding_text"])
                                for c in chunks])


def get_chunk(chunk_id):
    _load()
    return _state["by_id"].get(chunk_id)


# ------------------------------------------------------------------ building the index
def validate_chunks(chunks):
    """Fail fast (Session 2 Part A): non-empty list, required keys, doc+page, no empty text, unique ids."""
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks must be a non-empty list")
    seen, problems = set(), []
    for i, c in enumerate(chunks):
        for key in ("chunk_id", "text", "embedding_text", "metadata"):
            if key not in c:
                problems.append(f"chunk {i}: missing '{key}'")
        meta = c.get("metadata", {})
        if "doc" not in meta or "page" not in meta:
            problems.append(f"chunk {i}: metadata needs 'doc' and 'page'")
        if not str(c.get("text", "")).strip():
            problems.append(f"chunk {i}: empty text")
        if c.get("chunk_id") in seen:
            problems.append(f"duplicate chunk_id {c.get('chunk_id')}")
        seen.add(c.get("chunk_id"))
    if problems:
        raise ValueError(f"{len(problems)} chunk problems; first: {problems[:5]}")


def _chroma_metadata(meta):
    """Chroma accepts only str/int/float/bool: drop None (HTML pages), join lists (section_path)."""
    out = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = " > ".join(map(str, value))
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
    return out


def build_index(chunks, batch_size=32):
    """Embed every chunk's embedding_text and write index/ from scratch. Returns the number of records."""
    validate_chunks(chunks)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectors = _embedder().encode([c["embedding_text"] for c in chunks], normalize_embeddings=True,
                                 batch_size=batch_size, show_progress_bar=True)
    client = _chroma()
    try:
        client.delete_collection(COLLECTION)            # never leave stale chunks from an older build
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), 500):
        part = chunks[start:start + 500]
        col.upsert(ids=[c["chunk_id"] for c in part], embeddings=vectors[start:start + 500].tolist(),
                   documents=[c["text"] for c in part], metadatas=[_chroma_metadata(c["metadata"]) for c in part])
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    MANIFEST_FILE.write_text(json.dumps({"embedding_model": EMBED_MODEL, "query_instruction": QUERY_INSTRUCTION,
                                         "rerank_model": RERANK_MODEL, "chunks": len(chunks)}, indent=1), encoding="utf-8")
    _state.pop("chunks", None)                          # force a fresh load on the next search
    return col.count()


# ------------------------------------------------------------------ filters
def _where(filters):
    if not filters:
        return None
    conds = [{k: v} for k, v in filters.items()]
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _matches(meta, filters):
    return not filters or all(meta.get(k) == v for k, v in filters.items())


def _result(chunk, score):
    return {"chunk_id": chunk["chunk_id"], "text": chunk["text"], "metadata": chunk["metadata"], "score": float(score)}


# ------------------------------------------------------------------ retrievers
def search_dense(query, k=CANDIDATE_K, filters=None, add_instruction=True):
    """Meaning-based search. add_instruction=False for document-like text (HyDE passages)."""
    _load()
    vec = _embedder().encode((QUERY_INSTRUCTION if add_instruction else "") + query, normalize_embeddings=True)
    res = _state["collection"].query(query_embeddings=[vec.tolist()], n_results=min(k, len(_state["chunks"])),
                                     where=_where(filters), include=["distances"])
    return [_result(_state["by_id"][cid], 1 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]


def search_bm25(query, k=CANDIDATE_K, filters=None):
    """Keyword search: exact words, codes and numbers."""
    _load()
    scores = _state["bm25"].get_scores(tokenize(query))
    out = []
    for i in np.argsort(-scores):
        if scores[i] <= 0 or len(out) >= k:
            break
        chunk = _state["chunks"][i]
        if _matches(chunk["metadata"], filters):
            out.append(_result(chunk, scores[i]))
    return out


def rrf(result_lists, k=RRF_K):
    """Reciprocal Rank Fusion: each list gives 1/(k + rank) per chunk; points are summed; ranks only, no raw scores."""
    points, first = {}, {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            points[r["chunk_id"]] = points.get(r["chunk_id"], 0.0) + 1.0 / (k + rank)
            first.setdefault(r["chunk_id"], r)
    return [{**first[cid], "score": points[cid]} for cid in sorted(points, key=points.get, reverse=True)]


def dedupe(results):
    """Drop results whose text is the same as a better-ranked one (repeated notes, footers, overlaps)."""
    seen, out = set(), []
    for r in results:
        key = re.sub(r"\W+", " ", r["text"].lower()).strip()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _recency_penalties(question, candidates):
    """Recency safety (B4). Unless the user asks about the past:
    - a rule version marked "no longer applies" loses OLD_RULE_PENALTY points;
    - a chunk whose document is more than STALE_YEARS older than the newest document on the SAME topic
      in the pool loses STALE_DOC_PENALTY points, so a 2026 rule beats the 2015 one it replaced."""
    if PAST_RE.search(question):
        return [0.0] * len(candidates)
    newest = {}
    for c in candidates:
        m = c["metadata"]
        if m.get("topic") and m.get("doc_date"):
            newest[m["topic"]] = max(newest.get(m["topic"], ""), m["doc_date"])
    out = []
    for c in candidates:
        m, pen = c["metadata"], 0.0
        if m.get("rule_status") == "old":
            pen += OLD_RULE_PENALTY
        top = newest.get(m.get("topic"))
        if top and m.get("doc_date") and int(top[:4]) - int(m["doc_date"][:4]) > STALE_YEARS:
            pen += STALE_DOC_PENALTY
        out.append(pen)
    return out


def rerank(question, candidates, k=5):
    """Cross-encoder reads (question, chunk) together; always scored against the ORIGINAL question."""
    if not candidates:
        return []
    scores = _reranker().predict([(question, c[RERANK_FIELD] if RERANK_FIELD in c else get_chunk(c["chunk_id"])[RERANK_FIELD])
                                  for c in candidates], show_progress_bar=False)
    scores = [float(sc) - pen for sc, pen in zip(scores, _recency_penalties(question, candidates))]
    if not OVERVIEW_RE.search(question):
        scores = [sc - (OVERVIEW_PENALTY if c['metadata'].get('split_method') == 'doc_overview' else 0.0)
                  for sc, c in zip(scores, candidates)]
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [{**candidates[i], "score": float(scores[i]), "before_rank": i + 1} for i in order[:k]]


def _hybrid_lists(query, filters):
    return [search_dense(query, CANDIDATE_K, filters), search_bm25(query, CANDIDATE_K, filters)]


def search(question, k=5, mode="hybrid_rerank", filters=None, extra_queries=None, hyde_passage=None, prefer=None,
           neighbors=False):
    """Top-k chunks for a question. The question is first expanded (abbreviations, days->hours, am/pm->24h);
    in hybrid_rerank, superseded rules and much older documents lose points unless the question asks about the past.

    mode          : dense | bm25 | hybrid | hybrid_rerank (default, the best)
    filters       : hard metadata filter (only matching chunks can be returned)
    extra_queries : more phrasings / sub-questions; their hybrid lists are fused in (hybrid modes only)
    hyde_passage  : hypothetical answer text; its dense list is fused in (hybrid modes only)
    prefer        : soft metadata filter; a filtered hybrid list is fused in, unfiltered results stay possible
    neighbors     : True adds "context_text" (chunk + its neighbours in the same section) to every result
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    question = question.strip()
    if not question:
        return []
    original, question = question, expand_query(question)
    if mode == "dense":
        results = search_dense(question, k, filters)
    elif mode == "bm25":
        results = search_bm25(question, k, filters)
    else:
        lists = _hybrid_lists(question, filters)
        for q in extra_queries or []:
            if q and q.strip() and q.strip() != original:
                lists += _hybrid_lists(expand_query(q.strip()), filters)
        if hyde_passage:
            lists.append(search_dense(hyde_passage, CANDIDATE_K, filters, add_instruction=False))
        if prefer:
            lists += _hybrid_lists(question, {**(filters or {}), **prefer})
        fused = dedupe(rrf(lists))
        results = fused[:k] if mode == "hybrid" else rerank(question if RERANK_EXPANDED else original,
                                                            fused[:RERANK_POOL], k)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
        if neighbors:
            r["context_text"] = neighbor_context(r["chunk_id"])
    return results


def neighbor_context(chunk_id):
    """B5: the chunk with the chunk before and after it from the SAME document and section (via metadata.position).
    Use it when an answer may run over a chunk border; ranking and scores are not changed."""
    _load()
    if "by_pos" not in _state:
        _state["by_pos"] = {(c["metadata"]["doc"], c["metadata"].get("position")): c for c in _state["chunks"]}
    me = _state["by_id"][chunk_id]
    doc, pos = me["metadata"]["doc"], me["metadata"].get("position")
    if pos is None:
        return me["text"]
    parts = []
    for p in (pos - 1, pos, pos + 1):
        c = _state["by_pos"].get((doc, p))
        if c and c["metadata"].get("section") == me["metadata"].get("section"):
            parts.append(c["text"])
    return "\n...\n".join(parts)


# ------------------------------------------------------------------ CLI
def chat_loop(answer, first=None):
    """Ask question after question in ONE program, so the models load only once.
    answer(question) handles one question; empty line, 'exit' or Ctrl+C stops."""
    print("Loading models (once)...", flush=True)
    search("warm up", 1)
    print("Ready. Type a question (or 'exit').", flush=True)
    question = first
    while True:
        if question:
            answer(question)
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("", "exit", "quit", "q"):
            break


def _print_results(question, k, mode):
    for r in search(question, k, mode):
        m = r["metadata"]
        print(f"{r['rank']}. score {r['score']:.3f} | {m['doc']} | page {m.get('page')} | {r['chunk_id']}")
        print("   ", r["text"][:200].replace("\n", " "))


def _main():
    ap = argparse.ArgumentParser(description="Search the help desk index.")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--mode", default="hybrid_rerank", choices=MODES)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chat", action="store_true", help="keep the models loaded and ask many questions")
    args = ap.parse_args()
    if args.chat:
        chat_loop(lambda q: _print_results(q, args.k, args.mode), args.question)
    elif args.question:
        _print_results(args.question, args.k, args.mode)
    else:
        ap.print_help()


if __name__ == "__main__":
    _main()
