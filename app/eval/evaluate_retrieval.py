"""Retrieval metrics (Session 3 Part L / RAGAS Part 2): Hit@K, Recall@K, Precision@K, MRR per search mode.
No LLM, no API key: the numbers depend only on the index and search settings.

A retrieved result counts as relevant when its (doc, page) is one of the question's gold sources.
Several chunks from the same page count once (we rank unique pages).

Usage:
    python eval/evaluate_retrieval.py                      # eval/test_set.json if it exists, else the sample questions
    python eval/evaluate_retrieval.py --questions file.json --k 5 --verbose
Question file: a list of {"id", "question", "sources": [{"doc", "page"}]}; questions without sources are skipped.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from helpdesk.search import MODES, search  # noqa: E402


def default_questions():
    for p in [ROOT / "eval" / "test_set.json", ROOT / "data" / "questions" / "sample_questions.json",
              ROOT / "rail_capstone_pack" / "questions" / "sample_questions.json"]:
        if p.exists():
            return p
    sys.exit("No question file found.")


def unique_pages(results):
    pages = []
    for r in results:
        key = (r["metadata"]["doc"], r["metadata"].get("page"))
        if key not in pages:
            pages.append(key)
    return pages


def metrics(retrieved, relevant, k):
    top = retrieved[:k]
    found = [p for p in top if p in relevant]
    first = next((i for i, p in enumerate(top, 1) if p in relevant), None)
    return {"hit": 1.0 if found else 0.0, "recall": len(set(found)) / len(relevant),
            "precision": len(found) / k, "rr": 1 / first if first else 0.0, "first": first}


def answer_numbers(text):
    """Numbers/times in a reference answer: '00:20 to 23:45' -> {'00:20', '23:45'}; '25%' -> {'25'}."""
    return set(re.findall(r"\d+(?::\d+)?", text.replace("00.", "00:")))


def chunk_level(results, answer, k):
    """Chunk-level check: share of the reference answer's numbers found in the top-k CHUNK texts
    (what the agent really reads). Page-level scores can hide a missing chunk; this cannot. None if no numbers."""
    nums = answer_numbers(answer)
    if not nums:
        return None
    text = " ".join(r["text"] for r in results[:k]).replace(".", ":")
    have = set(re.findall(r"\d+(?::\d+)?", text))
    return len(nums & have) / len(nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    path = args.questions or default_questions()
    items = [q for q in json.loads(path.read_text(encoding="utf-8")) if q.get("sources")]
    print(f"{len(items)} questions with gold sources from {path.name}, K={args.k}\n")
    search("warm up", 1)                                   # load models once, so timings are fair

    print(f"{'mode':14} | Hit@{args.k} | Recall@{args.k} | Prec@{args.k} | MRR  | Ans@{args.k} | ms/query")
    print("-" * 73)
    per_mode = {}
    for mode in args.modes:
        rows, t0 = [], time.time()
        for q in items:
            relevant = {(s["doc"], s["page"]) for s in q["sources"]}
            # fetch extra chunks so that K unique pages are available
            results = search(q["question"], k=args.k * 3, mode=mode)
            m = metrics(unique_pages(results), relevant, args.k)
            m["ans"] = chunk_level(results, q.get("answer", ""), args.k)
            rows.append((q, m))
        ms = (time.time() - t0) * 1000 / len(items)
        n = len(rows)
        avg = {m: sum(r[m] for _, r in rows) / n for m in ("hit", "recall", "precision", "rr")}
        ans = [r["ans"] for _, r in rows if r["ans"] is not None]
        avg["ans"] = sum(ans) / len(ans) if ans else 0.0
        per_mode[mode] = rows
        print(f"{mode:14} | {avg['hit']:.2f}  | {avg['recall']:.2f}     | {avg['precision']:.2f}   | "
              f"{avg['rr']:.2f} | {avg['ans']:.2f}  | {ms:.0f}")

    if args.verbose:
        print("\nfirst relevant rank per question (- = not in top K):")
        print("id    | " + " | ".join(f"{m[:13]:13}" for m in args.modes) + " | question")
        for i, q in enumerate(items):
            ranks = [per_mode[m][i][1]["first"] for m in args.modes]
            low = [f"{m[:6]}:{per_mode[m][i][1]['ans']:.0%}" for m in args.modes
                   if per_mode[m][i][1]["ans"] is not None and per_mode[m][i][1]["ans"] < 1]
            print(f"{q.get('id', i):5} | " + " | ".join(f"{str(r or '-'):13}" for r in ranks)
                  + f" | {q['question'][:60]}" + (f"   Ans<100%: {' '.join(low)}" if low else ""))
    print(f"\nAns@{args.k}: share of the reference answer's numbers found in the top-{args.k} CHUNKS "
          f"(questions whose answer has numbers).")


if __name__ == "__main__":
    main()
