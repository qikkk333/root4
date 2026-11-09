"""End-to-end test: run an agent on the test set and score it like the graders do.
    correct    : an LLM judge compares our answer with the reference (yes/no)   -> "30 points" part
    source hit : at least one of our sources is a gold source                 -> "10 points" part
    dont_know  : we said dont_know exactly when the reference does            -> "10 points" part

Usage:
    python eval/evaluate_agent.py [--agent rag|multi|tools] [--questions file] [--no-judge] [--out results.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from helpdesk.agent import AGENTS, LLM, make_agent  # noqa: E402

JUDGE_PROMPT = """You grade a railway help desk. Compare the ANSWER with the REFERENCE for the QUESTION.
Reply "yes" if the answer states the same key facts (numbers, times, yes/no) as the reference and nothing that
contradicts it; extra correct detail is fine. Reply "no" otherwise. If the reference says "I don't know", reply "yes"
only if the answer also declines. Reply with one word: yes or no."""


def judge(llm, q, ref, ans):
    reply = llm.text([{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user", "content": f"QUESTION: {q}\nREFERENCE: {ref}\nANSWER: {ans}"}])
    return reply.strip().lower().startswith("yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=None, choices=list(AGENTS))
    ap.add_argument("--questions", type=Path, default=ROOT / "eval" / "test_set.json")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    items = [q for q in json.loads(args.questions.read_text(encoding="utf-8")) if not q.get("sql")]
    agent = make_agent(args.agent)
    judge_llm = None if args.no_judge else LLM()
    rows, t0 = [], time.time()
    for q in items:
        out = agent.ask(q["question"])
        gold_dk = not q["sources"]
        gold = {(s["doc"], s["page"]) for s in q["sources"]}
        row = {"id": q["id"], "question": q["question"], "reference": q["answer"], **out,
               "dk_ok": out["dont_know"] == gold_dk,
               "source_hit": any((s["doc"], s["page"]) in gold for s in out["sources"]) if gold else None,
               "correct": judge(judge_llm, q["question"], q["answer"], out["answer"]) if judge_llm else None}
        rows.append(row)
        mark = {True: "OK ", False: "BAD", None: " - "}[row["correct"]]
        print(f"{mark} {q['id']} src_hit={row['source_hit']} dk_ok={row['dk_ok']} | {out['answer'][:110]}", flush=True)

    n = len(rows)
    with_src = [r for r in rows if r["source_hit"] is not None]
    print(f"\nagent={args.agent or 'default'} | {n} questions | {time.time() - t0:.0f}s")
    if judge_llm:
        print(f"correct      : {sum(r['correct'] for r in rows)}/{n}")
    print(f"source hit   : {sum(r['source_hit'] for r in with_src)}/{len(with_src)}")
    print(f"dont_know ok : {sum(r['dk_ok'] for r in rows)}/{n}")
    wrong = [r for r in rows if r["correct"] is False or r["source_hit"] is False or not r["dk_ok"]]
    if wrong:
        print("\nto inspect:")
        for r in wrong:
            print(f"- {r['id']} {r['question']}\n    ours: {r['answer']}\n    ref : {r['reference']}\n    src : {r['sources']}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
