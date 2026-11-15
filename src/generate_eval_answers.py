import json
import os

from .cli import check, get
from .config import EVAL, INDEXES
from .golden import load_golden
from .llm import generate_answer, is_refusal
from .retrieval import HybridRetriever


def main():
    check("chunking", "limit", "refusal-limit")
    index_name = get("chunking", "heading")
    limit = get("limit", 10)  # answerable questions to run
    refusal_limit = get("refusal-limit", 15)  # unanswerable questions to run

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY first.")

    golden = load_golden()
    answerable = [q for q in golden if q.get("relevant_paths")][:limit]
    unanswerable = [q for q in golden if not q.get("relevant_paths")][:refusal_limit]

    target = EVAL / "eval_answers.jsonl"
    done = set()
    if target.exists():  # resume where we stopped
        lines = target.read_text(encoding="utf-8").splitlines()
        done = {json.loads(l)["user_input"] for l in lines if l.strip()}

    retriever = HybridRetriever(INDEXES / index_name)
    with target.open("a", encoding="utf-8") as f:
        for item in answerable + unanswerable:
            if item["question"] in done:
                continue
            contexts = retriever.retrieve(item["question"], mode="hybrid", k=5)
            try:
                answer = generate_answer(item["question"], contexts)
            except RuntimeError as e:
                print(f"Stopped: {e}\nRe-run the same command to resume.")
                break
            record = {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": [c["text"] for c in contexts],
                "reference": item["reference"],
                "answerable": bool(item.get("relevant_paths")),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    rows = [json.loads(l) for l in target.read_text(encoding="utf-8").splitlines() if l.strip()]
    refusal_rows = [r for r in rows if not r["answerable"]]
    if refusal_rows:
        rate = sum(is_refusal(r["response"]) for r in refusal_rows) / len(refusal_rows)
        print(f"Refusal rate on {len(refusal_rows)} unanswerable questions: {rate:.0%} "
              "(exact-phrase match, check misses by hand)")
    print(f"{len(rows)} records in {target}")


if __name__ == "__main__":
    main()
