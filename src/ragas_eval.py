import json
import os

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from .cli import check, get
from .config import EMBEDDING_MODEL, EVAL, GROQ_MODEL, REPORTS

COLUMNS = ("user_input", "response", "retrieved_contexts", "reference")


def main():
    check("input", "limit")
    input_path = get("input", str(EVAL / "eval_answers.jsonl"))
    limit = get("limit", 0)

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY before running RAGAS.")

    rows = [json.loads(x) for x in open(input_path, encoding="utf-8") if x.strip()]
    rows = [r for r in rows if r.get("answerable", True)]  # refusal questions have no real reference
    rows = [{c: r[c] for c in COLUMNS} for r in rows]
    if limit:
        rows = rows[:limit]

    REPORTS.mkdir(exist_ok=True)  # before the long run, not after

    # groq only does n=1 and answer_relevancy asks for several generations at once
    # (not verified on ragas 0.2.15, try --limit 2 first)
    answer_relevancy.strictness = 1

    # max_tokens: reasoning judges ran out of tokens and gave nan on faithfulness
    judge = LangchainLLMWrapper(ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=4096))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, timeout=120),
    )
    print(result)
    result.to_pandas().to_csv(REPORTS / "ragas_scores.csv", index=False)


if __name__ == "__main__":
    main()
