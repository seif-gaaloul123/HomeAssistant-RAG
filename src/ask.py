from .cli import check, flag, get
from .config import INDEXES
from .llm import generate_answer
from .retrieval import HybridRetriever


def main():
    check("chunking", "query", "k", "mode", "no-rerank")
    index_name = get("chunking", "heading")  # index folder: regular, semantic, heading...
    query = get("query")
    if not query:
        raise SystemExit('Usage: python -m src.ask --query "How do I configure MQTT?"')
    k = get("k", 5)
    mode = get("mode", "hybrid")
    if mode not in ("dense", "bm25", "hybrid"):
        raise SystemExit("--mode must be dense, bm25 or hybrid")
    no_rerank = flag("no-rerank")

    retriever = HybridRetriever(INDEXES / index_name, use_reranker=not no_rerank)
    docs = retriever.retrieve(query, mode=mode, k=k)

    print("\nRETRIEVED SOURCES")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d['source']} — {d['title']}\n{d['text'][:350]}...\n")

    try:
        print("\nANSWER\n" + generate_answer(query, docs))
    except RuntimeError as e:
        print(f"\n{e}")


if __name__ == "__main__":
    main()
