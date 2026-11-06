import json

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .chunking import heading_chunks, regular_chunks, semantic_chunks
from .cli import check, flag, get
from .config import EMBEDDING_MODEL, INDEXES, PROCESSED, STRATEGIES
from .retrieval import save_index


def load_docs():
    path = PROCESSED / "documents.jsonl"
    if not path.exists():
        raise SystemExit("Run ingestion first: python -m src.ingest --repo PATH")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    check("chunking", "no-header")
    chunking = get("chunking", "all")
    if chunking not in (*STRATEGIES, "all"):
        raise SystemExit(f"--chunking must be one of: {', '.join([*STRATEGIES, 'all'])}")
    no_header = flag("no-header")  # ablation: embed the chunk text without the title/section header

    docs = load_docs()
    model = SentenceTransformer(EMBEDDING_MODEL)
    todo = list(STRATEGIES) if chunking == "all" else [chunking]

    for strategy in todo:
        chunks = []
        for doc in tqdm(docs, desc=f"{strategy} chunking"):
            if strategy == "regular":
                chunks += regular_chunks(doc)
            elif strategy == "semantic":
                chunks += semantic_chunks(doc, model)
            else:
                chunks += heading_chunks(doc)

        if no_header:
            for c in chunks:
                c["embed_text"] = c["text"]

        name = strategy + ("_nohdr" if no_header else "")
        save_index(chunks, INDEXES / name, model)

        sizes = np.array([len(c["text"]) for c in chunks])
        print(f"{name}: {len(chunks)} chunks | mean {sizes.mean():.0f} chars, "
              f"median {np.median(sizes):.0f}, max {sizes.max()} -> {INDEXES / name}")


if __name__ == "__main__":
    main()
