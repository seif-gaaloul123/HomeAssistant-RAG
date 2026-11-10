import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import EMBEDDING_MODEL, RERANKER_MODEL


def tokens(text):
    words = re.findall(r"[a-z0-9_]+", text.lower())
    # mqtt_sensor should also match "mqtt" and "sensor"
    parts = [p for w in words if "_" in w for p in w.split("_") if p]
    return words + parts


class HybridRetriever:
    def __init__(self, index_dir, use_reranker=True, embedder=None):
        index_dir = Path(index_dir)
        self.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_dir / "dense.faiss"))
        self.embedder = embedder or SentenceTransformer(EMBEDDING_MODEL)
        # dense, bm25 and reranker all see the same text (title + section + body)
        self.texts = [c.get("embed_text", c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi([tokens(t) for t in self.texts])
        self.use_reranker = use_reranker
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    def dense(self, query, n=30):
        vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(vec, min(n, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def lexical(self, query, n=30):
        scores = self.bm25.get_scores(tokens(query))
        top = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def retrieve(self, query, mode="hybrid", candidate_k=30, k=5, rerank=None):
        if rerank is None:
            rerank = self.use_reranker

        dense = self.dense(query, candidate_k) if mode in ("dense", "hybrid") else []
        lexical = self.lexical(query, candidate_k) if mode in ("bm25", "hybrid") else []

        if mode == "dense":
            ranked = [i for i, _ in dense]
        elif mode == "bm25":
            ranked = [i for i, _ in lexical]
        else:
            # reciprocal rank fusion, constant 60
            fused = {}
            for ranking in (dense, lexical):
                for pos, (idx, _) in enumerate(ranking, start=1):
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + pos)
            ranked = [i for i, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

        ranked = ranked[:candidate_k]
        if rerank and ranked:
            scores = self.reranker.predict([(query, self.texts[i]) for i in ranked])
            ranked = [i for _, i in sorted(zip(scores, ranked), key=lambda x: x[0], reverse=True)]
        return [self.chunks[i] for i in ranked[:k]]


def save_index(chunks, out_dir, model=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = model or SentenceTransformer(EMBEDDING_MODEL)

    texts = [c.get("embed_text", c["text"]) for c in chunks]
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, str(out / "dense.faiss"))
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
