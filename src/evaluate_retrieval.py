import csv

import numpy as np

from .cli import check, flag, get, get_list
from .config import INDEXES, REPORTS, STRATEGIES
from .golden import check_golden, known_sources, load_golden
from .metrics import metrics
from .retrieval import HybridRetriever

METRIC_NAMES = ("hit_at_k", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k")
MODES = ("dense", "bm25", "hybrid")
POOL = 30  # chunks fetched per query, then deduped to pages before scoring


def bootstrap_ci(values, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, len(values)).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi


def main():
    check("k", "questions", "indexes", "allow-missing")
    k = get("k", 5)
    limit = get("questions", 0)  # 0 = all questions
    index_names = get_list("indexes", list(STRATEGIES))  # e.g. --indexes regular semantic heading
    allow_missing = flag("allow-missing")

    all_questions = load_golden()
    problems = check_golden(all_questions, known_sources())
    if problems and not allow_missing:
        raise SystemExit("Golden set problems (fix or use --allow-missing):\n" + "\n".join(problems[:20]))

    # unanswerable ones have no relevant pages, they're only for the refusal test
    questions = [q for q in all_questions if q.get("relevant_paths")]
    if limit:
        questions = questions[:limit]

    REPORTS.mkdir(exist_ok=True)
    rows = []
    for name in index_names:
        retriever = HybridRetriever(INDEXES / name, use_reranker=True)
        for rerank in (False, True):
            for mode in MODES:
                for q in questions:
                    results = retriever.retrieve(q["question"], mode=mode,
                                                 candidate_k=POOL, k=POOL, rerank=rerank)
                    rows.append({"index": name, "retrieval": mode, "reranker": rerank,
                                 "question": q["question"],
                                 **metrics(results, q["relevant_paths"], k)})

    csv_path = REPORTS / "retrieval_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for r in rows:
        groups.setdefault((r["index"], r["retrieval"], r["reranker"]), []).append(r)

    lines = ["# Retrieval benchmark summary", "",
             f"Answerable questions: {len(questions)} | K={k} pages (chunks deduplicated by page) | "
             "brackets = 95% bootstrap CI", "",
             "| Index | Retrieval | Reranker | Hit@K | Precision@K | Recall@K | MRR | nDCG@K |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (name, mode, rerank), vals in groups.items():
        avg = {m: float(np.mean([v[m] for v in vals])) for m in METRIC_NAMES}
        hit_lo, hit_hi = bootstrap_ci([v["hit_at_k"] for v in vals])
        mrr_lo, mrr_hi = bootstrap_ci([v["mrr"] for v in vals])
        lines.append(f"| {name} | {mode} | {rerank} | {avg['hit_at_k']:.3f} [{hit_lo:.2f}, {hit_hi:.2f}] | "
                     f"{avg['precision_at_k']:.3f} | {avg['recall_at_k']:.3f} | "
                     f"{avg['mrr']:.3f} [{mrr_lo:.2f}, {mrr_hi:.2f}] | {avg['ndcg_at_k']:.3f} |")

    (REPORTS / "retrieval_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(REPORTS / "retrieval_summary.md")


if __name__ == "__main__":
    main()
