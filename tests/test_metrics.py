from src.metrics import metrics


def test_metrics_never_exceed_one():
    results = [{"source": "a.md"}] * 3 + [{"source": "b.md"}, {"source": "c.md"}]
    m = metrics(results, ["a.md"], k=5)
    assert all(0.0 <= v <= 1.0 for v in m.values())
    assert m["recall_at_k"] == 1.0 and m["mrr"] == 1.0 and m["hit_at_k"] == 1.0


def test_miss():
    m = metrics([{"source": "x.md"}], ["a.md"], k=5)
    assert m["hit_at_k"] == 0.0 and m["mrr"] == 0.0
