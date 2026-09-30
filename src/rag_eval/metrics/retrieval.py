"""Retrieval metrics computed against labelled relevant documents.

These are exact, deterministic computations (not heuristics); their validity
depends entirely on the quality of the relevance labels.

Conventions:
* Retrieved lists are de-duplicated by document id, keeping the first (best
  ranked) occurrence, before the cut-off ``k`` is applied.
* A document is relevant when its grade is > 0. Only NDCG uses grade values.
* If a sample has no relevant-document labels every metric returns
  ``score=None`` (not applicable) instead of a misleading 0 or 1.
* Precision@K divides by ``k`` even when fewer than ``k`` documents were
  returned, so returning less context is not rewarded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..types import MetricResult


def dedupe(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def _na(metric: str, method: str) -> MetricResult:
    return MetricResult(metric, None, method, {"reason": "no relevant_docs labels for this sample"})


def _prep(
    retrieved: Sequence[str], relevant: Mapping[str, int], k: int
) -> tuple[list[str], set[str], list[str]]:
    _check_k(k)
    unique = dedupe(retrieved)
    top = unique[:k]
    rel = {d for d, g in relevant.items() if g > 0}
    hits = [d for d in top if d in rel]
    return top, rel, hits


def precision_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> MetricResult:
    name, method = f"precision@{k}", "hits in top-k / k (binary relevance)"
    top, rel, hits = _prep(retrieved, relevant, k)
    if not rel:
        return _na(name, method)
    return MetricResult(
        name,
        len(hits) / k,
        method,
        {
            "top_k": top,
            "relevant_hits": hits,
            "num_relevant": len(rel),
            "duplicates_removed": len(retrieved) - len(dedupe(retrieved)),
        },
    )


def recall_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> MetricResult:
    name, method = f"recall@{k}", "relevant docs in top-k / total relevant docs"
    top, rel, hits = _prep(retrieved, relevant, k)
    if not rel:
        return _na(name, method)
    return MetricResult(
        name,
        len(hits) / len(rel),
        method,
        {
            "top_k": top,
            "relevant_hits": hits,
            "missed": sorted(rel - set(hits)),
            "num_relevant": len(rel),
        },
    )


def hit_rate_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> MetricResult:
    name, method = f"hit_rate@{k}", "1 if any relevant doc in top-k else 0"
    top, rel, hits = _prep(retrieved, relevant, k)
    if not rel:
        return _na(name, method)
    return MetricResult(name, 1.0 if hits else 0.0, method, {"top_k": top, "relevant_hits": hits})


def mrr_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> MetricResult:
    name, method = f"mrr@{k}", "1 / rank of first relevant doc within top-k (0 if none)"
    top, rel, _ = _prep(retrieved, relevant, k)
    if not rel:
        return _na(name, method)
    for rank, doc in enumerate(top, start=1):
        if doc in rel:
            return MetricResult(
                name, 1.0 / rank, method, {"first_relevant": doc, "rank": rank, "top_k": top}
            )
    return MetricResult(name, 0.0, method, {"first_relevant": None, "rank": None, "top_k": top})


def ndcg_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> MetricResult:
    name = f"ndcg@{k}"
    method = "DCG/IDCG with linear gain = grade, log2(rank+1) discount"
    _check_k(k)
    grades = {d: g for d, g in relevant.items() if g > 0}
    if not grades:
        return _na(name, method)
    top = dedupe(retrieved)[:k]
    dcg = sum(grades.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    score = dcg / idcg if idcg > 0 else 0.0
    return MetricResult(
        name, min(1.0, score), method, {"dcg": round(dcg, 6), "idcg": round(idcg, 6), "top_k": top}
    )


RETRIEVAL_METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "hit_rate": hit_rate_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def compute_retrieval_metrics(
    retrieved: Sequence[str],
    relevant: Mapping[str, int],
    k_values: Sequence[int],
    names: Sequence[str] | None = None,
) -> list[MetricResult]:
    """Compute the selected retrieval metrics for every k in ``k_values``."""
    chosen = list(names) if names is not None else list(RETRIEVAL_METRICS)
    unknown = [n for n in chosen if n not in RETRIEVAL_METRICS]
    if unknown:
        raise ValueError(f"unknown retrieval metrics: {unknown}")
    return [RETRIEVAL_METRICS[n](retrieved, relevant, k) for n in chosen for k in k_values]
