from __future__ import annotations

import math

import pytest

from rag_eval.metrics.retrieval import (
    compute_retrieval_metrics,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag_eval.types import MetricResult

REL = {"b": 1, "d": 1}
RET = ["a", "b", "c", "d"]


def score(result: MetricResult) -> float:
    assert result.score is not None
    return result.score


def test_precision() -> None:
    assert score(precision_at_k(RET, REL, 1)) == 0.0
    assert score(precision_at_k(RET, REL, 2)) == 0.5
    assert score(precision_at_k(RET, REL, 4)) == 0.5


def test_recall() -> None:
    assert score(recall_at_k(RET, REL, 2)) == 0.5
    assert score(recall_at_k(RET, REL, 4)) == 1.0
    assert recall_at_k(RET, REL, 2).evidence["missed"] == ["d"]


def test_hit_rate() -> None:
    assert score(hit_rate_at_k(RET, REL, 1)) == 0.0
    assert score(hit_rate_at_k(RET, REL, 2)) == 1.0


def test_mrr() -> None:
    assert score(mrr_at_k(RET, REL, 4)) == 0.5
    assert score(mrr_at_k(RET, REL, 1)) == 0.0  # first relevant is outside top-1
    assert score(mrr_at_k(["b"], REL, 3)) == 1.0
    assert mrr_at_k(RET, REL, 4).evidence["rank"] == 2


def test_ndcg_binary_hand_computed() -> None:
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert score(ndcg_at_k(RET, REL, 4)) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert score(ndcg_at_k(["b", "d", "x"], REL, 3)) == pytest.approx(1.0)


def test_ndcg_graded_relevance() -> None:
    rel = {"a": 3, "b": 1}
    dcg = 1 / math.log2(2) + 3 / math.log2(3)  # ranked [b, a]
    idcg = 3 + 1 / math.log2(3)
    assert score(ndcg_at_k(["b", "a"], rel, 2)) == pytest.approx(dcg / idcg)


def test_ndcg_respects_cutoff() -> None:
    # Only rank 1 is inside k=1; the relevant doc at rank 2 must not count.
    assert score(ndcg_at_k(["x", "b"], {"b": 1}, 1)) == 0.0


# ---- edge cases -----------------------------------------------------------------------


@pytest.mark.parametrize("fn", [precision_at_k, recall_at_k, hit_rate_at_k, mrr_at_k, ndcg_at_k])
def test_empty_retrieval_scores_zero(fn: object) -> None:
    r = fn([], {"a": 1}, 5)  # type: ignore[operator]
    assert r.score == 0.0


@pytest.mark.parametrize("fn", [precision_at_k, recall_at_k, hit_rate_at_k, mrr_at_k, ndcg_at_k])
def test_no_labels_is_not_applicable(fn: object) -> None:
    r = fn(["a", "b"], {}, 3)  # type: ignore[operator]
    assert r.score is None
    assert "no relevant_docs" in r.evidence["reason"]


@pytest.mark.parametrize("fn", [precision_at_k, recall_at_k, hit_rate_at_k, mrr_at_k, ndcg_at_k])
def test_zero_grade_counts_as_unlabelled(fn: object) -> None:
    assert fn(["a"], {"a": 0}, 3).score is None  # type: ignore[operator]


@pytest.mark.parametrize("fn", [precision_at_k, recall_at_k, hit_rate_at_k, mrr_at_k, ndcg_at_k])
@pytest.mark.parametrize("bad_k", [0, -1, 2.5, True])
def test_invalid_k_raises(fn: object, bad_k: object) -> None:
    with pytest.raises(ValueError):
        fn(["a"], {"a": 1}, bad_k)  # type: ignore[operator]


def test_duplicates_are_collapsed_before_cutoff() -> None:
    r = precision_at_k(["a", "a", "b"], {"a": 1}, 3)
    assert r.score == pytest.approx(1 / 3)
    assert r.evidence["duplicates_removed"] == 1
    # Duplicates must not let the same doc fill the top-k twice.
    assert score(recall_at_k(["a", "a", "a"], {"a": 1, "b": 1}, 3)) == 0.5
    assert score(ndcg_at_k(["a", "a", "a"], {"a": 1}, 3)) == pytest.approx(1.0)
    # After de-duplication "b" is at rank 2, so it is inside k=2.
    assert score(hit_rate_at_k(["a", "a", "b"], {"b": 1}, 2)) == 1.0


def test_multiple_correct_documents() -> None:
    rel = {"a": 1, "b": 1, "c": 1}
    assert score(precision_at_k(["a", "b"], rel, 2)) == 1.0
    assert score(recall_at_k(["a", "b"], rel, 2)) == pytest.approx(2 / 3)
    assert score(mrr_at_k(["x", "c", "a"], rel, 3)) == 0.5


def test_k_larger_than_retrieved_does_not_reward_short_lists() -> None:
    assert score(precision_at_k(["a"], {"a": 1}, 10)) == pytest.approx(0.1)
    assert score(recall_at_k(["a"], {"a": 1}, 10)) == 1.0


def test_unrelated_ids_do_not_crash() -> None:
    assert score(recall_at_k(["zzz"], {"a": 1}, 3)) == 0.0


def test_long_retrieval_list() -> None:
    ids = [f"d{i}" for i in range(10_000)]
    assert score(recall_at_k(ids, {"d9999": 1}, 10_000)) == 1.0
    assert score(recall_at_k(ids, {"d9999": 1}, 100)) == 0.0


def test_ndcg_never_exceeds_one() -> None:
    for ret in (["a", "b"], ["b", "a"], ["a", "a", "b"]):
        assert score(ndcg_at_k(ret, {"a": 2, "b": 1}, 3)) <= 1.0


def test_compute_selected_metrics() -> None:
    res = compute_retrieval_metrics(RET, REL, [1, 3], ["precision", "mrr"])
    assert [r.metric for r in res] == ["precision@1", "precision@3", "mrr@1", "mrr@3"]
    with pytest.raises(ValueError):
        compute_retrieval_metrics(RET, REL, [1], ["bogus"])
