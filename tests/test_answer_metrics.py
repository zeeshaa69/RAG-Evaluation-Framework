from __future__ import annotations

import pytest

from rag_eval.metrics.answer import (
    answer_relevance,
    citation_coverage,
    compute_answer_metrics,
    contains_expected,
    context_relevance,
    exact_match,
    semantic_similarity,
    token_overlap,
)
from rag_eval.metrics.embeddings import CharNgramEmbedder, cosine
from tests.conftest import rd


def test_exact_match_normalizes() -> None:
    assert exact_match("The Eiffel Tower.", "eiffel tower").score == 1.0
    assert exact_match("Paris [d1]", "Paris").score == 1.0  # citations are stripped
    assert exact_match("Lyon", "Paris").score == 0.0


@pytest.mark.parametrize("expected", [None, ""])
def test_reference_metrics_not_applicable_without_expected(expected: str | None) -> None:
    for fn in (exact_match, token_overlap, contains_expected, semantic_similarity):
        r = fn("anything", expected)
        assert r.score is None
        assert "no expected_answer" in r.evidence["reason"]


def test_token_f1_values() -> None:
    r = token_overlap("the quick brown fox", "quick brown dog")
    # normalized: [quick brown fox] vs [quick brown dog] -> P=R=2/3
    assert r.score == pytest.approx(2 / 3)
    assert r.evidence["precision"] == pytest.approx(0.6667, abs=1e-4)


def test_token_f1_multiset_semantics() -> None:
    # "a a a" vs "a": common=1, p=1/3, r=1 -> f1=0.5 ("a" is an article; use real words)
    r = token_overlap("go go go", "go")
    assert r.score == pytest.approx(0.5)


def test_token_f1_disjoint_and_empty() -> None:
    assert token_overlap("alpha", "beta").score == 0.0
    assert token_overlap("", "beta").score == 0.0
    assert token_overlap("the", "a").score == 1.0  # both normalize to empty


def test_contains_expected() -> None:
    assert contains_expected("It is 8,849 meters tall [d1]", "8,849 meters").score == 1.0
    assert contains_expected("Mount Everest is tall", "K2").score == 0.0
    assert contains_expected("Paris is large", "Par").score == 0.0  # whole-token match only


def test_contains_expected_is_marked_heuristic() -> None:
    assert contains_expected("x", "x").heuristic


def test_semantic_similarity_identical_and_unrelated() -> None:
    assert semantic_similarity("Paris is nice", "Paris is nice").score == pytest.approx(1.0)
    low = semantic_similarity("zzzz qqqq", "Paris is nice").score
    assert low is not None and low < 0.2
    assert semantic_similarity("x", "y").heuristic  # default embedder is lexical


def test_semantic_similarity_custom_embedder() -> None:
    class Const:
        name = "const-neural-stub"

        def embed(self, text: str) -> dict[int, float]:
            return {0: 1.0}

    r = semantic_similarity("a", "b", Const())
    assert r.score == 1.0
    assert not r.heuristic


def test_cosine_edge_cases() -> None:
    assert cosine({}, {1: 1.0}) == 0.0
    assert cosine({1: 0.0}, {1: 1.0}) == 0.0
    assert CharNgramEmbedder().embed("") == {}


def test_embedder_is_deterministic() -> None:
    e = CharNgramEmbedder()
    assert e.embed("hello world") == e.embed("hello world")


def test_context_relevance_no_context_is_zero() -> None:
    r = context_relevance("What is the capital of France?", [])
    assert r.score == 0.0 and r.heuristic


def test_context_relevance_scores_per_chunk() -> None:
    ctx = [rd("d1", "Paris is the capital of France."), rd("d2", "Bananas are yellow.")]
    r = context_relevance("What is the capital of France?", ctx)
    cov = {c["doc_id"]: c["coverage"] for c in r.evidence["per_chunk"]}
    assert cov["d1"] == 1.0 and cov["d2"] == 0.0
    assert r.score == pytest.approx(0.5)


def test_context_relevance_question_without_content_terms() -> None:
    assert context_relevance("what is it?", [rd("d1", "x")]).score is None


def test_answer_relevance() -> None:
    hi = answer_relevance("What is the capital of France?", "The capital of France is Paris.")
    lo = answer_relevance("What is the capital of France?", "Bananas contain potassium.")
    assert hi.score is not None and lo.score is not None and hi.score > lo.score


def test_answer_relevance_empty_and_abstention() -> None:
    assert answer_relevance("q?", "").score == 0.0
    assert answer_relevance("q?", "I don't know.").score is None


def test_citation_coverage_partial_and_invalid() -> None:
    ans = "Paris is the capital [d1]. It is large. Tokyo is big [zzz]."
    r = citation_coverage(ans, ["d1", "d2"], {"d1": 1, "d2": 1})
    assert r.score == pytest.approx(1 / 3)
    assert r.evidence["invalid_citations"] == ["zzz"]
    assert r.evidence["cited_relevant"] == ["d1"]


def test_citation_coverage_full() -> None:
    assert citation_coverage("A is b. [d1] C is d. [d2]", ["d1", "d2"]).score == 1.0


def test_citation_coverage_abstention_empty_and_no_context() -> None:
    assert citation_coverage("I don't know.", ["d1"]).score is None
    assert citation_coverage("", ["d1"]).score == 0.0
    assert citation_coverage("Claim [d1].", []).score == 0.0  # cited doc was not retrieved


def test_compute_answer_metrics_subset_and_unknown() -> None:
    res = compute_answer_metrics("q", "a", "a", [], {}, ["exact_match"])
    assert [r.metric for r in res] == ["exact_match"]
    with pytest.raises(ValueError):
        compute_answer_metrics("q", "a", "a", [], {}, ["nope"])


def test_long_context_relevance_is_fast_and_bounded() -> None:
    big = " ".join(f"word{i}" for i in range(50_000)) + " capital France"
    r = context_relevance("capital of France", [rd("big", big)])
    assert r.score == 1.0
