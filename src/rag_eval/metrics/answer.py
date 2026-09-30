"""Answer-quality metrics.

Reference-based metrics (exact match, token overlap, semantic similarity) return
``score=None`` when the sample has no expected answer. Reference-free metrics
(context relevance, answer relevance, citation coverage) are lexical heuristics
and are flagged ``heuristic=True``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..text import (
    content_tokens,
    extract_citations,
    is_abstention,
    normalize_answer,
    split_sentences,
    strip_citations,
)
from ..types import MetricResult, RetrievedDoc
from .embeddings import CharNgramEmbedder, Embedder, cosine

_DEFAULT_EMBEDDER = CharNgramEmbedder()


def _no_reference(metric: str, method: str) -> MetricResult:
    return MetricResult(metric, None, method, {"reason": "sample has no expected_answer"})


def exact_match(answer: str, expected: str | None) -> MetricResult:
    method = "equality after SQuAD normalization (lowercase, strip punctuation/articles)"
    if not expected:
        return _no_reference("exact_match", method)
    a, e = normalize_answer(strip_citations(answer)), normalize_answer(expected)
    return MetricResult(
        "exact_match",
        1.0 if a == e else 0.0,
        method,
        {"normalized_answer": a, "normalized_expected": e},
    )


def token_overlap(answer: str, expected: str | None) -> MetricResult:
    """Token-level F1 between answer and expected answer (multiset overlap)."""
    method = "token F1 on normalized whitespace tokens (multiset overlap)"
    if not expected:
        return _no_reference("token_f1", method)
    a = normalize_answer(strip_citations(answer)).split()
    e = normalize_answer(expected).split()
    common = Counter(a) & Counter(e)
    n_common = sum(common.values())
    if not a or not e:
        f1 = 1.0 if a == e else 0.0
        return MetricResult(
            "token_f1", f1, method, {"precision": f1, "recall": f1, "common_tokens": []}
        )
    if n_common == 0:
        return MetricResult(
            "token_f1", 0.0, method, {"precision": 0.0, "recall": 0.0, "common_tokens": []}
        )
    p, r = n_common / len(a), n_common / len(e)
    return MetricResult(
        "token_f1",
        2 * p * r / (p + r),
        method,
        {"precision": round(p, 4), "recall": round(r, 4), "common_tokens": sorted(common)},
    )


def contains_expected(answer: str, expected: str | None) -> MetricResult:
    """1 if the normalized expected answer occurs as a whole-token span in the answer.

    Useful for verbose generators where exact match is too strict. A substring hit
    does not prove the answer *asserts* the expected text (e.g. "not 1991").
    """
    method = "normalized expected answer appears as a token span inside the normalized answer"
    if not expected:
        return _no_reference("contains_expected", method)
    a = " " + normalize_answer(strip_citations(answer)) + " "
    e = normalize_answer(expected)
    hit = bool(e) and f" {e} " in a
    return MetricResult(
        "contains_expected",
        1.0 if hit else 0.0,
        method,
        {"normalized_expected": e},
        heuristic=True,
        note="Containment can be satisfied by negated or hedged statements.",
    )


def semantic_similarity(
    answer: str, expected: str | None, embedder: Embedder | None = None
) -> MetricResult:
    emb = embedder or _DEFAULT_EMBEDDER
    method = f"cosine similarity of embeddings [{emb.name}]"
    if not expected:
        return _no_reference("semantic_similarity", method)
    score = cosine(emb.embed(strip_citations(answer)), emb.embed(expected))
    return MetricResult(
        "semantic_similarity",
        score,
        method,
        {"embedder": emb.name},
        heuristic=emb.name.startswith("char-"),
        note=(
            "Lexical character n-gram proxy; does not capture meaning beyond surface form."
            if emb.name.startswith("char-")
            else ""
        ),
    )


def context_relevance(question: str, contexts: Sequence[RetrievedDoc]) -> MetricResult:
    """Mean share of question content terms covered by each retrieved chunk."""
    method = "mean per-chunk coverage of question content terms (label-free)"
    q_terms = set(content_tokens(question))
    if not contexts:
        return MetricResult(
            "context_relevance",
            0.0,
            method,
            {"per_chunk": [], "reason": "no context retrieved"},
            True,
            "Empty retrieval scores 0.",
        )
    if not q_terms:
        return MetricResult(
            "context_relevance", None, method, {"reason": "question has no content terms"}, True
        )
    per_chunk = []
    for c in contexts:
        terms = set(content_tokens(c.text))
        matched = sorted(q_terms & terms)
        per_chunk.append(
            {
                "doc_id": c.doc_id,
                "coverage": round(len(matched) / len(q_terms), 4),
                "matched_terms": matched,
            }
        )
    score = sum(p["coverage"] for p in per_chunk) / len(per_chunk)  # type: ignore[misc]
    return MetricResult(
        "context_relevance",
        score,
        method,
        {"question_terms": sorted(q_terms), "per_chunk": per_chunk},
        heuristic=True,
        note="Term overlap, not judged relevance; synonyms are missed.",
    )


def answer_relevance(question: str, answer: str, embedder: Embedder | None = None) -> MetricResult:
    emb = embedder or _DEFAULT_EMBEDDER
    method = f"cosine similarity of question and answer embeddings [{emb.name}]"
    body = strip_citations(answer).strip()
    if not body:
        return MetricResult("answer_relevance", 0.0, method, {"reason": "empty answer"}, True)
    if is_abstention(body):
        return MetricResult(
            "answer_relevance",
            None,
            method,
            {"reason": "answer is an abstention"},
            True,
            "Abstentions are not scored for relevance.",
        )
    q_terms, a_terms = set(content_tokens(question)), set(content_tokens(body))
    return MetricResult(
        "answer_relevance",
        cosine(emb.embed(question), emb.embed(body)),
        method,
        {"shared_terms": sorted(q_terms & a_terms)},
        heuristic=True,
        note="Measures topical similarity to the question, not correctness.",
    )


def citation_coverage(
    answer: str,
    retrieved_ids: Sequence[str],
    relevant: dict[str, int] | None = None,
) -> MetricResult:
    """Share of answer sentences carrying at least one valid ``[doc_id]`` citation.

    A citation is *valid* if it names a document that was actually retrieved.
    When relevance labels exist, ``cited_relevant`` is reported as extra evidence.
    """
    method = "fraction of answer sentences with >=1 citation to a retrieved doc"
    if is_abstention(answer) and not extract_citations(answer):
        return MetricResult(
            "citation_coverage", None, method, {"reason": "answer is an abstention"}, True
        )
    sentences = split_sentences(answer)
    if not sentences:
        return MetricResult("citation_coverage", 0.0, method, {"reason": "empty answer"}, True)
    retrieved = set(retrieved_ids)
    rows: list[dict[str, Any]] = []
    covered = 0
    invalid: list[str] = []
    for s in sentences:
        cites = extract_citations(s)
        valid = [c for c in cites if c in retrieved]
        invalid.extend(c for c in cites if c not in retrieved)
        if valid:
            covered += 1
        rows.append({"sentence": s, "citations": cites, "valid": valid})
    evidence: dict[str, object] = {"sentences": rows, "invalid_citations": sorted(set(invalid))}
    if relevant:
        cited = {c for r in rows for c in r["valid"]}
        evidence["cited_relevant"] = sorted(cited & {d for d, g in relevant.items() if g > 0})
    return MetricResult(
        "citation_coverage",
        covered / len(sentences),
        method,
        evidence,
        heuristic=True,
        note="Checks citation presence/validity only, "
        "not whether the cited text supports the claim.",
    )


def compute_answer_metrics(
    question: str,
    answer: str,
    expected: str | None,
    contexts: Sequence[RetrievedDoc],
    relevant: dict[str, int],
    names: Sequence[str] | None = None,
    embedder: Embedder | None = None,
) -> list[MetricResult]:
    all_names = [
        "exact_match",
        "contains_expected",
        "token_f1",
        "semantic_similarity",
        "context_relevance",
        "answer_relevance",
        "citation_coverage",
    ]
    chosen = list(names) if names is not None else all_names
    unknown = [n for n in chosen if n not in all_names]
    if unknown:
        raise ValueError(f"unknown answer metrics: {unknown}")
    out: list[MetricResult] = []
    for n in chosen:
        if n == "exact_match":
            out.append(exact_match(answer, expected))
        elif n == "contains_expected":
            out.append(contains_expected(answer, expected))
        elif n == "token_f1":
            out.append(token_overlap(answer, expected))
        elif n == "semantic_similarity":
            out.append(semantic_similarity(answer, expected, embedder))
        elif n == "context_relevance":
            out.append(context_relevance(question, contexts))
        elif n == "answer_relevance":
            out.append(answer_relevance(question, answer, embedder))
        else:
            out.append(citation_coverage(answer, [c.doc_id for c in contexts], relevant))
    return out


__all__ = [
    "answer_relevance",
    "citation_coverage",
    "compute_answer_metrics",
    "context_relevance",
    "contains_expected",
    "exact_match",
    "semantic_similarity",
    "token_overlap",
]
