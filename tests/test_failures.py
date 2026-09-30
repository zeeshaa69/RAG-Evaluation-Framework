from __future__ import annotations

from typing import Any

from rag_eval.failures import analyze_sample, summarize_failures


def cats(**kw: Any) -> set[str]:
    base: dict[str, Any] = {
        "retrieved": ids("a"),
        "relevant": {"a": 1},
        "metrics": {},
        "answer": "a",
        "abstained": False,
    }
    base.update(kw)
    return {f["category"] for f in analyze_sample(**base)}


def ids(*names: str) -> list[dict[str, str]]:
    return [{"doc_id": n} for n in names]


def test_empty_retrieval() -> None:
    assert cats(retrieved=[], relevant={"a": 1}) == {"empty_retrieval"}
    assert cats(retrieved=[]) == {"empty_retrieval"}


def test_retrieval_miss_and_irrelevant_context() -> None:
    c = cats(retrieved=ids("x", "y"), relevant={"a": 1})
    assert c == {"retrieval_miss", "irrelevant_context"}


def test_partial_retrieval() -> None:
    assert cats(retrieved=ids("a", "x"), relevant={"a": 1, "b": 1}) == {"partial_retrieval"}


def test_relevant_not_at_top_is_irrelevant_context() -> None:
    assert cats(retrieved=ids("x", "a"), relevant={"a": 1}) == {"irrelevant_context"}


def test_clean_retrieval_has_no_failures() -> None:
    assert cats(retrieved=ids("a"), relevant={"a": 1}) == set()


def test_no_labels_no_retrieval_failures() -> None:
    assert cats(retrieved=ids("x"), relevant={}) == set()


def faith(labels: list[str]) -> dict[str, Any]:
    return {
        "faithfulness": {"score": 0.0, "evidence": {"sentences": [{"label": l_} for l_ in labels]}}
    }


def test_answer_failures_from_faithfulness_evidence() -> None:
    assert cats(metrics=faith(["supported", "unsupported"])) == {"unsupported_answer"}
    assert cats(metrics=faith(["contradicted"])) == {"contradicted_answer"}
    assert cats(metrics=faith(["supported"])) == set()
    assert cats(metrics={"faithfulness": {"score": None, "evidence": {}}}) == set()


def test_wrong_answer_requires_no_containment_and_low_recall() -> None:
    bad = {
        "token_f1": {"score": 0.0, "evidence": {"recall": 0.0}},
        "contains_expected": {"score": 0.0, "evidence": {}},
    }
    assert cats(metrics=bad) == {"wrong_answer"}
    verbose_but_right = {
        "token_f1": {"score": 0.1, "evidence": {"recall": 1.0}},
        "contains_expected": {"score": 1.0, "evidence": {}},
    }
    assert cats(metrics=verbose_but_right) == set()
    not_applicable: dict[str, Any] = {
        "token_f1": {"score": None, "evidence": {}},
        "contains_expected": {"score": None, "evidence": {}},
    }
    assert cats(metrics=not_applicable) == set()


def test_heuristic_flag_and_summary() -> None:
    f = analyze_sample(
        retrieved=ids("a"),
        relevant={"a": 1},
        metrics=faith(["unsupported"]),
        answer="x",
        abstained=False,
    )
    assert f[0]["heuristic"] is True
    counts = summarize_failures([{"failures": f}, {"failures": []}])
    assert counts["unsupported_answer"] == 1 and counts["retrieval_miss"] == 0
