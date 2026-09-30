"""Failure analysis: turn per-sample metric results into labelled failure cases.

Each failure carries the concrete evidence that triggered it. Categories that
depend on heuristic metrics (unsupported / contradicted answers) inherit their
limitations and are flagged accordingly.
"""

from __future__ import annotations

from typing import Any

CATEGORIES: dict[str, str] = {
    "empty_retrieval": "Retriever returned no documents",
    "retrieval_miss": "No labelled relevant document was retrieved",
    "partial_retrieval": "Some, but not all, labelled relevant documents were retrieved",
    "irrelevant_context": "Top-ranked retrieved document is not labelled relevant",
    "unsupported_answer": "Answer contains sentences not grounded in the retrieved context",
    "contradicted_answer": "Answer contradicts the retrieved context (number or negation mismatch)",
    "wrong_answer": "Answer lacks the expected answer and covers under half of its tokens",
}

WRONG_ANSWER_RECALL = 0.5


def analyze_sample(
    *,
    retrieved: list[dict[str, Any]],
    relevant: dict[str, int],
    metrics: dict[str, dict[str, Any]],
    answer: str,
    abstained: bool,
) -> list[dict[str, Any]]:
    """Return failure records for one sample. ``retrieved`` is the ranked list of dicts."""
    out: list[dict[str, Any]] = []
    rel = {d for d, g in relevant.items() if g > 0}
    ids = [r["doc_id"] for r in retrieved]

    def add(cat: str, severity: str, evidence: dict[str, Any], heuristic: bool = False) -> None:
        out.append(
            {
                "category": cat,
                "severity": severity,
                "description": CATEGORIES[cat],
                "evidence": evidence,
                "heuristic": heuristic,
            }
        )

    if not retrieved:
        add("empty_retrieval", "high", {"expected_relevant": sorted(rel)})
    elif rel:
        found = [i for i in dict.fromkeys(ids) if i in rel]
        if not found:
            add("retrieval_miss", "high", {"expected_relevant": sorted(rel), "retrieved": ids})
        else:
            if len(found) < len(rel):
                add(
                    "partial_retrieval",
                    "medium",
                    {"missed": sorted(rel - set(found)), "found": found, "retrieved": ids},
                )
            if ids[0] not in rel:
                add(
                    "irrelevant_context",
                    "medium",
                    {
                        "top_ranked": ids[0],
                        "first_relevant_rank": ids.index(found[0]) + 1,
                        "retrieved": ids,
                    },
                )
        if not found and ids:
            add("irrelevant_context", "medium", {"top_ranked": ids[0], "retrieved": ids})

    faith = metrics.get("faithfulness")
    if faith and faith["score"] is not None:
        sentences = faith["evidence"].get("sentences", [])
        bad = [s for s in sentences if s.get("label") == "unsupported"]
        contra = [s for s in sentences if s.get("label") == "contradicted"]
        if bad:
            add("unsupported_answer", "medium", {"sentences": bad}, heuristic=True)
        if contra:
            add("contradicted_answer", "high", {"sentences": contra}, heuristic=True)

    f1 = metrics.get("token_f1")
    contains = metrics.get("contains_expected")
    if f1 and f1["score"] is not None and contains and contains["score"] is not None:
        recall = f1["evidence"].get("recall", 0.0)
        if contains["score"] == 0.0 and recall < WRONG_ANSWER_RECALL:
            add(
                "wrong_answer",
                "medium",
                {"expected_token_recall": recall, "answer": answer, "abstained": abstained},
                heuristic=True,
            )
    return out


def summarize_failures(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(CATEGORIES, 0)
    for s in samples:
        for cat in {f["category"] for f in s["failures"]}:
            counts[cat] += 1
    return counts
