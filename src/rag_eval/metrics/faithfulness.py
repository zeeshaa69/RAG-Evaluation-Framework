"""Transparent, heuristic faithfulness (hallucination) check.

For each claim-bearing sentence in the answer we look for the retrieved context
sentence with the best content-term overlap, then label the sentence:

* ``supported``    - enough of its content terms (and all its numbers) appear in a
                     single context sentence,
* ``contradicted`` - it overlaps strongly with a context sentence but disagrees on
                     a number or on negation polarity,
* ``unsupported``  - neither.

This is a lexical approximation. It cannot verify paraphrases, reasoning across
sentences, or world knowledge, and it can mislabel in both directions. Every
label is returned with its evidence so a human can audit it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..text import (
    content_tokens,
    extract_numbers,
    has_negation,
    is_abstention,
    split_sentences,
    strip_citations,
)
from ..types import MetricResult, RetrievedDoc

SUPPORT_THRESHOLD = 0.6
CONTRADICTION_OVERLAP = 0.5


@dataclass(frozen=True)
class _ContextSentence:
    doc_id: str
    text: str
    terms: frozenset[str]
    numbers: frozenset[str]
    negated: bool


def _context_sentences(contexts: Sequence[RetrievedDoc]) -> list[_ContextSentence]:
    out: list[_ContextSentence] = []
    for c in contexts:
        for s in split_sentences(c.text):
            out.append(
                _ContextSentence(
                    c.doc_id,
                    s,
                    frozenset(content_tokens(s)),
                    frozenset(extract_numbers(s)),
                    has_negation(s),
                )
            )
    return out


def _judge_sentence(sentence: str, ctx: list[_ContextSentence]) -> dict[str, object]:
    clean = strip_citations(sentence).strip()
    terms = set(content_tokens(clean))
    numbers = set(extract_numbers(clean))
    neg = has_negation(clean)
    if not terms and not numbers:
        return {"sentence": sentence, "label": "skipped", "reason": "no content terms"}
    best: _ContextSentence | None = None
    best_overlap = 0.0
    for c in ctx:
        overlap = len(terms & c.terms) / len(terms) if terms else 0.0
        if overlap > best_overlap:
            best, best_overlap = c, overlap
    row: dict[str, object] = {
        "sentence": sentence,
        "overlap": round(best_overlap, 4),
        "best_context": best.text if best else None,
        "best_doc_id": best.doc_id if best else None,
    }
    if best is None:
        row.update(label="unsupported", reason="no overlapping context")
        return row
    missing_numbers = sorted(numbers - best.numbers)
    negation_flip = neg != best.negated
    if best_overlap >= CONTRADICTION_OVERLAP and (
        (missing_numbers and best.numbers) or negation_flip
    ):
        reasons = []
        if missing_numbers and best.numbers:
            reasons.append(
                f"number(s) {missing_numbers} not in context sentence "
                f"(context has {sorted(best.numbers)})"
            )
        if negation_flip:
            reasons.append("negation polarity differs from closest context sentence")
        row.update(label="contradicted", reason="; ".join(reasons))
    elif best_overlap >= SUPPORT_THRESHOLD and not missing_numbers:
        row.update(
            label="supported",
            reason=f"{best_overlap:.0%} of content terms found in one context sentence",
        )
    else:
        why = f"only {best_overlap:.0%} of content terms found in any context sentence"
        if missing_numbers:
            why += f"; number(s) {missing_numbers} absent from context"
        row.update(label="unsupported", reason=why)
    return row


def faithfulness(answer: str, contexts: Sequence[RetrievedDoc]) -> list[MetricResult]:
    """Return ``[faithfulness, contradiction_rate]`` results with per-sentence evidence."""
    method = (
        f"per-sentence lexical grounding: supported if >={SUPPORT_THRESHOLD:.0%} of content terms "
        "and all numbers occur in one retrieved sentence"
    )
    names = ("faithfulness", "contradiction_rate")
    note = "Heuristic. Not a proof of hallucination or of correctness; review the evidence."
    body = strip_citations(answer).strip()
    if not body or is_abstention(body):
        reason = "empty answer" if not body else "answer is an abstention (no claims to verify)"
        return [MetricResult(n, None, method, {"reason": reason}, True, note) for n in names]
    ctx = _context_sentences(contexts)
    rows = [_judge_sentence(s, ctx) for s in split_sentences(answer)]
    scored = [r for r in rows if r["label"] != "skipped"]
    if not scored:
        return [
            MetricResult(
                n,
                None,
                method,
                {"reason": "no claim-bearing sentences", "sentences": rows},
                True,
                note,
            )
            for n in names
        ]
    supported = sum(1 for r in scored if r["label"] == "supported")
    contradicted = sum(1 for r in scored if r["label"] == "contradicted")
    evidence = {
        "sentences": rows,
        "supported": supported,
        "contradicted": contradicted,
        "unsupported": len(scored) - supported - contradicted,
        "num_context_sentences": len(ctx),
    }
    return [
        MetricResult("faithfulness", supported / len(scored), method, evidence, True, note),
        MetricResult(
            "contradiction_rate", contradicted / len(scored), method, evidence, True, note
        ),
    ]
