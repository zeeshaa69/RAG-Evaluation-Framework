"""Core data types shared across the framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    """One evaluation example.

    ``relevant_docs`` maps document id to a graded relevance (>= 1). Binary
    labels are represented with grade 1.
    """

    id: str
    question: str
    expected_answer: str | None = None
    relevant_docs: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    rank: int  # 1-based


@dataclass(frozen=True)
class Generation:
    answer: str
    model_id: str
    used_doc_ids: tuple[str, ...] = ()


@dataclass
class MetricResult:
    """A single transparent measurement.

    ``score`` is ``None`` when the metric is not applicable to the sample
    (for example no expected answer). ``heuristic`` marks metrics that are
    approximations and must not be read as ground truth.
    """

    metric: str
    score: float | None
    method: str
    evidence: dict[str, Any] = field(default_factory=dict)
    heuristic: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "score": self.score,
            "method": self.method,
            "evidence": self.evidence,
            "heuristic": self.heuristic,
            "note": self.note,
        }
