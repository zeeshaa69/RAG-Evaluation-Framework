from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.types import RetrievedDoc

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "examples" / "demo"


def rd(doc_id: str, text: str = "", rank: int = 1, score: float = 1.0) -> RetrievedDoc:
    return RetrievedDoc(doc_id, text, score, rank)


def write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    d = tmp_path / "ds"
    d.mkdir()
    write_jsonl(
        d / "corpus.jsonl",
        [
            {"id": "d1", "text": "The capital of France is Paris. Paris lies on the Seine."},
            {"id": "d2", "text": "The capital of Japan is Tokyo. Tokyo has 14 million residents."},
            {"id": "d3", "text": "Bananas are rich in potassium."},
        ],
    )
    write_jsonl(
        d / "samples.jsonl",
        [
            {
                "id": "s1",
                "question": "What is the capital of France?",
                "expected_answer": "Paris",
                "relevant_docs": ["d1"],
            },
            {
                "id": "s2",
                "question": "What is the capital of Japan?",
                "expected_answer": "Tokyo",
                "relevant_docs": ["d2"],
            },
        ],
    )
    return d
