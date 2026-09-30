"""Pluggable text embedders used for similarity metrics.

The default :class:`CharNgramEmbedder` is a deterministic lexical vectoriser
(hashed character n-grams). It measures surface overlap with some robustness to
inflection and typos; it is *not* a neural semantic model. Install the
``embeddings`` extra to use :class:`SentenceTransformerEmbedder` with a locally
cached model.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from ..text import normalize_answer


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> dict[int, float]: ...


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


class CharNgramEmbedder:
    """Hashed character n-gram term-frequency vectors (sub-linear TF)."""

    def __init__(self, n: int = 3, dim: int = 2**18) -> None:
        self.n = n
        self.dim = dim
        self.name = f"char-{n}gram-hash{dim}"

    def embed(self, text: str) -> dict[int, float]:
        norm = " " + normalize_answer(text) + " "
        counts: dict[int, int] = {}
        for i in range(max(0, len(norm) - self.n + 1)):
            gram = norm[i : i + self.n]
            h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=4).digest(), "big")
            idx = h % self.dim
            counts[idx] = counts.get(idx, 0) + 1
        return {k: 1.0 + math.log(v) for k, v in counts.items()}


class SentenceTransformerEmbedder:  # pragma: no cover - requires optional dependency
    """Dense embeddings from a locally available sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("install rag-eval[embeddings] to use this embedder") from exc
        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"

    def embed(self, text: str) -> dict[int, float]:
        vec = self._model.encode(text)
        return {i: float(v) for i, v in enumerate(vec)}
