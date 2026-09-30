"""Deterministic, dependency-free retrievers for offline use and as baselines."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..metrics.embeddings import CharNgramEmbedder, cosine
from ..text import tokenize
from ..types import Document, RetrievedDoc


def _rank(scored: list[tuple[str, float]], docs: dict[str, Document], k: int) -> list[RetrievedDoc]:
    # Ties broken by document id so results never depend on dict/hash ordering.
    scored.sort(key=lambda x: (-x[1], x[0]))
    out: list[RetrievedDoc] = []
    for rank, (doc_id, score) in enumerate(scored[:k], start=1):
        out.append(RetrievedDoc(doc_id, docs[doc_id].text, score, rank))
    return out


class BM25Retriever:
    """Okapi BM25 over lowercase word tokens."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.name = "bm25"
        self.params: dict[str, Any] = {"k1": k1, "b": b}
        self._k1, self._b = k1, b
        self._docs: dict[str, Document] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._len: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._avg = 0.0

    def index(self, documents: Sequence[Document]) -> None:
        self._docs = {d.id: d for d in documents}
        self._tf = {d.id: Counter(tokenize(d.text)) for d in documents}
        self._len = {i: sum(tf.values()) for i, tf in self._tf.items()}
        self._df = Counter(t for tf in self._tf.values() for t in tf)
        self._avg = sum(self._len.values()) / len(self._len) if self._len else 0.0

    def retrieve(self, query: str, k: int) -> list[RetrievedDoc]:
        n = len(self._docs)
        q_terms = tokenize(query)
        scored: list[tuple[str, float]] = []
        for doc_id, tf in self._tf.items():
            score = 0.0
            for t in set(q_terms):
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                denom = f + self._k1 * (
                    1 - self._b + self._b * self._len[doc_id] / (self._avg or 1)
                )
                score += idf * f * (self._k1 + 1) / denom
            if score > 0:
                scored.append((doc_id, score))
        return _rank(scored, self._docs, k)


class TfidfRetriever:
    """Cosine similarity over TF-IDF word vectors."""

    def __init__(self) -> None:
        self.name = "tfidf"
        self.params: dict[str, Any] = {}
        self._docs: dict[str, Document] = {}
        self._vecs: dict[str, dict[str, float]] = {}
        self._idf: dict[str, float] = {}

    def _vec(self, text: str) -> dict[str, float]:
        tf = Counter(tokenize(text))
        return {t: (1 + math.log(c)) * self._idf.get(t, 0.0) for t, c in tf.items()}

    def index(self, documents: Sequence[Document]) -> None:
        self._docs = {d.id: d for d in documents}
        df: Counter[str] = Counter()
        for d in documents:
            df.update(set(tokenize(d.text)))
        n = len(documents)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vecs = {d.id: self._vec(d.text) for d in documents}

    def retrieve(self, query: str, k: int) -> list[RetrievedDoc]:
        q = self._vec(query)
        nq = math.sqrt(sum(v * v for v in q.values()))
        scored: list[tuple[str, float]] = []
        if nq > 0:
            for doc_id, v in self._vecs.items():
                nd = math.sqrt(sum(x * x for x in v.values()))
                if nd == 0:
                    continue
                dot = sum(w * v.get(t, 0.0) for t, w in q.items())
                if dot > 0:
                    scored.append((doc_id, dot / (nq * nd)))
        return _rank(scored, self._docs, k)


class CharNgramRetriever:
    """Dense-style retrieval using hashed character n-gram vectors (lexical, not neural)."""

    def __init__(self, n: int = 3) -> None:
        self.name = "char-ngram"
        self.params: dict[str, Any] = {"n": n}
        self._emb = CharNgramEmbedder(n=n)
        self._docs: dict[str, Document] = {}
        self._vecs: dict[str, dict[int, float]] = {}

    def index(self, documents: Sequence[Document]) -> None:
        self._docs = {d.id: d for d in documents}
        self._vecs = {d.id: self._emb.embed(d.text) for d in documents}

    def retrieve(self, query: str, k: int) -> list[RetrievedDoc]:
        q = self._emb.embed(query)
        scored = [(i, cosine(q, v)) for i, v in self._vecs.items()]
        return _rank([s for s in scored if s[1] > 0], self._docs, k)


class FirstKRetriever:
    """Naive baseline: the first ``k`` documents in corpus order, ignoring the query."""

    def __init__(self) -> None:
        self.name = "first-k"
        self.params: dict[str, Any] = {}
        self._order: list[Document] = []

    def index(self, documents: Sequence[Document]) -> None:
        self._order = list(documents)

    def retrieve(self, query: str, k: int) -> list[RetrievedDoc]:
        return [
            RetrievedDoc(d.id, d.text, 1.0 / r, r) for r, d in enumerate(self._order[:k], start=1)
        ]


class NullRetriever:
    """Returns no context; measures what a generator does without retrieval."""

    def __init__(self) -> None:
        self.name = "none"
        self.params: dict[str, Any] = {}

    def index(self, documents: Sequence[Document]) -> None:
        return None

    def retrieve(self, query: str, k: int) -> list[RetrievedDoc]:
        return []
