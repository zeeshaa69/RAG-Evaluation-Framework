from __future__ import annotations

from typing import Any

from .base import Retriever, RetrieverFactory
from .builtin import (
    BM25Retriever,
    CharNgramRetriever,
    FirstKRetriever,
    NullRetriever,
    TfidfRetriever,
)

_REGISTRY: dict[str, RetrieverFactory] = {
    "bm25": lambda p: BM25Retriever(**p),
    "tfidf": lambda p: TfidfRetriever(**p),
    "char-ngram": lambda p: CharNgramRetriever(**p),
    "first-k": lambda p: FirstKRetriever(**p),
    "none": lambda p: NullRetriever(**p),
}


def register_retriever(name: str, factory: RetrieverFactory) -> None:
    """Register a custom retriever type usable from experiment configs."""
    _REGISTRY[name] = factory


def build_retriever(kind: str, params: dict[str, Any] | None = None) -> Retriever:
    if kind not in _REGISTRY:
        raise ValueError(f"unknown retriever {kind!r}; available: {sorted(_REGISTRY)}")
    try:
        return _REGISTRY[kind](dict(params or {}))
    except TypeError as exc:
        raise ValueError(f"invalid parameters for retriever {kind!r}: {exc}") from exc


def available_retrievers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "BM25Retriever",
    "CharNgramRetriever",
    "FirstKRetriever",
    "NullRetriever",
    "Retriever",
    "TfidfRetriever",
    "available_retrievers",
    "build_retriever",
    "register_retriever",
]
