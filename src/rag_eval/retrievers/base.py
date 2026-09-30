from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from ..types import Document, RetrievedDoc


class Retriever(Protocol):
    """Anything with ``index`` and ``retrieve`` can be evaluated."""

    name: str
    params: dict[str, Any]

    def index(self, documents: Sequence[Document]) -> None: ...

    def retrieve(self, query: str, k: int) -> list[RetrievedDoc]: ...


RetrieverFactory = Callable[[dict[str, Any]], Retriever]
