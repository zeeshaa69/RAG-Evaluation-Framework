"""Evaluate your own retriever and generator with the Python SDK (no network, no API keys).

Run from the repository root:  python examples/python_api.py
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rag_eval import CallableGenerator, Document, RagEval, RetrievedDoc

HERE = Path(__file__).resolve().parent


class KeywordRetriever:
    """Toy retriever: ranks documents by how many query words they contain."""

    name = "keyword-count"
    params: dict[str, object] = {}

    def index(self, documents: Sequence[Document]) -> None:
        self._docs = list(documents)

    def retrieve(self, query: str, k: int) -> list[RetrievedDoc]:
        words = set(query.lower().split())
        scored = [(len(words & set(d.text.lower().split())), d) for d in self._docs]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], s[1].id))
        return [RetrievedDoc(d.id, d.text, float(n), r) for r, (n, d) in enumerate(scored[:k], 1)]


def first_sentence_answer(question: str, contexts: Sequence[RetrievedDoc]) -> str:
    if not contexts:
        return "I don't know based on the provided context."
    return contexts[0].text.split(". ")[0].rstrip(".") + f". [{contexts[0].doc_id}]"


def main(results_dir: str | Path = "runs") -> None:
    ev = RagEval(results_dir=results_dir)
    print("dataset valid:", ev.validate_dataset(HERE / "demo" / "dataset").ok)

    mine = ev.run(
        {"name": "my-keyword-retriever", "dataset": str(HERE / "demo" / "dataset"), "top_k": 3},
        retriever=KeywordRetriever(),
        generator=CallableGenerator(first_sentence_answer, model_id="first-sentence"),
    )
    baseline = ev.run(HERE / "demo" / "experiments" / "bm25-extractive.json")

    for name in ("recall@3", "mrr@3", "faithfulness"):
        print(f"{name}: mine={mine.metric_means[name]}  bm25={baseline.metric_means[name]}")
    print(ev.format_comparison(ev.compare(baseline, mine)))


if __name__ == "__main__":
    main()
