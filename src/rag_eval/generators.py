"""Answer generators.

Built-ins are deterministic and need no network. :class:`OllamaGenerator` talks
to a *local* Ollama server if you have one running; it is optional.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .text import content_tokens, split_sentences
from .types import Generation, RetrievedDoc

_NUMBER_TOKEN = re.compile(r"(?<![\w\[.,])[0-9][0-9,]*(?:\.[0-9]+)?(?![\w\]])")

ABSTAIN_TEXT = "I don't know based on the provided context."


class Generator(Protocol):
    model_id: str
    params: dict[str, Any]

    def generate(self, question: str, contexts: Sequence[RetrievedDoc]) -> Generation: ...


class ExtractiveGenerator:
    """Deterministic "model": returns the context sentences most similar to the question.

    Each selected sentence is suffixed with ``[doc_id]``. If no sentence shares at
    least ``min_overlap`` of the question's content terms it abstains.
    """

    def __init__(
        self, max_sentences: int = 1, min_overlap: float = 0.25, cite: bool = True
    ) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        self.params: dict[str, Any] = {
            "max_sentences": max_sentences,
            "min_overlap": min_overlap,
            "cite": cite,
        }
        self.model_id = "extractive-v1"
        self._n, self._min, self._cite = max_sentences, min_overlap, cite

    def generate(self, question: str, contexts: Sequence[RetrievedDoc]) -> Generation:
        q = set(content_tokens(question))
        cands: list[tuple[float, int, int, str, str]] = []
        for ci, c in enumerate(contexts):
            for si, s in enumerate(split_sentences(c.text)):
                terms = set(content_tokens(s))
                if not q or not terms:
                    continue
                overlap = len(q & terms) / len(q)
                if overlap >= self._min:
                    cands.append((-overlap, ci, si, s, c.doc_id))
        if not cands:
            return Generation(ABSTAIN_TEXT, self.model_id, ())
        cands.sort(key=lambda x: x[:3])
        chosen = cands[: self._n]
        parts, used = [], []
        for _, _, _, s, doc_id in chosen:
            s = s.rstrip()
            parts.append(f"{s} [{doc_id}]" if self._cite else s)
            used.append(doc_id)
        return Generation(" ".join(parts), self.model_id, tuple(dict.fromkeys(used)))


class LeadGenerator:
    """Baseline: first sentence of the top-ranked context, regardless of the question."""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.model_id = "lead-sentence-v1"

    def generate(self, question: str, contexts: Sequence[RetrievedDoc]) -> Generation:
        if not contexts:
            return Generation(ABSTAIN_TEXT, self.model_id, ())
        sents = split_sentences(contexts[0].text)
        first = sents[0] if sents else contexts[0].text
        return Generation(f"{first} [{contexts[0].doc_id}]", self.model_id, (contexts[0].doc_id,))


class FaultInjectionGenerator:
    """Test utility: wraps the extractive generator and corrupts its answer on purpose.

    Use it to check that the evaluation pipeline *detects* known faults. It is a
    control for the evaluator, not a model to be benchmarked.

    ``mode``: ``"number"`` increments the first number, ``"negate"`` inserts "not"
    after the first copula, ``"unsupported"`` appends an invented sentence.
    """

    def __init__(self, mode: str = "number", max_sentences: int = 1) -> None:
        if mode not in {"number", "negate", "unsupported"}:
            raise ValueError("mode must be one of: number, negate, unsupported")
        self.mode = mode
        self.params: dict[str, Any] = {"mode": mode, "max_sentences": max_sentences}
        self.model_id = f"fault-injection-{mode}-v1"
        self._inner = ExtractiveGenerator(max_sentences=max_sentences)

    def generate(self, question: str, contexts: Sequence[RetrievedDoc]) -> Generation:
        g = self._inner.generate(question, contexts)
        ans = g.answer
        if ans == ABSTAIN_TEXT:
            return Generation(ans, self.model_id, ())
        if self.mode == "number":
            match = _NUMBER_TOKEN.search(ans)
            if match:
                raw = match.group(0).replace(",", "")
                shifted = str(int(raw) + 7) if raw.isdigit() else str(round(float(raw) + 7, 6))
                ans = ans[: match.start()] + shifted + ans[match.end() :]
        elif self.mode == "negate":
            ans = re.sub(r"\b(is|are|was|were)\b", r"\1 not", ans, count=1)
        else:
            ans = ans + " The committee ratified it unanimously in a secret vote."
        return Generation(ans, self.model_id, g.used_doc_ids)


class CallableGenerator:
    """Adapt any ``fn(question, contexts) -> str`` to the Generator interface."""

    def __init__(
        self, fn: Callable[[str, Sequence[RetrievedDoc]], str], model_id: str = "custom-callable"
    ) -> None:
        self._fn = fn
        self.model_id = model_id
        self.params: dict[str, Any] = {}

    def generate(self, question: str, contexts: Sequence[RetrievedDoc]) -> Generation:
        return Generation(self._fn(question, contexts), self.model_id, ())


class OllamaGenerator:
    """Generate with a model served by a local Ollama instance (``/api/generate``)."""

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 0,
        timeout: float = 120.0,
    ) -> None:
        self.params: dict[str, Any] = {
            "model": model,
            "host": host,
            "temperature": temperature,
            "seed": seed,
        }
        self.model_id = f"ollama:{model}"
        self._timeout = timeout

    def build_prompt(self, question: str, contexts: Sequence[RetrievedDoc]) -> str:
        ctx = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in contexts) or "(no context)"
        return (
            "Answer the question using only the context. Cite sources as [doc_id]. "
            f'If the context is insufficient, say "{ABSTAIN_TEXT}"\n\n'
            f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        )

    def generate(self, question: str, contexts: Sequence[RetrievedDoc]) -> Generation:
        body = json.dumps(
            {
                "model": self.params["model"],
                "prompt": self.build_prompt(question, contexts),
                "stream": False,
                "options": {"temperature": self.params["temperature"], "seed": self.params["seed"]},
            }
        ).encode()
        req = urllib.request.Request(
            str(self.params["host"]).rstrip("/") + "/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request to {self.params['host']} failed: {exc}") from exc
        answer = str(payload.get("response", "")).strip()
        return Generation(answer, self.model_id, ())


_GENERATORS: dict[str, Callable[[dict[str, Any]], Generator]] = {
    "extractive": lambda p: ExtractiveGenerator(**p),
    "lead": lambda p: LeadGenerator(**p),
    "fault-injection": lambda p: FaultInjectionGenerator(**p),
    "ollama": lambda p: OllamaGenerator(**p),
}


def register_generator(name: str, factory: Callable[[dict[str, Any]], Generator]) -> None:
    _GENERATORS[name] = factory


def build_generator(kind: str, params: dict[str, Any] | None = None) -> Generator:
    if kind not in _GENERATORS:
        raise ValueError(f"unknown generator {kind!r}; available: {sorted(_GENERATORS)}")
    try:
        return _GENERATORS[kind](dict(params or {}))
    except TypeError as exc:
        raise ValueError(f"invalid parameters for generator {kind!r}: {exc}") from exc


def available_generators() -> list[str]:
    return sorted(_GENERATORS)
