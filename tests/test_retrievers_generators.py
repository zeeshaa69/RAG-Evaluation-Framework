from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from rag_eval.generators import (
    ABSTAIN_TEXT,
    CallableGenerator,
    ExtractiveGenerator,
    FaultInjectionGenerator,
    LeadGenerator,
    OllamaGenerator,
    build_generator,
)
from rag_eval.retrievers import available_retrievers, build_retriever, register_retriever
from rag_eval.types import Document
from tests.conftest import rd

DOCS = [
    Document("paris", "Paris is the capital of France. It lies on the Seine."),
    Document("tokyo", "Tokyo is the capital of Japan."),
    Document("banana", "Bananas are rich in potassium."),
]


@pytest.mark.parametrize("kind", ["bm25", "tfidf", "char-ngram"])
def test_lexical_retrievers_rank_relevant_first(kind: str) -> None:
    r = build_retriever(kind)
    r.index(DOCS)
    res = r.retrieve("capital of France", 3)
    assert res[0].doc_id == "paris"
    assert [x.rank for x in res] == list(range(1, len(res) + 1))
    assert all(a.score >= b.score for a, b in zip(res, res[1:], strict=False))


@pytest.mark.parametrize("kind", ["bm25", "tfidf", "char-ngram", "first-k", "none"])
def test_retrievers_are_deterministic_and_respect_k(kind: str) -> None:
    r = build_retriever(kind)
    r.index(DOCS)
    a, b = r.retrieve("capital", 2), r.retrieve("capital", 2)
    assert a == b and len(a) <= 2


def test_bm25_no_overlap_returns_nothing_and_empty_query() -> None:
    r = build_retriever("bm25")
    r.index(DOCS)
    assert r.retrieve("zzzz qqqq", 3) == []
    assert r.retrieve("", 3) == []


def test_retrievers_on_empty_corpus() -> None:
    for kind in ("bm25", "tfidf", "char-ngram", "first-k"):
        r = build_retriever(kind)
        r.index([])
        assert r.retrieve("anything", 3) == []


def test_ties_break_by_doc_id() -> None:
    r = build_retriever("bm25")
    r.index([Document("b", "same words"), Document("a", "same words")])
    assert [x.doc_id for x in r.retrieve("same words", 2)] == ["a", "b"]


def test_first_k_ignores_query_and_none_is_empty() -> None:
    f = build_retriever("first-k")
    f.index(DOCS)
    assert [d.doc_id for d in f.retrieve("whatever", 2)] == ["paris", "tokyo"]
    n = build_retriever("none")
    n.index(DOCS)
    assert n.retrieve("capital", 3) == []


def test_unknown_retriever_and_bad_params() -> None:
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever("magic")
    with pytest.raises(ValueError, match="invalid parameters"):
        build_retriever("bm25", {"nope": 1})


def test_register_custom_retriever() -> None:
    register_retriever("const-test", lambda p: build_retriever("first-k"))
    assert "const-test" in available_retrievers()


def test_extractive_generator_cites_and_selects_best_sentence() -> None:
    g = ExtractiveGenerator()
    out = g.generate("What is the capital of France?", [rd("paris", DOCS[0].text)])
    assert out.answer == "Paris is the capital of France. [paris]"
    assert out.used_doc_ids == ("paris",)


def test_extractive_generator_abstains_without_support() -> None:
    g = ExtractiveGenerator()
    assert g.generate("Who won?", []).answer == ABSTAIN_TEXT
    assert g.generate("quantum chromodynamics", [rd("banana", DOCS[2].text)]).answer == ABSTAIN_TEXT


def test_extractive_generator_options() -> None:
    ctx = [rd("paris", DOCS[0].text), rd("tokyo", DOCS[1].text)]
    out = ExtractiveGenerator(max_sentences=2, cite=False).generate(
        "capital of France and Japan", ctx
    )
    assert "[" not in out.answer and out.answer.count(".") == 2
    with pytest.raises(ValueError):
        ExtractiveGenerator(max_sentences=0)


def test_lead_generator() -> None:
    g = LeadGenerator()
    assert g.generate("q", []).answer == ABSTAIN_TEXT
    assert g.generate("q", [rd("tokyo", DOCS[1].text)]).answer.endswith("[tokyo]")


@pytest.mark.parametrize("mode", ["number", "negate", "unsupported"])
def test_fault_injection_modes(mode: str) -> None:
    ctx = [rd("e", "Mount Everest is 8,849 meters above sea level.")]
    clean = ExtractiveGenerator().generate("How tall is Mount Everest meters?", ctx).answer
    bad = FaultInjectionGenerator(mode).generate("How tall is Mount Everest meters?", ctx).answer
    assert bad != clean


def test_fault_injection_invalid_mode_and_abstain() -> None:
    with pytest.raises(ValueError):
        FaultInjectionGenerator("explode")
    assert FaultInjectionGenerator().generate("q", []).answer == ABSTAIN_TEXT


def test_callable_generator_and_registry() -> None:
    g = CallableGenerator(lambda q, c: f"{q}!{len(c)}", model_id="mine")
    out = g.generate("hi", [rd("a")])
    assert out.answer == "hi!1" and out.model_id == "mine"
    with pytest.raises(ValueError, match="unknown generator"):
        build_generator("gpt-magic")


class _FakeOllama(BaseHTTPRequestHandler):
    seen: dict[str, Any] = {}

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.seen = {"path": self.path, "body": body}
        data = json.dumps({"response": " Paris [paris] "}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None


def test_ollama_generator_protocol_against_local_fake_server() -> None:
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        g = OllamaGenerator(model="m", host=f"http://127.0.0.1:{srv.server_port}", seed=3)
        out = g.generate("Capital of France?", [rd("paris", "Paris is the capital of France.")])
    finally:
        srv.shutdown()
    assert out.answer == "Paris [paris]" and out.model_id == "ollama:m"
    body = _FakeOllama.seen["body"]
    assert _FakeOllama.seen["path"] == "/api/generate"
    assert body["stream"] is False and body["options"] == {"temperature": 0.0, "seed": 3}
    assert "[paris] Paris is the capital" in body["prompt"]


def test_ollama_connection_failure_is_reported() -> None:
    g = OllamaGenerator(host="http://127.0.0.1:9", timeout=1)
    with pytest.raises(RuntimeError, match="Ollama request"):
        g.generate("q", [])
