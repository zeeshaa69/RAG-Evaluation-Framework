from __future__ import annotations

from rag_eval.text import (
    content_tokens,
    extract_citations,
    extract_numbers,
    has_negation,
    is_abstention,
    normalize_answer,
    split_sentences,
    strip_citations,
    tokenize,
)


def test_tokenize_keeps_decimals_and_lowercases() -> None:
    assert tokenize("Pi is 3.14159, roughly!") == ["pi", "is", "3.14159", "roughly"]


def test_content_tokens_drop_stopwords_and_negations() -> None:
    assert content_tokens("The cat is not on the mat") == ["cat", "mat"]


def test_normalize_answer_squad_style() -> None:
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"


def test_split_sentences_attaches_trailing_citation() -> None:
    parts = split_sentences("Paris is big. [d1] Tokyo is bigger. [d2]")
    assert parts == ["Paris is big. [d1]", "Tokyo is bigger. [d2]"]


def test_split_sentences_empty() -> None:
    assert split_sentences("   ") == []


def test_citations() -> None:
    text = "A fact [d1]. Another [doc-2] and [d1]."
    assert extract_citations(text) == ["d1", "doc-2", "d1"]
    assert "[" not in strip_citations(text)


def test_extract_numbers_handles_commas_and_ignores_citations() -> None:
    assert extract_numbers("It is 8,849 m and 3.5 km [d2]") == ["8849", "3.5"]


def test_negation_and_abstention() -> None:
    assert has_negation("It is not true")
    assert has_negation("They didn't go")
    assert not has_negation("It is true")
    assert is_abstention("I don't know based on the provided context.")
    assert not is_abstention("Paris")
