"""Text utilities shared by retrievers, generators and metrics.

Everything here is deterministic and dependency-free.
"""

from __future__ import annotations

import re
import string

STOPWORDS: frozenset[str] = frozenset(
    """a an the and or but if then else of to in on at by for with from as is are was were be
    been being am do does did doing have has had having it its this that these those there here
    which who whom whose what when where why how not no nor so than too very can could will would
    should may might must shall about into over under again further once than such own same
    i you he she we they them his her their our your my me him us""".split()
)

NEGATIONS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "cannot",
        "without",
        "neither",
        "nor",
        "n't",
        "isn't",
        "wasn't",
        "doesn't",
        "didn't",
        "won't",
        "aren't",
        "weren't",
        "hasn't",
        "haven't",
        "can't",
    }
)

ABSTENTION_PATTERNS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "not enough information",
    "no relevant information",
    "insufficient context",
    "unable to answer",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?(?:\.[0-9]+)?")
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:\-]+)\]")
_SENTENCE_RE = re.compile(
    r"\S.*?[.!?]+(?:\s*\[[A-Za-z0-9_.:\-]+\])*(?=\s+[A-Z0-9\"'\[(]|\s*$)|\S.*$"
)
_NUMBER_RE = re.compile(r"(?<![a-z0-9])[0-9][0-9,]*(?:\.[0-9]+)?(?![a-z0-9])")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Keeps digits and simple decimals ("3.14")."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed (negations are kept by callers separately)."""
    return [t for t in tokenize(text) if t not in STOPWORDS and t not in NEGATIONS]


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation; a trailing ``[doc_id]`` stays with its sentence."""
    out: list[str] = []
    for line in text.splitlines():
        out.extend(m.strip() for m in _SENTENCE_RE.findall(line) if m.strip())
    return out


def extract_citations(text: str) -> list[str]:
    """Return ids cited as ``[doc_id]`` in order of appearance (with repeats)."""
    return _CITATION_RE.findall(text)


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text)


def extract_numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in _NUMBER_RE.findall(strip_citations(text).lower())]


def has_negation(text: str) -> bool:
    return any(t in NEGATIONS for t in re.findall(r"[a-z']+", text.lower()))


def is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in ABSTENTION_PATTERNS)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/extra space."""
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    no_articles = _ARTICLES_RE.sub(" ", no_punct)
    return " ".join(no_articles.split())
