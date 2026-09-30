"""Dataset loading, validation and quality analysis.

A dataset is either a directory::

    dataset/
      dataset.json      (optional: {"name": ..., "description": ...})
      corpus.jsonl      one {"id", "text", "metadata"?} per line
      samples.jsonl     one {"id", "question", "expected_answer"?,
                             "relevant_docs"?, "metadata"?} per line

or a single ``.json`` file ``{"name", "documents": [...], "samples": [...]}``.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .text import content_tokens
from .types import Document, Sample


class DatasetError(Exception):
    """Raised when a dataset cannot be loaded because it has validation errors."""

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    location: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "location": self.location,
        }


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def error(self, code: str, message: str, location: str = "") -> None:
        self.issues.append(Issue("error", code, message, location))

    def warn(self, code: str, message: str, location: str = "") -> None:
        self.issues.append(Issue("warning", code, message, location))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class Dataset:
    name: str
    documents: list[Document]
    samples: list[Sample]
    description: str = ""

    def __post_init__(self) -> None:
        self._doc_index = {d.id: d for d in self.documents}

    def get_document(self, doc_id: str) -> Document | None:
        return self._doc_index.get(doc_id)

    @property
    def corpus_hash(self) -> str:
        payload = [[d.id, d.text, d.metadata] for d in self.documents]
        return _hash(payload)

    @property
    def samples_hash(self) -> str:
        payload = [
            [s.id, s.question, s.expected_answer, sorted(s.relevant_docs.items()), s.metadata]
            for s in self.samples
        ]
        return _hash(payload)

    @property
    def version(self) -> str:
        """Content hash identifying this exact dataset (corpus + samples)."""
        return _hash([self.corpus_hash, self.samples_hash])


def _hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- parsing


def _read_jsonl(path: Path, report: ValidationReport) -> list[tuple[int, Any]]:
    rows: list[tuple[int, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        report.error("file_unreadable", f"cannot read {path.name}: {exc}", path.name)
        return rows
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append((lineno, json.loads(line)))
        except json.JSONDecodeError as exc:
            report.error("malformed_json", f"invalid JSON: {exc.msg}", f"{path.name}:{lineno}")
    return rows


def _parse_relevant(raw: Any, loc: str, report: ValidationReport) -> dict[str, int]:
    out: dict[str, int] = {}
    if raw is None:
        return out
    items: Iterable[Any]
    if isinstance(raw, dict):
        items = [{"id": k, "grade": v} for k, v in raw.items()]
    elif isinstance(raw, list):
        items = raw
    else:
        report.error("bad_relevant_docs", "relevant_docs must be a list or object", loc)
        return out
    for item in items:
        doc_id: Any
        grade: Any = 1
        if isinstance(item, str):
            doc_id = item
        elif isinstance(item, dict) and "id" in item:
            doc_id, grade = item["id"], item.get("grade", 1)
        else:
            report.error("bad_relevant_docs", f"unsupported relevant_docs entry: {item!r}", loc)
            continue
        if not isinstance(doc_id, str) or not doc_id:
            report.error(
                "bad_relevant_docs", f"document id must be a non-empty string: {doc_id!r}", loc
            )
            continue
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
            report.error("bad_grade", f"grade for {doc_id!r} must be an integer >= 0", loc)
            continue
        if doc_id in out:
            report.warn("duplicate_relevant_doc", f"relevant doc {doc_id!r} listed twice", loc)
            out[doc_id] = max(out[doc_id], grade)
        else:
            out[doc_id] = grade
    return {k: v for k, v in out.items() if v > 0}


def _parse_documents(
    rows: list[tuple[int, Any]], src: str, report: ValidationReport
) -> list[Document]:
    docs: list[Document] = []
    seen: set[str] = set()
    for lineno, row in rows:
        loc = f"{src}:{lineno}"
        if not isinstance(row, dict):
            report.error("bad_record", "document must be a JSON object", loc)
            continue
        doc_id, text, meta = row.get("id"), row.get("text"), row.get("metadata", {})
        if not isinstance(doc_id, str) or not doc_id.strip():
            report.error("missing_id", "document requires a non-empty string 'id'", loc)
            continue
        if not isinstance(text, str):
            report.error("missing_text", f"document {doc_id!r} requires string 'text'", loc)
            continue
        if not isinstance(meta, dict):
            report.error("bad_metadata", f"document {doc_id!r} metadata must be an object", loc)
            continue
        if doc_id in seen:
            report.error("duplicate_doc_id", f"duplicate document id {doc_id!r}", loc)
            continue
        if not text.strip():
            report.warn("empty_document", f"document {doc_id!r} has empty text", loc)
        seen.add(doc_id)
        docs.append(Document(doc_id, text, meta))
    return docs


def _parse_samples(rows: list[tuple[int, Any]], src: str, report: ValidationReport) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    for lineno, row in rows:
        loc = f"{src}:{lineno}"
        if not isinstance(row, dict):
            report.error("bad_record", "sample must be a JSON object", loc)
            continue
        sid, question = row.get("id"), row.get("question")
        if not isinstance(sid, str) or not sid.strip():
            report.error("missing_id", "sample requires a non-empty string 'id'", loc)
            continue
        if not isinstance(question, str) or not question.strip():
            report.error("missing_question", f"sample {sid!r} requires a non-empty 'question'", loc)
            continue
        answer = row.get("expected_answer")
        if answer is not None and not isinstance(answer, str):
            report.error(
                "bad_answer", f"sample {sid!r} expected_answer must be a string or null", loc
            )
            continue
        meta = row.get("metadata", {})
        if not isinstance(meta, dict):
            report.error("bad_metadata", f"sample {sid!r} metadata must be an object", loc)
            continue
        if sid in seen:
            report.error("duplicate_sample_id", f"duplicate sample id {sid!r}", loc)
            continue
        seen.add(sid)
        if answer is not None and not answer.strip():
            answer = None
        rel = _parse_relevant(row.get("relevant_docs"), loc, report)
        samples.append(Sample(sid, question, answer, rel, meta))
    return samples


def _cross_check(ds: Dataset, report: ValidationReport) -> None:
    ids = {d.id for d in ds.documents}
    for s in ds.samples:
        missing = sorted(d for d in s.relevant_docs if d not in ids)
        if missing:
            report.error(
                "unknown_relevant_doc",
                f"sample {s.id!r} references documents not in corpus: {missing}",
                s.id,
            )
        if not s.relevant_docs:
            report.warn("no_relevant_docs", f"sample {s.id!r} has no relevant_docs labels", s.id)
        if s.expected_answer is None:
            report.warn("no_expected_answer", f"sample {s.id!r} has no expected_answer", s.id)
    if not ds.documents:
        report.error("empty_corpus", "corpus contains no documents")
    if not ds.samples:
        report.error("empty_samples", "dataset contains no samples")
    questions: dict[str, str] = {}
    for s in ds.samples:
        key = " ".join(s.question.lower().split())
        if key in questions:
            report.warn(
                "duplicate_question",
                f"sample {s.id!r} has the same question as {questions[key]!r}",
                s.id,
            )
        else:
            questions[key] = s.id


def inspect_dataset(path: str | Path) -> tuple[Dataset | None, ValidationReport]:
    """Parse ``path`` and return ``(dataset, report)``.

    ``dataset`` is ``None`` when the structure is unusable (no parsable
    corpus/samples at all); otherwise it contains every record that parsed.
    Always check ``report.ok`` before trusting results.
    """
    p = Path(path)
    report = ValidationReport()
    name, description = p.stem if p.is_file() else p.name, ""
    docs: list[Document] = []
    samples: list[Sample] = []

    if p.is_dir():
        meta_file = p / "dataset.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    name = str(meta.get("name", name))
                    description = str(meta.get("description", ""))
                else:
                    report.error("bad_record", "dataset.json must be an object", "dataset.json")
            except json.JSONDecodeError as exc:
                report.error("malformed_json", f"invalid JSON: {exc.msg}", "dataset.json")
        for fname in ("corpus.jsonl", "samples.jsonl"):
            if not (p / fname).exists():
                report.error("missing_file", f"required file {fname} not found", fname)
        if (p / "corpus.jsonl").exists():
            docs = _parse_documents(_read_jsonl(p / "corpus.jsonl", report), "corpus.jsonl", report)
        if (p / "samples.jsonl").exists():
            samples = _parse_samples(
                _read_jsonl(p / "samples.jsonl", report), "samples.jsonl", report
            )
    elif p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            report.error("malformed_json", f"cannot parse {p.name}: {exc}", p.name)
            return None, report
        if not isinstance(data, dict):
            report.error("bad_record", "top-level JSON must be an object", p.name)
            return None, report
        name = str(data.get("name", name))
        description = str(data.get("description", ""))
        raw_docs, raw_samples = data.get("documents"), data.get("samples")
        if not isinstance(raw_docs, list):
            report.error("missing_field", "'documents' must be a list", p.name)
            raw_docs = []
        if not isinstance(raw_samples, list):
            report.error("missing_field", "'samples' must be a list", p.name)
            raw_samples = []
        docs = _parse_documents(list(enumerate(raw_docs, 1)), "documents", report)
        samples = _parse_samples(list(enumerate(raw_samples, 1)), "samples", report)
    else:
        report.error("missing_file", f"path does not exist: {p}")
        return None, report

    ds = Dataset(name=name, documents=docs, samples=samples, description=description)
    _cross_check(ds, report)
    return ds, report


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset, raising :class:`DatasetError` if it has validation errors."""
    ds, report = inspect_dataset(path)
    if ds is None or not report.ok:
        first = report.errors[0] if report.errors else None
        detail = f": {first.message} ({first.location})" if first else ""
        raise DatasetError(
            f"dataset at {path} is invalid ({len(report.errors)} error(s)){detail}", report
        )
    return ds


# --------------------------------------------------------------------------- quality


def dataset_quality(ds: Dataset) -> dict[str, Any]:
    """Descriptive statistics and label-quality signals. Nothing here is a model score."""
    n = len(ds.samples)
    with_answer = [s for s in ds.samples if s.expected_answer]
    with_rel = [s for s in ds.samples if s.relevant_docs]
    referenced: set[str] = {d for s in ds.samples for d in s.relevant_docs}
    doc_lens = [len(d.text.split()) for d in ds.documents]
    ans_lens = [len(s.expected_answer.split()) for s in with_answer if s.expected_answer]

    seen_text: dict[str, str] = {}
    dup_docs: list[list[str]] = []
    for d in ds.documents:
        key = " ".join(d.text.lower().split())
        if key in seen_text:
            dup_docs.append([seen_text[key], d.id])
        else:
            seen_text[key] = d.id

    # Answerability: share of expected-answer content tokens found in the labelled docs.
    low_support: list[dict[str, Any]] = []
    support_scores: list[float] = []
    for s in with_answer:
        if not s.relevant_docs or s.expected_answer is None:
            continue
        ans = set(content_tokens(s.expected_answer))
        if not ans:
            continue
        pool: set[str] = set()
        for did in s.relevant_docs:
            doc = ds.get_document(did)
            if doc:
                pool.update(content_tokens(doc.text))
        cov = len(ans & pool) / len(ans)
        support_scores.append(cov)
        if cov < 0.5:
            low_support.append({"sample_id": s.id, "coverage": round(cov, 3)})

    def dist(values: list[int]) -> dict[str, float]:
        if not values:
            return {"min": 0, "median": 0, "max": 0, "mean": 0}
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "mean": round(statistics.fmean(values), 2),
        }

    rel_counts = [len(s.relevant_docs) for s in ds.samples]
    return {
        "name": ds.name,
        "version": ds.version,
        "corpus_hash": ds.corpus_hash,
        "samples_hash": ds.samples_hash,
        "num_samples": n,
        "num_documents": len(ds.documents),
        "samples_with_expected_answer": len(with_answer),
        "samples_with_relevant_docs": len(with_rel),
        "samples_multi_relevant": sum(1 for c in rel_counts if c > 1),
        "relevant_docs_per_sample": dist(rel_counts),
        "documents_never_relevant": len(ds.documents)
        - len(referenced & {d.id for d in ds.documents}),
        "document_length_words": dist(doc_lens),
        "answer_length_words": dist(ans_lens),
        "duplicate_documents": dup_docs,
        "answer_support_mean": round(statistics.fmean(support_scores), 3)
        if support_scores
        else None,
        "low_answer_support": low_support,
        "note": "Label-quality signals are lexical heuristics; "
        "low support may be a paraphrase, not an error.",
    }
