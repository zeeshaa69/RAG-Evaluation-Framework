"""Experiment definition and runner."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import failures as failure_mod
from .datasets import Dataset, dataset_quality, load_dataset
from .generators import Generator, build_generator
from .metrics.answer import compute_answer_metrics, contains_expected, token_overlap
from .metrics.embeddings import CharNgramEmbedder, Embedder
from .metrics.faithfulness import faithfulness
from .metrics.retrieval import RETRIEVAL_METRICS, compute_retrieval_metrics
from .retrievers import Retriever, build_retriever
from .text import is_abstention
from .types import MetricResult, Sample
from .version import EVAL_VERSION, __version__

ANSWER_METRICS = (
    "exact_match",
    "contains_expected",
    "token_f1",
    "semantic_similarity",
    "context_relevance",
    "answer_relevance",
    "citation_coverage",
)
ALL_METRICS: tuple[str, ...] = (*RETRIEVAL_METRICS, *ANSWER_METRICS, "faithfulness")


class ConfigError(ValueError):
    """Raised for invalid experiment configurations."""


@dataclass
class ExperimentConfig:
    name: str
    dataset: str
    retriever: dict[str, Any] = field(default_factory=lambda: {"type": "bm25", "params": {}})
    generator: dict[str, Any] = field(default_factory=lambda: {"type": "extractive", "params": {}})
    top_k: int = 5
    k_values: list[int] | None = None
    metrics: list[str] | None = None
    seed: int = 0
    description: str = ""
    base_dir: str | None = None  # directory the dataset path is resolved against; not serialized

    def __post_init__(self) -> None:
        if not self.name or not re.fullmatch(r"[A-Za-z0-9._\- ]+", self.name):
            raise ConfigError("name must be non-empty and use letters, digits, space, . _ -")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ConfigError("top_k must be a positive integer")
        for key in ("retriever", "generator"):
            spec = getattr(self, key)
            if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
                raise ConfigError(f"{key} must be an object with a string 'type'")
            if not isinstance(spec.get("params", {}), dict):
                raise ConfigError(f"{key}.params must be an object")
        if self.k_values is None:
            self.k_values = sorted({k for k in (1, 3, 5, self.top_k) if k <= self.top_k})
        if not self.k_values or any(
            isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > self.top_k
            for k in self.k_values
        ):
            raise ConfigError(f"k_values must be integers in [1, top_k={self.top_k}]")
        self.k_values = sorted(set(self.k_values))
        if self.metrics is not None:
            bad = [m for m in self.metrics if m not in ALL_METRICS]
            if bad or not self.metrics:
                raise ConfigError(f"unknown metrics {bad}; available: {list(ALL_METRICS)}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        known = set(cls.__dataclass_fields__) - {"base_dir"}
        extra = set(data) - known
        if extra:
            raise ConfigError(f"unknown config keys: {sorted(extra)}")
        missing = [k for k in ("name", "dataset") if k not in data]
        if missing:
            raise ConfigError(f"missing required config keys: {missing}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        p = Path(path)
        try:
            raw = p.read_bytes()
            data = tomllib.loads(raw.decode()) if p.suffix == ".toml" else json.loads(raw)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read config {p}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("config must be an object")
        cfg = cls.from_dict(data)
        cfg.base_dir = str(p.parent.resolve())
        return cfg

    @property
    def dataset_path(self) -> Path:
        """Dataset location: relative paths resolve against the config file, then the CWD."""
        ds = Path(self.dataset)
        if not ds.is_absolute() and self.base_dir and (Path(self.base_dir) / ds).exists():
            return (Path(self.base_dir) / ds).resolve()
        return ds

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "dataset": self.dataset,
            "retriever": {
                "type": self.retriever["type"],
                "params": self.retriever.get("params", {}),
            },
            "generator": {
                "type": self.generator["type"],
                "params": self.generator.get("params", {}),
            },
            "top_k": self.top_k,
            "k_values": self.k_values,
            "metrics": self.metrics,
            "seed": self.seed,
        }

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class RunResult:
    manifest: dict[str, Any]
    config: dict[str, Any]
    summary: dict[str, Any]
    samples: list[dict[str, Any]]
    dataset_quality: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def metric_means(self) -> dict[str, float | None]:
        return {k: v["mean"] for k, v in self.summary["metrics"].items()}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "run"


def _select(names: Sequence[str] | None) -> tuple[list[str], list[str], bool]:
    chosen = list(names) if names is not None else list(ALL_METRICS)
    ret = [n for n in chosen if n in RETRIEVAL_METRICS]
    ans = [n for n in chosen if n in ANSWER_METRICS]
    return ret, ans, "faithfulness" in chosen


def evaluate_sample(
    sample: Sample,
    retrieved: list[Any],
    answer: str,
    *,
    k_values: Sequence[int],
    metric_names: Sequence[str] | None = None,
    embedder: Embedder | None = None,
) -> tuple[dict[str, MetricResult], list[dict[str, Any]]]:
    """Score one sample; returns ``(metrics_by_name, failures)``."""
    ret_names, ans_names, do_faith = _select(metric_names)
    results: list[MetricResult] = []
    results += compute_retrieval_metrics(
        [r.doc_id for r in retrieved], sample.relevant_docs, k_values, ret_names
    )
    if ans_names:
        results += compute_answer_metrics(
            sample.question,
            answer,
            sample.expected_answer,
            retrieved,
            sample.relevant_docs,
            ans_names,
            embedder,
        )
    if do_faith:
        results += faithfulness(answer, retrieved)
    by_name = {r.metric: r for r in results}
    # Failure analysis always needs grounding and reference-overlap evidence, even
    # when those metrics were not requested for reporting.
    analysis = {k: v.to_dict() for k, v in by_name.items()}
    for r in faithfulness(answer, retrieved):
        analysis.setdefault(r.metric, r.to_dict())
    analysis.setdefault("token_f1", token_overlap(answer, sample.expected_answer).to_dict())
    analysis.setdefault(
        "contains_expected", contains_expected(answer, sample.expected_answer).to_dict()
    )
    fails = failure_mod.analyze_sample(
        retrieved=[{"doc_id": r.doc_id} for r in retrieved],
        relevant=sample.relevant_docs,
        metrics=analysis,
        answer=answer,
        abstained=is_abstention(answer),
    )
    return by_name, fails


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    names: list[str] = []
    for s in samples:
        for n in s["metrics"]:
            if n not in names:
                names.append(n)
    out: dict[str, Any] = {}
    for n in names:
        vals = [s["metrics"][n]["score"] for s in samples if s["metrics"][n]["score"] is not None]
        first = next(s["metrics"][n] for s in samples if n in s["metrics"])
        mean = sum(vals) / len(vals) if vals else None
        var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals and mean is not None else 0.0
        out[n] = {
            "mean": mean,
            "std": var**0.5 if vals else None,
            "n": len(vals),
            "n_total": len(samples),
            "not_applicable": len(samples) - len(vals),
            "method": first["method"],
            "heuristic": first["heuristic"],
        }
    return out


def run_experiment(
    config: ExperimentConfig,
    *,
    dataset: Dataset | None = None,
    retriever: Retriever | None = None,
    generator: Generator | None = None,
    embedder: Embedder | None = None,
    created_at: datetime | None = None,
) -> RunResult:
    """Execute an experiment end to end (dataset -> retrieval -> generation -> metrics)."""
    started = time.perf_counter()
    ds = dataset or load_dataset(config.dataset_path)
    ret = retriever or build_retriever(config.retriever["type"], config.retriever.get("params"))
    gen = generator or build_generator(config.generator["type"], config.generator.get("params"))
    emb = embedder or CharNgramEmbedder()
    ret.index(ds.documents)
    assert config.k_values is not None

    rows: list[dict[str, Any]] = []
    for s in ds.samples:
        retrieved = ret.retrieve(s.question, config.top_k)
        generation = gen.generate(s.question, retrieved)
        by_name, fails = evaluate_sample(
            s,
            retrieved,
            generation.answer,
            k_values=config.k_values,
            metric_names=config.metrics,
            embedder=emb,
        )
        rel = {d for d, g in s.relevant_docs.items() if g > 0}
        rows.append(
            {
                "sample_id": s.id,
                "question": s.question,
                "expected_answer": s.expected_answer,
                "relevant_docs": sorted(rel),
                "metadata": s.metadata,
                "retrieved": [
                    {
                        "doc_id": r.doc_id,
                        "rank": r.rank,
                        "score": round(r.score, 6),
                        "relevant": r.doc_id in rel,
                        "text": r.text,
                    }
                    for r in retrieved
                ],
                "answer": generation.answer,
                "metrics": {k: v.to_dict() for k, v in by_name.items()},
                "failures": fails,
            }
        )

    now = created_at or datetime.now(UTC)
    manifest = {
        "run_id": f"{_slug(config.name)}-{now.strftime('%Y%m%dT%H%M%SZ')}-{config.config_hash[:6]}",
        "name": config.name,
        "created_at": now.isoformat(),
        "eval_version": EVAL_VERSION,
        "framework_version": __version__,
        "config_hash": config.config_hash,
        "dataset": {
            "name": ds.name,
            "version": ds.version,
            "corpus_hash": ds.corpus_hash,
            "samples_hash": ds.samples_hash,
            "num_samples": len(ds.samples),
            "num_documents": len(ds.documents),
        },
        "retriever": {"type": config.retriever["type"], "name": ret.name, "params": ret.params},
        "generator": {
            "type": config.generator["type"],
            "model_id": gen.model_id,
            "params": gen.params,
        },
        "embedder": emb.name,
        "top_k": config.top_k,
        "k_values": config.k_values,
        "metrics": config.metrics or list(ALL_METRICS),
        "seed": config.seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    summary = {
        "num_samples": len(rows),
        "metrics": _aggregate(rows),
        "failure_counts": failure_mod.summarize_failures(rows),
    }
    return RunResult(manifest, config.to_dict(), summary, rows, dataset_quality(ds))
