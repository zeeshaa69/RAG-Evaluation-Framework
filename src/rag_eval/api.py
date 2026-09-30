"""High-level SDK facade.

from rag_eval import RagEval
ev = RagEval(results_dir="runs")
run = ev.run("examples/demo/experiments/bm25.json")
print(ev.compare(run.run_id, other.run_id))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .compare import compare_runs
from .datasets import Dataset, ValidationReport, dataset_quality, inspect_dataset, load_dataset
from .experiment import ExperimentConfig, RunResult, run_experiment
from .generators import Generator
from .metrics.embeddings import Embedder
from .report import comparison_text, run_html, run_markdown
from .retrievers import Retriever
from .store import ResultStore


class RagEval:
    def __init__(self, results_dir: str | Path = "runs") -> None:
        self.store = ResultStore(results_dir)

    # datasets
    @staticmethod
    def load_dataset(path: str | Path) -> Dataset:
        return load_dataset(path)

    @staticmethod
    def validate_dataset(path: str | Path) -> ValidationReport:
        return inspect_dataset(path)[1]

    @staticmethod
    def dataset_quality(path: str | Path) -> dict[str, Any]:
        return dataset_quality(load_dataset(path))

    # experiments
    def run(
        self,
        config: str | Path | ExperimentConfig | dict[str, Any],
        *,
        dataset: Dataset | None = None,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        embedder: Embedder | None = None,
        save: bool = True,
    ) -> RunResult:
        """Run an experiment from a config file, dict or object.

        Pass ``retriever``/``generator`` objects to evaluate your own components;
        the config then only supplies names, ``top_k``, metrics and the dataset.
        """
        if isinstance(config, (str, Path)):
            cfg = ExperimentConfig.from_file(config)
        elif isinstance(config, dict):
            cfg = ExperimentConfig.from_dict(config)
        else:
            cfg = config
        result = run_experiment(
            cfg, dataset=dataset, retriever=retriever, generator=generator, embedder=embedder
        )
        if save:
            self.store.save(result)
        return result

    def load_run(self, ref: str) -> RunResult:
        return self.store.load(ref)

    def list_runs(self) -> list[dict[str, Any]]:
        return self.store.list_manifests()

    def compare(self, a: str | RunResult, b: str | RunResult, *, seed: int = 0) -> dict[str, Any]:
        ra = self.store.load(a) if isinstance(a, str) else a
        rb = self.store.load(b) if isinstance(b, str) else b
        return compare_runs(ra, rb, seed=seed)

    # reports
    def report(self, run: str | RunResult, fmt: str = "md") -> str:
        r = self.store.load(run) if isinstance(run, str) else run
        if fmt == "md":
            return run_markdown(r)
        if fmt == "html":
            return run_html(r)
        raise ValueError("fmt must be 'md' or 'html'")

    @staticmethod
    def format_comparison(cmp: dict[str, Any], markdown: bool = False) -> str:
        return comparison_text(cmp, markdown)
