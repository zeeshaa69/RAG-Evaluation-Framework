"""rag-eval: reproducible, offline-first evaluation of RAG systems."""

from .api import RagEval
from .compare import compare_runs
from .datasets import Dataset, DatasetError, dataset_quality, inspect_dataset, load_dataset
from .experiment import ConfigError, ExperimentConfig, RunResult, run_experiment
from .generators import (
    CallableGenerator,
    ExtractiveGenerator,
    Generator,
    OllamaGenerator,
    build_generator,
    register_generator,
)
from .retrievers import Retriever, build_retriever, register_retriever
from .store import ResultStore
from .types import Document, Generation, MetricResult, RetrievedDoc, Sample
from .version import EVAL_VERSION, __version__

__all__ = [
    "EVAL_VERSION",
    "CallableGenerator",
    "ConfigError",
    "Dataset",
    "DatasetError",
    "Document",
    "ExperimentConfig",
    "ExtractiveGenerator",
    "Generation",
    "Generator",
    "MetricResult",
    "OllamaGenerator",
    "RagEval",
    "ResultStore",
    "RetrievedDoc",
    "Retriever",
    "RunResult",
    "Sample",
    "__version__",
    "build_generator",
    "build_retriever",
    "compare_runs",
    "dataset_quality",
    "inspect_dataset",
    "load_dataset",
    "register_generator",
    "register_retriever",
    "run_experiment",
]
