from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rag_eval import (
    CallableGenerator,
    ConfigError,
    ExperimentConfig,
    RagEval,
    ResultStore,
    build_retriever,
    compare_runs,
    load_dataset,
    run_experiment,
)
from rag_eval.report import comparison_text, run_html, run_markdown
from rag_eval.store import RunNotFoundError
from tests.conftest import DEMO

FIXED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def cfg(ds: Path, **over: Any) -> ExperimentConfig:
    base: dict[str, Any] = {"name": "t", "dataset": str(ds), "top_k": 3}
    base.update(over)
    return ExperimentConfig.from_dict(base)


def test_run_produces_metrics_and_evidence(tiny_dataset: Path) -> None:
    run = run_experiment(cfg(tiny_dataset), created_at=FIXED)
    m = run.summary["metrics"]
    assert m["hit_rate@1"]["mean"] == 1.0
    assert run.samples[0]["metrics"]["faithfulness"]["evidence"]["sentences"]
    for row in run.samples:
        for res in row["metrics"].values():
            assert {"metric", "score", "evidence", "method", "heuristic"} <= set(res)
    assert {"precision@1", "ndcg@3", "token_f1", "faithfulness", "citation_coverage"} <= set(m)


def test_manifest_records_reproducibility_fields(tiny_dataset: Path) -> None:
    ds = load_dataset(tiny_dataset)
    run = run_experiment(cfg(tiny_dataset, seed=7), created_at=FIXED)
    man = run.manifest
    assert man["created_at"] == "2026-01-02T03:04:05+00:00"
    assert man["dataset"]["version"] == ds.version
    assert man["generator"]["model_id"] == "extractive-v1"
    assert man["eval_version"] and man["framework_version"] and man["config_hash"]
    assert man["seed"] == 7 and man["top_k"] == 3 and man["retriever"]["params"]
    assert man["run_id"].startswith("t-20260102T030405Z-")


def test_runs_are_deterministic(tiny_dataset: Path) -> None:
    a = run_experiment(cfg(tiny_dataset), created_at=FIXED)
    b = run_experiment(cfg(tiny_dataset), created_at=FIXED)
    assert a.samples == b.samples and a.summary == b.summary
    assert a.manifest["config_hash"] == b.manifest["config_hash"]


def test_config_hash_changes_with_parameters(tiny_dataset: Path) -> None:
    a = cfg(tiny_dataset).config_hash
    b = cfg(tiny_dataset, top_k=2).config_hash
    assert a != b


def test_metric_selection(tiny_dataset: Path) -> None:
    run = run_experiment(cfg(tiny_dataset, metrics=["recall", "exact_match"], k_values=[1, 2]))
    assert set(run.summary["metrics"]) == {"recall@1", "recall@2", "exact_match"}
    # failure analysis still works when faithfulness was not requested
    assert "unsupported_answer" in run.summary["failure_counts"]


@pytest.mark.parametrize(
    "bad",
    [
        {"name": ""},
        {"name": "bad/name"},
        {"top_k": 0},
        {"top_k": True},
        {"k_values": [9]},
        {"k_values": []},
        {"metrics": ["nope"]},
        {"metrics": []},
        {"retriever": "bm25"},
        {"generator": {"type": 3}},
        {"retriever": {"type": "bm25", "params": []}},
    ],
)
def test_invalid_config(bad: dict[str, Any], tiny_dataset: Path) -> None:
    with pytest.raises(ConfigError):
        cfg(tiny_dataset, **bad)


def test_unknown_and_missing_config_keys() -> None:
    with pytest.raises(ConfigError, match="unknown config keys"):
        ExperimentConfig.from_dict({"name": "a", "dataset": "d", "bogus": 1})
    with pytest.raises(ConfigError, match="missing"):
        ExperimentConfig.from_dict({"name": "a"})


def test_config_files_json_toml_and_relative_dataset(tiny_dataset: Path, tmp_path: Path) -> None:
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"name": "j", "dataset": "ds", "top_k": 2}))
    assert ExperimentConfig.from_file(j).dataset_path == tiny_dataset.resolve()
    t = tmp_path / "c.toml"
    t.write_text('name = "t"\ndataset = "ds"\ntop_k = 2\n[retriever]\ntype = "tfidf"\n')
    c = ExperimentConfig.from_file(t)
    assert c.retriever["type"] == "tfidf"
    run = run_experiment(c)
    assert run.manifest["retriever"]["type"] == "tfidf"
    # the stored config keeps the portable relative path
    assert run.config["dataset"] == "ds"


def test_bad_config_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ExperimentConfig.from_file(tmp_path / "missing.json")
    bad = tmp_path / "b.json"
    bad.write_text("[1]")
    with pytest.raises(ConfigError):
        ExperimentConfig.from_file(bad)
    bad.write_text("{oops")
    with pytest.raises(ConfigError):
        ExperimentConfig.from_file(bad)


def test_bring_your_own_components(tiny_dataset: Path) -> None:
    gen = CallableGenerator(lambda q, c: "Paris [d1]", model_id="stub")
    run = run_experiment(cfg(tiny_dataset), retriever=build_retriever("tfidf"), generator=gen)
    assert run.manifest["generator"]["model_id"] == "stub"
    assert run.samples[0]["metrics"]["exact_match"]["score"] == 1.0
    assert run.samples[1]["metrics"]["exact_match"]["score"] == 0.0


def test_no_retrieval_run_reports_failures(tiny_dataset: Path) -> None:
    run = run_experiment(cfg(tiny_dataset, retriever={"type": "none"}))
    assert run.summary["failure_counts"]["empty_retrieval"] == 2
    assert run.summary["metrics"]["recall@1"]["mean"] == 0.0
    assert run.summary["metrics"]["faithfulness"]["mean"] is None  # abstentions: nothing to verify


def test_samples_without_labels_are_excluded_not_zeroed(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    (d / "corpus.jsonl").write_text(json.dumps({"id": "a", "text": "Cats purr loudly."}) + "\n")
    (d / "samples.jsonl").write_text(
        json.dumps(
            {
                "id": "s1",
                "question": "Do cats purr?",
                "expected_answer": "yes",
                "relevant_docs": ["a"],
            }
        )
        + "\n"
        + json.dumps({"id": "s2", "question": "Do dogs bark?"})
        + "\n"
    )
    run = run_experiment(cfg(d))
    r = run.summary["metrics"]["recall@1"]
    assert (r["n"], r["n_total"], r["not_applicable"]) == (1, 2, 1)
    assert run.summary["metrics"]["exact_match"]["n"] == 1


# ---- store ---------------------------------------------------------------------------------


def test_store_roundtrip_and_resolution(tiny_dataset: Path, tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "runs")
    run = run_experiment(cfg(tiny_dataset), created_at=FIXED)
    path = store.save(run)
    for name in (
        "manifest.json",
        "config.json",
        "summary.json",
        "samples.jsonl",
        "dataset_quality.json",
    ):
        assert (path / name).is_file()
    loaded = store.load(run.run_id)
    assert loaded.samples == run.samples and loaded.manifest == run.manifest
    assert store.load("t").run_id == run.run_id  # by experiment name
    assert store.load(run.run_id[:8]).run_id == run.run_id  # by prefix


def test_store_never_overwrites(tiny_dataset: Path, tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    a = run_experiment(cfg(tiny_dataset), created_at=FIXED)
    b = run_experiment(cfg(tiny_dataset), created_at=FIXED)
    store.save(a)
    store.save(b)
    assert a.run_id != b.run_id and len(store.list_manifests()) == 2


def test_store_missing_and_ambiguous(tiny_dataset: Path, tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "none")
    assert store.list_manifests() == []
    with pytest.raises(RunNotFoundError):
        store.load("x")
    s2 = ResultStore(tmp_path / "r2")
    s2.save(run_experiment(cfg(tiny_dataset, name="alpha-1"), created_at=FIXED))
    s2.save(run_experiment(cfg(tiny_dataset, name="alpha-2"), created_at=FIXED))
    with pytest.raises(RunNotFoundError, match="ambiguous"):
        s2.load("alpha")


def test_bundle_contains_runs(tiny_dataset: Path, tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    store.save(run_experiment(cfg(tiny_dataset), created_at=FIXED))
    bundle = store.bundle()
    assert len(bundle["runs"]) == 1 and bundle["runs"][0]["samples"]
    json.dumps(bundle)  # must be serializable


# ---- compare & report ---------------------------------------------------------------------


def test_compare_reports_measured_deltas(tiny_dataset: Path) -> None:
    a = run_experiment(cfg(tiny_dataset, name="good"), created_at=FIXED)
    b = run_experiment(
        cfg(tiny_dataset, name="blind", retriever={"type": "none"}), created_at=FIXED
    )
    cmp = compare_runs(a, b)
    row = next(r for r in cmp["metrics"] if r["metric"] == "recall@1")
    assert row["a"] == a.summary["metrics"]["recall@1"]["mean"]
    assert row["b"] == b.summary["metrics"]["recall@1"]["mean"]
    assert row["delta"] == pytest.approx(row["b"] - row["a"])
    assert row["a_better"] == 2 and row["b_better"] == 0 and row["paired_n"] == 2
    assert cmp["warnings"] == []
    text = comparison_text(cmp)
    assert "good" in text and "blind" in text and "heuristic" in text


def test_compare_identical_runs_all_ties(tiny_dataset: Path) -> None:
    a = run_experiment(cfg(tiny_dataset), created_at=FIXED)
    cmp = compare_runs(a, a)
    assert all(r["delta"] in (0.0, None) for r in cmp["metrics"])
    assert all(r["a_better"] == 0 and r["b_better"] == 0 for r in cmp["metrics"])


def test_compare_warns_on_dataset_mismatch(tiny_dataset: Path, tmp_path: Path) -> None:
    a = run_experiment(cfg(tiny_dataset))
    other = tmp_path / "other"
    other.mkdir()
    (other / "corpus.jsonl").write_text(json.dumps({"id": "d1", "text": "Paris is nice."}) + "\n")
    (other / "samples.jsonl").write_text(
        json.dumps(
            {"id": "z", "question": "Paris?", "expected_answer": "nice", "relevant_docs": ["d1"]}
        )
        + "\n"
    )
    b = run_experiment(cfg(other))
    w = " ".join(compare_runs(a, b)["warnings"])
    assert "dataset versions differ" in w and "sample sets differ" in w


def test_bootstrap_is_seeded() -> None:
    from rag_eval.compare import bootstrap_ci

    d = [0.1, -0.2, 0.3, 0.0, 0.4]
    assert bootstrap_ci(d, seed=1) == bootstrap_ci(d, seed=1)
    assert bootstrap_ci([1.0]) is None


def test_reports_render(tiny_dataset: Path) -> None:
    run = run_experiment(cfg(tiny_dataset, retriever={"type": "none"}), created_at=FIXED)
    md = run_markdown(run)
    assert "# RAG evaluation report" in md and "empty_retrieval" in md and "Dataset" in md
    html_doc = run_html(run)
    assert html_doc.startswith("<!doctype html>") and "<script" not in html_doc


def test_html_report_escapes_content(tmp_path: Path) -> None:
    d = tmp_path / "x"
    d.mkdir()
    (d / "corpus.jsonl").write_text(json.dumps({"id": "a", "text": "Safe text."}) + "\n")
    (d / "samples.jsonl").write_text(
        json.dumps(
            {
                "id": "s",
                "question": "<script>alert(1)</script>?",
                "expected_answer": "zzz",
                "relevant_docs": ["a"],
            }
        )
        + "\n"
    )
    run = run_experiment(cfg(d, retriever={"type": "none"}))
    assert "<script>alert" not in run_html(run)


def test_sdk_facade(tiny_dataset: Path, tmp_path: Path) -> None:
    ev = RagEval(tmp_path / "runs")
    assert ev.validate_dataset(tiny_dataset).ok
    a = ev.run({"name": "a", "dataset": str(tiny_dataset)})
    b = ev.run({"name": "b", "dataset": str(tiny_dataset), "retriever": {"type": "tfidf"}})
    assert len(ev.list_runs()) == 2
    cmp = ev.compare(a.run_id, b.run_id)
    assert "a" in RagEval.format_comparison(cmp) and ev.report(a.run_id, "html")
    assert ev.report(a, "md") and ev.dataset_quality(tiny_dataset)["num_samples"] == 2
    with pytest.raises(ValueError):
        ev.report(a, "pdf")
    ev.run({"name": "c", "dataset": str(tiny_dataset)}, save=False)
    assert len(ev.list_runs()) == 2


# ---- the bundled offline demo ---------------------------------------------------------


@pytest.fixture(scope="module")
def demo_runs() -> dict[str, Any]:
    out = {}
    for f in sorted((DEMO / "experiments").glob("*.json")):
        c = ExperimentConfig.from_file(f)
        out[c.name] = run_experiment(c, created_at=FIXED)
    return out


def test_demo_configs_all_run(demo_runs: dict[str, Any]) -> None:
    assert len(demo_runs) >= 5
    assert len({r.manifest["dataset"]["version"] for r in demo_runs.values()}) == 1


def test_demo_blind_baseline_is_worse_at_retrieval(demo_runs: dict[str, Any]) -> None:
    good = demo_runs["bm25-extractive"].summary["metrics"]["recall@3"]["mean"]
    blind = demo_runs["first-k-baseline"].summary["metrics"]["recall@3"]["mean"]
    assert good > blind


def test_demo_fault_control_is_detected(demo_runs: dict[str, Any]) -> None:
    clean = demo_runs["bm25-extractive"].summary
    faulty = demo_runs["control-number-fault"].summary
    assert clean["failure_counts"]["contradicted_answer"] == 0
    assert faulty["failure_counts"]["contradicted_answer"] > 0
    assert faulty["metrics"]["faithfulness"]["mean"] < clean["metrics"]["faithfulness"]["mean"]


def test_readme_experiment_example_is_valid() -> None:
    readme = (DEMO.parent.parent / "README.md").read_text(encoding="utf-8")
    block = readme.split("## Experiment format")[1].split("```json")[1].split("```")[0]
    cfg = ExperimentConfig.from_dict({**json.loads(block), "dataset": str(DEMO / "dataset")})
    assert cfg.k_values == [1, 3, 5] and cfg.metrics is not None and "faithfulness" in cfg.metrics
