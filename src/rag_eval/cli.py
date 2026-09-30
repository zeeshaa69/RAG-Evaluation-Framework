"""Command line interface: ``rag-eval <group> <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .compare import compare_runs
from .datasets import DatasetError, dataset_quality, inspect_dataset, load_dataset
from .experiment import ConfigError, ExperimentConfig, run_experiment
from .report import comparison_text, run_html, run_markdown
from .store import ResultStore, RunNotFoundError

DEFAULT_RESULTS = "runs"


def _add_results(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--results-dir", default=DEFAULT_RESULTS, help="where runs are stored (default: runs)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval", description="Evaluate RAG systems reproducibly."
    )
    groups = parser.add_subparsers(dest="group", required=True)

    ds = groups.add_parser("dataset", help="dataset tools").add_subparsers(
        dest="cmd", required=True
    )
    v = ds.add_parser("validate", help="validate a dataset (exit 1 on errors)")
    v.add_argument("path")
    v.add_argument("--strict", action="store_true", help="treat warnings as errors")
    v.add_argument("--json", action="store_true")
    q = ds.add_parser("quality", help="print dataset statistics and label-quality signals")
    q.add_argument("path")

    ex = groups.add_parser("experiment", help="run and compare experiments").add_subparsers(
        dest="cmd", required=True
    )
    r = ex.add_parser("run", help="run an experiment config (.json or .toml)")
    r.add_argument("config")
    _add_results(r)
    r.add_argument("--no-save", action="store_true")
    ls = ex.add_parser("list", help="list stored runs")
    _add_results(ls)
    c = ex.add_parser("compare", help="compare two runs (id, id prefix, or experiment name)")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--format", choices=["text", "md", "json"], default="text")
    c.add_argument("--seed", type=int, default=0)
    _add_results(c)

    rp = groups.add_parser("report", help="reports").add_subparsers(dest="cmd", required=True)
    g = rp.add_parser("generate", help="render a stored run as Markdown or HTML")
    g.add_argument("run")
    g.add_argument("--format", choices=["md", "html"], default="md")
    g.add_argument("--out", help="output file (default: stdout)")
    _add_results(g)

    dash = groups.add_parser("dashboard", help="web dashboard").add_subparsers(
        dest="cmd", required=True
    )
    s = dash.add_parser("serve", help="serve the built dashboard with live results")
    s.add_argument("--dashboard-dir", default="dashboard/dist")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    _add_results(s)
    e = dash.add_parser("export", help="write a static dashboard with a frozen snapshot of results")
    e.add_argument("out")
    e.add_argument("--dashboard-dir", default="dashboard/dist")
    _add_results(e)
    return parser


def _cmd_dataset_validate(args: argparse.Namespace) -> int:
    ds, report = inspect_dataset(args.path)
    failed = bool(report.errors) or (args.strict and bool(report.warnings))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for i in report.issues:
            print(f"{i.severity.upper():7} {i.code:22} {i.location:24} {i.message}")
        if ds is not None:
            print(
                f"\n{len(ds.samples)} samples, {len(ds.documents)} documents, version {ds.version}"
            )
        print(
            f"{len(report.errors)} error(s), {len(report.warnings)} warning(s): "
            f"{'FAILED' if failed else 'OK'}"
        )
    return 1 if failed else 0


def _cmd_dataset_quality(args: argparse.Namespace) -> int:
    print(json.dumps(dataset_quality(load_dataset(args.path)), indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig.from_file(args.config)
    result = run_experiment(cfg)
    if not args.no_save:
        path = ResultStore(args.results_dir).save(result)
        print(f"Saved run to {path}")
    m = result.manifest
    print(
        f"Run {result.run_id}: dataset {m['dataset']['name']}@{m['dataset']['version']}, "
        f"{result.summary['num_samples']} samples, {m['duration_seconds']}s"
    )
    for name, s in result.summary["metrics"].items():
        mean = "n/a" if s["mean"] is None else f"{s['mean']:.4f}"
        print(f"  {name:22} {mean:>8}  (n={s['n']}/{s['n_total']}){' *' if s['heuristic'] else ''}")
    print("  * heuristic metric")
    shown = {k: v for k, v in result.summary["failure_counts"].items() if v}
    print(f"Failures (samples per category): {shown or 'none'}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    for m in ResultStore(args.results_dir).list_manifests():
        print(
            f"{m['run_id']}  {m['created_at']}  dataset={m['dataset']['version']}  "
            f"model={m['generator']['model_id']}"
        )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    store = ResultStore(args.results_dir)
    cmp = compare_runs(store.load(args.a), store.load(args.b), seed=args.seed)
    if args.format == "json":
        print(json.dumps(cmp, indent=2))
    else:
        print(comparison_text(cmp, markdown=args.format == "md"), end="")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    run = ResultStore(args.results_dir).load(args.run)
    text = run_html(run) if args.format == "html" else run_markdown(run)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .serve import make_server

    server = make_server(args.results_dir, args.dashboard_dir, args.host, args.port)
    print(f"Dashboard on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .serve import export_static

    print(
        f"Wrote static dashboard to {export_static(args.results_dir, args.dashboard_dir, args.out)}"
    )
    return 0


_DISPATCH = {
    ("dataset", "validate"): _cmd_dataset_validate,
    ("dataset", "quality"): _cmd_dataset_quality,
    ("experiment", "run"): _cmd_run,
    ("experiment", "list"): _cmd_list,
    ("experiment", "compare"): _cmd_compare,
    ("report", "generate"): _cmd_report,
    ("dashboard", "serve"): _cmd_serve,
    ("dashboard", "export"): _cmd_export,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[(args.group, args.cmd)](args)
    except (
        DatasetError,
        ConfigError,
        RunNotFoundError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
