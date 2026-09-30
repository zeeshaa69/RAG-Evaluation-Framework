from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from rag_eval.cli import main
from rag_eval.serve import export_static, make_server
from tests.conftest import DEMO, write_jsonl

CFG = DEMO / "experiments" / "bm25-extractive.json"
CFG2 = DEMO / "experiments" / "first-k-baseline.json"


def test_dataset_validate_ok_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["dataset", "validate", str(DEMO / "dataset")]) == 0
    assert "0 error(s)" in capsys.readouterr().out
    assert main(["dataset", "validate", str(DEMO / "dataset"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_dataset_validate_strict_fails_on_warnings() -> None:
    # The demo dataset intentionally contains unanswerable questions (warnings only).
    assert main(["dataset", "validate", str(DEMO / "dataset"), "--strict"]) == 1


def test_dataset_validate_malformed_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = tmp_path / "bad"
    d.mkdir()
    write_jsonl(d / "corpus.jsonl", [{"id": "a", "text": "x"}])
    (d / "samples.jsonl").write_text("{broken\n")
    assert main(["dataset", "validate", str(d)]) == 1
    assert "malformed_json" in capsys.readouterr().out


def test_dataset_quality(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["dataset", "quality", str(DEMO / "dataset")]) == 0
    assert json.loads(capsys.readouterr().out)["num_documents"] == 32


def test_run_list_compare_report_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rd_ = str(tmp_path / "runs")
    assert main(["experiment", "run", str(CFG), "--results-dir", rd_]) == 0
    out = capsys.readouterr().out
    assert "recall@3" in out and "heuristic metric" in out
    assert main(["experiment", "run", str(CFG2), "--results-dir", rd_]) == 0
    capsys.readouterr()

    assert main(["experiment", "list", "--results-dir", rd_]) == 0
    listing = capsys.readouterr().out
    assert "bm25-extractive" in listing and "first-k-baseline" in listing

    assert (
        main(["experiment", "compare", "bm25-extractive", "first-k-baseline", "--results-dir", rd_])
        == 0
    )
    text = capsys.readouterr().out
    assert "recall@3" in text and "Delta" in text

    assert (
        main(
            [
                "experiment",
                "compare",
                "bm25-extractive",
                "first-k-baseline",
                "--results-dir",
                rd_,
                "--format",
                "json",
            ]
        )
        == 0
    )
    cmp = json.loads(capsys.readouterr().out)
    assert any(r["metric"] == "recall@3" and r["delta"] < 0 for r in cmp["metrics"])

    assert (
        main(
            [
                "experiment",
                "compare",
                "bm25-extractive",
                "first-k-baseline",
                "--results-dir",
                rd_,
                "--format",
                "md",
            ]
        )
        == 0
    )
    assert "| Metric |" in capsys.readouterr().out

    out_file = tmp_path / "r.html"
    assert (
        main(
            [
                "report",
                "generate",
                "bm25-extractive",
                "--format",
                "html",
                "--out",
                str(out_file),
                "--results-dir",
                rd_,
            ]
        )
        == 0
    )
    assert "<table>" in out_file.read_text()
    assert main(["report", "generate", "bm25-extractive", "--results-dir", rd_]) == 0
    assert "# RAG evaluation report" in capsys.readouterr().out


def test_no_save_flag(tmp_path: Path) -> None:
    rd_ = tmp_path / "runs"
    assert main(["experiment", "run", str(CFG), "--results-dir", str(rd_), "--no-save"]) == 0
    assert not rd_.exists()


def test_errors_return_exit_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rd_ = str(tmp_path)
    assert main(["experiment", "compare", "a", "b", "--results-dir", rd_]) == 2
    assert main(["experiment", "run", str(tmp_path / "nope.json"), "--results-dir", rd_]) == 2
    assert main(["report", "generate", "zzz", "--results-dir", rd_]) == 2
    assert main(["dataset", "quality", str(tmp_path / "nope")]) == 2
    err = capsys.readouterr().err
    assert err.count("error:") == 4


def test_dashboard_needs_build(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "dashboard",
                "export",
                str(tmp_path / "o"),
                "--dashboard-dir",
                str(tmp_path / "missing"),
            ]
        )
        == 2
    )
    assert "index.html not found" in capsys.readouterr().err


def test_server_serves_bundle_and_static_files(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    assert main(["experiment", "run", str(CFG), "--results-dir", str(runs)]) == 0
    dash = tmp_path / "dist"
    dash.mkdir()
    (dash / "index.html").write_text("<html>ok</html>")
    server = make_server(runs, dash, port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/api/bundle.json") as r:
            bundle = json.loads(r.read())
        with urllib.request.urlopen(base + "/") as r:
            assert b"ok" in r.read()
    finally:
        server.shutdown()
        server.server_close()
    assert len(bundle["runs"]) == 1
    assert bundle["runs"][0]["manifest"]["name"] == "bm25-extractive"


def test_export_static(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    main(["experiment", "run", str(CFG), "--results-dir", str(runs)])
    dash = tmp_path / "dist"
    dash.mkdir()
    (dash / "index.html").write_text("x")
    out = export_static(runs, dash, tmp_path / "site")
    assert json.loads((out / "api" / "bundle.json").read_text())["runs"]
    with pytest.raises(FileNotFoundError):
        make_server(runs, tmp_path / "absent")


def test_python_api_example_runs(tmp_path: Path) -> None:
    import runpy

    ns = runpy.run_path(str(DEMO.parent / "python_api.py"))
    ns["main"](tmp_path / "runs")
    assert len(list((tmp_path / "runs").iterdir())) == 2
