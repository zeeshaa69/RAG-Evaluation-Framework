"""On-disk storage of experiment runs.

Layout (one directory per run)::

    runs/<run_id>/
      manifest.json         reproducibility record (versions, hashes, timestamp, ...)
      config.json           the experiment configuration as executed
      summary.json          aggregate metrics and failure counts
      samples.jsonl         per-sample retrieval, answer, metrics with evidence, failures
      dataset_quality.json  dataset statistics at run time
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .experiment import RunResult


class RunNotFoundError(KeyError):
    pass


class ResultStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def save(self, run: RunResult) -> Path:
        run_id = run.run_id
        path = self.root / run_id
        n = 2
        while path.exists():  # never overwrite an earlier result
            run.manifest["run_id"] = f"{run_id}-{n}"
            path = self.root / run.manifest["run_id"]
            n += 1
        path.mkdir(parents=True)
        _dump(path / "manifest.json", run.manifest)
        _dump(path / "config.json", run.config)
        _dump(path / "summary.json", run.summary)
        _dump(path / "dataset_quality.json", run.dataset_quality)
        with (path / "samples.jsonl").open("w", encoding="utf-8") as fh:
            for row in run.samples:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def list_manifests(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            m = d / "manifest.json"
            if m.is_file():
                out.append(json.loads(m.read_text(encoding="utf-8")))
        return sorted(out, key=lambda m: (m["created_at"], m["run_id"]))

    def resolve(self, ref: str) -> str:
        """Resolve an exact id, a unique id prefix, or an experiment name (latest run)."""
        manifests = self.list_manifests()
        ids = [m["run_id"] for m in manifests]
        if ref in ids:
            return str(ref)
        prefix = [i for i in ids if i.startswith(ref)]
        if len(prefix) == 1:
            return str(prefix[0])
        if len(prefix) > 1:
            by_name = [m["run_id"] for m in manifests if m["name"] == ref]
            if by_name:
                return str(by_name[-1])
            raise RunNotFoundError(f"ambiguous run reference {ref!r}: {prefix}")
        by_name = [m["run_id"] for m in manifests if m["name"] == ref]
        if by_name:
            return str(by_name[-1])
        raise RunNotFoundError(f"no run matches {ref!r} in {self.root}")

    def load(self, ref: str) -> RunResult:
        run_id = self.resolve(ref)
        d = self.root / run_id
        rows = [
            json.loads(line)
            for line in (d / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        quality = d / "dataset_quality.json"
        return RunResult(
            manifest=_load(d / "manifest.json"),
            config=_load(d / "config.json"),
            summary=_load(d / "summary.json"),
            samples=rows,
            dataset_quality=_load(quality) if quality.exists() else {},
        )

    def bundle(self) -> dict[str, Any]:
        """Everything the dashboard needs, as one JSON-serializable object."""
        runs = []
        for m in self.list_manifests():
            r = self.load(m["run_id"])
            runs.append(
                {
                    "manifest": r.manifest,
                    "config": r.config,
                    "summary": r.summary,
                    "samples": r.samples,
                    "dataset_quality": r.dataset_quality,
                }
            )
        return {"generated_at": datetime.now(UTC).isoformat(), "runs": runs}


def _dump(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
