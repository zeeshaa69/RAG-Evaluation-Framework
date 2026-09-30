"""Compare two runs using their measured values only."""

from __future__ import annotations

import random
from typing import Any

from .experiment import RunResult

TIE_EPS = 1e-9


def _paired(a: RunResult, b: RunResult, metric: str) -> list[tuple[float, float]]:
    bmap = {s["sample_id"]: s for s in b.samples}
    pairs = []
    for s in a.samples:
        o = bmap.get(s["sample_id"])
        if o is None:
            continue
        x, y = s["metrics"].get(metric), o["metrics"].get(metric)
        if x and y and x["score"] is not None and y["score"] is not None:
            pairs.append((float(x["score"]), float(y["score"])))
    return pairs


def bootstrap_ci(
    diffs: list[float], *, seed: int = 0, n_boot: int = 2000
) -> tuple[float, float] | None:
    """Seeded percentile bootstrap 95% CI for the mean of paired differences."""
    if len(diffs) < 2:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def compare_runs(
    a: RunResult, b: RunResult, *, seed: int = 0, n_boot: int = 2000
) -> dict[str, Any]:
    warnings: list[str] = []
    da, db = a.manifest["dataset"], b.manifest["dataset"]
    if da["version"] != db["version"]:
        warnings.append(
            f"dataset versions differ ({da['version']} vs {db['version']}); "
            "differences may come from the data, not the system"
        )
    if a.manifest["eval_version"] != b.manifest["eval_version"]:
        warnings.append(
            f"evaluation versions differ ({a.manifest['eval_version']} vs "
            f"{b.manifest['eval_version']}); metric definitions may have changed"
        )
    ids_a, ids_b = {s["sample_id"] for s in a.samples}, {s["sample_id"] for s in b.samples}
    if ids_a != ids_b:
        warnings.append(
            f"sample sets differ: {len(ids_a & ids_b)} shared of {len(ids_a)} / {len(ids_b)}"
        )

    ma, mb = a.summary["metrics"], b.summary["metrics"]
    rows = []
    for name in ma:
        if name not in mb:
            continue
        pairs = _paired(a, b, name)
        diffs = [y - x for x, y in pairs]
        wins = sum(1 for d in diffs if d > TIE_EPS)
        losses = sum(1 for d in diffs if d < -TIE_EPS)
        ci = bootstrap_ci(diffs, seed=seed, n_boot=n_boot)
        va, vb = ma[name]["mean"], mb[name]["mean"]
        rows.append(
            {
                "metric": name,
                "a": va,
                "b": vb,
                "delta": None if va is None or vb is None else vb - va,
                "paired_n": len(pairs),
                "b_better": wins,
                "a_better": losses,
                "ties": len(diffs) - wins - losses,
                "ci95": list(ci) if ci else None,
                "ci_excludes_zero": bool(ci and (ci[0] > 0 or ci[1] < 0)),
                "heuristic": bool(ma[name]["heuristic"] or mb[name]["heuristic"]),
            }
        )
    return {
        "a": {"run_id": a.run_id, "name": a.manifest["name"], "manifest": a.manifest},
        "b": {"run_id": b.run_id, "name": b.manifest["name"], "manifest": b.manifest},
        "warnings": warnings,
        "metrics": rows,
        "metrics_only_in_a": sorted(set(ma) - set(mb)),
        "metrics_only_in_b": sorted(set(mb) - set(ma)),
        "failure_counts": {
            k: [a.summary["failure_counts"].get(k, 0), b.summary["failure_counts"].get(k, 0)]
            for k in a.summary["failure_counts"]
        },
        "notes": [
            "Deltas are b - a. CI is a seeded paired bootstrap over samples; "
            "with few samples it is wide and unreliable.",
            "Heuristic metrics are approximations; treat their deltas as indicative only.",
        ],
    }
