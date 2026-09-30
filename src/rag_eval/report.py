"""Render runs and comparisons as text, Markdown or self-contained HTML."""

from __future__ import annotations

import html
from typing import Any

from .experiment import RunResult
from .failures import CATEGORIES


def _fmt(v: float | None, digits: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _manifest_rows(m: dict[str, Any]) -> list[list[str]]:
    return [
        ["Run ID", f"`{m['run_id']}`"],
        ["Created (UTC)", m["created_at"]],
        [
            "Dataset",
            f"{m['dataset']['name']} (version `{m['dataset']['version']}`, "
            f"{m['dataset']['num_samples']} samples, {m['dataset']['num_documents']} docs)",
        ],
        ["Retriever", f"{m['retriever']['name']} {m['retriever']['params']}"],
        ["Generator / model", f"{m['generator']['model_id']} {m['generator']['params']}"],
        ["Embedder", m["embedder"]],
        ["top_k / k values", f"{m['top_k']} / {m['k_values']}"],
        ["Evaluation version", m["eval_version"]],
        ["Framework version", m["framework_version"]],
        ["Config hash", f"`{m['config_hash']}`"],
        ["Python / platform", f"{m['python']} / {m['platform']}"],
    ]


def run_markdown(run: RunResult, max_failures: int = 10) -> str:
    m = run.manifest
    lines = [
        f"# RAG evaluation report: {m['name']}",
        "",
        _md_table(["Field", "Value"], _manifest_rows(m)),
        "",
        "## Metrics",
        "",
    ]
    rows = []
    for name, s in run.summary["metrics"].items():
        rows.append(
            [
                name,
                _fmt(s["mean"]),
                _fmt(s["std"]),
                f"{s['n']}/{s['n_total']}",
                "heuristic" if s["heuristic"] else "formula",
                s["method"],
            ]
        )
    lines += [
        _md_table(["Metric", "Mean", "Std", "n scored", "Type", "Method"], rows),
        "",
        "Scores are means over the samples where the metric applies. Heuristic metrics "
        "are approximations, not ground truth.",
        "",
        "## Failure analysis",
        "",
    ]
    counts = run.summary["failure_counts"]
    lines += [
        _md_table(
            ["Category", "Samples", "Meaning"],
            [[k, str(v), CATEGORIES[k]] for k, v in counts.items()],
        ),
        "",
    ]
    shown = 0
    for s in run.samples:
        for f in s["failures"]:
            if shown >= max_failures:
                break
            tag = " (heuristic)" if f["heuristic"] else ""
            lines += [
                f"### {s['sample_id']} - {f['category']}{tag}",
                "",
                f"- Question: {s['question']}",
                f"- Answer: {s['answer']}",
                f"- Expected: {s['expected_answer']}",
                f"- Evidence: `{_short(f['evidence'])}`",
                "",
            ]
            shown += 1
    if shown == 0:
        lines.append("No failures detected by the configured checks.")
    return "\n".join(lines).rstrip() + "\n"


def _short(obj: Any, limit: int = 400) -> str:
    import json

    text = json.dumps(obj, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def run_html(run: RunResult, max_failures: int = 25) -> str:
    e = html.escape
    m = run.manifest
    meta = "".join(f"<tr><th>{e(a)}</th><td>{e(b)}</td></tr>" for a, b in _manifest_rows(m))
    metrics = "".join(
        f"<tr><td>{e(n)}</td><td class=num>{_fmt(s['mean'])}</td><td class=num>{s['n']}/{s['n_total']}</td>"
        f"<td>{'heuristic' if s['heuristic'] else 'formula'}</td><td>{e(s['method'])}</td></tr>"
        for n, s in run.summary["metrics"].items()
    )
    fails = ""
    shown = 0
    for s in run.samples:
        for f in s["failures"]:
            if shown >= max_failures:
                break
            fails += (
                f"<div class=card><b>{e(s['sample_id'])}</b> &middot; {e(f['category'])}"
                f"{' (heuristic)' if f['heuristic'] else ''}<br>Q: {e(s['question'])}<br>"
                f"A: {e(s['answer'])}<br>Expected: {e(str(s['expected_answer']))}"
                f"<pre>{e(_short(f['evidence'], 600))}</pre></div>"
            )
            shown += 1
    counts = "".join(
        f"<tr><td>{e(k)}</td><td class=num>{v}</td><td>{e(CATEGORIES[k])}</td></tr>"
        for k, v in run.summary["failure_counts"].items()
    )
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>RAG evaluation: {e(m["name"])}</title><style>
body{{font:14px/1.5 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1b1f24}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}td,th{{border:1px solid #d0d7de;padding:.35rem .6rem;text-align:left;vertical-align:top}}
th{{background:#f6f8fa}}.num{{text-align:right;font-variant-numeric:tabular-nums}}.card{{border:1px solid #d0d7de;border-radius:6px;padding:.6rem .8rem;margin:.5rem 0}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:.5rem;border-radius:4px;font-size:12px}}</style></head><body>
<h1>RAG evaluation report: {e(m["name"])}</h1><table>{meta}</table>
<h2>Metrics</h2><table><tr><th>Metric</th><th>Mean</th><th>n scored</th><th>Type</th><th>Method</th></tr>{metrics}</table>
<p>Heuristic metrics are approximations, not ground truth.</p>
<h2>Failure analysis</h2><table><tr><th>Category</th><th>Samples</th><th>Meaning</th></tr>{counts}</table>{fails or "<p>No failures detected.</p>"}
</body></html>
"""


def comparison_text(cmp: dict[str, Any], markdown: bool = False) -> str:
    a, b = cmp["a"], cmp["b"]
    rows = []
    for r in cmp["metrics"]:
        ci = "n/a" if r["ci95"] is None else f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        rows.append(
            [
                r["metric"] + ("*" if r["heuristic"] else ""),
                _fmt(r["a"]),
                _fmt(r["b"]),
                "n/a" if r["delta"] is None else f"{r['delta']:+.4f}",
                f"{r['b_better']}/{r['a_better']}/{r['ties']}",
                ci,
            ]
        )
    headers = [
        "Metric",
        a["name"],
        b["name"],
        "Delta (B-A)",
        "B better/A better/ties",
        "95% CI of delta",
    ]
    lines = [f"Comparison: A = {a['run_id']}  vs  B = {b['run_id']}", ""]
    for w in cmp["warnings"]:
        lines.append(f"WARNING: {w}")
    if cmp["warnings"]:
        lines.append("")
    if markdown:
        lines.append(_md_table(headers, rows))
    else:
        widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows, strict=True)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        lines += [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
        lines += [fmt.format(*r) for r in rows]
    lines += ["", "* = heuristic metric", "", "Failure counts (A / B):"]
    lines += [f"  {k}: {v[0]} / {v[1]}" for k, v in cmp["failure_counts"].items()]
    lines += [""] + [f"Note: {n}" for n in cmp["notes"]]
    return "\n".join(lines) + "\n"
