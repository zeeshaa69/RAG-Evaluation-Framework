import { useState } from "react";
import { PairedBars } from "../charts";
import { compareRuns, fmt, signed } from "../lib";
import type { Run } from "../types";

export function Compare({ runs, initialA }: { runs: Run[]; initialA: string }) {
  const ids = runs.map((r) => r.manifest.run_id);
  const [a, setA] = useState(initialA);
  const [b, setB] = useState(ids.find((i) => i !== initialA) ?? initialA);
  const ra = runs.find((r) => r.manifest.run_id === a);
  const rb = runs.find((r) => r.manifest.run_id === b);
  if (runs.length < 2 || !ra || !rb) return <div className="empty">Comparison needs at least two stored runs.</div>;
  const deltas = compareRuns(ra, rb);
  const warnings: string[] = [];
  if (ra.manifest.dataset.version !== rb.manifest.dataset.version) warnings.push("Dataset versions differ; differences may come from the data.");
  if (ra.manifest.eval_version !== rb.manifest.eval_version) warnings.push("Evaluation versions differ; metric definitions may have changed.");
  const scalarOrK = (k: number) => deltas.filter((d) => d.metric.endsWith("@" + k));
  const headline = [...scalarOrK(Math.max(...ra.manifest.k_values.filter((k) => rb.manifest.k_values.includes(k)), 1)), ...deltas.filter((d) => !d.metric.includes("@"))];
  return (
    <div>
      <h2>Experiment comparison</h2>
      <div className="row">
        <label className="f">A <select value={a} onChange={(e) => setA(e.target.value)}>{ids.map((i) => <option key={i}>{i}</option>)}</select></label>
        <label className="f">B <select value={b} onChange={(e) => setB(e.target.value)}>{ids.map((i) => <option key={i}>{i}</option>)}</select></label>
      </div>
      {warnings.map((w) => <div key={w} className="note warn-box">{w}</div>)}
      <div className="grid">
        <div className="card">
          <h3>Measured values (* = heuristic)</h3>
          <PairedBars nameA={ra.manifest.name} nameB={rb.manifest.name}
            items={headline.map((d) => ({ label: d.metric, a: d.a, b: d.b, heuristic: d.heuristic }))} />
        </div>
        <div className="card">
          <h3>Failure counts (samples)</h3>
          <table>
            <thead><tr><th>Category</th><th className="num">A</th><th className="num">B</th></tr></thead>
            <tbody>
              {Object.keys(ra.summary.failure_counts).map((c) => (
                <tr key={c}><td>{c}</td><td className="num">{ra.summary.failure_counts[c]}</td><td className="num">{rb.summary.failure_counts[c] ?? 0}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <h2 style={{ marginTop: 16 }}>All metrics</h2>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Metric</th><th className="num">A</th><th className="num">B</th><th className="num">Δ (B−A)</th><th className="num">B better</th><th className="num">A better</th><th className="num">Ties</th></tr></thead>
          <tbody>
            {deltas.map((d) => (
              <tr key={d.metric}>
                <td>{d.metric}{d.heuristic ? " *" : ""}</td>
                <td className="num">{fmt(d.a, 4)}</td><td className="num">{fmt(d.b, 4)}</td>
                <td className="num">{signed(d.delta, 4)}</td>
                <td className="num">{d.bBetter}</td><td className="num">{d.aBetter}</td><td className="num">{d.ties}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="note">
        Wins/ties are per-sample paired comparisons. Differences on small datasets can be noise; use{" "}
        <span className="mono">rag-eval experiment compare</span> for bootstrap confidence intervals.
      </div>
    </div>
  );
}
