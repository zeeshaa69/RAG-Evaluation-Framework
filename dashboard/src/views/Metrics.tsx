import { HBarChart, LineChart } from "../charts";
import { fmt, metricFamilies, scalarMetrics } from "../lib";
import type { Run } from "../types";
import { Manifest } from "./Experiments";

export function Metrics({ run }: { run: Run }) {
  const fam = metricFamilies(run.summary.metrics);
  const scalars = scalarMetrics(run.summary.metrics);
  const retrievalFamilies = Object.keys(fam);
  return (
    <div>
      <h2>Metrics · {run.manifest.name}</h2>
      <div className="grid">
        <div className="card">
          <h3>Retrieval metrics by cut-off k (higher is better)</h3>
          <LineChart
            title="Retrieval metrics by k"
            series={retrievalFamilies.map((f) => ({ name: f, points: (fam[f] ?? []).map((p) => ({ k: p.k, v: p.mean })) }))}
          />
        </div>
        <div className="card">
          <h3>Answer &amp; grounding metrics (hatched = heuristic)</h3>
          <HBarChart
            title="Answer metrics"
            items={scalars.map(([name, s]) => ({
              label: name, value: s.mean, heuristic: s.heuristic,
              detail: `n=${s.n}/${s.n_total}`,
            }))}
          />
        </div>
        <Manifest run={run} />
      </div>
      <h2 style={{ marginTop: 16 }}>All metrics</h2>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Metric</th><th className="num">Mean</th><th className="num">Std</th><th className="num">n scored</th><th>Type</th><th>Method</th></tr></thead>
          <tbody>
            {Object.entries(run.summary.metrics).map(([name, s]) => (
              <tr key={name}>
                <td>{name}</td><td className="num">{fmt(s.mean, 4)}</td><td className="num">{fmt(s.std, 4)}</td>
                <td className="num">{s.n}/{s.n_total}</td>
                <td><span className={"badge" + (s.heuristic ? " heur" : "")}>{s.heuristic ? "heuristic" : "formula"}</span></td>
                <td className="muted">{s.method}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="note">
        Means exclude samples where a metric is not applicable (for example no expected answer or no relevance labels); the
        “n scored” column shows how many samples each mean covers.
      </div>
    </div>
  );
}
