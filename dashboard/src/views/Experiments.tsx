import { fmt, shortId } from "../lib";
import type { Run } from "../types";

const HEADLINE = ["recall@3", "mrr@3", "ndcg@3", "token_f1", "faithfulness"];

export function Experiments({ runs, selected, onSelect }: { runs: Run[]; selected: string; onSelect: (id: string) => void }) {
  return (
    <div>
      <h2>Experiments</h2>
      <p className="muted">All values are measured from stored runs. Click a row to inspect it in the other tabs.</p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run</th><th>Retriever</th><th>Model</th><th className="num">top_k</th>
              {HEADLINE.map((h) => <th key={h} className="num">{h}</th>)}
              <th>Dataset</th><th>Created (UTC)</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.manifest.run_id} className={"click" + (r.manifest.run_id === selected ? " sel" : "")} onClick={() => onSelect(r.manifest.run_id)}>
                <td title={r.manifest.run_id}>{shortId(r.manifest.run_id)}</td>
                <td>{r.manifest.retriever.name}</td>
                <td>{r.manifest.generator.model_id}</td>
                <td className="num">{r.manifest.top_k}</td>
                {HEADLINE.map((h) => <td key={h} className="num">{fmt(r.summary.metrics[h]?.mean)}</td>)}
                <td className="mono">{r.manifest.dataset.version}</td>
                <td>{r.manifest.created_at.replace("T", " ").slice(0, 19)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted">Faithfulness is a heuristic lexical-grounding score, not a measure of truth.</p>
    </div>
  );
}

export function Manifest({ run }: { run: Run }) {
  const m = run.manifest;
  const rows: [string, string][] = [
    ["Run ID", m.run_id],
    ["Created", m.created_at],
    ["Dataset", `${m.dataset.name} @ ${m.dataset.version} (${m.dataset.num_samples} samples, ${m.dataset.num_documents} docs)`],
    ["Retriever", `${m.retriever.name} ${JSON.stringify(m.retriever.params)}`],
    ["Generator / model", `${m.generator.model_id} ${JSON.stringify(m.generator.params)}`],
    ["Embedder", m.embedder],
    ["Evaluation version", m.eval_version],
    ["Framework version", m.framework_version],
    ["Config hash", m.config_hash],
    ["Seed", String(m.seed)],
    ["Python / platform", `${m.python} / ${m.platform}`],
  ];
  return (
    <div className="card">
      <h3>Reproducibility manifest</h3>
      <table><tbody>{rows.map(([k, v]) => <tr key={k}><th>{k}</th><td className="mono">{v}</td></tr>)}</tbody></table>
    </div>
  );
}
