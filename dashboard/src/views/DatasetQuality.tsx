import type { Distribution, Run } from "../types";

function dist(d?: Distribution): string {
  return d ? `min ${d.min} · median ${d.median} · mean ${d.mean} · max ${d.max}` : "n/a";
}

export function DatasetQuality({ run }: { run: Run }) {
  const q = run.dataset_quality;
  const n = q.num_samples ?? 0;
  const stats: [string, string][] = [
    ["Dataset", `${q.name ?? run.manifest.dataset.name} @ ${q.version ?? run.manifest.dataset.version}`],
    ["Samples", String(n)],
    ["Documents", String(q.num_documents ?? "n/a")],
    ["Samples with expected answer", `${q.samples_with_expected_answer ?? 0} / ${n}`],
    ["Samples with relevance labels", `${q.samples_with_relevant_docs ?? 0} / ${n}`],
    ["Samples with multiple relevant docs", String(q.samples_multi_relevant ?? 0)],
    ["Documents never labelled relevant (distractors)", String(q.documents_never_relevant ?? 0)],
    ["Relevant docs per sample", dist(q.relevant_docs_per_sample)],
    ["Document length (words)", dist(q.document_length_words)],
    ["Expected answer length (words)", dist(q.answer_length_words)],
    ["Mean lexical answer support in labelled docs", q.answer_support_mean === null || q.answer_support_mean === undefined ? "n/a" : q.answer_support_mean.toFixed(3)],
  ];
  return (
    <div>
      <h2>Dataset quality</h2>
      <div className="grid">
        <div className="card">
          <h3>Statistics</h3>
          <table><tbody>{stats.map(([k, v]) => <tr key={k}><th>{k}</th><td>{v}</td></tr>)}</tbody></table>
        </div>
        <div className="card">
          <h3>Label-quality flags</h3>
          <p><b>Duplicate documents:</b> {(q.duplicate_documents ?? []).length === 0 ? "none" : (q.duplicate_documents ?? []).map((p) => p.join(" = ")).join(", ")}</p>
          <p><b>Low answer support (&lt; 0.5 lexical coverage):</b></p>
          {(q.low_answer_support ?? []).length === 0 ? <p className="muted">none</p> : (
            <table><thead><tr><th>Sample</th><th className="num">Coverage</th></tr></thead>
              <tbody>{(q.low_answer_support ?? []).map((s) => <tr key={s.sample_id}><td>{s.sample_id}</td><td className="num">{s.coverage}</td></tr>)}</tbody></table>
          )}
          <div className="note">{q.note ?? ""}</div>
        </div>
      </div>
    </div>
  );
}
