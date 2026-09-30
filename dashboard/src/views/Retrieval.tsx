import { useMemo, useState } from "react";
import { fmt } from "../lib";
import type { Run } from "../types";

export function Retrieval({ run }: { run: Run }) {
  const [q, setQ] = useState("");
  const [onlyMisses, setOnlyMisses] = useState(false);
  const rows = useMemo(() => run.samples.filter((s) => {
    if (q && !`${s.sample_id} ${s.question}`.toLowerCase().includes(q.toLowerCase())) return false;
    if (onlyMisses && !s.failures.some((f) => f.category.includes("retrieval") || f.category === "irrelevant_context" || f.category === "empty_retrieval")) return false;
    return true;
  }), [run, q, onlyMisses]);
  return (
    <div>
      <h2>Retrieval examples · {run.manifest.name}</h2>
      <div className="row">
        <input type="search" placeholder="Filter by id or question" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter samples" />
        <label className="f"><input type="checkbox" checked={onlyMisses} onChange={(e) => setOnlyMisses(e.target.checked)} /> only retrieval problems</label>
        <span className="muted">{rows.length} of {run.samples.length} samples</span>
      </div>
      {rows.length === 0 && <div className="empty">No samples match.</div>}
      {rows.map((s) => (
        <div key={s.sample_id} className="card" style={{ marginBottom: 10 }}>
          <div><b>{s.sample_id}</b> · {s.question}</div>
          <div className="muted">Expected: {s.expected_answer ?? "—"} · relevant: {s.relevant_docs.join(", ") || "—"}</div>
          {s.retrieved.length === 0 && <div className="doc irr">No documents retrieved.</div>}
          {s.retrieved.map((d) => (
            <div key={d.doc_id} className={"doc " + (d.relevant ? "rel" : "irr")}>
              <span className="mono">#{d.rank} {d.doc_id}</span> · score {fmt(d.score)}{" "}
              <span className={"badge " + (d.relevant ? "ok" : "")}>{d.relevant ? "labelled relevant" : "not labelled relevant"}</span>
              <div className="t">{d.text}</div>
            </div>
          ))}
          <div style={{ marginTop: 6 }}><b>Answer:</b> {s.answer}</div>
        </div>
      ))}
    </div>
  );
}
