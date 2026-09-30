import { useState } from "react";
import type { Run } from "../types";

export function Failures({ run }: { run: Run }) {
  const counts = run.summary.failure_counts;
  const [cat, setCat] = useState<string>("all");
  const cases = run.samples.flatMap((s) => s.failures.map((f) => ({ s, f })));
  const shown = cases.filter(({ f }) => cat === "all" || f.category === cat);
  return (
    <div>
      <h2>Failure analysis · {run.manifest.name}</h2>
      <div className="row">
        <div className="seg" role="group" aria-label="Failure category">
          <button aria-pressed={cat === "all"} onClick={() => setCat("all")}>all ({cases.length})</button>
          {Object.entries(counts).filter(([, n]) => n > 0).map(([c, n]) => (
            <button key={c} aria-pressed={cat === c} onClick={() => setCat(c)}>{c} ({n})</button>
          ))}
        </div>
      </div>
      <div className="note">
        Retrieval failures are computed from relevance labels. “unsupported”, “contradicted” and “wrong answer” rely on heuristics
        and can be false positives or negatives; always read the evidence.
      </div>
      {shown.length === 0 && <div className="empty">No failures in this category.</div>}
      {shown.map(({ s, f }, i) => (
        <div className="card" key={s.sample_id + f.category + i} style={{ marginBottom: 10 }}>
          <div>
            <span className={"badge " + f.severity}>{f.severity}</span>{" "}
            <b>{f.category}</b>{f.heuristic && <span className="badge heur" style={{ marginLeft: 6 }}>heuristic</span>} · {s.sample_id}
          </div>
          <div className="muted">{f.description}</div>
          <div>Q: {s.question}</div>
          <div>Answer: {s.answer}</div>
          <div className="muted">Expected: {s.expected_answer ?? "—"}</div>
          <pre className="ev">{JSON.stringify(f.evidence, null, 2)}</pre>
        </div>
      ))}
    </div>
  );
}
