import { useEffect, useState } from "react";
import type { Bundle, Run } from "./types";
import { Compare } from "./views/Compare";
import { DatasetQuality } from "./views/DatasetQuality";
import { Experiments } from "./views/Experiments";
import { Failures } from "./views/Failures";
import { Metrics } from "./views/Metrics";
import { Retrieval } from "./views/Retrieval";

const TABS = ["Experiments", "Metrics", "Dataset quality", "Retrieval examples", "Failure cases", "Compare"] as const;
type Tab = (typeof TABS)[number];

export async function loadBundle(url = "./api/bundle.json"): Promise<Bundle> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
  return (await res.json()) as Bundle;
}

export default function App() {
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("Experiments");
  const [selected, setSelected] = useState("");

  useEffect(() => {
    loadBundle().then((b) => {
      setBundle(b);
      setSelected(b.runs[b.runs.length - 1]?.manifest.run_id ?? "");
    }).catch((e: unknown) => setError(String(e)));
  }, []);

  const runs: Run[] = bundle?.runs ?? [];
  const run = runs.find((r) => r.manifest.run_id === selected) ?? runs[runs.length - 1];

  let body: React.ReactNode;
  if (error) {
    body = <div className="empty">Could not load results: {error}<br />Start the dashboard with <span className="mono">rag-eval dashboard serve</span>.</div>;
  } else if (!bundle) {
    body = <div className="empty">Loading…</div>;
  } else if (!run) {
    body = <div className="empty">No runs stored yet. Run <span className="mono">rag-eval experiment run examples/demo/experiments/bm25-extractive.json</span>.</div>;
  } else {
    body = (
      <>
        {tab === "Experiments" && <Experiments runs={runs} selected={run.manifest.run_id} onSelect={setSelected} />}
        {tab === "Metrics" && <Metrics run={run} />}
        {tab === "Dataset quality" && <DatasetQuality run={run} />}
        {tab === "Retrieval examples" && <Retrieval run={run} />}
        {tab === "Failure cases" && <Failures run={run} />}
        {tab === "Compare" && <Compare runs={runs} initialA={run.manifest.run_id} />}
      </>
    );
  }

  return (
    <>
      <header className="top">
        <h1>rag-eval</h1>
        <span className="sub">Measured results only · heuristic metrics are marked</span>
        {run && (
          <label className="f">Selected run
            <select value={run.manifest.run_id} onChange={(e) => setSelected(e.target.value)}>
              {runs.map((r) => <option key={r.manifest.run_id}>{r.manifest.run_id}</option>)}
            </select>
          </label>
        )}
      </header>
      <nav role="tablist">
        {TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>{t}</button>)}
      </nav>
      <main><div className="panel" role="tabpanel">{body}</div></main>
    </>
  );
}
