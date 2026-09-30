import type { MetricSummary, Run, SampleRow } from "./types";

export function fmt(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined ? "n/a" : v.toFixed(digits);
}

export function signed(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined) return "n/a";
  return (v >= 0 ? "+" : "") + v.toFixed(digits);
}

/** Metrics named "<family>@<k>" grouped by family, sorted by k. */
export function metricFamilies(
  metrics: Record<string, MetricSummary>,
): Record<string, { k: number; mean: number | null }[]> {
  const out: Record<string, { k: number; mean: number | null }[]> = {};
  for (const [name, s] of Object.entries(metrics)) {
    const m = /^(.+)@(\d+)$/.exec(name);
    if (!m || !m[1] || !m[2]) continue;
    (out[m[1]] ??= []).push({ k: Number(m[2]), mean: s.mean });
  }
  for (const list of Object.values(out)) list.sort((a, b) => a.k - b.k);
  return out;
}

export function scalarMetrics(metrics: Record<string, MetricSummary>): [string, MetricSummary][] {
  return Object.entries(metrics).filter(([name]) => !name.includes("@"));
}

export interface Delta {
  metric: string;
  a: number | null;
  b: number | null;
  delta: number | null;
  heuristic: boolean;
  bBetter: number;
  aBetter: number;
  ties: number;
  pairedN: number;
}

const EPS = 1e-9;

/** Same definition as the Python comparison: delta = B - A, per-sample paired wins. */
export function compareRuns(a: Run, b: Run): Delta[] {
  const bMap = new Map<string, SampleRow>(b.samples.map((s) => [s.sample_id, s]));
  const out: Delta[] = [];
  for (const [name, sa] of Object.entries(a.summary.metrics)) {
    const sb = b.summary.metrics[name];
    if (!sb) continue;
    let bBetter = 0;
    let aBetter = 0;
    let ties = 0;
    for (const s of a.samples) {
      const other = bMap.get(s.sample_id);
      const x = s.metrics[name]?.score;
      const y = other?.metrics[name]?.score;
      if (x === null || x === undefined || y === null || y === undefined) continue;
      const d = y - x;
      if (d > EPS) bBetter++;
      else if (d < -EPS) aBetter++;
      else ties++;
    }
    out.push({
      metric: name,
      a: sa.mean,
      b: sb.mean,
      delta: sa.mean === null || sb.mean === null ? null : sb.mean - sa.mean,
      heuristic: sa.heuristic || sb.heuristic,
      bBetter,
      aBetter,
      ties,
      pairedN: bBetter + aBetter + ties,
    });
  }
  return out;
}

export function shortId(runId: string): string {
  return runId.length > 34 ? runId.slice(0, 31) + "…" : runId;
}
