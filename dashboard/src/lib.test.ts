import { describe, expect, it } from "vitest";
import { compareRuns, fmt, metricFamilies, signed } from "./lib";
import type { MetricSummary, Run, SampleRow } from "./types";

const summary = (mean: number | null, heuristic = false): MetricSummary => ({
  mean, std: 0, n: 1, n_total: 1, not_applicable: 0, method: "m", heuristic,
});

const sample = (id: string, score: number | null): SampleRow => ({
  sample_id: id, question: "", expected_answer: null, relevant_docs: [], metadata: {},
  retrieved: [], answer: "", failures: [],
  metrics: { "recall@1": { metric: "recall@1", score, method: "", evidence: {}, heuristic: false, note: "" } },
});

const run = (mean: number | null, scores: (number | null)[]): Run =>
  ({
    summary: { num_samples: scores.length, metrics: { "recall@1": summary(mean) }, failure_counts: {} },
    samples: scores.map((s, i) => sample(`s${i}`, s)),
  }) as unknown as Run;

describe("lib", () => {
  it("formats", () => {
    expect(fmt(null)).toBe("n/a");
    expect(fmt(0.5)).toBe("0.500");
    expect(signed(0.25)).toBe("+0.250");
    expect(signed(-0.25)).toBe("-0.250");
  });
  it("groups metric families by k", () => {
    const f = metricFamilies({ "recall@3": summary(0.5), "recall@1": summary(0.1), exact_match: summary(0) });
    expect(f.recall?.map((x) => x.k)).toEqual([1, 3]);
    expect(Object.keys(f)).toEqual(["recall"]);
  });
  it("computes paired deltas and ignores non-applicable samples", () => {
    const d = compareRuns(run(0.5, [1, 0, null]), run(0.75, [1, 1, 1]))[0]!;
    expect(d.delta).toBeCloseTo(0.25);
    expect([d.bBetter, d.aBetter, d.ties, d.pairedN]).toEqual([1, 0, 1, 2]);
  });
  it("delta is null when a mean is missing", () => {
    expect(compareRuns(run(null, [null]), run(0.5, [1]))[0]!.delta).toBeNull();
  });
});
