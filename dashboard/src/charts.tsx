import { useState } from "react";
import { fmt } from "./lib";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

interface TipState { x: number; y: number; html: string }

function useTip() {
  const [tip, setTip] = useState<TipState | null>(null);
  const show = (e: React.MouseEvent, html: string) => setTip({ x: e.clientX + 12, y: e.clientY + 12, html });
  const hide = () => setTip(null);
  const el = tip ? (
    <div className="tip" style={{ left: tip.x, top: tip.y }} role="tooltip">
      {tip.html}
    </div>
  ) : null;
  return { show, hide, el };
}

export interface BarItem { label: string; value: number | null; heuristic?: boolean; detail?: string }

/** Horizontal bars on a fixed 0..1 axis. Heuristic metrics are hatched. */
export function HBarChart({ items, title }: { items: BarItem[]; title: string }) {
  const { show, hide, el } = useTip();
  const rowH = 26;
  const labelW = 150;
  const W = 520;
  const H = items.length * rowH + 22;
  const plotW = W - labelW - 56;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={title}>
        <defs>
          <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="var(--series-1)" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--surface-1)" strokeWidth="2" />
          </pattern>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={labelW + t * plotW} x2={labelW + t * plotW} y1={0} y2={H - 18} stroke="var(--grid)" />
            <text x={labelW + t * plotW} y={H - 4} textAnchor="middle">{t}</text>
          </g>
        ))}
        {items.map((it, i) => {
          const y = i * rowH + 4;
          const w = it.value === null ? 0 : Math.max(0, Math.min(1, it.value)) * plotW;
          return (
            <g key={it.label} onMouseMove={(e) => show(e, `${it.label}: ${fmt(it.value)}${it.heuristic ? " (heuristic)" : ""}${it.detail ? " · " + it.detail : ""}`)} onMouseLeave={hide}>
              <rect x={0} y={y - 2} width={W} height={rowH - 2} fill="transparent" />
              <text x={labelW - 8} y={y + 13} textAnchor="end">{it.label}</text>
              {it.value === null ? (
                <text x={labelW + 4} y={y + 13}>n/a</text>
              ) : (
                <>
                  <rect x={labelW} y={y} width={w} height={16} rx={3} fill={it.heuristic ? "url(#hatch)" : "var(--series-1)"} />
                  <text x={labelW + w + 6} y={y + 13}>{fmt(it.value)}</text>
                </>
              )}
            </g>
          );
        })}
      </svg>
      {el}
    </div>
  );
}

export interface LineSeries { name: string; points: { k: number; v: number | null }[] }

/** Metric-vs-k lines, y fixed at 0..1, with legend and per-point hover. */
export function LineChart({ series, title }: { series: LineSeries[]; title: string }) {
  const { show, hide, el } = useTip();
  const W = 520, H = 240, L = 36, R = 16, T = 10, B = 28;
  const ks = [...new Set(series.flatMap((s) => s.points.map((p) => p.k)))].sort((a, b) => a - b);
  const x = (k: number) => (ks.length <= 1 ? L + (W - L - R) / 2 : L + ((ks.indexOf(k)) / (ks.length - 1)) * (W - L - R));
  const y = (v: number) => T + (1 - v) * (H - T - B);
  return (
    <div>
      <div className="legend">
        {series.map((s, i) => (
          <span key={s.name}><i style={{ background: SERIES[i % SERIES.length] }} />{s.name}</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={title}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
            <text x={L - 6} y={y(t) + 4} textAnchor="end">{t}</text>
          </g>
        ))}
        {ks.map((k) => <text key={k} x={x(k)} y={H - 8} textAnchor="middle">@{k}</text>)}
        {series.map((s, i) => {
          const color = SERIES[i % SERIES.length];
          const pts = s.points.filter((p) => p.v !== null) as { k: number; v: number }[];
          return (
            <g key={s.name}>
              <polyline fill="none" stroke={color} strokeWidth={2} points={pts.map((p) => `${x(p.k)},${y(p.v)}`).join(" ")} />
              {pts.map((p) => (
                <circle key={p.k} cx={x(p.k)} cy={y(p.v)} r={4} fill={color} stroke="var(--surface-1)" strokeWidth={2}
                  onMouseMove={(e) => show(e, `${s.name}@${p.k}: ${fmt(p.v)}`)} onMouseLeave={hide} />
              ))}
            </g>
          );
        })}
      </svg>
      {el}
    </div>
  );
}

export interface PairItem { label: string; a: number | null; b: number | null; heuristic: boolean }

/** Grouped bars A (series 1) vs B (series 2) on a shared 0..1 axis. */
export function PairedBars({ items, nameA, nameB }: { items: PairItem[]; nameA: string; nameB: string }) {
  const { show, hide, el } = useTip();
  const rowH = 34, labelW = 150, W = 560, plotW = W - labelW - 50;
  const H = items.length * rowH + 8;
  const bar = (v: number | null) => (v === null ? 0 : Math.max(0, Math.min(1, v)) * plotW);
  return (
    <div>
      <div className="legend">
        <span><i style={{ background: "var(--series-1)" }} />A: {nameA}</span>
        <span><i style={{ background: "var(--series-2)" }} />B: {nameB}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Metric comparison">
        {items.map((it, i) => {
          const y = i * rowH + 4;
          return (
            <g key={it.label} onMouseMove={(e) => show(e, `${it.label}${it.heuristic ? " (heuristic)" : ""} — A: ${fmt(it.a)}  B: ${fmt(it.b)}`)} onMouseLeave={hide}>
              <rect x={0} y={y - 2} width={W} height={rowH - 2} fill="transparent" />
              <text x={labelW - 8} y={y + 18} textAnchor="end">{it.label}{it.heuristic ? "*" : ""}</text>
              <rect x={labelW} y={y} width={bar(it.a)} height={11} rx={3} fill="var(--series-1)" />
              <text x={labelW + bar(it.a) + 5} y={y + 10}>{fmt(it.a)}</text>
              <rect x={labelW} y={y + 14} width={bar(it.b)} height={11} rx={3} fill="var(--series-2)" />
              <text x={labelW + bar(it.b) + 5} y={y + 24}>{fmt(it.b)}</text>
            </g>
          );
        })}
      </svg>
      {el}
    </div>
  );
}
