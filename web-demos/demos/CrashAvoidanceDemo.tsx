"use client";

import * as React from "react";

/**
 * Crash-avoidance demo.
 *
 * The one signal in the study that beats its null is a cross-sectional ML
 * ranker, but its skill lives in the LEFT tail: sorting coins by predicted
 * rank, every decile's median forward return is negative, and the top
 * predicted decile is not the top-return decile. So a long-only trader cannot
 * monetise it (top decile is still a loser net of cost); the information is
 * "which coins crash", which needs avoidance or a short, not a long.
 */

const BENCH = -2.0; // benchmark median forward return (%), per period
const COST = 1.64; // round-trip cost (%)

function decileReturn(d: number, skill: number): number {
  // d in 1..10 (1 = worst predicted, 10 = best predicted)
  // at skill=0 every decile = benchmark; at skill=1 they fan from ~-6% to ~-0.2%
  const t = (d - 1) / 9; // 0..1
  const lo = BENCH - 4.0 * skill; // bottom decile
  const hi = BENCH + 1.8 * skill; // top decile (still < 0)
  return lo + (hi - lo) * t;
}

export default function CrashAvoidanceDemo() {
  const [skill, setSkill] = React.useState(0.8);
  const [allowShort, setAllowShort] = React.useState(false);

  const deciles = Array.from({ length: 10 }, (_, i) => decileReturn(i + 1, skill));
  const topDecileNet = deciles[9] - COST;
  const shortBottomNet = -deciles[0] - COST; // shorting the predicted-worst decile

  const W = 560,
    H = 240,
    PADL = 40,
    PADB = 34,
    PADT = 14,
    PADR = 12;
  const yMin = -7,
    yMax = 7;
  const ly = (v: number) => PADT + (1 - (v - yMin) / (yMax - yMin)) * (H - PADT - PADB);
  const bw = (W - PADL - PADR) / 10;

  return (
    <div className="not-prose my-8 border-hair bg-[color:var(--color-surface)]/30">
      <div className="px-5 py-4 border-b border-[color:var(--color-border)]">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-accent)]">
          Interactive · crash avoidance
        </p>
        <p className="mt-1 text-sm text-[color:var(--color-fg-muted)] leading-relaxed">
          Coins sorted into deciles by the ranker&apos;s predicted score. Raise the model skill and
          watch the deciles fan out, yet every one stays below zero. The skill is in the left tail.
        </p>
      </div>

      <div className="bg-[color:var(--color-bg)] p-3">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Median forward return by predicted decile">
          {/* zero line */}
          <line x1={PADL} y1={ly(0)} x2={W - PADR} y2={ly(0)} stroke="var(--color-border)" />
          {/* benchmark */}
          <line x1={PADL} y1={ly(BENCH)} x2={W - PADR} y2={ly(BENCH)} stroke="var(--color-fg-subtle)" strokeDasharray="3 3" opacity={0.7} />
          <text x={W - PADR} y={ly(BENCH) - 3} textAnchor="end" fontSize="9" fontFamily="monospace" fill="var(--color-fg-subtle)">
            equal-weight benchmark
          </text>
          {/* y ticks */}
          {[-6, -3, 0, 3, 6].map((v) => (
            <text key={v} x={PADL - 6} y={ly(v) + 3} textAnchor="end" fontSize="9" fontFamily="monospace" fill="var(--color-fg-subtle)">
              {v > 0 ? `+${v}` : v}
            </text>
          ))}
          {/* bars */}
          {deciles.map((v, i) => {
            const x = PADL + i * bw + 2;
            const top = v >= 0 ? ly(v) : ly(0);
            const h = Math.abs(ly(v) - ly(0));
            const isTop = i === 9;
            const isBottom = i === 0;
            return (
              <g key={i}>
                <rect
                  x={x}
                  y={top}
                  width={bw - 4}
                  height={Math.max(h, 0.5)}
                  fill={isTop ? "var(--color-accent)" : isBottom ? "var(--color-fg)" : "var(--color-fg-muted)"}
                  opacity={isTop || isBottom ? 0.9 : 0.45}
                />
                <text x={x + (bw - 4) / 2} y={H - PADB + 12} textAnchor="middle" fontSize="8" fontFamily="monospace" fill="var(--color-fg-subtle)">
                  {i + 1}
                </text>
              </g>
            );
          })}
          <text x={PADL} y={H - 4} fontSize="9" fontFamily="monospace" fill="var(--color-fg-subtle)">
            predicted decile (1 = worst, 10 = best) →
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-[color:var(--color-border)] border-t border-[color:var(--color-border)]">
        <div className="bg-[color:var(--color-bg)] p-5">
          <div className="flex justify-between font-mono text-[11px] text-[color:var(--color-fg-muted)]">
            <span>model skill</span>
            <span className="tnum text-[color:var(--color-fg)]">{(skill * 100).toFixed(0)}%</span>
          </div>
          <input type="range" min={0} max={100} value={skill * 100} onChange={(e) => setSkill(Number(e.target.value) / 100)} className="w-full mt-2 accent-[color:var(--color-accent)]" />
          <label className="mt-4 flex items-center gap-2 font-mono text-[11px] text-[color:var(--color-fg-muted)] cursor-pointer">
            <input type="checkbox" checked={allowShort} onChange={(e) => setAllowShort(e.target.checked)} className="accent-[color:var(--color-accent)]" />
            imagine a short were possible (it is not, on an AMM)
          </label>
        </div>
        <div className="bg-[color:var(--color-bg)] p-5 flex flex-col justify-center">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-[11px] text-[color:var(--color-fg-muted)]">long top decile, net of cost</span>
            <span className="font-display text-xl tnum" style={{ color: "var(--color-fg)" }}>
              {topDecileNet >= 0 ? "+" : ""}
              {topDecileNet.toFixed(2)}%
            </span>
          </div>
          {allowShort && (
            <div className="mt-3 flex items-baseline justify-between">
              <span className="font-mono text-[11px] text-[color:var(--color-fg-muted)]">short bottom decile, net of cost</span>
              <span className="font-display text-xl tnum" style={{ color: "var(--color-accent)" }}>
                {shortBottomNet >= 0 ? "+" : ""}
                {shortBottomNet.toFixed(2)}%
              </span>
            </div>
          )}
          <p className="mt-4 font-mono text-[11px] leading-relaxed" style={{ color: "var(--color-fg-subtle)" }}>
            {allowShort
              ? "The edge is real, but it lives on the short side the AMM denies you."
              : "Long-only, the top decile is still a loser. The model knows what falls, not what rises."}
          </p>
        </div>
      </div>
    </div>
  );
}
