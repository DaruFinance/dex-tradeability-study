"use client";

import * as React from "react";

/**
 * Null-control demo.
 *
 * Generates a synthetic DEX-like market (negative drift, fat tails,
 * anti-momentum) and runs a simple long-only momentum strategy on (a) the real
 * series and (b) a bar-shuffled copy of each coin. The shuffle destroys the
 * temporal structure but keeps each coin's return distribution. Because that
 * structure (clustered crashes on a downward, mean-reverting drift) is adverse
 * to longs, the strategy scores WORSE on real data than on its own null: the
 * "edge" a naive backtest reports is below noise. Drag the knobs and re-draw.
 */

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeRandn(rng: () => number) {
  return () => {
    let u = 0,
      v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

type Stats = { pctPF1: number; medPF: number; meanNet: number };

function pf(trades: number[]): number {
  let g = 0,
    l = 0;
  for (const t of trades) {
    if (t > 0) g += t;
    else l -= t;
  }
  return l > 0 ? g / l : g > 0 ? Infinity : 0;
}

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function simulate(seed: number, drift: number, phi: number, costBp: number): { real: Stats; nul: Stats } {
  const rng = mulberry32(seed);
  const randn = makeRandn(rng);
  const N_COINS = 180;
  const N_BARS = 170;
  const cost = costBp / 1e4;
  const sigma = 0.06;
  const realPF: number[] = [];
  const nulPF: number[] = [];
  const realNet: number[] = [];
  const nulNet: number[] = [];

  for (let c = 0; c < N_COINS; c++) {
    const r = new Float64Array(N_BARS);
    let prev = 0;
    for (let i = 0; i < N_BARS; i++) {
      let noise = randn() * sigma;
      // fat tail + crash skew: occasional large, mostly-downward jumps
      if (rng() < 0.05) noise -= Math.abs(randn()) * sigma * 4;
      // anti-momentum: pull against the previous bar
      r[i] = drift + noise - phi * prev;
      prev = r[i];
    }
    // long-only momentum: after an up bar, hold next bar (causal)
    const realTrades: number[] = [];
    for (let i = 1; i < N_BARS; i++) if (r[i - 1] > 0) realTrades.push(r[i] - cost);
    if (realTrades.length >= 4) {
      realPF.push(pf(realTrades));
      realNet.push(realTrades.reduce((a, b) => a + b, 0) / realTrades.length);
    }
    // bar-shuffle the same returns, run the identical rule
    const sh = Float64Array.from(r);
    for (let i = sh.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = sh[i];
      sh[i] = sh[j];
      sh[j] = tmp;
    }
    const nulTrades: number[] = [];
    for (let i = 1; i < N_BARS; i++) if (sh[i - 1] > 0) nulTrades.push(sh[i] - cost);
    if (nulTrades.length >= 4) {
      nulPF.push(pf(nulTrades));
      nulNet.push(nulTrades.reduce((a, b) => a + b, 0) / nulTrades.length);
    }
  }
  const stat = (pfs: number[], nets: number[]): Stats => ({
    pctPF1: (100 * pfs.filter((x) => x > 1).length) / Math.max(pfs.length, 1),
    medPF: median(pfs.filter((x) => isFinite(x))),
    meanNet: (100 * nets.reduce((a, b) => a + b, 0)) / Math.max(nets.length, 1),
  });
  return { real: stat(realPF, realNet), nul: stat(nulPF, nulNet) };
}

function Col({ title, s, accent }: { title: string; s: Stats; accent: boolean }) {
  return (
    <div className="bg-[color:var(--color-bg)] p-5 text-center">
      <p
        className="font-mono text-[10px] uppercase tracking-[0.18em]"
        style={{ color: accent ? "var(--color-accent)" : "var(--color-fg-subtle)" }}
      >
        {title}
      </p>
      <p className="mt-3 font-display text-4xl font-semibold tnum">{s.pctPF1.toFixed(0)}%</p>
      <p className="font-mono text-[10px] text-[color:var(--color-fg-subtle)]">strategies with PF &gt; 1</p>
      <div className="mt-4 grid grid-cols-2 gap-px bg-[color:var(--color-border)]">
        <div className="bg-[color:var(--color-bg)] py-2">
          <p className="font-display text-lg tnum">{s.medPF.toFixed(2)}</p>
          <p className="font-mono text-[9px] text-[color:var(--color-fg-subtle)]">median PF</p>
        </div>
        <div className="bg-[color:var(--color-bg)] py-2">
          <p className="font-display text-lg tnum">{s.meanNet >= 0 ? "+" : ""}{s.meanNet.toFixed(2)}%</p>
          <p className="font-mono text-[9px] text-[color:var(--color-fg-subtle)]">mean net/trade</p>
        </div>
      </div>
    </div>
  );
}

export default function NullControlDemo() {
  const [seed, setSeed] = React.useState(42);
  const [drift, setDrift] = React.useState(-0.002);
  const [phi, setPhi] = React.useState(0.16);
  const [costBp, setCostBp] = React.useState(164);
  const { real, nul } = React.useMemo(() => simulate(seed, drift, phi, costBp), [seed, drift, phi, costBp]);
  const realWorse = real.pctPF1 < nul.pctPF1;

  return (
    <div className="not-prose my-8 border-hair bg-[color:var(--color-surface)]/30">
      <div className="px-5 py-4 border-b border-[color:var(--color-border)] flex items-start justify-between gap-4">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-accent)]">
            Interactive · null control
          </p>
          <p className="mt-1 text-sm text-[color:var(--color-fg-muted)] leading-relaxed">
            A long-only momentum strategy on a simulated DEX market vs. the same strategy on a
            bar-shuffled copy. If real beats noise, real should win. It does not.
          </p>
        </div>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="shrink-0 font-mono text-[11px] px-3 py-2 border border-[color:var(--color-border)] hover:border-[color:var(--color-accent)] hover:text-[color:var(--color-accent)] transition-colors"
        >
          re-draw
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-[color:var(--color-border)]">
        <Col title="REAL data" s={real} accent={false} />
        <Col title="NULL (bar-shuffled)" s={nul} accent />
      </div>

      <div className="px-5 py-3 border-t border-[color:var(--color-border)] text-center">
        <p className="font-mono text-[12px]" style={{ color: realWorse ? "var(--color-accent)" : "var(--color-fg-muted)" }}>
          {realWorse
            ? "→ Real data is WORSE than its own shuffle. The structure is adverse to longs."
            : "→ With these knobs the structure is no longer adverse; raise anti-momentum or lower drift."}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-[color:var(--color-border)] border-t border-[color:var(--color-border)]">
        <div className="bg-[color:var(--color-bg)] p-4">
          <div className="flex justify-between font-mono text-[11px] text-[color:var(--color-fg-muted)]">
            <span>drift / bar</span>
            <span className="tnum text-[color:var(--color-fg)]">{(drift * 100).toFixed(2)}%</span>
          </div>
          <input type="range" min={-50} max={0} value={drift * 10000} onChange={(e) => setDrift(Number(e.target.value) / 10000)} className="w-full mt-2 accent-[color:var(--color-accent)]" />
        </div>
        <div className="bg-[color:var(--color-bg)] p-4">
          <div className="flex justify-between font-mono text-[11px] text-[color:var(--color-fg-muted)]">
            <span>anti-momentum φ</span>
            <span className="tnum text-[color:var(--color-fg)]">{phi.toFixed(2)}</span>
          </div>
          <input type="range" min={0} max={40} value={phi * 100} onChange={(e) => setPhi(Number(e.target.value) / 100)} className="w-full mt-2 accent-[color:var(--color-accent)]" />
        </div>
        <div className="bg-[color:var(--color-bg)] p-4">
          <div className="flex justify-between font-mono text-[11px] text-[color:var(--color-fg-muted)]">
            <span>cost / trade</span>
            <span className="tnum text-[color:var(--color-fg)]">{costBp} bp</span>
          </div>
          <input type="range" min={0} max={300} value={costBp} onChange={(e) => setCostBp(Number(e.target.value))} className="w-full mt-2 accent-[color:var(--color-accent)]" />
        </div>
      </div>
    </div>
  );
}
