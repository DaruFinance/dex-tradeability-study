"use client";

import * as React from "react";

/**
 * DEX cost-frontier demo.
 *
 * The per-fill round-trip cost on an AMM is endogenous to trade size:
 *   c = 2f (swap fee)  +  slippage(size/reserve) on both legs  +  2g/size (gas).
 * Small trades are dominated by fixed gas (2g/size); large trades by
 * constant-product slippage (~size/reserve). The cost is U-shaped in size with
 * a minimum near the study's default sizing (0.25% of pool reserve), and on a
 * deep pool that minimum sits around ~160 bp round-trip. Because the study's
 * gross long-only timing edge is statistically zero, net return is essentially
 * minus this number.
 */

type ChainId = "eth" | "bsc" | "base" | "arbitrum" | "solana" | "polygon";
const CHAINS: { id: ChainId; label: string; gasUsd: number }[] = [
  { id: "eth", label: "Ethereum", gasUsd: 4.0 },
  { id: "arbitrum", label: "Arbitrum", gasUsd: 0.15 },
  { id: "bsc", label: "BSC", gasUsd: 0.2 },
  { id: "polygon", label: "Polygon", gasUsd: 0.01 },
  { id: "base", label: "Base", gasUsd: 0.02 },
  { id: "solana", label: "Solana", gasUsd: 0.02 },
];

const FEE_PER_LEG = 0.003; // 30 bp Uniswap-style fee, each leg
const SLIP_FLOOR = 0.0002; // 2 bp minimum slippage per leg

function roundTripBp(sizeUsd: number, reserveUsd: number, gasUsd: number): {
  fee: number;
  slip: number;
  gas: number;
  total: number;
} {
  const oneSided = reserveUsd / 2;
  const ratio = sizeUsd / oneSided;
  const impactBuy = Math.max(ratio, SLIP_FLOOR);
  const impactSell = Math.max(ratio / (1 + ratio), SLIP_FLOOR);
  const fee = 2 * FEE_PER_LEG;
  const slip = impactBuy + impactSell;
  const gas = sizeUsd > 0 ? (2 * gasUsd) / sizeUsd : 1;
  return {
    fee: fee * 1e4,
    slip: slip * 1e4,
    gas: gas * 1e4,
    total: (fee + slip + gas) * 1e4,
  };
}

function fmt(n: number): string {
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(0)}k`;
  return `$${n.toFixed(0)}`;
}

export default function CostFrontierDemo() {
  const [chainId, setChainId] = React.useState<ChainId>("eth");
  // log-scale sliders (0..100) mapped to USD
  const [reservePos, setReservePos] = React.useState(62); // ~$370k
  const [sizePos, setSizePos] = React.useState(40);

  const chain = CHAINS.find((c) => c.id === chainId)!;
  const reserve = Math.round(Math.pow(10, 3.7 + (reservePos / 100) * (7 - 3.7))); // $5k .. $10M
  const size = Math.max(10, Math.round(Math.pow(10, 1 + (sizePos / 100) * (5.5 - 1)))); // $10 .. ~$316k

  const cur = roundTripBp(size, reserve, chain.gasUsd);
  const defaultSize = Math.max(50, 0.0025 * reserve);
  const atDefault = roundTripBp(defaultSize, reserve, chain.gasUsd);

  // sweep size for the curve (log axis)
  const W = 560,
    H = 230,
    PADL = 46,
    PADB = 30,
    PADT = 12,
    PADR = 12;
  const sMin = 10,
    sMax = 5e5;
  const lx = (s: number) =>
    PADL + ((Math.log10(s) - Math.log10(sMin)) / (Math.log10(sMax) - Math.log10(sMin))) * (W - PADL - PADR);
  const yMaxBp = 1200;
  const ly = (bp: number) => PADT + (1 - Math.min(bp, yMaxBp) / yMaxBp) * (H - PADT - PADB);

  const pts: string[] = [];
  for (let i = 0; i <= 120; i++) {
    const s = Math.pow(10, Math.log10(sMin) + (i / 120) * (Math.log10(sMax) - Math.log10(sMin)));
    pts.push(`${lx(s).toFixed(1)},${ly(roundTripBp(s, reserve, chain.gasUsd).total).toFixed(1)}`);
  }

  const tradeable = cur.total <= 300;

  return (
    <div className="not-prose my-8 border-hair bg-[color:var(--color-surface)]/30">
      <div className="px-5 py-4 border-b border-[color:var(--color-border)]">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-accent)]">
          Interactive · cost frontier
        </p>
        <p className="mt-1 text-sm text-[color:var(--color-fg-muted)] leading-relaxed">
          Drag pool depth and trade size. Watch fee, slippage, and gas trade off into a U-shaped
          round-trip cost. On a deep pool the floor sits near 160&nbsp;bp; gross timing edge is ~0,
          so net return is roughly minus this.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-[color:var(--color-border)]">
        {/* controls */}
        <div className="bg-[color:var(--color-bg)] p-5 space-y-5">
          <div>
            <div className="flex justify-between font-mono text-[11px] text-[color:var(--color-fg-muted)]">
              <span>Pool reserve (TVL)</span>
              <span className="tnum text-[color:var(--color-fg)]">{fmt(reserve)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={reservePos}
              onChange={(e) => setReservePos(Number(e.target.value))}
              className="w-full mt-2 accent-[color:var(--color-accent)]"
            />
          </div>
          <div>
            <div className="flex justify-between font-mono text-[11px] text-[color:var(--color-fg-muted)]">
              <span>Trade size</span>
              <span className="tnum text-[color:var(--color-fg)]">{fmt(size)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={sizePos}
              onChange={(e) => setSizePos(Number(e.target.value))}
              className="w-full mt-2 accent-[color:var(--color-accent)]"
            />
            <p className="mt-1 font-mono text-[10px] text-[color:var(--color-fg-subtle)]">
              study default sizing (0.25% of reserve) = {fmt(defaultSize)} → {atDefault.total.toFixed(0)} bp
            </p>
          </div>
          <div>
            <div className="font-mono text-[11px] text-[color:var(--color-fg-muted)] mb-2">Chain (gas)</div>
            <div className="flex flex-wrap gap-1">
              {CHAINS.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setChainId(c.id)}
                  className={`font-mono text-[11px] px-2 py-1 border transition-colors ${
                    c.id === chainId
                      ? "border-[color:var(--color-accent)] text-[color:var(--color-accent)]"
                      : "border-[color:var(--color-border)] text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-fg)]"
                  }`}
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>

          <div className="pt-2 grid grid-cols-3 gap-px bg-[color:var(--color-border)] text-center">
            {[
              { k: "fee", v: cur.fee, l: "swap fee" },
              { k: "slip", v: cur.slip, l: "slippage" },
              { k: "gas", v: cur.gas, l: "gas" },
            ].map((c) => (
              <div key={c.k} className="bg-[color:var(--color-bg)] py-3">
                <p className="font-mono text-[9px] uppercase tracking-[0.16em] text-[color:var(--color-fg-subtle)]">
                  {c.l}
                </p>
                <p className="mt-1 font-display text-lg tnum">{c.v.toFixed(0)}</p>
                <p className="font-mono text-[9px] text-[color:var(--color-fg-subtle)]">bp</p>
              </div>
            ))}
          </div>
          <div className="text-center border-hair py-3">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-[color:var(--color-fg-subtle)]">
              round-trip cost
            </p>
            <p className="mt-1 font-display text-3xl font-semibold tnum">
              {cur.total.toFixed(0)} <span className="text-base font-normal">bp</span>
            </p>
            <p
              className="mt-1 font-mono text-[11px]"
              style={{ color: tradeable ? "var(--color-accent)" : "var(--color-fg-subtle)" }}
            >
              {tradeable ? "below the ~300 bp tradeability line" : "above the tradeability line"}
            </p>
          </div>
        </div>

        {/* chart */}
        <div className="bg-[color:var(--color-bg)] p-3 flex flex-col">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Round-trip cost versus trade size">
            {/* axes */}
            <line x1={PADL} y1={H - PADB} x2={W - PADR} y2={H - PADB} stroke="var(--color-border)" />
            <line x1={PADL} y1={PADT} x2={PADL} y2={H - PADB} stroke="var(--color-border)" />
            {/* 164 bp reference */}
            <line x1={PADL} y1={ly(164)} x2={W - PADR} y2={ly(164)} stroke="var(--color-accent)" strokeDasharray="3 3" opacity={0.5} />
            <text x={W - PADR} y={ly(164) - 4} textAnchor="end" fontSize="9" fontFamily="monospace" fill="var(--color-accent)">
              164 bp
            </text>
            {/* y ticks */}
            {[0, 300, 600, 900, 1200].map((bp) => (
              <text key={bp} x={PADL - 6} y={ly(bp) + 3} textAnchor="end" fontSize="9" fontFamily="monospace" fill="var(--color-fg-subtle)">
                {bp}
              </text>
            ))}
            {/* x ticks */}
            {[100, 1000, 10000, 100000].map((s) => (
              <text key={s} x={lx(s)} y={H - PADB + 12} textAnchor="middle" fontSize="9" fontFamily="monospace" fill="var(--color-fg-subtle)">
                {fmt(s)}
              </text>
            ))}
            {/* cost curve */}
            <polyline points={pts.join(" ")} fill="none" stroke="var(--color-fg)" strokeWidth={1.6} />
            {/* current marker */}
            <line x1={lx(size)} y1={PADT} x2={lx(size)} y2={H - PADB} stroke="var(--color-accent)" opacity={0.35} />
            <circle cx={lx(size)} cy={ly(cur.total)} r={4} fill="var(--color-accent)" />
          </svg>
          <p className="mt-2 px-2 font-mono text-[10px] text-[color:var(--color-fg-subtle)] leading-relaxed">
            x: trade size (log) · y: round-trip cost (bp). Left arm = gas-dominated, right arm =
            slippage-dominated. The minimum is the cheapest you can trade this pool.
          </p>
        </div>
      </div>
    </div>
  );
}
