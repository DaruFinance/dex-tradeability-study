"""Phase 1: buy-and-hold baseline + cost/edge breakeven on the DEX-only, non-CEX universe.

Reads ohlcv_gt/*.parquet + _gt_universe*.jsonl (reserve/age) + _cex_verdicts_universe.jsonl
(dex_only filter). Long-only. Applies the real per-chain DEX round-trip cost (fee+slippage+gas)
via chainscope.costs, sizing as a fraction of pool reserve.

⚠ SURVIVORSHIP: the universe is GT-survivor-seeded (coins live NOW), so buy-and-hold returns are
OPTIMISTIC (these survived). Reported with that caveat; the WFO/exit test (phase3) is less
survivor-sensitive (it's about in-window timing). Died-in-window coins to be added for production.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
# per-chain gas + native price for the cost model (BSC/Base/Arb cheap; ETH expensive)
CHAIN_GAS = {"bsc": 0.20, "base": 0.02, "eth": 3.0, "arbitrum": 0.05, "solana": 0.02}
CHAIN_NATIVE = {"bsc": 640.0, "base": 3500.0, "eth": 3500.0, "arbitrum": 3500.0, "solana": 180.0}
DEX_DEFAULT = {"bsc": "pancakeswap", "base": "uniswap", "eth": "uniswap", "arbitrum": "uniswap"}


def load_universe():
    recs = {}
    for f in glob.glob(f"{DATA}/_gt_universe*.jsonl"):
        for l in open(f):
            r = json.loads(l); recs[r["pair"]] = r
    return recs


def load_cex_verdicts():
    v = {}
    f = DATA / "_cex_verdicts_universe.jsonl"
    if f.exists():
        for l in open(f):
            r = json.loads(l); v[r["pair"]] = r["status"]
    return v


def rt_cost(chain, reserve, size):
    # gas_usd is the PER-SWAP gas; round_trip_cost_frac already applies it on both legs (2x/size).
    return round_trip_cost_frac(size, reserve, dex=DEX_DEFAULT.get(chain, "uniswap"), chain=chain,
                                gas_usd=CHAIN_GAS.get(chain, 0.10),
                                native_usd=CHAIN_NATIVE.get(chain))


def main():
    univ = load_universe(); cexv = load_cex_verdicts()
    files = glob.glob(f"{DATA}/ohlcv_gt/*.parquet")
    if not files:
        print("no ohlcv parquet yet"); return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["pair_address", "ts"])
    print(f"OHLCV: {df.pair_address.nunique()} pools, {len(df)} bars, chains={sorted(df.chain.unique())}")
    print(f"CEX verdicts available: {len(cexv)} | dex_only={sum(v=='dex_only' for v in cexv.values())} "
          f"major_cex={sum(v=='major_cex' for v in cexv.values())} major_pair={sum(v=='major_pair' for v in cexv.values())}")

    rows = []
    for pair, g in df.groupby("pair_address"):
        g = g.dropna(subset=["close"]).sort_values("ts")
        if len(g) < 10:
            continue
        u = univ.get(pair, {}); chain = u.get("chain", g.chain.iloc[0]); reserve = u.get("reserve_usd")
        verdict = cexv.get(pair, "unknown")
        c = g.close.to_numpy(); first, last = c[0], c[-1]
        if first <= 0:
            continue
        bh = last / first - 1.0                      # buy & hold full window
        mfe = g.close.max() / first - 1.0            # best favorable exit (upside potential, not tradeable)
        mae = g.close.min() / first - 1.0            # worst (downside)
        size = max(50.0, 0.01 * (reserve or 0))      # 1% of reserve, min $50
        cost = rt_cost(chain, reserve, size) if reserve else np.nan
        rows.append({"pair": pair, "chain": chain, "verdict": verdict, "bars": len(g),
                     "reserve": reserve, "size": size, "rt_cost": cost,
                     "bh_ret": bh, "bh_net": bh - (cost or 0), "mfe": mfe, "mae": mae})
    R = pd.DataFrame(rows)

    def report(name, d):
        if not len(d):
            print(f"\n## {name}: EMPTY"); return
        q = lambda s: {p: round(float(np.nanpercentile(s, p)), 3) for p in (10, 25, 50, 75, 90)}
        print(f"\n## {name}  (n={len(d)})")
        print(f"  rt_cost frac:   med={np.nanmedian(d.rt_cost):.4f} ({np.nanmedian(d.rt_cost)*1e4:.0f}bp)  pctiles={q(d.rt_cost)}")
        print(f"  buy&hold ret:   med={np.nanmedian(d.bh_ret):.3f}  mean={np.nanmean(d.bh_ret):.3f}  %>0={100*(d.bh_ret>0).mean():.0f}%  pctiles={q(d.bh_ret)}")
        print(f"  b&h NET cost:   med={np.nanmedian(d.bh_net):.3f}  mean={np.nanmean(d.bh_net):.3f}  %>0={100*(d.bh_net>0).mean():.0f}%")
        print(f"  MFE (best exit):med={np.nanmedian(d.mfe):.3f}  %MFE>rt_cost={100*(d.mfe>d.rt_cost).mean():.0f}%   MAE med={np.nanmedian(d.mae):.3f}")

    report("ALL pools", R)
    for ch in sorted(R.chain.dropna().unique()):
        report(f"chain={ch}", R[R.chain == ch])
    dexonly = R[R.verdict == "dex_only"]
    report("DEX-ONLY non-CEX (the target universe)", dexonly)
    print("\n(NOTE: survivor-seeded universe -> buy&hold is OPTIMISTIC. MFE>rt_cost = fraction of coins whose "
          "best in-window move clears round-trip cost = upper bound on opportunity if timing were perfect.)")
    R.to_parquet(DATA / "_phase1_baseline.parquet", index=False)
    print(f"\nsaved -> {DATA}/_phase1_baseline.parquet")


if __name__ == "__main__":
    main()
