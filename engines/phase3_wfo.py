"""Phase 3 prototype: long-only archetypes + pooled calendar-time WFO on the DEX universe.

Signals from OHLCV+volume (the fields GT gives): momentum, breakout(Donchian), reversion,
vol-surge. Long-only. Causal (every feature uses .shift(1)). Per-fill DEX cost applied at
entry+exit (reserve-fraction sizing, per-chain gas). Pooled calendar-time WFO: rolling IS / disjoint
forward OOS; IS selects param combos (PF>1 & min-trades), OOS evaluates them. Reports OOS PF,
%positive, and cross-combo decorrelation on per-OOS-window PnL (the playbook metric).

⚠ survivor-seeded universe -> optimistic; this is a CONCEPT proof, not a production result.

Usage: python3 _research/phase3_wfo.py <tf day|hour> <is_frac> <oos_frac>
"""
from __future__ import annotations
import json, glob, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
TF = sys.argv[1] if len(sys.argv) > 1 else "day"
SIZE_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0025   # ticket as frac of reserve (0.25% default)
IS_SPLIT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6       # first IS_SPLIT of calendar = IS, rest = OOS
CHAIN_GAS = {"bsc": 0.20, "base": 0.02, "eth": 3.0, "arbitrum": 0.05, "solana": 0.02}
CHAIN_NATIVE = {"bsc": 640.0, "base": 3500.0, "eth": 3500.0, "arbitrum": 3500.0, "solana": 180.0}
DEX_DEFAULT = {"bsc": "pancakeswap", "base": "uniswap", "eth": "uniswap", "arbitrum": "uniswap"}
MIN_TRADES_IS = int(sys.argv[4]) if len(sys.argv) > 4 else 20   # pooled IS trades to consider a combo
MIN_BARS = 60


def load():
    univ = {}
    for f in glob.glob(f"{DATA}/_gt_universe*.jsonl"):
        for l in open(f):
            r = json.loads(l); univ[r["pair"]] = r
    cexv = {}
    cf = DATA / "_cex_verdicts_universe.jsonl"
    if cf.exists():
        for l in open(cf):
            r = json.loads(l); cexv[r["pair"]] = r["status"]
    files = glob.glob(f"{DATA}/ohlcv_gt/*{TF}*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df, univ, cexv


# --- archetypes: each returns a boolean long-entry signal array (causal) given close/vol arrays ---
def sig_momentum(c, v, L, thr):
    pr = pd.Series(c).shift(1).pct_change(L).to_numpy()
    return pr > thr

def sig_breakout(c, v, L, thr):
    s = pd.Series(c).shift(1)
    dch = s.rolling(L).max().to_numpy()
    return (s.to_numpy() >= dch) & ~np.isnan(dch)

def sig_reversion(c, v, L, thr):
    s = pd.Series(c).shift(1)
    ma = s.rolling(L).mean(); sd = s.rolling(L).std()
    z = ((s - ma) / sd).to_numpy()
    return z < -thr

def sig_volsurge(c, v, L, thr):
    vs = pd.Series(v).shift(1)
    base = vs.rolling(L).mean().to_numpy()
    ret1 = pd.Series(c).shift(1).pct_change().to_numpy()
    return (vs.to_numpy() > thr * base) & (ret1 > 0) & ~np.isnan(base)

ARCHETYPES = {"momentum": sig_momentum, "breakout": sig_breakout,
              "reversion": sig_reversion, "volsurge": sig_volsurge}
GRID = {
    "momentum":  [(L, t) for L in (3, 5, 10, 20) for t in (0.05, 0.15, 0.30)],
    "breakout":  [(L, 0) for L in (5, 10, 20, 30)],
    "reversion": [(L, t) for L in (5, 10, 20) for t in (1.0, 1.5, 2.0)],
    "volsurge":  [(L, t) for L in (5, 10, 20) for t in (2.0, 3.0, 5.0)],
}
MAXHOLD = [3, 7, 14]
STOP = [0.15, 0.30]      # hard stop as frac drawdown from entry


def simulate(c, entry_sig, max_hold, stop, cost):
    """Long-only state machine -> list of trade net returns (frac). Causal: entry_sig[i] decided
    from data <= i-1; we enter at close[i], exit at close[j]."""
    n = len(c); i = 0; trades = []
    while i < n - 1:
        if entry_sig[i] and c[i] > 0:
            ep = c[i]; exit_i = min(i + max_hold, n - 1); stop_px = ep * (1 - stop)
            j = i + 1
            while j <= exit_i:
                if c[j] <= stop_px:
                    break
                j += 1
            j = min(j, exit_i)
            trades.append(c[j] / ep - 1.0 - cost)
            i = j + 1
        else:
            i += 1
    return trades


def main():
    df, univ, cexv = load()
    df = df.dropna(subset=["close"]).sort_values(["pair_address", "ts"])
    pools = []
    for pair, g in df.groupby("pair_address"):
        if len(g) < MIN_BARS:
            continue
        if cexv and cexv.get(pair) != "dex_only":   # restrict to DEX-only non-CEX when verdicts exist
            continue
        u = univ.get(pair, {}); chain = u.get("chain", g.chain.iloc[0]); reserve = u.get("reserve_usd")
        size = max(50.0, SIZE_FRAC * (reserve or 0))
        cost = round_trip_cost_frac(size, reserve, dex=DEX_DEFAULT.get(chain, "uniswap"), chain=chain,
                                    gas_usd=CHAIN_GAS.get(chain, 0.1), native_usd=CHAIN_NATIVE.get(chain)) if reserve else 0.05
        pools.append({"pair": pair, "chain": chain, "c": g.close.to_numpy(), "v": g.volume.fillna(0).to_numpy(),
                      "ts": g.ts.to_numpy(), "cost": cost})
    print(f"pooled DEX-only universe: {len(pools)} coins, tf={TF}")
    if len(pools) < 5:
        print("too few coins (need OHLCV + CEX verdicts present); rerun after ingest completes."); return

    tmin = min(p["ts"].min() for p in pools); tmax = max(p["ts"].max() for p in pools)
    split = tmin + IS_SPLIT * (tmax - tmin)   # single disjoint forward split: IS < split <= OOS
    print(f"size_frac={SIZE_FRAC} (median rt_cost={np.median([p['cost'] for p in pools])*1e4:.0f}bp); "
          f"IS/OOS calendar split at {IS_SPLIT:.0%}")

    combos = [(a, L, thr, mh, st) for a in ARCHETYPES for (L, thr) in GRID[a]
              for mh in MAXHOLD for st in STOP]
    print(f"{len(combos)} structural combos")

    res = []   # per combo: IS/OOS pf, trades, oos net; plus per-coin OOS pnl for decorrelation
    oos_by_coin = {}
    for (arch, L, thr, mh, st) in combos:
        fn = ARCHETYPES[arch]; key = f"{arch}_L{L}_t{thr}_h{mh}_s{st}"
        is_tr, oos_tr = [], []; coin_vec = []
        for p in pools:
            sig = fn(p["c"], p["v"], L, thr)
            s_is = sig & (p["ts"] < split); s_oo = sig & (p["ts"] >= split)
            it = simulate(p["c"], s_is, mh, st, p["cost"]); ot = simulate(p["c"], s_oo, mh, st, p["cost"])
            is_tr += it; oos_tr += ot
            coin_vec.append(sum(ot))   # per-coin OOS net (for cross-combo decorrelation)
        def pf(tr):
            g = sum(t for t in tr if t > 0); l = -sum(t for t in tr if t < 0)
            return (g / l) if l > 0 else (np.inf if g > 0 else 0.0)
        res.append({"combo": key, "arch": arch, "is_n": len(is_tr), "oos_n": len(oos_tr),
                    "is_pf": pf(is_tr), "oos_pf": pf(oos_tr),
                    "oos_net": sum(oos_tr), "oos_avg": np.mean(oos_tr) if oos_tr else 0.0})
        oos_by_coin[key] = coin_vec
    R = pd.DataFrame(res)
    R.to_parquet(DATA / f"_phase3_wfo_{TF}.parquet", index=False)

    def line(d, label):
        if not len(d): print(f"  {label}: none"); return
        print(f"  {label}: n={len(d)}  med_oos_pf={d.oos_pf.replace(np.inf,np.nan).median():.2f}  "
              f"%oos_pf>1={100*(d.oos_pf>1).mean():.0f}%  med_oos_avg/trade={d.oos_avg.median():+.4f}  "
              f"%oos_net>0={100*(d.oos_net>0).mean():.0f}%")
    print(f"\n## ALL {len(R)} combos"); line(R, "all")
    sel = R[(R.is_pf > 1.0) & (R.is_n >= MIN_TRADES_IS)]
    print(f"\n## IS-POSITIVE subset (IS PF>1 & IS_n>={MIN_TRADES_IS}): {len(sel)} combos")
    line(sel, "is-positive -> OOS")
    if len(sel):
        print("  (does IS edge persist OOS? compare med_oos_pf vs all-combos above)")
        for a in sorted(sel.arch.unique()):
            line(sel[sel.arch == a], f"  arch={a}")
        # decorrelation on per-coin OOS pnl among IS-positive combos
        keys = sel.combo.tolist()
        M = np.array([oos_by_coin[k] for k in keys])
        if len(keys) >= 2:
            C = np.corrcoef(M); iu = np.triu_indices(len(keys), 1)
            rr = np.abs(C[iu]); rr = rr[~np.isnan(rr)]
            if len(rr):
                print(f"\n  cross-combo |corr| (per-coin OOS PnL): median={np.median(rr):.3f}  %<0.5={100*(rr<0.5).mean():.0f}%")
        top = sel.sort_values("oos_net", ascending=False).head(8)
        print("\n  top IS-positive combos by OOS net:")
        for _, r in top.iterrows():
            print(f"    {r.combo:32} IS_pf={r.is_pf:.2f} OOS_pf={r.oos_pf:.2f} OOS_net={r.oos_net:+.2f} n={int(r.oos_n)}")
    print(f"\nsaved -> {DATA}/_phase3_wfo_{TF}.parquet")


if __name__ == "__main__":
    main()
