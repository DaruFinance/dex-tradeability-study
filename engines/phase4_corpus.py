"""Phase 4: build the per-(archetype,params,coin) strategy corpus and measure decorrelation.

The unit of a 'strategy' here is (archetype, params) applied to ONE coin. Decorrelation comes from
the COIN cross-section (idiosyncratic price paths), not param variants. IS->OOS gate per strategy;
decorrelation measured on time-bucketed OOS PnL (the playbook's per-bar-PnL correlation, not summary
stats). Reports: # kept strategies, OOS PF distribution, median pairwise |corr|, %<0.5, and effective
independent dimensions (participation ratio).

⚠ survivor-seeded universe -> optimistic & drift-confounded; concept measurement, not production.

Usage: python3 _research/phase4_corpus.py <tf> <size_frac> <is_split> <min_is_trades_per_coin>
"""
from __future__ import annotations
import json, glob, sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
TF = sys.argv[1] if len(sys.argv) > 1 else "hour"
SIZE_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0025
IS_SPLIT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
MIN_IS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
CHAIN_GAS = {"bsc": 0.20, "base": 0.02, "eth": 3.0, "arbitrum": 0.05, "solana": 0.02}
CHAIN_NATIVE = {"bsc": 640.0, "base": 3500.0, "eth": 3500.0, "arbitrum": 3500.0, "solana": 180.0}
DEX_DEFAULT = {"bsc": "pancakeswap", "base": "uniswap", "eth": "uniswap", "arbitrum": "uniswap"}

def sig_momentum(c, v, L, thr): return pd.Series(c).shift(1).pct_change(L).to_numpy() > thr
def sig_breakout(c, v, L, thr):
    s = pd.Series(c).shift(1); d = s.rolling(L).max().to_numpy(); return (s.to_numpy() >= d) & ~np.isnan(d)
def sig_reversion(c, v, L, thr):
    s = pd.Series(c).shift(1); z = ((s - s.rolling(L).mean()) / s.rolling(L).std()).to_numpy(); return z < -thr
def sig_volsurge(c, v, L, thr):
    vs = pd.Series(v).shift(1); b = vs.rolling(L).mean().to_numpy()
    return (vs.to_numpy() > thr * b) & (pd.Series(c).shift(1).pct_change().to_numpy() > 0) & ~np.isnan(b)
ARCH = {"momentum": sig_momentum, "breakout": sig_breakout, "reversion": sig_reversion, "volsurge": sig_volsurge}
GRID = {"momentum": [(L, t) for L in (6, 12, 24, 48) for t in (0.05, 0.15, 0.30)],
        "breakout": [(L, 0) for L in (12, 24, 48, 96)],
        "reversion": [(L, t) for L in (12, 24, 48) for t in (1.0, 1.5, 2.0)],
        "volsurge": [(L, t) for L in (12, 24, 48) for t in (2.0, 3.0, 5.0)]}
MAXHOLD = [6, 12, 24]; STOP = [0.15, 0.30]

def simulate_t(c, ts, sig, max_hold, stop, cost):
    """Return list of (exit_ts, net_ret). Causal entries at bar i (sig from <=i-1), exit by hold/stop."""
    n = len(c); i = 0; out = []
    while i < n - 1:
        if sig[i] and c[i] > 0:
            ep = c[i]; ex = min(i + max_hold, n - 1); sp = ep * (1 - stop); j = i + 1
            while j <= ex and c[j] > sp: j += 1
            j = min(j, ex); out.append((ts[j], c[j] / ep - 1.0 - cost)); i = j + 1
        else: i += 1
    return out

def main():
    univ = {}; cexv = {}
    for f in glob.glob(f"{DATA}/_gt_universe*.jsonl"):
        for l in open(f):
            r = json.loads(l); univ[r["pair"]] = r
    cf = DATA / "_cex_verdicts_universe.jsonl"
    for l in open(cf):
        r = json.loads(l); cexv[r["pair"]] = r["status"]
    files = glob.glob(f"{DATA}/ohlcv_gt/*{TF}*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["close"])
    pools = []
    for pair, g in df.groupby("pair_address"):
        if len(g) < 80 or cexv.get(pair) != "dex_only": continue
        g = g.sort_values("ts"); u = univ.get(pair, {}); chain = u.get("chain", g.chain.iloc[0]); reserve = u.get("reserve_usd")
        size = max(50.0, SIZE_FRAC * (reserve or 0))
        cost = round_trip_cost_frac(size, reserve, dex=DEX_DEFAULT.get(chain, "uniswap"), chain=chain,
                                    gas_usd=CHAIN_GAS.get(chain, 0.1), native_usd=CHAIN_NATIVE.get(chain)) if reserve else 0.05
        pools.append({"pair": pair, "chain": chain, "c": g.close.to_numpy(), "v": g.volume.fillna(0).to_numpy(),
                      "ts": g.ts.to_numpy(), "cost": cost})
    if len(pools) < 5:
        print(f"only {len(pools)} dex_only coins with >=80 {TF} bars; need more (rerun after hourly ingest)."); return
    tmin = min(p["ts"].min() for p in pools); tmax = max(p["ts"].max() for p in pools)
    split = tmin + IS_SPLIT * (tmax - tmin)
    combos = [(a, L, thr, mh, st) for a in ARCH for (L, thr) in GRID[a] for mh in MAXHOLD for st in STOP]
    print(f"{len(pools)} dex_only coins, tf={TF}, {len(combos)} combos -> up to {len(pools)*len(combos)} (combo,coin) strategies")

    # OOS time buckets for correlation (daily)
    bucket = lambda t: int(t // 86400)
    all_buckets = sorted({bucket(t) for p in pools for t in p["ts"] if t >= split})
    bidx = {b: i for i, b in enumerate(all_buckets)}
    strategies = []  # dicts with is_pf, oos_pf, oos_n, oos_net, pnl_vec
    for (arch, L, thr, mh, st) in combos:
        fn = ARCH[arch]
        for p in pools:
            sig = fn(p["c"], p["v"], L, thr)
            is_tr = simulate_t(p["c"], p["ts"], sig & (p["ts"] < split), mh, st, p["cost"])
            if len(is_tr) < MIN_IS: continue
            gi = sum(r for _, r in is_tr if r > 0); li = -sum(r for _, r in is_tr if r < 0)
            is_pf = gi / li if li > 0 else (np.inf if gi > 0 else 0)
            if is_pf <= 1.0: continue
            oos_tr = simulate_t(p["c"], p["ts"], sig & (p["ts"] >= split), mh, st, p["cost"])
            if not oos_tr: continue
            go = sum(r for _, r in oos_tr if r > 0); lo = -sum(r for _, r in oos_tr if r < 0)
            oos_pf = go / lo if lo > 0 else (np.inf if go > 0 else 0)
            vec = np.zeros(len(all_buckets))
            for tts, r in oos_tr:
                vec[bidx[bucket(tts)]] += r
            strategies.append({"key": f"{arch}_L{L}_t{thr}_h{mh}_s{st}@{p['pair'][:10]}", "arch": arch,
                               "coin": p["pair"], "is_pf": is_pf, "oos_pf": oos_pf,
                               "oos_n": len(oos_tr), "oos_net": sum(r for _, r in oos_tr), "vec": vec})
    print(f"\nIS-positive (combo,coin) strategies: {len(strategies)}")
    if len(strategies) < 2:
        print("too few; rerun on hourly with more coins."); return
    S = pd.DataFrame([{k: v for k, v in s.items() if k != "vec"} for s in strategies])
    finite_pf = S.oos_pf.replace(np.inf, np.nan)
    print(f"  OOS PF: median={finite_pf.median():.2f}  %OOS_pf>1={100*(S.oos_pf>1).mean():.0f}%  "
          f"%OOS_net>0={100*(S.oos_net>0).mean():.0f}%  median_oos_n={S.oos_n.median():.0f}")
    print(f"  archetype mix: {S.arch.value_counts().to_dict()}")
    print(f"  distinct coins represented: {S.coin.nunique()}")
    # decorrelation on time-bucketed OOS PnL
    M = np.array([s["vec"] for s in strategies])
    # keep buckets with some activity
    M = M[:, M.any(axis=0)]
    Csub = M - M.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Csub, axis=1, keepdims=True); norm[norm == 0] = 1
    Cn = Csub / norm
    corr = Cn @ Cn.T
    iu = np.triu_indices(len(strategies), 1)
    rr = np.abs(corr[iu]); rr = rr[~np.isnan(rr)]
    if len(rr):
        print(f"\n  pairwise |corr| on OOS daily PnL: median={np.median(rr):.3f}  mean={np.mean(rr):.3f}  %<0.5={100*(rr<0.5).mean():.0f}%")
    # effective independent dimensions via participation ratio of eigenvalues of correlation matrix
    try:
        ev = np.linalg.eigvalsh(corr + 1e-9 * np.eye(len(strategies)))
        ev = ev[ev > 0]; pr = (ev.sum() ** 2) / (ev ** 2).sum()
        print(f"  effective independent dimensions (participation ratio): {pr:.1f}  out of {len(strategies)} strategies")
    except Exception as e:
        print("  eig failed:", e)
    S.assign().to_parquet(DATA / f"_phase4_corpus_{TF}.parquet", index=False)
    print(f"\nsaved -> {DATA}/_phase4_corpus_{TF}.parquet")

if __name__ == "__main__":
    main()
