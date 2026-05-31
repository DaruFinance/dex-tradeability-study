"""Close-to-close (no-bracket) execution bound (referee fix R2).

The bracketed timing tests read intrabar high/low, the least trustworthy field on a thin pool
(one MEV or wash print sets the extreme). This variant removes the dependence entirely: enter at
the close of the signal bar, exit at the close after max_hold, with NO intrabar TP/SL. Same 16
families, same causal signals, same disjoint WFO, same per-fill cost, real vs bar-shuffle null.

Usage: python3 run_c2c.py <max_coins> <ns> <day|hour> [null]
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
sys.path.insert(0, ".")
import run_bracket as rb
from chainscope.costs import round_trip_cost_frac

DATA = rb.DATA; OUT = rb.OUT
MAXC = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
NS   = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TF   = sys.argv[3] if len(sys.argv) > 3 else "day"
NULL = (len(sys.argv) > 4 and sys.argv[4] == "null")
if TF == "hour": MIN_BARS, IS_LEN, OOS_LEN = 360, 240, 120
else:            MIN_BARS, IS_LEN, OOS_LEN = 80, 90, 45
MIN_IS_TR, MIN_OOS_TR = 5, 4
SIZE_FRAC = 0.0025
rng = np.random.default_rng(42)

def sim_c2c(c, entry, max_hold, cost):
    """Enter at close i, exit at close min(i+max_hold, n-1). No intrabar touch."""
    n = len(c); i = 0; out = []
    while i < n - 1:
        if entry[i] and c[i] > 0:
            end = min(i + max_hold, n - 1)
            out.append(c[end] / c[i] - 1.0 - cost); i = end + 1
        else:
            i += 1
    return out

def main():
    univ = {}
    for ln in open(DATA / "_mega_universe.jsonl"):
        r = json.loads(ln); univ[r["pair"]] = r
    files = glob.glob(f"{DATA}/ohlcv_gt/mega_*_{TF}.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["close","high","low"])
    df = df[df.close > 0].sort_values(["pair_address","ts"])
    coins = set([p for p, _ in df.groupby("pair_address")][:MAXC])
    print(f"{len(coins)} coins, close-to-close (no bracket), tf={TF}, NULL={NULL}", flush=True)

    t0 = time.time(); evals = []; n_eval = 0
    for ci, (pair, g) in enumerate(df.groupby("pair_address")):
        if pair not in coins: continue
        o, h, l, c, v = (g.open.to_numpy(float), g.high.to_numpy(float), g.low.to_numpy(float),
                         g.close.to_numpy(float), g.volume.fillna(0).to_numpy(float))
        n = len(c)
        if n < MIN_BARS: continue
        if NULL:
            perm = rng.permutation(n); o, h, l, c, v = o[perm], h[perm], l[perm], c[perm], v[perm]
        u = univ.get(pair, {}); res = u.get("reserve_usd") or 0; ch = u.get("chain", "bsc")
        size = max(50., SIZE_FRAC * res)
        cost = round_trip_cost_frac(size, res, dex="uniswap", chain=ch, gas_usd=rb.CHAIN_GAS.get(ch, .05),
                                    native_usd=rb.CHAIN_NATIVE.get(ch, 100)) if res else .05
        starts = list(range(0, n - IS_LEN - OOS_LEN + 1, OOS_LEN))
        if not starts: continue
        for fname, (fn, sampler) in rb.FAMILIES.items():
            n_eval += 1; oos = []
            for st in starts:
                isl = slice(st, st + IS_LEN); osl = slice(st + IS_LEN, st + IS_LEN + OOS_LEN)
                best = None
                for _ in range(NS):
                    p = sampler(); mh = int(rng.choice([5, 10, 20, 40]))
                    sig = fn(o, h, l, c, v, p)
                    is_tr = sim_c2c(c[isl], sig[isl], mh, cost)
                    if len(is_tr) < MIN_IS_TR: continue
                    s = rb.pf(is_tr)
                    if best is None or s > best[0]: best = (s, p, mh)
                if best is None: continue
                _, p, mh = best; sig = fn(o, h, l, c, v, p)
                oos += sim_c2c(c[osl], sig[osl], mh, cost)
            if len(oos) >= MIN_OOS_TR:
                P = rb.pf(oos); net = sum(oos)
                evals.append({"coin": pair, "chain": ch, "family": fname, "oos_pf": P,
                              "oos_net": net, "oos_n": len(oos), "avg_per_trade": net / len(oos)})
        if (ci + 1) % 300 == 0:
            print(f"  [{ci+1}] n_eval={n_eval} ({time.time()-t0:.0f}s)", flush=True)

    tag = "null" if NULL else "real"
    A = pd.DataFrame(evals); A.to_csv(OUT / f"evals_c2c_{tag}_{TF}.csv", index=False)
    apf = A.oos_pf.replace(np.inf, np.nan)
    row = {"tag": tag, "tf": TF, "rule": "c2c", "n_eval": len(A),
           "median_oos_pf": apf.median(), "mean_oos_pf_cap10": apf.clip(upper=10).mean(),
           "pct_pf_gt1": 100 * (A.oos_pf > 1).mean(),
           "median_net_per_trade": A.avg_per_trade.median(), "mean_net_per_trade": A.avg_per_trade.mean()}
    pd.DataFrame([row]).to_csv(OUT / f"summary_c2c_{tag}_{TF}.csv", index=False)
    print(f"=== c2c {tag} {TF} ===  {row}", flush=True)

if __name__ == "__main__":
    main()
