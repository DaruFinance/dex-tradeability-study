"""Close-to-close (no-bracket) bound, parallelized over coins (all cores).

Same as run_c2c.py but processes coins with a fork-based multiprocessing pool (data shared
copy-on-write), and does day+hour x real+null in one driver. Enter at close of signal bar, exit
at close after max_hold; NO intrabar TP/SL. 16 families, causal signals, disjoint WFO, per-fill
cost, bar-shuffle null. Per-coin RNG (crc32 seed) => deterministic and parallel-safe.
"""
from __future__ import annotations
import glob, json, sys, time, zlib, os
from pathlib import Path
import numpy as np, pandas as pd
import multiprocessing as mp
sys.path.insert(0, ".")
sys.path.insert(0, ".")
import run_bracket as rb
from chainscope.costs import round_trip_cost_frac

DATA = rb.DATA; OUT = rb.OUT
NS = 12; MIN_IS_TR, MIN_OOS_TR = 5, 4; SIZE_FRAC = 0.0025
NPROC = min(30, os.cpu_count() or 8)
COIN = {}; PAR = {}   # globals inherited by forked workers

def sim_c2c(c, entry, max_hold, cost):
    n = len(c); i = 0; out = []
    while i < n - 1:
        if entry[i] and c[i] > 0:
            end = min(i + max_hold, n - 1)
            out.append(c[end] / c[i] - 1.0 - cost); i = end + 1
        else:
            i += 1
    return out

def worker(args):
    pair, null = args
    o, h, l, c, v, res, ch = COIN[pair]
    o, h, l, c, v = o.copy(), h.copy(), l.copy(), c.copy(), v.copy()
    r = np.random.default_rng(zlib.crc32(pair.encode()) ^ (0x9e3779b9 if null else 0))
    rb.rng = r
    if null:
        perm = r.permutation(len(c)); o, h, l, c, v = o[perm], h[perm], l[perm], c[perm], v[perm]
    IS_LEN, OOS_LEN, MIN_BARS = PAR["IS_LEN"], PAR["OOS_LEN"], PAR["MIN_BARS"]
    n = len(c)
    if n < MIN_BARS: return []
    cost = round_trip_cost_frac(max(50., SIZE_FRAC*res), res, dex="uniswap", chain=ch,
            gas_usd=rb.CHAIN_GAS.get(ch, .05), native_usd=rb.CHAIN_NATIVE.get(ch, 100)) if res else .05
    starts = list(range(0, n - IS_LEN - OOS_LEN + 1, OOS_LEN))
    if not starts: return []
    rows = []
    for fname, (fn, sampler) in rb.FAMILIES.items():
        oos = []
        for st in starts:
            isl = slice(st, st+IS_LEN); osl = slice(st+IS_LEN, st+IS_LEN+OOS_LEN)
            best = None
            for _ in range(NS):
                p = sampler(); mh = int(r.choice([5, 10, 20, 40]))
                sig = fn(o, h, l, c, v, p)
                is_tr = sim_c2c(c[isl], sig[isl], mh, cost)
                if len(is_tr) < MIN_IS_TR: continue
                s = rb.pf(is_tr)
                if best is None or s > best[0]: best = (s, p, mh)
            if best is None: continue
            _, p, mh = best; sig = fn(o, h, l, c, v, p)
            oos += sim_c2c(c[osl], sig[osl], mh, cost)
        if len(oos) >= MIN_OOS_TR:
            rows.append({"coin": pair, "chain": ch, "oos_pf": rb.pf(oos), "oos_net": sum(oos),
                         "oos_n": len(oos), "avg_per_trade": sum(oos)/len(oos)})
    return rows

def run_tf(tf):
    global COIN, PAR
    if tf == "hour": PAR = {"MIN_BARS":360, "IS_LEN":240, "OOS_LEN":120}
    else:            PAR = {"MIN_BARS":80,  "IS_LEN":90,  "OOS_LEN":45}
    univ = {}
    for ln in open(DATA/"_mega_universe.jsonl"):
        rr = json.loads(ln); univ[rr["pair"]] = rr
    files = glob.glob(f"{DATA}/ohlcv_gt/mega_*_{tf}.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["close","high","low"])
    df = df[df.close > 0].sort_values(["pair_address","ts"])
    COIN = {}
    for pair, g in df.groupby("pair_address"):
        u = univ.get(pair, {})
        COIN[pair] = (g.open.to_numpy(float), g.high.to_numpy(float), g.low.to_numpy(float),
                      g.close.to_numpy(float), g.volume.fillna(0).to_numpy(float),
                      u.get("reserve_usd") or 0, u.get("chain", "bsc"))
    pairs = list(COIN)
    print(f"[{tf}] {len(pairs)} coins, {NPROC} workers", flush=True)
    for null in (False, True):
        t0 = time.time()
        with mp.Pool(NPROC) as pool:
            res = pool.map(worker, [(p, null) for p in pairs], chunksize=20)
        rows = [x for sub in res for x in sub]
        A = pd.DataFrame(rows); tag = "null" if null else "real"
        A.to_csv(OUT/f"evals_c2c_{tag}_{tf}.csv", index=False)
        apf = A.oos_pf.replace(np.inf, np.nan)
        summ = {"tag":tag, "tf":tf, "rule":"c2c", "n_eval":len(A),
                "median_oos_pf":apf.median(), "mean_oos_pf_cap10":apf.clip(upper=10).mean(),
                "pct_pf_gt1":100*(A.oos_pf>1).mean(),
                "median_net_per_trade":A.avg_per_trade.median(), "mean_net_per_trade":A.avg_per_trade.mean()}
        pd.DataFrame([summ]).to_csv(OUT/f"summary_c2c_{tag}_{tf}.csv", index=False)
        print(f"[{tf} {tag}] {len(A)} evals in {time.time()-t0:.0f}s | "
              f"medPF={summ['median_oos_pf']:.3f} %PF>1={summ['pct_pf_gt1']:.1f} "
              f"med_net/trade={summ['median_net_per_trade']:+.4f}", flush=True)

def main():
    for tf in ("day", "hour"):
        run_tf(tf)
    print("C2C_PAR_DONE", flush=True)

if __name__ == "__main__":
    main()
