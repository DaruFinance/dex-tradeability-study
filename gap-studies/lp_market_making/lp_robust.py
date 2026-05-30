"""Robustness layer for passive-LP gap:
 (1) cap implausible daily volume (wash/reporting artifacts) at a realistic turnover,
 (2) apply LVR (loss-versus-rebalancing / adverse selection) haircut from realized variance:
     LVR per bar ~ 0.5 * (fee-adjusted) sigma^2 * lp_share-scaled position value.
     Milionis-Moallemi-Roughgarden: a v2 LP's instantaneous arbitrage loss rate = (1/8) * sigma^2
     of price (per unit time) on the FULL position value (not just our share fraction of fees).
     We compute realized per-bar log-return variance from OHLCV and charge LVR = sum 0.125*ret^2
     against the position. This is the dominant cost real LPs face beyond close-to-close IL.
 (3) re-run the FULL DISTRIBUTION + WFO-vs-NULL on the haircut net.

We recompute net from stored components: fees are recomputed with capped volume; IL & gas reused;
LVR subtracted. Reproduces the headline under a defensible toxic-flow model.
"""
import numpy as np, pandas as pd, glob, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from chainscope.costs import swap_fee_bps, gas_cost_usd
OUT = Path("./data")
G = Path("./_gaps/lp_market_making")
rng = np.random.default_rng(42)
TF = "day"; MIN_BARS = 80; IS_LEN, OOS_LEN = 90, 45; SIZE_FRAC = 0.0025; TOPK = 5
CONC_BAND = 0.30
CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05, "polygon_pos": .01,
             "optimism": .05, "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "optimism": 3500, "sui-network": 3.5, "solana": 180, "tron": .3}
# Realistic max daily turnover an LP can actually earn fees on. Healthy AMM pools rarely
# sustain >2-3x TVL/day from organic flow; cap at 3x (anything above flagged as wash/artifact).
VOL_CAP_DAILY = 3.0


def gas_per_swap(chain):
    return gas_cost_usd(chain, native_usd=CHAIN_NATIVE.get(chain, 100), dex=None) if chain == "bsc" \
        else CHAIN_GAS.get(chain, .05)


def il_v2(r):
    return 2.0 * np.sqrt(r) / (1.0 + r) - 1.0


def conc_mult(band):
    return 1.0 / (1.0 - np.sqrt(1.0 / (1.0 + band)))


def load_universe():
    meta = {}
    for line in open(OUT / "_mega_universe.jsonl"):
        d = json.loads(line); meta[(d["chain"], d["pair"])] = d
    return meta


def sim(c, v, reserve, fee_frac, chain, mode, vol_cap=True, lvr=True):
    c = np.asarray(c, float); v = np.asarray(v, float); n = len(c)
    if n < 2 or not reserve or reserve <= 0 or c[0] <= 0 or np.nanmin(c) <= 0:
        return None
    P = max(50.0, SIZE_FRAC * reserve); lp_share = P / (reserve + P)
    gas1 = gas_per_swap(chain)
    vv = v[1:].copy()
    if vol_cap:
        vv = np.minimum(vv, VOL_CAP_DAILY * reserve)   # cap per-bar volume at turnover ceiling
    r = c[-1] / c[0]
    # realized per-bar log returns for LVR / variance
    lr = np.diff(np.log(c))
    var_sum = float(np.nansum(lr ** 2))
    lvr_frac = 0.125 * var_sum if lvr else 0.0          # v2 arbitrage-loss rate integral
    if mode == "full":
        fees = np.nansum(vv) * fee_frac * lp_share / P
        il = il_v2(r); boost = 1.0
    else:
        boost = conc_mult(CONC_BAND)
        pa, pb = c[0] / (1.0 + CONC_BAND), c[0] * (1.0 + CONC_BAND)
        lo, hi = pa, pb; inr = np.zeros(n - 1, bool); reb = 0
        for i in range(1, n):
            px = c[i]
            if lo <= px <= hi:
                inr[i - 1] = True
            else:
                reb += 1; lo, hi = px / (1.0 + CONC_BAND), px * (1.0 + CONC_BAND); inr[i - 1] = True
        fees = np.nansum(vv * inr) * fee_frac * lp_share * boost / P
        il = np.clip(il_v2(r) * boost, -1.0, 0.0)
        gas1_total = 2.0 * gas1 + reb * gas1
        gas_frac = -gas1_total / P
        lvr_frac = lvr_frac * boost
        net = fees + il + gas_frac - lvr_frac
        apr = net * 365.0 / n
        return dict(fees=fees, il=il, gas=gas_frac, lvr=-lvr_frac, net=net, apr=apr, n=n, P=P,
                    lp_share=lp_share, reb=reb)
    gas_frac = -2.0 * gas1 / P
    net = fees + il + gas_frac - lvr_frac
    apr = net * 365.0 / n
    return dict(fees=fees, il=il, gas=gas_frac, lvr=-lvr_frac, net=net, apr=apr, n=n, P=P,
                lp_share=lp_share, reb=0)


def run(vol_cap, lvr, tag):
    meta = load_universe()
    rows = {"full": [], "conc": []}
    for f in sorted(glob.glob(str(OUT / f"ohlcv_gt/mega_*_{TF}.parquet"))):
        df = pd.read_parquet(f)
        for (chain, pair), g in df.groupby(["chain", "pair_address"]):
            g = g.sort_values("ts")
            if len(g) < MIN_BARS:
                continue
            m = meta.get((chain, pair))
            if not m:
                continue
            reserve = m.get("reserve_usd"); fee_frac = swap_fee_bps(m.get("dex")) / 1e4
            c = g.close.to_numpy(); v = g.volume.to_numpy()
            if np.any(~np.isfinite(c)) or np.nanmin(c) <= 0:
                continue
            nb = len(c); start = 0; win = 0
            while start + IS_LEN + OOS_LEN <= nb:
                isv = np.nansum(v[start:start + IS_LEN])
                is_yield = isv * fee_frac / reserve / IS_LEN
                oos = slice(start + IS_LEN, start + IS_LEN + OOS_LEN)
                for mode in ("full", "conc"):
                    res = sim(c[oos], v[oos], reserve, fee_frac, chain, mode, vol_cap, lvr)
                    if res:
                        res.update(chain=chain, pair=pair, win=win, is_yield=is_yield,
                                   fee_bps=fee_frac * 1e4, reserve=reserve)
                        rows[mode].append(res)
                start += OOS_LEN; win += 1
    out = {}
    for mode in ("full", "conc"):
        d = pd.DataFrame(rows[mode]); d.to_csv(G / f"lp_{mode}_{tag}.csv", index=False); out[mode] = d
    return out


def wfo_vs_null(df, col="apr", n_null=300):
    real = []; nullm = []
    for win, g in df.groupby("win"):
        g = g.dropna(subset=["is_yield", col])
        if len(g) < TOPK + 2:
            continue
        top = g.sort_values("is_yield", ascending=False).head(TOPK)
        real.extend(top[col].tolist())
        pool = g[col].to_numpy()
        for _ in range(n_null):
            nullm.append(rng.choice(pool, TOPK, replace=False).mean())
    real = np.array(real); nullm = np.array(nullm)
    return real, nullm


def report(out, tag):
    print(f"\n{'='*92}\nROBUST RUN [{tag}] — volume cap={VOL_CAP_DAILY}x/day, LVR haircut applied\n{'='*92}")
    rows = []
    for mode in ("full", "conc"):
        d = out[mode]
        med = d.apr.median(); mean = d.apr.mean(); fp = (d.apr > 0).mean()
        fee = d.fees.median(); il = d.il.median(); lv = d.lvr.median(); ga = d.gas.median()
        print(f"  {mode:5s} APR median={med*100:7.2f}% mean={mean*100:9.2f}% frac_pos={fp*100:5.1f}% "
              f"| comps(med/win) fee=+{fee*100:6.3f}% IL={il*100:6.3f}% LVR={lv*100:6.3f}% gas={ga*100:6.3f}%")
        real, nullm = wfo_vs_null(d, "apr")
        rmean = real.mean(); rmed = np.median(real); nmu = nullm.mean(); nsd = nullm.std()
        z = (rmean - nmu) / (nsd + 1e-12); p = float((nullm >= rmean).mean())
        print(f"        WFO-select REAL median={rmed*100:8.2f}% mean={rmean*100:9.2f}% frac_pos={(real>0).mean()*100:5.1f}%"
              f" | NULL mean={nmu*100:8.2f}% sd={nsd*100:7.2f}% z={z:+5.2f} p={p:.3f}")
        rows.append(dict(tag=tag, mode=mode, apr_med=med, apr_mean=mean, frac_pos=fp,
                         fee_med=fee, il_med=il, lvr_med=lv, gas_med=ga,
                         sel_real_med=rmed, sel_real_mean=rmean, sel_real_fp=float((real > 0).mean()),
                         null_mean=nmu, null_sd=nsd, z=z, p=p))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    allrows = []
    # uncapped, no-LVR (theoretical max — matches lp_backtest)
    out0 = run(vol_cap=False, lvr=False, tag="theoretical"); allrows += report(out0, "theoretical_max")
    # capped volume only
    out1 = run(vol_cap=True, lvr=False, tag="volcap"); allrows += report(out1, "volcap_only")
    # capped + LVR (the realistic case)
    out2 = run(vol_cap=True, lvr=True, tag="realistic"); allrows += report(out2, "volcap+LVR")
    # LVR only (uncapped)
    out3 = run(vol_cap=False, lvr=True, tag="lvronly"); allrows += report(out3, "LVR_only")
    pd.DataFrame(allrows).to_csv(G / "lp_robust_summary.csv", index=False)
    print(f"\ntotal t={time.time()-t0:.1f}s  -> lp_robust_summary.csv")
