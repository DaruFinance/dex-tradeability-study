"""Median-based WFO-selection-vs-null test (robust to the heavy APR tail) on the realistic
(volcap+LVR) LP outcomes. Also tests an alternative selection signal (high reserve = blue-chip)
and reports selected-portfolio MEDIAN APR vs the null distribution of medians.
"""
import numpy as np, pandas as pd
from pathlib import Path
G = Path("./_gaps/lp_market_making")
rng = np.random.default_rng(7)
TOPK = 5; NNULL = 1000


def sel_test(df, sig, col="apr", asc=False):
    """Portfolio = window-wise top-K by signal `sig`. Compare REAL portfolio MEDIAN of `col`
    to null distribution of medians (random K per window)."""
    real_vals = []; per_win_real = []; per_win_pools = []
    for win, g in df.groupby("win"):
        g = g.dropna(subset=[sig, col])
        if len(g) < TOPK + 2:
            continue
        top = g.sort_values(sig, ascending=asc).head(TOPK)
        real_vals.extend(top[col].tolist())
        per_win_pools.append(g[col].to_numpy())
    real_med = float(np.median(real_vals)); real_mean = float(np.mean(real_vals))
    real_fp = float((np.array(real_vals) > 0).mean())
    null_meds = []
    for _ in range(NNULL):
        picks = []
        for pool in per_win_pools:
            picks.extend(rng.choice(pool, TOPK, replace=False))
        null_meds.append(np.median(picks))
    null_meds = np.array(null_meds)
    p = float((null_meds >= real_med).mean())
    return dict(real_med=real_med, real_mean=real_mean, real_fp=real_fp,
                null_med_mean=float(null_meds.mean()), null_med_sd=float(null_meds.std()),
                p_null_ge_real=p, n=len(real_vals))


rows = []
for tag in ("realistic",):
    for mode in ("full", "conc"):
        d = pd.read_csv(G / f"lp_{mode}_{tag}.csv")
        for sig, asc, name in (("is_yield", False, "IS_fee_yield"),
                               ("reserve", False, "high_reserve_bluechip")):
            r = sel_test(d, sig)
            r.update(mode=mode, signal=name)
            rows.append(r)
            print(f"  {mode:5s} sel={name:22s}: REAL median={r['real_med']*100:8.2f}% "
                  f"mean={r['real_mean']*100:9.2f}% frac_pos={r['real_fp']*100:5.1f}% | "
                  f"NULL median~{r['null_med_mean']*100:7.2f}% (sd {r['null_med_sd']*100:.2f}%) "
                  f"p(null>=real)={r['p_null_ge_real']:.3f}")
pd.DataFrame(rows).to_csv(G / "lp_select_robust.csv", index=False)
print("\nsaved lp_select_robust.csv")
