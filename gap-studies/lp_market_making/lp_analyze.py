"""WFO selection + null + full-distribution analysis for passive LP.

Two views of LP PnL (both reported):
  - net        : MARKET-NEUTRAL view = fees + IL + gas (the pure market-making edge,
                 i.e. assuming the directional move of the volatile leg is hedged/ignored).
                 This is the cleanest test of "is the spread harvestable".
  - net_total  : UNHEDGED view = directional price move of 50/50 deposit + fees + IL + gas
                 (what a real passive LP without a hedge actually earns).

WFO selection: per (chain) pool universe, in each window rank pools by IS causal
fee-yield signal, pick TOPK, realize their OOS net APR. NULL: permute which pool's OOS
outcome fills each selected slot (across all pools in that window) -> selection skill destroyed.
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path("./_gaps/lp_market_making")
rng = np.random.default_rng(42)
TOPK = 5

full = pd.read_csv(OUT / "lp_full_allwindows.csv")
conc = pd.read_csv(OUT / "lp_conc_allwindows.csv")


def dist(s):
    s = pd.Series(s).dropna()
    return dict(n=len(s), median=float(s.median()), mean=float(s.mean()),
                p25=float(s.quantile(.25)), p75=float(s.quantile(.75)),
                frac_pos=float((s > 0).mean()))


def summarize(df, label, col):
    d = dist(df[col])
    print(f"  [{label:14s} {col:9s}] n={d['n']:5d} median={d['median']*100:7.2f}% "
          f"mean={d['mean']*100:8.2f}% frac_pos={d['frac_pos']*100:5.1f}%  "
          f"p25={d['p25']*100:7.2f}% p75={d['p75']*100:7.2f}%")
    return dict(label=label, col=col, **d)


print("=" * 90)
print("FULL DISTRIBUTION over ALL pool-windows (no selection), APR (annualized)")
print("=" * 90)
rows = []
for df, name in ((full, "full_range"), (conc, "concentrated")):
    print(f"-- {name} --")
    rows.append(summarize(df, name, "apr"))        # market-neutral fees+IL+gas, annualized
    rows.append(summarize(df, name, "apr_total"))  # unhedged incl directional
# also raw per-window (non-annualized) net
print("-- per-window NET (not annualized), market-neutral view --")
for df, name in ((full, "full_range"), (conc, "concentrated")):
    rows.append(summarize(df, name, "net"))
pd.DataFrame(rows).to_csv(OUT / "lp_distribution.csv", index=False)

# component decomposition (full range, market-neutral)
print("\n" + "=" * 90)
print("COMPONENT DECOMPOSITION (full range, per-window, fraction of capital)")
print("=" * 90)
comp = []
for df, name in ((full, "full_range"), (conc, "concentrated")):
    fee_med = df.fee_pnl.median(); il_med = df.il.median(); gas_med = df.gas_pnl.median()
    print(f"  {name:14s} median fee=+{fee_med*100:6.3f}%  IL={il_med*100:7.3f}%  "
          f"gas={gas_med*100:7.3f}%  -> net={ (fee_med+il_med+gas_med)*100:7.3f}%")
    comp.append(dict(mode=name, fee_pnl_med=fee_med, il_med=il_med, gas_med=gas_med,
                     fee_pnl_mean=df.fee_pnl.mean(), il_mean=df.il.mean(),
                     gas_mean=df.gas_pnl.mean()))
pd.DataFrame(comp).to_csv(OUT / "lp_components.csv", index=False)


# ---------------- WFO SELECTION vs NULL ----------------
def wfo_select(df, col, null=False, n_null=200):
    """For each window, select TOPK pools by IS fee-yield signal; realize OOS `col`.
    REAL: pick top-K by is_yield. NULL: assign each selected SLOT a random pool's OOS
    outcome from the same window pool set (selection skill destroyed)."""
    sel_real = []
    null_means = []
    for win, g in df.groupby("win"):
        g = g.dropna(subset=["is_yield", col])
        if len(g) < TOPK + 2:
            continue
        gg = g.sort_values("is_yield", ascending=False)
        top = gg.head(TOPK)
        sel_real.extend(top[col].tolist())
        if null:
            pool = g[col].to_numpy()
            for _ in range(n_null):
                pick = rng.choice(pool, size=TOPK, replace=False)
                null_means.append(pick.mean())
    return np.array(sel_real), np.array(null_means)


print("\n" + "=" * 90)
print("WFO SELECTION (pick TOPK=5 by IS fee-yield) vs NULL (random pool selection)")
print("=" * 90)
sel_rows = []
for df, name in ((full, "full_range"), (conc, "concentrated")):
    for col in ("apr", "apr_total"):
        real, _ = wfo_select(df, col, null=False)
        _, nullm = wfo_select(df, col, null=True, n_null=300)
        # null distribution of the SELECTED-portfolio mean APR
        real_mean = real.mean(); real_med = np.median(real)
        null_mu = nullm.mean(); null_sd = nullm.std()
        # p-value: fraction of null portfolio-means >= real mean
        # approximate real selected-portfolio mean against null mean distribution
        z = (real_mean - null_mu) / (null_sd + 1e-12)
        pval = float((nullm >= real_mean).mean())
        print(f"  {name:14s} {col:9s}: REAL median={real_med*100:7.2f}% mean={real_mean*100:8.2f}% "
              f"frac_pos={(real>0).mean()*100:5.1f}%  | NULL mean={null_mu*100:8.2f}% "
              f"sd={null_sd*100:6.2f}%  z={z:+5.2f} p(null>=real)={pval:.3f}")
        sel_rows.append(dict(mode=name, col=col, real_median=real_med, real_mean=real_mean,
                             real_frac_pos=float((real > 0).mean()), null_mean=null_mu,
                             null_sd=null_sd, z=z, p_null_ge_real=pval, n_sel=len(real)))
pd.DataFrame(sel_rows).to_csv(OUT / "lp_wfo_vs_null.csv", index=False)

# ---------------- Compare to directional negative result (phase9 corpus) ----------------
print("\n" + "=" * 90)
print("COMPARISON: passive LP vs the directional-timing OOS result")
print("=" * 90)
try:
    ph9 = pd.read_parquet("./data/_phase9_corpus.parquet")
    print(f"  phase9 surviving (coin,family): n={len(ph9)} median OOS PF={ph9.oos_pf.median():.3f} "
          f"frac net>0={(ph9.oos_net>0).mean()*100:.1f}%")
except Exception as e:
    print("  phase9 load failed:", e)

# survivorship note + fee-tier breakdown
print("\n-- LP net APR (market-neutral) by fee tier (full range) --")
ft = full.groupby("fee_bps").apr.agg(["median", "mean", "count"])
print(ft.to_string())
ft.to_csv(OUT / "lp_by_feetier.csv")
print("\nDONE")
