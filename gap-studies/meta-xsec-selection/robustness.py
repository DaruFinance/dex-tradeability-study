"""Robustness + survivorship attribution for the metadata cross-sectional selection gap.

(1) IS-split sweep: does the (weak, negative) ranking hold across 0.5/0.6/0.7 splits?
(2) Survivorship attribution for snap_res (highest-reserve pick), the only feature with z>2 vs
    null: is its marginal ranking ability just 'big pools didn't rug'? We split coins into
    SURVIVORS (final price > -50% from start) vs CASUALTIES and report the feature's mean/median
    within each group + a within-survivor-only selection test (rank only among survivors).
Reuses the same engine logic, daily panel, per-coin cost, winsorized picks, median-coin benchmark,
across-coin null.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
OUT = Path("./_gaps/meta-xsec-selection")
SIZE_FRAC = 0.0025
rng = np.random.default_rng(11)
N_NULL = 20
CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05,
             "polygon_pos": .01, "optimism": .05, "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "optimism": 3500, "sui-network": 3.5, "solana": 180, "tron": .3}

def coin_cost(chain, reserve):
    if not reserve: return .05
    size = max(50., SIZE_FRAC * reserve)
    return round_trip_cost_frac(size, reserve, dex="uniswap", chain=chain,
                                gas_usd=CHAIN_GAS.get(chain, .05), native_usd=CHAIN_NATIVE.get(chain, 100))

univ = {}
for l in open(DATA / "_mega_universe.jsonl"):
    r = json.loads(l); univ[r["pair"]] = r
df = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/ohlcv_gt/mega_*_day.parquet")], ignore_index=True)
df = df.dropna(subset=["close", "volume"]); df = df[df.close > 0]
df["day"] = (df.ts // 86400).astype(int)
keep = {p for p, g in df.groupby("pair_address") if g.close.pct_change().std() > 0.01 and len(g) >= 60}
df = df[df.pair_address.isin(keep)]
px = df.pivot_table(index="day", columns="pair_address", values="close", aggfunc="last").sort_index()
dv = df.pivot_table(index="day", columns="pair_address", values="volume", aggfunc="sum").reindex_like(px)
pairs = list(px.columns); days = px.index.to_numpy()
cost = pd.Series({p: coin_cost(univ.get(p, {}).get("chain", "bsc"), univ.get(p, {}).get("reserve_usd") or 0) for p in pairs})

def winsor(ff): return ff.clip(upper=ff.quantile(0.95))
def fwd(R): return px.shift(-R) / px - 1.0

def run(svec, K, R, asc, split_day, restrict=None, null=False):
    f = fwd(R); rec = []
    sv0 = svec.dropna()
    if restrict is not None: sv0 = sv0[sv0.index.isin(restrict)]
    for i in range(0, len(days) - R, R):
        ff = f.iloc[i]; alive = ff.dropna().index
        cand = sv0.index.intersection(alive)
        if len(cand) < 3 * K: continue
        sv = sv0.loc[cand]
        if null: sv = pd.Series(rng.permutation(sv.values), index=sv.index)
        picks = sv.sort_values(ascending=asc).index[:K]
        ffa = winsor(ff.loc[cand]); costv = cost.reindex(picks).fillna(.05)
        strat = (ffa.reindex(picks).fillna(ffa.median()) - costv).mean()
        bench = ff.loc[cand].median()
        rec.append((days[i] < split_day, strat - bench, strat))
    if not rec: return None
    a = np.array(rec); m = a[:, 0] == 1
    return dict(oos_edge=a[~m, 1].mean() if (~m).any() else np.nan,
                oos_strat=a[~m, 2].mean() if (~m).any() else np.nan,
                is_edge=a[m, 1].mean() if m.any() else np.nan)

# ---- (1) IS-split sweep on snap_res desc (best feature) and turnover asc ----
snap_res = pd.Series({p: univ.get(p, {}).get("reserve_usd") or np.nan for p in pairs})
turnover = (dv.rolling(14).mean() / snap_res).shift(1)
rows = []
for split_frac in (0.5, 0.6, 0.7):
    sd = days[int(split_frac * len(days))]
    for K in (5, 10, 20):
        for R in (7, 14, 28):
            for name, sv, asc, tv in [("snap_res", snap_res, False, False), ("turnover", turnover, True, True)]:
                r = run(sv, K, R, asc, sd)
                if r is None: continue
                nulls = [run(sv, K, R, asc, sd, null=True) for _ in range(N_NULL)]
                ne = np.nanmean([x["oos_edge"] for x in nulls if x])
                ns = np.nanstd([x["oos_edge"] for x in nulls if x])
                rows.append(dict(feature=name, split=split_frac, K=K, R=R,
                                 oos_edge=r["oos_edge"], oos_strat=r["oos_strat"],
                                 null_oos=ne, z=(r["oos_edge"]-ne)/ns if ns>0 else np.nan))
RB = pd.DataFrame(rows)
RB.to_csv(OUT / "robustness_split.csv", index=False)
print("== IS-split robustness (median OOS by feature x split) ==")
print(RB.groupby(["feature", "split"]).agg(oos_edge_med=("oos_edge", "median"),
      oos_strat_med=("oos_strat", "median"), z_med=("z", "median")).round(4).to_string())

# ---- (2) survivorship attribution for snap_res ----
finals = {}
for p in pairs:
    c = px[p].dropna().to_numpy()
    if len(c) >= 10: finals[p] = c[-1] / c[0] - 1.0
fin = pd.Series(finals)
survivors = set(fin[fin > -0.5].index)
casualties = set(fin[fin <= -0.5].index)
res_surv = snap_res.reindex(list(survivors)).dropna()
res_cas = snap_res.reindex(list(casualties)).dropna()
print("\n== survivorship attribution: reserve_usd of survivors vs casualties ==")
print(f"survivors (n={len(res_surv)}): median reserve ${res_surv.median():,.0f}")
print(f"casualties(n={len(res_cas)}): median reserve ${res_cas.median():,.0f}")
print(f"-> ratio survivor/casualty median reserve = {res_surv.median()/max(res_cas.median(),1):.2f}x")

# within-survivor only selection: does snap_res STILL rank among survivors?
sd = days[int(0.6 * len(days))]
rws = []
for K in (5, 10, 20):
    for R in (7, 14, 28):
        full = run(snap_res, K, R, False, sd)
        surv_only = run(snap_res, K, R, False, sd, restrict=survivors)
        if full and surv_only:
            rws.append(dict(K=K, R=R, full_oos_edge=full["oos_edge"], full_strat=full["oos_strat"],
                            survonly_oos_edge=surv_only["oos_edge"], survonly_strat=surv_only["oos_strat"]))
WS = pd.DataFrame(rws)
WS.to_csv(OUT / "within_survivor.csv", index=False)
print("\n== snap_res selection: FULL panel vs SURVIVORS-ONLY (within-survivor dispersion) ==")
print(WS.round(4).to_string(index=False))
print(f"\nfull-panel median OOS strat   = {WS.full_strat.median():+.4f}")
print(f"survivors-only median OOS strat= {WS.survonly_strat.median():+.4f}")
print("(if survivors-only edge collapses toward 0, the full-panel 'edge' was survivorship: big pools = not-yet-rugged)")
json.dump(dict(survivor_med_reserve=float(res_surv.median()), casualty_med_reserve=float(res_cas.median()),
               n_survivors=len(survivors), n_casualties=len(casualties),
               full_strat_med=float(WS.full_strat.median()), survonly_strat_med=float(WS.survonly_strat.median())),
          open(OUT / "survivorship_attribution.json", "w"), indent=2)
print("\nwrote robustness_split.csv, within_survivor.csv, survivorship_attribution.json")
