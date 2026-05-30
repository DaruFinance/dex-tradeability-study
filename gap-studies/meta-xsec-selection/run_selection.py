"""GAP: metadata-derived CROSS-SECTIONAL SELECTION features for DEX-only coins.

age_days / vol24 / reserve_usd / dex were used ONLY for the cost model, never as SELECTION
signals. This builds CAUSAL cross-sectional features and asks: each rebalance, rank coins by a
feature, long the top-K, forward return vs the equal-weight universe. IS/OOS calendar split,
per-coin reserve-based cost on both legs, and an ACROSS-COIN permutation null (shuffle the feature
across coins each rebalance -> destroys feature->return link -> chance baseline).

Features
  Time-varying (from the daily panel, all causal via .shift(1)):
    amihud      : Amihud illiquidity = mean(|ret|/dollar_vol) over lookback  (low = liquid)
    turnover    : dollar_vol / reserve_usd (velocity); rolling mean
    voltrend    : dollar_vol_recent / dollar_vol_older  (volume trend)
    dvaccel     : dollar-volume acceleration = d(rolling dollar_vol)
    volgrowth   : realized growth of dollar volume (proxy for reserve growth; reserve not time-series)
  Static snapshot (universe meta, sorted ONCE — flagged as snapshot, mild lookahead in TIMING but
  legitimate as a coin-PICKING prior; we report separately):
    age_days    : coin age cohort
    snap_turn   : vol24/reserve_usd snapshot
    snap_res    : reserve_usd

For each feature we test ascending & descending picks (some signals predict in either direction).
Composite = mean of per-feature cross-sectional ranks of the best-direction features (IS-chosen).

NULL: across-coin permutation of the feature each rebalance.
Real edge ONLY if real OOS edge materially > null OOS edge AND real OOS strat-vs-bench > 0.

Usage: python3 run_selection.py
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
OUT = Path("./_gaps/meta-xsec-selection")
IS_SPLIT = 0.6
SIZE_FRAC = 0.0025
SEED = 11
rng = np.random.default_rng(SEED)
N_NULL = 30  # null reps per config (avg) for stable chance baseline

CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05,
             "polygon_pos": .01, "optimism": .05, "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "optimism": 3500, "sui-network": 3.5, "solana": 180, "tron": .3}


def coin_cost(chain, reserve):
    if not reserve:
        return .05
    size = max(50., SIZE_FRAC * reserve)
    return round_trip_cost_frac(size, reserve, dex="uniswap", chain=chain,
                                gas_usd=CHAIN_GAS.get(chain, .05),
                                native_usd=CHAIN_NATIVE.get(chain, 100))


def load():
    univ = {}
    for l in open(DATA / "_mega_universe.jsonl"):
        r = json.loads(l)
        univ[r["pair"]] = r
    frames = [pd.read_parquet(f) for f in glob.glob(f"{DATA}/ohlcv_gt/mega_*_day.parquet")]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["close", "volume"])
    df = df[df.close > 0]
    df["day"] = (df.ts // 86400).astype(int)
    return df, univ


def build_panels(df):
    px = df.pivot_table(index="day", columns="pair_address", values="close", aggfunc="last").sort_index()
    dv = df.pivot_table(index="day", columns="pair_address", values="volume", aggfunc="sum").reindex_like(px)
    return px, dv


def main():
    df, univ = load()

    # --- exclude stablecoins / too-short (need a real cross-section to rank) ---
    keep = {p for p, g in df.groupby("pair_address")
            if g.close.pct_change().std() > 0.01 and len(g) >= 60}
    df = df[df.pair_address.isin(keep)]
    px, dv = build_panels(df)
    pairs = list(px.columns)
    days = px.index.to_numpy()
    split_day = days[int(IS_SPLIT * len(days))]
    print(f"panel: {len(pairs)} coins x {len(days)} days; IS<{split_day} OOS>=; cost=per-coin reserve-based")

    # per-coin cost vector
    cost = pd.Series({p: coin_cost(univ.get(p, {}).get("chain", "bsc"),
                                   univ.get(p, {}).get("reserve_usd") or 0) for p in pairs})
    print(f"cost: median={cost.median()*1e4:.0f}bp mean={cost.mean()*1e4:.0f}bp")

    ret = px.pct_change(fill_method=None)
    LB = 14  # feature lookback

    # ---- TIME-VARYING causal features (shift(1)) ----
    feats = {}
    # Amihud illiquidity: mean(|ret|/dollar_vol); LOW=liquid. (replace 0 dv with nan)
    dv_safe = dv.replace(0, np.nan)
    amihud = (ret.abs() / dv_safe).rolling(LB).mean().shift(1)
    feats["amihud"] = amihud
    # turnover proxy: dollar_vol / reserve (reserve static snapshot) rolling
    res = pd.Series({p: univ.get(p, {}).get("reserve_usd") or np.nan for p in pairs})
    turnover = (dv.rolling(LB).mean() / res).shift(1)
    feats["turnover"] = turnover
    # volume trend: recent dollar vol / older dollar vol
    voltrend = (dv.rolling(7).mean() / dv.rolling(28).mean().replace(0, np.nan)).shift(1)
    feats["voltrend"] = voltrend
    # dollar-volume acceleration: change in rolling dollar vol
    dvroll = dv.rolling(7).mean()
    dvaccel = (dvroll - dvroll.shift(7)).shift(1)
    feats["dvaccel"] = dvaccel
    # realized dollar-volume growth (reserve-growth proxy): log dv now vs LB ago
    volgrowth = (np.log(dvroll.replace(0, np.nan)) - np.log(dvroll.replace(0, np.nan).shift(LB))).shift(1)
    feats["volgrowth"] = volgrowth

    # ---- STATIC snapshot features (coin-level constant; flagged) ----
    age = pd.Series({p: univ.get(p, {}).get("age_days") or np.nan for p in pairs})
    snap_turn = pd.Series({p: ((univ.get(p, {}).get("vol24") or 0) /
                               (univ.get(p, {}).get("reserve_usd") or np.nan)) for p in pairs})
    snap_res = res
    static_feats = {"age_days": age, "snap_turn": snap_turn, "snap_res": snap_res}

    fwd_cache = {}
    def fwd(R):
        if R not in fwd_cache:
            fwd_cache[R] = px.shift(-R) / px - 1.0
        return fwd_cache[R]

    # Microcap DEX forward returns have monstrous right tails (a few coins 10-100x per window),
    # so a raw equal-weight MEAN benchmark is dominated by 2-3 lottery coins and is meaningless.
    # We winsorize each window's cross-section at the 99th pct (cap, applied IDENTICALLY to the
    # picks and the benchmark and to the null) so the real-vs-null comparison is on a robust metric.
    # Cap at 95th pct AND record the MEDIAN-coin benchmark. Even the 99th-pct cap leaves a mean of
    # ~+3200%/window because hundreds of coins sit in the extreme right tail; the equal-weight MEAN
    # of 2940 microcaps is not an achievable or meaningful benchmark. We therefore (a) winsorize the
    # picks at the 95th pct of the live cross-section, and (b) benchmark against the MEDIAN coin
    # return (the typical coin you'd land on at random). The rigorous test remains the across-coin
    # permutation null, which is a random-K portfolio run through the identical winsorized pipeline.
    WINSOR_Q = 0.95
    def winsor_row(ff_alive):
        cap = ff_alive.quantile(WINSOR_Q)
        return ff_alive.clip(upper=cap)

    # ---------- runner for time-varying feature panel ----------
    def run_tv(sig, K, R, asc, null_rep=0):
        f = fwd(R)
        # precompute null permutation seeds deterministically
        rec = []  # (is/oos, strat-bench)
        for i in range(0, len(days) - R, R):
            s = sig.iloc[i].dropna()
            ff = f.iloc[i]
            alive = ff.dropna().index
            cand = s.index.intersection(alive)
            if len(cand) < 3 * K:
                continue
            sv = s.loc[cand]
            if null_rep:
                sv = pd.Series(rng.permutation(sv.values), index=sv.index)
            picks = sv.sort_values(ascending=asc).index[:K]
            ffa = winsor_row(ff.loc[alive])              # robust cross-section (cap right tail)
            costv = cost.reindex(picks).fillna(.05)
            strat = (ffa.reindex(picks).fillna(ffa.median()) - costv).mean()
            bench = ff.loc[alive].median()               # typical coin (tail-immune)
            rec.append((days[i] < split_day, strat - bench, strat, bench))
        if not rec:
            return None
        rec = np.array([(int(a), b, c, d) for a, b, c, d in rec])
        is_m = rec[:, 0] == 1
        return dict(is_edge=rec[is_m, 1].mean() if is_m.any() else np.nan,
                    oos_edge=rec[~is_m, 1].mean() if (~is_m).any() else np.nan,
                    oos_strat=rec[~is_m, 2].mean() if (~is_m).any() else np.nan,
                    oos_bench=rec[~is_m, 3].mean() if (~is_m).any() else np.nan,
                    ni=is_m.sum(), no=(~is_m).sum())

    # ---------- runner for static feature (constant sort, rebalance for forward windows) ----------
    def run_static(svec, K, R, asc, null_rep=0):
        f = fwd(R)
        rec = []
        sv0 = svec.dropna()
        for i in range(0, len(days) - R, R):
            ff = f.iloc[i]
            alive = ff.dropna().index
            cand = sv0.index.intersection(alive)
            if len(cand) < 3 * K:
                continue
            sv = sv0.loc[cand]
            if null_rep:
                sv = pd.Series(rng.permutation(sv.values), index=sv.index)
            picks = sv.sort_values(ascending=asc).index[:K]
            ffa = winsor_row(ff.loc[alive])
            costv = cost.reindex(picks).fillna(.05)
            strat = (ffa.reindex(picks).fillna(ffa.median()) - costv).mean()
            bench = ff.loc[alive].median()
            rec.append((days[i] < split_day, strat - bench, strat, bench))
        if not rec:
            return None
        rec = np.array([(int(a), b, c, d) for a, b, c, d in rec])
        is_m = rec[:, 0] == 1
        return dict(is_edge=rec[is_m, 1].mean() if is_m.any() else np.nan,
                    oos_edge=rec[~is_m, 1].mean() if (~is_m).any() else np.nan,
                    oos_strat=rec[~is_m, 2].mean() if (~is_m).any() else np.nan,
                    oos_bench=rec[~is_m, 3].mean() if (~is_m).any() else np.nan,
                    ni=is_m.sum(), no=(~is_m).sum())

    Ks = (5, 10, 20)
    Rs = (7, 14, 28)
    rows = []

    def eval_feature(name, kind, runner, sigobj):
        for K in Ks:
            for R in Rs:
                for asc in (True, False):
                    real = runner(sigobj, K, R, asc, 0)
                    if real is None or real["ni"] < 3 or real["no"] < 2:
                        continue
                    # null: average over N_NULL permutations
                    nulls = []
                    for _ in range(N_NULL):
                        nr = runner(sigobj, K, R, asc, 1)
                        if nr is not None and np.isfinite(nr["oos_edge"]):
                            nulls.append(nr["oos_edge"])
                    null_mean = np.mean(nulls) if nulls else np.nan
                    null_std = np.std(nulls) if nulls else np.nan
                    z = (real["oos_edge"] - null_mean) / null_std if null_std and null_std > 0 else np.nan
                    rows.append(dict(feature=name, kind=kind, K=K, R=R,
                                     direction="asc" if asc else "desc",
                                     is_edge=real["is_edge"], oos_edge=real["oos_edge"],
                                     oos_strat=real["oos_strat"], oos_bench=real["oos_bench"],
                                     null_oos_edge=null_mean, null_std=null_std, z_vs_null=z,
                                     ni=real["ni"], no=real["no"]))

    print("\n== time-varying causal features ==")
    for name, sig in feats.items():
        eval_feature(name, "timevarying", run_tv, sig)
        print(f"  done {name}")
    print("== static snapshot features ==")
    for name, sv in static_feats.items():
        eval_feature(name, "static", run_static, sv)
        print(f"  done {name}")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "selection_results.csv", index=False)
    print(f"\nwrote {len(R)} configs -> selection_results.csv")

    # ---- per-feature summary: best IS-direction picked, then OOS real vs null ----
    summ = []
    for (name, kind), g in R.groupby(["feature", "kind"]):
        # IS-selected best direction (the honest WFO choice)
        best_dir = g.groupby("direction").is_edge.mean().idxmax()
        gd = g[g.direction == best_dir]
        summ.append(dict(feature=name, kind=kind, is_dir=best_dir,
                         is_edge_med=gd.is_edge.median(),
                         oos_edge_med=gd.oos_edge.median(),
                         oos_edge_mean=gd.oos_edge.mean(),
                         null_oos_med=gd.null_oos_edge.median(),
                         oos_strat_med=gd.oos_strat.median(),
                         oos_bench_med=gd.oos_bench.median(),
                         z_med=gd.z_vs_null.median(),
                         frac_real_gt_null=(gd.oos_edge > gd.null_oos_edge).mean(),
                         frac_oos_pos=(gd.oos_strat > gd.oos_bench).mean(),
                         n=len(gd)))
    S = pd.DataFrame(summ).sort_values("oos_edge_med", ascending=False)
    S.to_csv(OUT / "feature_summary.csv", index=False)
    print("\n== PER-FEATURE SUMMARY (IS-selected direction; OOS real vs null) ==")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(S.to_string(index=False))

    # ---- composite rank (IS-chosen best-direction time-varying feats, z-scored cross-section) ----
    print("\n== composite multi-feature rank ==")
    # choose features with IS edge>0 in their best direction
    chosen = []
    for _, r in S.iterrows():
        if r.kind == "timevarying" and r.is_edge_med > 0:
            chosen.append((r.feature, r.is_dir))
    print("IS-chosen TV features for composite:", chosen)
    comp = None
    for fname, d in chosen:
        sg = feats[fname]
        # cross-sectional z each day; sign so HIGHER=better-pick
        z = sg.sub(sg.mean(axis=1), axis=0).div(sg.std(axis=1).replace(0, np.nan), axis=0)
        if d == "asc":
            z = -z  # ascending pick => lower raw is better => invert so high=pick
        comp = z if comp is None else comp.add(z, fill_value=0)
    comp_rows = []
    if comp is not None:
        for K in Ks:
            for R in Rs:
                real = run_tv(comp, K, R, False, 0)  # high composite = pick (desc)
                if real is None or real["no"] < 2:
                    continue
                nulls = [run_tv(comp, K, R, False, 1)["oos_edge"] for _ in range(N_NULL)]
                nulls = [x for x in nulls if np.isfinite(x)]
                nm = np.mean(nulls); ns = np.std(nulls)
                comp_rows.append(dict(K=K, R=R, is_edge=real["is_edge"], oos_edge=real["oos_edge"],
                                      oos_strat=real["oos_strat"], oos_bench=real["oos_bench"],
                                      null_oos=nm, z=(real["oos_edge"]-nm)/ns if ns>0 else np.nan))
        C = pd.DataFrame(comp_rows)
        C.to_csv(OUT / "composite_results.csv", index=False)
        print(C.to_string(index=False))
        print(f"composite: median OOS edge={C.oos_edge.median():+.4f} vs null={C.null_oos.median():+.4f}; "
              f"OOS strat>bench in {100*(C.oos_strat>C.oos_bench).mean():.0f}%")

    # ---- SURVIVORSHIP / within-survivor dispersion diagnostic ----
    print("\n== SURVIVORSHIP diagnostic ==")
    finals, dd = [], []
    for p in pairs:
        c = px[p].dropna().to_numpy()
        if len(c) < 10:
            continue
        peak = np.maximum.accumulate(c)
        finals.append(c[-1] / c[0] - 1.0)
        dd.append(c[-1] / peak[-1] - 1.0)
    finals, dd = np.array(finals), np.array(dd)
    surv = dict(n=len(finals), median_total_ret=float(np.median(finals)),
                mean_total_ret=float(np.mean(finals)),
                frac_pos=float((finals > 0).mean()),
                frac_down50=float((finals < -.5).mean()),
                frac_down90=float((finals < -.9).mean()),
                median_dd_from_peak=float(np.median(dd)))
    print(json.dumps(surv, indent=2))
    pd.DataFrame([surv]).to_csv(OUT / "survivorship.csv", index=False)

    return R, S, comp_rows if comp is not None else [], surv, finals, dd, px, cost


if __name__ == "__main__":
    main()
