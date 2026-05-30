"""
GAP: Cross-sectional ML SELECTION ranker on DEX-only coins.

Thesis (falsifiable): A gradient-boosted-tree ranker trained on ONLY causal cross-sectional
features (age, turnover, illiquidity/Amihud, volume velocity, reserve, recent past returns/vol,
+ static bitquery trader-count flow where joinable) predicting forward N-day cross-sectional
return RANK does NOT, net of ~164bp per-fill cost, beat an equal-weight benchmark or its own
label-permuted null. If a learned multi-feature selector also fails (as single-feature timing
and momentum did), the no-edge conclusion is robust to model class.

Pipeline:
  - Daily OHLCV panel (long-only AMM spot). Causal features at time t use data <= t-1.
  - Label = forward R-day return rank (cross-sectional), computed at rebalance dates.
  - STRICT WFO: rolling expanding-IS train -> predict next OOS block. Never fit on OOS.
  - Long top-K by predicted score, equal weight, costed per-fill (per-coin reserve-based cost).
  - Benchmark = equal-weight over all alive coins that rebalance date (also long-only, costed
    only when entering -> we cost the top-K picks; benchmark is the costless universe drift AND
    a costed-universe variant for fairness).
  - NULL = permute labels ACROSS coins within each IS training window (destroys feature->return
    link) through the IDENTICAL pipeline. Reported side-by-side.
  - Metrics: OOS rank-IC (Spearman pred vs realized fwd ret), top-K net edge vs benchmark,
    real vs null, feature importances.

Survivorship: universe sourced from currently-live GT pools => survivor-tilted. This MOST
compromises cross-sectional selection (dead/rugged coins under-represented). Flagged in output.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
OUT = Path("./_gaps/ml_xsec_ranker")
SEED = 7
rng = np.random.default_rng(SEED)

CHAIN_GAS = {'bsc':.2,'base':.02,'eth':3.,'arbitrum':.05,'avax':.05,'polygon_pos':.01,
             'optimism':.05,'sui-network':.01,'solana':.02,'tron':.01}
CHAIN_NATIVE = {'bsc':640,'base':3500,'eth':3500,'arbitrum':3500,'avax':35,'polygon_pos':.5,
                'optimism':3500,'sui-network':3.5,'solana':180,'tron':.3}

def coin_cost(chain, reserve):
    size = max(50., 0.0025*(reserve or 0))
    if not reserve:
        return .05
    return round_trip_cost_frac(size, reserve, dex='uniswap', chain=chain,
                                gas_usd=CHAIN_GAS.get(chain, .05),
                                native_usd=CHAIN_NATIVE.get(chain, 100))

# ----------------------------------------------------------------------------- load
def load_universe():
    univ = {}
    # preserve case for solana/tron/ton/aptos base58; lowercase EVM
    for fn in ["_mega_universe.jsonl"]:
        p = DATA / fn
        if not p.exists(): continue
        for l in open(p):
            r = json.loads(l)
            univ[r["pair"]] = r
    return univ

def load_panel(min_bars=120):
    frames = [pd.read_parquet(f) for f in glob.glob(f"{DATA}/ohlcv_gt/mega_*_day.parquet")]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["close","open","high","low"])
    df = df[df.close > 0]
    df["day"] = (df.ts // 86400).astype(int)
    # drop near-stable / illiquid-history coins; require length & volatility
    keep = []
    for pair, g in df.groupby("pair_address"):
        if len(g) < min_bars: continue
        if g.close.pct_change().std() <= 0.01: continue   # stablecoins
        keep.append(pair)
    df = df[df.pair_address.isin(keep)].copy()
    return df

def load_flow_static():
    """Bitquery flow only spans 5 days (snapshot) -> use as a STATIC recent-flow cross-sectional
    feature joined per coin. buy_usd/sell_usd are degenerate (0) so we use trader/tx COUNTS."""
    flow = pd.read_parquet(DATA / "_bitquery_flow_daily.parquet")
    # map (chain, token) -> pair
    tok2pair = {}
    for l in open(DATA / "_mega_universe.jsonl"):
        r = json.loads(l); bt = r.get("base_token") or ""
        if "_" in bt:
            ch, tok = bt.split("_", 1)
            # EVM lowercase; base58 preserve -> store both forms
            tok2pair[(r["chain"], tok)] = r["pair"]
            tok2pair[(r["chain"], tok.lower())] = r["pair"]
    flow["pair"] = [tok2pair.get((c, t)) or tok2pair.get((c, str(t).lower()))
                    for c, t in zip(flow.chain, flow.token)]
    flow = flow.dropna(subset=["pair"])
    for c in ["buy_traders","sell_traders","buy_n","sell_n"]:
        flow[c] = flow[c].fillna(0.0)
    agg = flow.groupby("pair").agg(
        f_buy_tr=("buy_traders","sum"), f_sell_tr=("sell_traders","sum"),
        f_buy_n=("buy_n","sum"), f_sell_n=("sell_n","sum")).reset_index()
    agg["f_net_tr"] = (agg.f_buy_tr - agg.f_sell_tr) / (agg.f_buy_tr + agg.f_sell_tr + 1e-9)
    agg["f_net_n"]  = (agg.f_buy_n  - agg.f_sell_n)  / (agg.f_buy_n  + agg.f_sell_n  + 1e-9)
    agg["f_tot_tr"] = np.log1p(agg.f_buy_tr + agg.f_sell_tr)
    agg["f_tot_n"]  = np.log1p(agg.f_buy_n  + agg.f_sell_n)
    return agg[["pair","f_net_tr","f_net_n","f_tot_tr","f_tot_n"]].set_index("pair")

# ----------------------------------------------------------------------------- features
# All features are CAUSAL: computed from bars strictly < rebalance day (we .shift after pivot).
PXFEATS = ["ret_5","ret_10","ret_20","ret_60","vol_20","vol_60","amihud_20",
           "turn_20","vvel_10","mom_accel","maxdd_20","rng_20","abv_ma20"]

def build_feature_panels(df):
    """Return dict feature -> (day x coin) DataFrame, plus close/high/low/vol panels.
    Every returned feature panel is already lagged by 1 day (causal)."""
    px = df.pivot_table(index="day", columns="pair_address", values="close", aggfunc="last").sort_index()
    vol = df.pivot_table(index="day", columns="pair_address", values="volume", aggfunc="sum").reindex_like(px).fillna(0)
    days = px.index.to_numpy()

    logret = np.log(px).diff()
    feats = {}
    feats["ret_5"]   = px.pct_change(5,  fill_method=None)
    feats["ret_10"]  = px.pct_change(10, fill_method=None)
    feats["ret_20"]  = px.pct_change(20, fill_method=None)
    feats["ret_60"]  = px.pct_change(60, fill_method=None)
    feats["vol_20"]  = logret.rolling(20).std()
    feats["vol_60"]  = logret.rolling(60).std()
    # Amihud illiquidity: mean(|ret|/dollar_vol)
    illq = (logret.abs() / vol.replace(0, np.nan))
    feats["amihud_20"] = np.log1p(illq.rolling(20).mean())
    # turnover proxy: dollar vol (no reserve in panel; reserve added as static below)
    feats["turn_20"] = np.log1p(vol.rolling(20).mean())
    # volume velocity: recent vol vs longer vol
    feats["vvel_10"] = (vol.rolling(10).mean() / (vol.rolling(40).mean() + 1.0))
    # momentum acceleration: short mom - long mom
    feats["mom_accel"] = px.pct_change(10, fill_method=None) - px.pct_change(40, fill_method=None)
    # max drawdown over 20d (negative)
    roll_max = px.rolling(20).max()
    feats["maxdd_20"] = px / roll_max - 1.0
    # high-low range proxy (use close range)
    feats["rng_20"] = (px.rolling(20).max() / px.rolling(20).min().replace(0, np.nan)) - 1.0
    # above 20d MA flag
    feats["abv_ma20"] = (px > px.rolling(20).mean()).astype(float)

    # LAG every feature by 1 (data <= t-1 only)
    for k in feats:
        feats[k] = feats[k].shift(1)
    return px, feats, days

# ----------------------------------------------------------------------------- WFO
def make_model(kind):
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, random_state=SEED, n_jobs=4, verbosity=-1)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             random_state=SEED)

def assemble_samples(px, feats, days, static_df, rebal_idx, R):
    """For each rebalance index i build (X rows per alive coin, y=fwd R-day ret, rank label).
    Returns long DataFrame: rebal_i, day, pair, feat..., fwd_ret."""
    fwd = px.shift(-R) / px - 1.0
    coins = px.columns.to_numpy()
    # static features aligned to coins
    st = static_df.reindex(coins)
    rows = []
    for i in rebal_idx:
        d = days[i]
        fr = fwd.iloc[i]
        alive = fr.dropna().index
        if len(alive) < 30: continue
        rec = {"rebal_i": i, "day": d}
        block = {}
        for k in PXFEATS:
            block[k] = feats[k].iloc[i]
        fd = pd.DataFrame(block)
        # add static (reserve, age, flow) broadcast
        for c in static_df.columns:
            fd[c] = st[c]
        fd["fwd_ret"] = fr
        fd["pair"] = fd.index
        fd["rebal_i"] = i; fd["day"] = d
        fd = fd.loc[alive]
        fd = fd.dropna(subset=["fwd_ret"])
        rows.append(fd)
    if not rows: return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def run_wfo(samples, feat_cols, cost_map, chain_map, K, R, model_kind,
            n_folds=5, permute_labels=False):
    """Strict expanding-window WFO. Returns dict of OOS metrics aggregated across folds."""
    reb = np.sort(samples.rebal_i.unique())
    if len(reb) < n_folds + 2:
        return None
    # split rebalance dates into IS (expanding) and OOS blocks
    fold_edges = np.linspace(int(0.4*len(reb)), len(reb), n_folds+1).astype(int)
    fold_edges = np.unique(fold_edges)
    ics, real_edges, bench_rets, topk_rets, topk_net = [], [], [], [], []
    oos_pred_real, oos_y = [], []
    importances = np.zeros(len(feat_cols))
    nfit = 0
    for fi in range(len(fold_edges)-1):
        oos_lo, oos_hi = fold_edges[fi], fold_edges[fi+1]
        oos_reb = reb[oos_lo:oos_hi]
        is_reb = reb[:oos_lo]
        if len(is_reb) < 2 or len(oos_reb) < 1: continue
        tr = samples[samples.rebal_i.isin(is_reb)].copy()
        te = samples[samples.rebal_i.isin(oos_reb)].copy()
        if len(tr) < 200 or len(te) < 30: continue
        Xtr = tr[feat_cols].to_numpy(np.float64)
        ytr = tr["fwd_ret"].to_numpy(np.float64)
        # rank-transform label within each IS rebalance date (cross-sectional rank target)
        ytr_rank = tr.groupby("rebal_i")["fwd_ret"].rank(pct=True).to_numpy()
        if permute_labels:
            # permute labels ACROSS coins within each IS date -> destroy feature->return link
            yp = tr.copy()
            yp["yr"] = ytr_rank
            yp["yr"] = yp.groupby("rebal_i")["yr"].transform(lambda s: rng.permutation(s.values))
            ytr_rank = yp["yr"].to_numpy()
        m = make_model(model_kind)
        # impute NaNs in features (median per column from IS)
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        m.fit(Xtr, ytr_rank)
        if hasattr(m, "feature_importances_"):
            fi_arr = np.asarray(m.feature_importances_, float)
            if fi_arr.sum() > 0: importances += fi_arr / fi_arr.sum()
            nfit += 1
        # OOS: predict per rebalance date, pick top-K, cost, vs benchmark
        Xte = te[feat_cols].to_numpy(np.float64)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        te = te.assign(pred=m.predict(Xte))
        for ri, grp in te.groupby("rebal_i"):
            if len(grp) < 3*K: continue
            # WINSORIZE forward returns per rebalance date: a single >1000x outlier otherwise
            # dominates the equal-weight mean and makes the edge comparison meaningless.
            # Cap at +300% / -95% (realistic single-position bound for a long).
            grp = grp.copy()
            grp["fwd_c"] = grp.fwd_ret.clip(-0.95, 3.0)
            # rank-IC (rank metric is outlier-robust by construction)
            ic = spearmanr(grp.pred, grp.fwd_ret).correlation
            if np.isfinite(ic): ics.append(ic)
            oos_pred_real.append(grp.pred.values); oos_y.append(grp.fwd_ret.values)
            picks = grp.sort_values("pred", ascending=False).head(K)
            costs = np.array([cost_map.get(p, 0.05) for p in picks.pair])
            net_picks = picks.fwd_c.values - costs
            topk = net_picks.mean()
            # costed equal-weight benchmark over all alive coins (also long, also pays cost)
            bench_costs = np.array([cost_map.get(p, 0.05) for p in grp.pair])
            bench_net = (grp.fwd_c.values - bench_costs).mean()
            topk_rets.append(topk); bench_rets.append(bench_net)
            topk_net.append(topk); real_edges.append(topk - bench_net)
    if not ics:
        return None
    real_edges = np.array(real_edges)
    out = {
        "n_oos_rebals": len(ics),
        "rank_ic_mean": float(np.mean(ics)),
        "rank_ic_median": float(np.median(ics)),
        "rank_ic_t": float(np.mean(ics)/ (np.std(ics)/np.sqrt(len(ics)) + 1e-12)),
        "topk_net_mean": float(np.mean(topk_net)),
        "topk_net_median": float(np.median(topk_net)),
        "bench_net_mean": float(np.mean(bench_rets)),
        "edge_mean": float(np.mean(real_edges)),
        "edge_median": float(np.median(real_edges)),
        "edge_t": float(np.mean(real_edges)/(np.std(real_edges)/np.sqrt(len(real_edges))+1e-12)),
        "edge_winrate": float((real_edges > 0).mean()),
        "importances": (importances/max(nfit,1)).tolist(),
        "_oos_pred": np.concatenate(oos_pred_real) if oos_pred_real else np.array([]),
        "_oos_y": np.concatenate(oos_y) if oos_y else np.array([]),
    }
    return out

# ----------------------------------------------------------------------------- main
def main():
    print("loading panel...", flush=True)
    univ = load_universe()
    df = load_panel(min_bars=120)
    flow = load_flow_static()
    print(f"panel: {df.pair_address.nunique()} coins, {df.day.nunique()} days", flush=True)

    # static per-coin features
    coins = df.pair_address.unique()
    static_rows = {}
    cost_map = {}; chain_map = {}
    for p in coins:
        u = univ.get(p, {})
        ch = u.get("chain", "bsc"); chain_map[p] = ch
        res = u.get("reserve_usd") or 0.0
        cost_map[p] = coin_cost(ch, res)
        static_rows[p] = {
            "log_reserve": np.log1p(res),
            "log_age": np.log1p(u.get("age_days") or 0.0),
            "log_vol24": np.log1p(u.get("vol24") or 0.0),
        }
    static_df = pd.DataFrame(static_rows).T
    # join flow (static snapshot); coins w/o flow get NaN (model handles via imputation)
    static_df = static_df.join(flow, how="left")
    flow_cols = ["f_net_tr","f_net_n","f_tot_tr","f_tot_n"]
    n_flow = static_df[flow_cols].notna().any(axis=1).sum()
    print(f"coins with bitquery flow join: {n_flow}/{len(coins)}", flush=True)

    px, feats, days = build_feature_panels(df)
    static_cols = ["log_reserve","log_age","log_vol24"] + flow_cols
    feat_cols = PXFEATS + static_cols

    results = []
    for model_kind in ["lgbm", "histgb"]:
        for R in [7, 14, 30]:
            # rebalance every R days, leave room for fwd window
            rebal_idx = list(range(60, len(days)-R, R))   # warmup 60 for 60d features
            samples = assemble_samples(px, feats, days, static_df, rebal_idx, R)
            if samples.empty:
                print(f"[{model_kind} R={R}] no samples"); continue
            for K in [10, 20, 40]:
                real = run_wfo(samples, feat_cols, cost_map, chain_map, K, R, model_kind,
                               permute_labels=False)
                null = run_wfo(samples, feat_cols, cost_map, chain_map, K, R, model_kind,
                               permute_labels=True)
                if real is None or null is None:
                    print(f"[{model_kind} R={R} K={K}] insufficient folds"); continue
                row = {"model": model_kind, "R": R, "K": K,
                       "n_oos": real["n_oos_rebals"],
                       "ic_mean_real": real["rank_ic_mean"], "ic_t_real": real["rank_ic_t"],
                       "ic_mean_null": null["rank_ic_mean"], "ic_t_null": null["rank_ic_t"],
                       "topk_net_real": real["topk_net_mean"], "topk_net_med_real": real["topk_net_median"],
                       "bench_net": real["bench_net_mean"],
                       "edge_real": real["edge_mean"], "edge_med_real": real["edge_median"],
                       "edge_t_real": real["edge_t"], "edge_winrate": real["edge_winrate"],
                       "edge_null": null["edge_mean"], "edge_med_null": null["edge_median"],
                       "real_gt_null": real["edge_mean"] > null["edge_mean"]}
                results.append(row)
                # stash importances from lgbm only (interpretable gain)
                if model_kind == "lgbm" and R == 14 and K == 20:
                    imp = pd.DataFrame({"feature": feat_cols, "importance": real["importances"]})
                    imp = imp.sort_values("importance", ascending=False)
                    imp.to_csv(OUT / "feature_importances.csv", index=False)
                    np.savez(OUT / "oos_ic_scatter.npz", pred=real["_oos_pred"], y=real["_oos_y"])
                print(f"[{model_kind} R={R} K={K}] IC real={real['rank_ic_mean']:+.4f}(t={real['rank_ic_t']:+.2f}) "
                      f"null={null['rank_ic_mean']:+.4f} | edge_med real={real['edge_median']:+.4f} "
                      f"null={null['edge_median']:+.4f} win={real['edge_winrate']:.2f} "
                      f"topk_med={real['topk_net_median']:+.4f}",
                      flush=True)

    res = pd.DataFrame(results)
    res.to_csv(OUT / "ml_ranker_results.csv", index=False)
    print("\n=== SUMMARY ===")
    print(res.to_string(index=False))
    print(f"\nconfigs where real MEDIAN edge > null median edge: {(res.edge_med_real>res.edge_med_null).sum()}/{len(res)}")
    print(f"configs net-positive median top-K (topk_net_med>0): {(res.topk_net_med_real>0).sum()}/{len(res)}")
    print(f"configs where median top-K beats costed benchmark (edge_med>0): {(res.edge_med_real>0).sum()}/{len(res)}")
    print(f"median rank-IC real: {res.ic_mean_real.median():+.4f}  median null IC: {res.ic_mean_null.median():+.4f}")

    # survivorship diagnostic
    finals = []
    for pair, g in df.groupby("pair_address"):
        c = g.sort_values("day").close.to_numpy()
        finals.append(c[-1]/c[0]-1.0)
    finals = np.array(finals)
    surv = {"n_coins": len(finals), "median_total_ret": float(np.median(finals)),
            "pct_up": float((finals>0).mean()*100),
            "pct_down_50": float((finals<-0.5).mean()*100),
            "pct_down_90": float((finals<-0.9).mean()*100)}
    pd.Series(surv).to_csv(OUT / "survivorship.csv")
    print("\nSURVIVORSHIP:", surv)
    return res

if __name__ == "__main__":
    main()
