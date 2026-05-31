"""Leakage-free re-run of the cross-sectional ranker (referee fix).

Removes the three end-of-sample SNAPSHOT features (log_reserve, log_age, log_vol24) that were
broadcast across all rebalance dates and are themselves survival markers. Replaces age with a
strictly point-in-time, causal cumulative-bar count; drops the snapshot reserve/vol and the
snapshot flow features entirely (rely on the time-varying OHLCV features only). Reports, for the
six distinct (model, horizon) cells (rank-IC is K-invariant, so K is fixed at 20), the OOS rank-IC,
its t over disjoint rebalances, and a label-permutation p-value over N_NULL permutations.

Usage:
  python3 ml_ranker_pit.py <n_null>                 # all six cells, write ml_pit_results.csv
  python3 ml_ranker_pit.py <n_null> <model> <R>     # one cell, write ml_pit_cell_<model>_<R>.csv
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import ml_ranker as ml

OUT = ml.OUT
_args = sys.argv[1:]
N_NULL = int(_args[0]) if _args else 200
ONE = (_args[1], int(_args[2])) if len(_args) >= 3 else None

def main():
    univ = ml.load_universe()
    df = ml.load_panel(min_bars=120)
    coins = df.pair_address.unique()
    cost_map, chain_map = {}, {}
    for p in coins:
        u = univ.get(p, {}); ch = u.get("chain", "bsc"); chain_map[p] = ch
        cost_map[p] = ml.coin_cost(ch, u.get("reserve_usd") or 0.0)

    px, feats, days = ml.build_feature_panels(df)
    feats["log_age_pit"] = np.log1p(px.notna().cumsum()).shift(1)   # point-in-time age, causal
    if "log_age_pit" not in ml.PXFEATS:
        ml.PXFEATS.append("log_age_pit")
    feat_cols = list(ml.PXFEATS)                # PXFEATS + log_age_pit; NO snapshot statics, NO flow
    static_df = pd.DataFrame(index=px.columns)  # empty -> no static columns added

    cells = [ONE] if ONE else [(m, R) for m in ["lgbm", "histgb"] for R in [7, 14, 30]]
    rows = []
    for model_kind, R in cells:
        rebal_idx = list(range(60, len(days) - R, R))
        samples = ml.assemble_samples(px, feats, days, static_df, rebal_idx, R)
        if samples.empty:
            print(f"[{model_kind} R={R}] no samples", flush=True); continue
        K = 20
        ml.rng = np.random.default_rng(7)
        real = ml.run_wfo(samples, feat_cols, cost_map, chain_map, K, R, model_kind, permute_labels=False)
        if real is None:
            print(f"[{model_kind} R={R}] insufficient folds", flush=True); continue
        real_ic = real["rank_ic_mean"]
        null_ics = []
        for s in range(N_NULL):
            ml.rng = np.random.default_rng(1000 + s)
            nl = ml.run_wfo(samples, feat_cols, cost_map, chain_map, K, R, model_kind, permute_labels=True)
            if nl: null_ics.append(nl["rank_ic_mean"])
        null_ics = np.array(null_ics)
        perm_p = float((1 + (null_ics >= real_ic).sum()) / (len(null_ics) + 1))
        rows.append({"model": model_kind, "R": R, "n_oos": real["n_oos_rebals"],
                     "ic_mean_real": real_ic, "ic_t_real": real["rank_ic_t"],
                     "perm_p": perm_p, "null_ic_mean": float(null_ics.mean()),
                     "null_ic_p95": float(np.percentile(null_ics, 95)),
                     "topk_net_med": real["topk_net_median"], "edge_med": real["edge_median"],
                     "edge_winrate": real["edge_winrate"]})
        print(f"[{model_kind} R={R}] PIT ic={real_ic:+.4f}(t={real['rank_ic_t']:+.2f}) "
              f"perm_p={perm_p:.4f} null_mean={null_ics.mean():+.4f} n_oos={real['n_oos_rebals']}", flush=True)
        if model_kind == "lgbm" and R == 14:
            imp = pd.DataFrame({"feature": feat_cols, "importance": real["importances"]}).sort_values("importance", ascending=False)
            imp.to_csv(OUT / "feature_importances_pit.csv", index=False)
            pred, y = real["_oos_pred"], np.clip(real["_oos_y"], -0.95, 3.0)
            if len(pred) > 50:
                dec = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
                pd.DataFrame({"dec": dec, "y": y}).groupby("dec").y.median().to_csv(OUT / "decile_pit.csv")

    out = pd.DataFrame(rows)
    fn = f"ml_pit_cell_{ONE[0]}_{ONE[1]}.csv" if ONE else "ml_pit_results.csv"
    out.to_csv(OUT / fn, index=False)
    print(f"\nwrote {fn}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
