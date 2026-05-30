"""Figures for the cross-sectional ML ranker gap study."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

OUT = Path("./_gaps/ml_xsec_ranker")
FIG = OUT / "figs"
res = pd.read_csv(OUT / "ml_ranker_results.csv")

# ---- Fig 1: rank-IC real vs null, per config ----
fig, ax = plt.subplots(figsize=(9,5))
labels = [f"{r.model}\nR{r.R} K{r.K}" for r in res.itertuples()]
x = np.arange(len(res)); w = 0.38
ax.bar(x-w/2, res.ic_mean_real, w, label="real IC", color="#2166ac")
ax.bar(x+w/2, res.ic_mean_null, w, label="null IC (labels permuted)", color="#b2182b", alpha=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=0)
ax.set_ylabel("OOS cross-sectional rank-IC (Spearman pred vs fwd ret)")
ax.set_title("Learned ranker has real OOS rank-IC vs ~0 label-permuted null\n(DEX-only coins, daily, strict WFO, all features causal)")
ax.legend()
fig.tight_layout(); fig.savefig(FIG/"fig1_rank_ic_real_vs_null.pdf"); plt.close(fig)

# ---- Fig 2: net top-K edge vs costed benchmark (median, winsorized) ----
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(x-w/2, res.edge_med_real, w, label="real median edge", color="#2166ac")
ax.bar(x+w/2, res.edge_med_null, w, label="null median edge", color="#b2182b", alpha=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Median per-rebalance net edge: top-K minus costed equal-weight")
ax.set_title("Net tradeable edge (top-K vs costed benchmark) straddles zero, ~= null\n164bp/leg cost, fwd returns winsorized [-95%,+300%]")
ax.legend()
fig.tight_layout(); fig.savefig(FIG/"fig2_net_edge_real_vs_null.pdf"); plt.close(fig)

# ---- Fig 3: feature importances ----
imp = pd.read_csv(OUT / "feature_importances.csv")
fig, ax = plt.subplots(figsize=(8,6))
ax.barh(imp.feature[::-1], imp.importance[::-1], color="#4d4d4d")
ax.set_xlabel("Mean normalized gain importance (LGBM, R=14 K=20, across WFO folds)")
ax.set_title("Feature importances of the cross-sectional ranker")
fig.tight_layout(); fig.savefig(FIG/"fig3_feature_importances.pdf"); plt.close(fig)

# ---- Fig 4: OOS pred-rank vs realized fwd return decile (the IC, visualized) ----
d = np.load(OUT / "oos_ic_scatter.npz")
pred, y = d["pred"], d["y"]
yc = np.clip(y, -0.95, 3.0)
# decile of prediction -> mean realized winsorized fwd return
order = np.argsort(pred)
ranks = np.argsort(order).astype(float) / max(len(pred)-1,1)
bins = np.clip((ranks*10).astype(int), 0, 9)
mean_ret = [yc[bins==b].mean() for b in range(10)]
med_ret  = [np.median(yc[bins==b]) for b in range(10)]
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(1,11), mean_ret, "o-", label="mean fwd ret (winsorized)", color="#2166ac")
ax.plot(range(1,11), med_ret, "s--", label="median fwd ret", color="#762a83")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Predicted-score decile (1=lowest, 10=highest)")
ax.set_ylabel("Realized forward 14-day return")
sp = spearmanr(pred, y).correlation
ax.set_title(f"Monotone OOS pred-decile vs realized return (pooled rank-IC={sp:+.3f})\nsignal is real but small; top decile is where cost eats it")
ax.legend()
fig.tight_layout(); fig.savefig(FIG/"fig4_pred_decile_vs_return.pdf"); plt.close(fig)

print("figs written")
