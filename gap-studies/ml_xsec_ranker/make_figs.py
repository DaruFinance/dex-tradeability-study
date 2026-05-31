"""Figures for the cross-sectional ML ranker gap study.

LEAKAGE-FREE: consumes the point-in-time re-run outputs
  ml_pit_results.csv        (six model x horizon cells; rank-IC, t, perm-p, null p95)
  feature_importances_pit.csv
  decile_pit.csv            (median forward return by predicted decile)
NOT the earlier leaky snapshot-feature outputs. Run `python3 ml_ranker_pit.py 200`
first to (re)generate those inputs.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(".")
FIG = OUT / "figs"; FIG.mkdir(exist_ok=True)
ACC = "#b07a2e"; MUT = "#888888"; FGC = "#222222"

res = pd.read_csv(OUT / "ml_pit_results.csv").sort_values(["R", "model"]).reset_index(drop=True)
lab = [f"{m}\n{R}d" for m, R in zip(res.model, res.R)]
x = np.arange(len(res))

# ---- Fig 1: leakage-free rank-IC vs label-permutation null (p95 band + perm-p) ----
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.bar(x, res.ic_mean_real, color=[ACC if p < 0.05 else MUT for p in res.perm_p], width=0.6, label="real rank-IC")
ax.plot(x, res.null_ic_p95, "k_", ms=18, mew=2, label="null 95th pct (200 shuffles)")
for i, (ic, p) in enumerate(zip(res.ic_mean_real, res.perm_p)):
    ax.text(i, ic + 0.004, f"p={p:.3f}", ha="center", fontsize=7.5, color=FGC)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("out-of-sample rank-IC")
ax.set_title("Leakage-free cross-sectional rank-IC vs label-permutation null", fontsize=10)
ax.legend(fontsize=8, frameon=False); fig.tight_layout()
fig.savefig(FIG / "fig1_rank_ic_real_vs_null.pdf"); plt.close(fig)

# ---- Fig 2: top-K net edge per cell (leakage-free) ----
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.bar(x, res.edge_med, color=MUT, width=0.6)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("median per-rebalance net edge\n(top-K minus costed equal-weight)")
ax.set_title("Top-K net edge (leakage-free): small, winsorization/survivorship-driven", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "fig2_net_edge_real_vs_null.pdf"); plt.close(fig)

# ---- Fig 3: leakage-free feature importances (point-in-time age in gold) ----
imp = pd.read_csv(OUT / "feature_importances_pit.csv").head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.barh(imp.feature, imp.importance, color=[ACC if f == "log_age_pit" else MUT for f in imp.feature])
ax.set_xlabel("gain importance")
ax.set_title("Feature importances (leakage-free; point-in-time age in gold)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig3_feature_importances.pdf"); plt.close(fig)

# ---- Fig 4: predicted decile vs forward return (U-shaped, every decile negative) ----
dec = pd.read_csv(OUT / "decile_pit.csv")
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.bar(dec.dec, dec.y * 100, color=[ACC if d == dec.dec.max() else MUT for d in dec.dec])
ax.axhline(0, color="k", lw=.6)
ax.set_xlabel("predicted decile (0 = lowest score, 9 = highest)")
ax.set_ylabel("median forward return (%)")
ax.set_title("Every predicted decile is negative; top decile is 'least bad' (leakage-free)", fontsize=9.5)
ax.set_xticks(range(10)); fig.tight_layout()
fig.savefig(FIG / "fig4_pred_decile_vs_return.pdf"); plt.close(fig)

print("figs written (leakage-free, from *_pit.csv)")
