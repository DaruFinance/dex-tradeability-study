"""Publication figures for the metadata cross-sectional selection gap test."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from pathlib import Path

OUT = Path("./_gaps/meta-xsec-selection")
FIG = OUT / "figs"
S = pd.read_csv(OUT / "feature_summary.csv")
R = pd.read_csv(OUT / "selection_results.csv")
surv = pd.read_csv(OUT / "survivorship.csv").iloc[0]

# ---- Fig 1: per-feature OOS real vs null (median edge, IS-selected direction) ----
S2 = S.sort_values("oos_edge_med")
fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(S2))
ax.barh(y - 0.2, S2.oos_edge_med, height=0.4, label="REAL OOS edge", color="#2c6fbb")
ax.barh(y + 0.2, S2.null_oos_med, height=0.4, label="NULL (across-coin shuffle)", color="#c44")
ax.set_yticks(y)
ax.set_yticklabels([f"{f}\n({k}, {d})" for f, k, d in zip(S2.feature, S2.kind, S2.is_dir)], fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Median OOS edge (top-K minus winsorized equal-weight universe, per window)")
ax.set_title("Cross-sectional selection by universe-metadata features:\nreal vs across-coin permutation null (winsorized returns, per-fill cost)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig1_feature_real_vs_null.pdf"); plt.close(fig)

# ---- Fig 2: z vs null per feature ----
fig, ax = plt.subplots(figsize=(8, 4.5))
S3 = S.sort_values("z_med")
ax.barh(np.arange(len(S3)), S3.z_med, color=["#2c6fbb" if z > 0 else "#c44" for z in S3.z_med])
ax.set_yticks(np.arange(len(S3))); ax.set_yticklabels(S3.feature, fontsize=9)
ax.axvline(0, color="k", lw=0.8)
for x in (2, -2):
    ax.axvline(x, color="grey", ls="--", lw=0.7)
ax.set_xlabel("Median z-score of REAL OOS edge vs null distribution (|z|>2 = beyond chance)")
ax.set_title("How far each feature's OOS edge sits above its across-coin null")
fig.tight_layout(); fig.savefig(FIG / "fig2_z_vs_null.pdf"); plt.close(fig)

# ---- Fig 3: scatter is_edge vs oos_edge (persistence) across all configs ----
fig, ax = plt.subplots(figsize=(7, 6))
for kind, mk in [("timevarying", "o"), ("static", "s")]:
    d = R[R.kind == kind]
    ax.scatter(d.is_edge, d.oos_edge, marker=mk, alpha=0.6, label=kind, s=30)
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("IS edge (median window)"); ax.set_ylabel("OOS edge")
ax.set_title("IS->OOS edge persistence across all (feature,K,R,direction) configs")
ax.legend()
fig.tight_layout(); fig.savefig(FIG / "fig3_is_oos_persistence.pdf"); plt.close(fig)

# ---- Fig 4: survivorship of the panel ----
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["%coins\nended >0", "%down\n>50%", "%down\n>90% (rug)"]
vals = [100*surv.frac_pos, 100*surv.frac_down50, 100*surv.frac_down90]
ax.bar(labels, vals, color=["#2a8", "#e80", "#c33"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=10)
ax.set_ylabel("% of coins")
ax.set_title(f"Survivorship of the live-pool panel (n={int(surv.n)} coins)\n"
             f"median total return {surv.median_total_ret:+.0%}, median DD-from-peak {surv.median_dd_from_peak:+.0%}")
fig.tight_layout(); fig.savefig(FIG / "fig4_survivorship.pdf"); plt.close(fig)

print("wrote 4 figures to", FIG)
