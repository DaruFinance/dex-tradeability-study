import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
G = Path("./_gaps/lp_market_making")
FIG = G / "figs"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ---- Fig 1: APR distribution, theoretical vs realistic (volcap+LVR), full & conc ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, mode, title in zip(axes, ("full", "conc"), ("Full-range (v2-style)", "Concentrated (+-30%)")):
    th = pd.read_csv(G / f"lp_{mode}_theoretical.csv").apr.clip(-2, 3)
    rl = pd.read_csv(G / f"lp_{mode}_realistic.csv").apr.clip(-2, 3)
    bins = np.linspace(-2, 3, 60)
    ax.hist(th, bins=bins, alpha=.5, label="theoretical (vol x fee)", color="#4C9F70")
    ax.hist(rl, bins=bins, alpha=.5, label="realistic (volcap + LVR)", color="#C0504D")
    ax.axvline(0, color="k", lw=.8, ls="--")
    ax.axvline(th.median(), color="#4C9F70", lw=1.5)
    ax.axvline(rl.median(), color="#C0504D", lw=1.5)
    ax.set_title(title); ax.set_xlabel("OOS net LP APR (annualized, clipped [-2,3])")
    ax.set_ylabel("pool-windows"); ax.legend(fontsize=8)
fig.suptitle("Passive LP net APR: naive theory vs adverse-selection-aware (DEX-only coins, WFO OOS)")
fig.tight_layout(); fig.savefig(FIG / "fig1_apr_distribution.pdf"); plt.close(fig)

# ---- Fig 2: component decomposition (median per-window, fraction of capital) ----
rob = pd.read_csv(G / "lp_robust_summary.csv")
real = rob[rob.tag == "volcap+LVR"]
fig, ax = plt.subplots(figsize=(8, 4.2))
modes = ["full", "conc"]
comp_cols = ["fee_med", "il_med", "lvr_med", "gas_med"]
labels = ["fees (+)", "impermanent loss", "LVR / adverse selection", "gas"]
colors = ["#4C9F70", "#E0A458", "#C0504D", "#7F7F7F"]
x = np.arange(len(modes)); w = 0.2
for i, (cc, lab, col) in enumerate(zip(comp_cols, labels, colors)):
    ax.bar(x + (i - 1.5) * w, real.set_index("mode").loc[modes, cc].values * 100, w, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels(["full-range", "concentrated"])
ax.axhline(0, color="k", lw=.8)
ax.set_ylabel("median per-window PnL (% of capital)")
ax.set_title("LP PnL decomposition (realistic model): fees vs IL vs LVR vs gas")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig2_components.pdf"); plt.close(fig)

# ---- Fig 3: WFO selection vs null (median APR), both signals ----
sel = pd.read_csv(G / "lp_select_robust.csv")
fig, ax = plt.subplots(figsize=(9, 4.4))
order = sel.copy()
order["lab"] = order["mode"] + " / " + order["signal"]
xs = np.arange(len(order))
ax.bar(xs - 0.2, order.real_med * 100, 0.4, label="REAL selected (median APR)", color="#4C9F70")
ax.bar(xs + 0.2, order.null_med_mean * 100, 0.4, label="NULL random (median APR)", color="#999999")
ax.errorbar(xs + 0.2, order.null_med_mean * 100, yerr=order.null_med_sd * 100, fmt="none",
            ecolor="k", capsize=3, lw=.8)
for i, p in enumerate(order.p_null_ge_real):
    ax.text(xs[i], max(order.real_med.iloc[i], order.null_med_mean.iloc[i]) * 100 + 5,
            f"p={p:.3f}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(order.lab, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("OOS median LP APR (%)"); ax.set_yscale("symlog")
ax.set_title("WFO pool selection vs null (realistic LP). Fee-yield persists; blue-chip = null")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig3_selection_vs_null.pdf"); plt.close(fig)

# ---- Fig 4: vol/TVL turnover distribution (the wash-flow caveat) ----
f = pd.read_csv(G / "lp_full_allwindows.csv")
f["dturn"] = f.vol_sum / f.reserve / f.n
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(np.log10(f.dturn.clip(1e-4, 1e4)), bins=70, color="#4C72B0")
for thr, c in [(3.0, "#C0504D")]:
    ax.axvline(np.log10(thr), color=c, lw=1.5, label=f"vol cap = {thr}x TVL/day")
ax.set_xlabel("log10(daily volume / TVL)"); ax.set_ylabel("pool-windows")
ax.set_title(f"Daily turnover (median {f.dturn.median():.3f}x): heavy wash/artifact tail above cap")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig4_turnover.pdf"); plt.close(fig)
print("figs written:", [p.name for p in FIG.glob("*.pdf")])
