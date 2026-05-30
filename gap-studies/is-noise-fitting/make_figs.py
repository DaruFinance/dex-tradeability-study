import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path("./_gaps/is-noise-fitting")
T=pd.read_csv(OUT/"real_vs_null_by_setting.csv")
R=pd.read_csv(OUT/"sweep_results.csv")

plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.3})

# --- FIG 1: real vs null median OOS PF, all settings ---
order=["IS240_TR5_NS12_pf","IS360_TR5_NS12_pf","IS480_TR5_NS12_pf",
       "IS240_TR15_NS12_pf","IS240_TR30_NS12_pf","IS240_TR5_NS40_pf",
       "IS240_TR5_NS12_pf_pen","IS240_TR5_NS12_netpt",
       "STRICT_IS360_TR15_NS40_pfpen","STRICT_IS480_TR30_NS40_pfpen"]
T2=T.set_index("setting").loc[order].reset_index()
x=np.arange(len(T2)); w=0.38
fig,ax=plt.subplots(figsize=(11,5))
ax.bar(x-w/2, T2.real_medPF, w, label="REAL", color="#c0392b")
ax.bar(x+w/2, T2.null_medPF, w, label="NULL (bar-shuffle)", color="#7f8c8d")
ax.axhline(1.0, color="k", ls="--", lw=1, label="break-even PF=1")
ax.set_xticks(x); ax.set_xticklabels(T2.setting, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("median OOS profit factor (all evals)")
ax.set_title("Noise-fitting gap test: stricter/longer IS selection does NOT lift real OOS PF\n"
             "Real is below its own bar-shuffle null in EVERY setting (long-only DEX timing, hourly, per-fill cost)")
ax.legend()
fig.tight_layout(); fig.savefig(OUT/"figs/fig1_real_vs_null_medPF.pdf"); plt.close(fig)

# --- FIG 2: IS_LEN sweep & MIN_IS_TR sweep trends ---
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
# IS length: IS240/360/480 at TR5 NS12 pf
isl=T[T.setting.isin(["IS240_TR5_NS12_pf","IS360_TR5_NS12_pf","IS480_TR5_NS12_pf"])].copy()
isl["IS"]=isl.setting.str.extract(r"IS(\d+)").astype(int)
isl=isl.sort_values("IS")
axs[0].plot(isl.IS, isl.real_medPF, "o-", color="#c0392b", label="REAL")
axs[0].plot(isl.IS, isl.null_medPF, "s--", color="#7f8c8d", label="NULL")
axs[0].axhline(1.0,color="k",ls=":",lw=1)
axs[0].set_xlabel("IS window length (hourly bars)"); axs[0].set_ylabel("median OOS PF")
axs[0].set_title("Longer IS does not help real OOS"); axs[0].legend()
# MIN_IS_TR: TR5/15/30 at IS240 NS12 pf
mt=T[T.setting.isin(["IS240_TR5_NS12_pf","IS240_TR15_NS12_pf","IS240_TR30_NS12_pf"])].copy()
mt["TR"]=mt.setting.str.extract(r"TR(\d+)").astype(int)
mt=mt.sort_values("TR")
axs[1].plot(mt.TR, mt.real_medPF, "o-", color="#c0392b", label="REAL")
axs[1].plot(mt.TR, mt.null_medPF, "s--", color="#7f8c8d", label="NULL")
axs[1].axhline(1.0,color="k",ls=":",lw=1)
axs[1].set_xlabel("MIN_IS_TR (min IS trades to select)"); axs[1].set_ylabel("median OOS PF")
axs[1].set_title("Stricter IS-trade floor does not help real OOS"); axs[1].legend()
fig.tight_layout(); fig.savefig(OUT/"figs/fig2_is_strictness_trends.pdf"); plt.close(fig)

# --- FIG 3: mean net per trade, real vs null ---
fig,ax=plt.subplots(figsize=(11,4.5))
ax.bar(x-w/2, 100*T2.real_meanNet, w, label="REAL", color="#c0392b")
ax.bar(x+w/2, 100*T2.null_meanNet, w, label="NULL", color="#7f8c8d")
ax.axhline(0,color="k",lw=1)
ax.set_xticks(x); ax.set_xticklabels(T2.setting, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("mean net return per trade (%)")
ax.set_title("Mean net per trade: real is negative everywhere and below null in every setting")
ax.legend(); fig.tight_layout(); fig.savefig(OUT/"figs/fig3_mean_net_per_trade.pdf"); plt.close(fig)

# --- FIG 4: PF distribution baseline real vs null (CDF) ---
import glob
r=pd.read_parquet(OUT/"_detail_IS240_TR5_NS12_pf_REAL.parquet").oos_pf.replace(np.inf,np.nan).dropna().clip(upper=5)
n=pd.read_parquet(OUT/"_detail_IS240_TR5_NS12_pf_NULL.parquet").oos_pf.replace(np.inf,np.nan).dropna().clip(upper=5)
fig,ax=plt.subplots(figsize=(7,5))
for d,lab,col in [(r,"REAL","#c0392b"),(n,"NULL","#7f8c8d")]:
    xs=np.sort(d.values); ys=np.arange(1,len(xs)+1)/len(xs)
    ax.plot(xs,ys,label=f"{lab} (median {np.median(d):.2f})",color=col,lw=2)
ax.axvline(1.0,color="k",ls="--",lw=1,label="break-even")
ax.set_xlabel("OOS profit factor (capped at 5)"); ax.set_ylabel("cumulative fraction of (coin,family) units")
ax.set_title("Baseline setting (IS240/TR5/NS12/pf): OOS PF distribution\nReal stochastically dominated by its own bar-shuffle null")
ax.legend(); fig.tight_layout(); fig.savefig(OUT/"figs/fig4_pf_cdf_baseline.pdf"); plt.close(fig)
print("figs written")
