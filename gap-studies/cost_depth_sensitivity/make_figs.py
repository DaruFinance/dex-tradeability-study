"""Publication figures for the cost-depth sensitivity gap study."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path("./_gaps/cost_depth_sensitivity"); F=OUT/"figs"

gr=pd.read_csv(OUT/"grid_real.csv"); gn=pd.read_csv(OUT/"grid_null.csv")
sr=pd.read_csv(OUT/"sweep_real.csv"); sn=pd.read_csv(OUT/"sweep_null.csv")
br=pd.read_csv(OUT/"buyhold_real.csv"); bn=pd.read_csv(OUT/"buyhold_null.csv")

# --- Fig 1: cost-vs-edge frontier (multiplier sweep) real vs null, median avg net return ---
fig,ax=plt.subplots(figsize=(7,4.6))
ax.plot(sr.median_cost_bp,sr.median_avg_ret_bp,"-o",color="#1f77b4",label="Real (median net ret/trade)",ms=4)
ax.plot(sn.median_cost_bp,sn.median_avg_ret_bp,"-s",color="#d62728",label="Null (bar-shuffled)",ms=4)
ax.axhline(0,color="k",lw=.8,ls="--")
ax.axvline(170,color="gray",lw=.8,ls=":"); ax.text(175,ax.get_ylim()[1]*0.9,"v2 baseline ~170bp",fontsize=8,color="gray")
ax.set_xlabel("Round-trip cost (bp, median across coin-strategies)")
ax.set_ylabel("Median net return per trade (bp)")
ax.set_title("Cost-vs-edge frontier: timing edge stays negative even at zero cost")
ax.legend(); fig.tight_layout(); fig.savefig(F/"fig1_cost_edge_frontier.pdf")

# --- Fig 2: % strategies OOS PF>1 vs cost, real vs null ---
fig,ax=plt.subplots(figsize=(7,4.6))
ax.plot(sr.median_cost_bp,sr.pct_pf_gt1,"-o",color="#1f77b4",label="Real",ms=4)
ax.plot(sn.median_cost_bp,sn.pct_pf_gt1,"-s",color="#d62728",label="Null",ms=4)
ax.axhline(50,color="k",lw=.8,ls="--",label="coin-flip (50%)")
ax.set_xlabel("Round-trip cost (bp)"); ax.set_ylabel("% coin-strategies with OOS PF>1")
ax.set_title("Fraction of profitable timing strategies vs cost")
ax.legend(); fig.tight_layout(); fig.savefig(F/"fig2_pct_profitable.pdf")

# --- Fig 3: depth-fraction heatmap of median net ret/trade (real) ---
piv=gr.pivot(index="stale_bp",columns="depth_frac",values="median_avg_ret_bp")
fig,ax=plt.subplots(figsize=(6.5,4.2))
im=ax.imshow(piv.values,cmap="RdBu",aspect="auto",vmin=-np.abs(piv.values).max(),vmax=np.abs(piv.values).max())
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:.2f}" for c in piv.columns])
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{int(i)}" for i in piv.index])
ax.set_xlabel("v3 active-tick depth fraction of TVL"); ax.set_ylabel("stale-close penalty (bp/leg)")
ax.set_title("Median net return/trade (bp) under depth mis-specification (real)")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j,i,f"{piv.values[i,j]:.0f}",ha="center",va="center",fontsize=8)
fig.colorbar(im,label="bp"); fig.tight_layout(); fig.savefig(F/"fig3_depth_heatmap.pdf")

# --- Fig 4: buy&hold net vs cost, real vs null ---
fig,ax=plt.subplots(figsize=(7,4.6))
ax.plot(br.cost_mult*170,br.median_bh_net*100,"-o",color="#1f77b4",label="Real B&H median",ms=4)
ax.plot(br.cost_mult*170,br.mean_bh_net*100,"--o",color="#1f77b4",label="Real B&H mean",ms=3,alpha=.6)
ax.axhline(0,color="k",lw=.8,ls="--")
ax.set_xlabel("Round-trip cost (bp, approx)"); ax.set_ylabel("Net buy&hold return (%)")
ax.set_title("Buy&hold is already deeply negative gross (survivorship-tilted universe)")
ax.legend(); fig.tight_layout(); fig.savefig(F/"fig4_buyhold.pdf")
print("figs written")
