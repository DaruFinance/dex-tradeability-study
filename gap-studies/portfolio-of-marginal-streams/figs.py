"""Publication figures for the portfolio-of-marginal-streams gap."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("./_gaps/portfolio-of-marginal-streams")
FIG = OUT/"figs"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi":120,"font.size":10,"axes.grid":True,"grid.alpha":0.3})

C = pd.read_csv(OUT/"capacity_sweep.csv")

# Fig 1: per-stream expectancy distribution real vs null
fig, ax = plt.subplots(1,2, figsize=(11,4.2))
for mode,col in [("real","#c1121f"),("null","#457b9d")]:
    T=pd.read_parquet(OUT/f"trades_{mode}.parquet")
    se=T.groupby("stream").net_ret.mean()
    ax[0].hist(se.clip(-0.3,0.3), bins=80, alpha=0.55, label=f"{mode} (med {se.median():+.3f})", color=col, density=True)
ax[0].axvline(0,color="k",lw=1,ls="--"); ax[0].set_xlabel("per-stream mean net return / trade")
ax[0].set_ylabel("density"); ax[0].set_title("Per-stream expectancy: REAL is negative, NULL is positive")
ax[0].legend()
# pct positive-sum streams bar
pos={}
for mode in ["real","null"]:
    T=pd.read_parquet(OUT/f"trades_{mode}.parquet"); s=T.groupby("stream").net_ret.sum()
    pos[mode]=100*(s>0).mean()
ax[1].bar(["real","null"],[pos["real"],pos["null"]],color=["#c1121f","#457b9d"])
ax[1].set_ylabel("% of streams with positive cumulative PnL")
ax[1].set_title("Profitable-stream share (net of cost)")
for i,(k,v) in enumerate(pos.items()): ax[1].text(i,v+1,f"{v:.1f}%",ha="center")
ax[1].set_ylim(0,75)
fig.tight_layout(); fig.savefig(FIG/"fig1_per_stream_expectancy.pdf"); plt.close(fig)

# Fig 2: Sharpe and ann_ret vs AUM (capacity decay), ew_pos, real vs null
fig, ax = plt.subplots(1,2, figsize=(11,4.2))
for mode,col in [("real","#c1121f"),("null","#457b9d")]:
    d=C[(C["mode"]==mode)&(C.config=="ew_pos")].sort_values("aum")
    ax[0].plot(d.aum,d.sharpe,"o-",color=col,label=mode)
    ax[1].plot(d.aum,d.ann_ret,"o-",color=col,label=mode)
for a in ax: a.set_xscale("log"); a.set_xlabel("AUM (USD)"); a.legend()
ax[0].set_ylabel("OOS Sharpe (annualized)"); ax[0].set_title("Sharpe vs AUM (EW positive-selection book)")
ax[1].set_ylabel("annualized return"); ax[1].set_title("Return decays toward cash as AUM grows")
fig.tight_layout(); fig.savefig(FIG/"fig2_capacity_decay.pdf"); plt.close(fig)

# Fig 3: deployed fraction vs AUM (capacity ceiling), all configs real
fig, ax = plt.subplots(figsize=(7,4.5))
for cfg,ls in [("ew_pos","-"),("ew_all","--"),("invvol_pos",":")]:
    d=C[(C["mode"]=="real")&(C.config==cfg)].sort_values("aum")
    ax.plot(d.aum,100*d.deployed_frac,ls,marker="o",label=f"real {cfg}")
ax.set_xscale("log"); ax.set_xlabel("AUM (USD)"); ax.set_ylabel("% of intended capital actually deployed")
ax.set_title("Capacity ceiling: median per-stream cap = $103, so capital sits in cash")
ax.axhline(50,color="gray",lw=0.8,ls=":"); ax.legend()
fig.tight_layout(); fig.savefig(FIG/"fig3_deployed_fraction.pdf"); plt.close(fig)

# Fig 4: equity curves of the $10k book real vs null (ew_pos)
B = pd.read_parquet(OUT/"book_daily_10k.parquet")
fig, ax = plt.subplots(figsize=(8,4.5))
for mode,col in [("real","#c1121f"),("null","#457b9d")]:
    key=f"{mode}|ew_pos"
    if key in B.columns:
        r=B[key].dropna(); eq=(1+r).cumprod()
        ax.plot(eq.index, eq.values, color=col, label=f"{mode} ew_pos $10k")
ax.set_yscale("log"); ax.set_ylabel("equity (log, start=1)"); ax.set_xlabel("date")
ax.set_title("$10k book equity: NULL dominates REAL (no genuine edge)")
ax.legend()
fig.tight_layout(); fig.savefig(FIG/"fig4_equity_curves.pdf"); plt.close(fig)

print("figures written to", FIG)
