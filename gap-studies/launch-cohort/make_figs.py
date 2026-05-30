"""Publication figures for the launch-cohort gap study."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path("./_gaps/launch-cohort"); F=OUT/"figs"

# ---- Fig 1: event-study quantile fan ----
es=pd.read_csv(OUT/"event_study_path.csv")
fig,ax=plt.subplots(figsize=(8,5))
a=es["age"]
ax.fill_between(a,es["p5"],es["p95"],color="#cfe3f5",alpha=.7,label="5-95 pct")
ax.fill_between(a,es["p25"],es["p75"],color="#7fb3e0",alpha=.7,label="25-75 pct")
ax.plot(a,es["p50"],color="#0b3d66",lw=2,label="median")
ax.plot(a,es["mean"],color="#c1272d",lw=2,ls="--",label="mean (skew)")
ax.axhline(1.0,color="k",lw=.8,ls=":")
ax.set_xlabel("Days since pool launch (age)"); ax.set_ylabel("Price (normalized, entry=1.0)")
ax.set_title("DEX-only launch cohort: average price path (n=323 launch-captured pools)")
ax.legend(loc="upper left"); ax.set_ylim(0,4.5); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(F/"fig1_event_study_fan.pdf"); plt.close(fig)

# log-scale companion to show median is flat while mean explodes
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(a,es["p50"],color="#0b3d66",lw=2,label="median")
ax.plot(a,es["mean"],color="#c1272d",lw=2,ls="--",label="mean")
ax.plot(a,es["p25"],color="#888",lw=1,label="p25"); ax.plot(a,es["p75"],color="#888",lw=1,ls=":",label="p75")
ax.axhline(1.0,color="k",lw=.8,ls=":"); ax.set_yscale("log")
ax.set_xlabel("Days since pool launch"); ax.set_ylabel("Price (norm, log)")
ax.set_title("Median path is flat; mean is dragged by a few moonshots (right-skew)")
ax.legend(); ax.grid(alpha=.3,which="both"); fig.tight_layout()
fig.savefig(F/"fig2_median_vs_mean_log.pdf"); plt.close(fig)

# ---- Fig 3: age-filter real vs null ----
R=pd.read_csv(OUT/"age_filter_results.csv")
real=R[~R.null]; null=R[R.null]
fig,axs=plt.subplots(1,2,figsize=(12,5))
axs[0].plot(real.skipN,real.median_oos_pf,"o-",color="#0b3d66",label="REAL median OOS PF")
axs[0].plot(null.skipN,null.median_oos_pf,"s--",color="#c1272d",label="NULL (bar-shuffle) median OOS PF")
axs[0].axhline(1.0,color="k",lw=.8,ls=":"); axs[0].set_xlabel("Skip first N launch days")
axs[0].set_ylabel("Median OOS profit factor"); axs[0].set_title("Age filter does NOT lift edge; null beats real")
axs[0].legend(); axs[0].grid(alpha=.3)
axs[1].plot(real.skipN,real.median_avg_ret,"o-",color="#0b3d66",label="REAL median net ret/trade")
axs[1].plot(null.skipN,null.median_avg_ret,"s--",color="#c1272d",label="NULL median net ret/trade")
axs[1].axhline(0.0,color="k",lw=.8,ls=":"); axs[1].set_xlabel("Skip first N launch days")
axs[1].set_ylabel("Median net return per trade"); axs[1].set_title("Real timing is net-negative at every offset")
axs[1].legend(); axs[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(F/"fig3_age_filter_real_vs_null.pdf"); plt.close(fig)
print("figs written:", [p.name for p in F.glob("*.pdf")])
