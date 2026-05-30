"""Analyze regime-conditioned evals: real vs null, full distribution, figures + CSVs."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("./_gaps/btc_regime_timing")
VARIANTS = ["UNCOND","RISK_ON_TREND","RISK_OFF_TREND","RISK_ON_LOWVOL","RISK_OFF_LOWVOL","MASTER_ON","MASTER_OFF"]
LBL = {"UNCOND":"Unconditional","RISK_ON_TREND":"Risk-on (BTC uptrend)","RISK_OFF_TREND":"Risk-off (BTC downtrend)",
       "RISK_ON_LOWVOL":"Risk-on (low BTC vol)","RISK_OFF_LOWVOL":"Risk-off (high BTC vol)",
       "MASTER_ON":"Master ON (uptrend & low vol)","MASTER_OFF":"Master OFF"}

real=pd.read_parquet(OUT/"evals_real.parquet")
null=pd.read_parquet(OUT/"evals_null.parquet")

def summ(df):
    out=[]
    for vr in VARIANTS:
        s=df[df.variant==vr]
        if not len(s):
            out.append(dict(variant=vr,n=0)); continue
        apf=s.oos_pf.replace(np.inf,np.nan)
        out.append(dict(variant=vr,n=len(s),
            n_coins=s.coin.nunique(),
            med_oos_pf=apf.median(),
            mean_oos_pf_cap10=apf.clip(upper=10).mean(),
            pct_pf_gt1=100*(s.oos_pf>1).mean(),
            med_net_per_trade=s.avg_per_trade.median(),
            mean_net_per_trade=s.avg_per_trade.mean(),
            med_oos_n=s.oos_n.median()))
    return pd.DataFrame(out)

sr=summ(real); sn=summ(null)
sr["run"]="real"; sn["run"]="null"
comb=pd.concat([sr,sn],ignore_index=True)
comb.to_csv(OUT/"summary_real_vs_null.csv",index=False)

# side-by-side table
m=sr.merge(sn,on="variant",suffixes=("_real","_null"))
m["delta_med_pf"]=m.med_oos_pf_real-m.med_oos_pf_null
m["delta_mean_net"]=m.mean_net_per_trade_real-m.mean_net_per_trade_null
m=m[["variant","n_real","n_coins_real","med_oos_pf_real","med_oos_pf_null","delta_med_pf",
     "pct_pf_gt1_real","pct_pf_gt1_null","mean_net_per_trade_real","mean_net_per_trade_null","delta_mean_net"]]
m.to_csv(OUT/"real_vs_null_sidebyside.csv",index=False)
print(m.to_string(index=False))

# bootstrap CI on mean net-per-trade per variant (real), and real-vs-null diff
def boot_ci(x,n=2000,seed=1):
    x=np.asarray(x); rng=np.random.default_rng(seed)
    bs=[rng.choice(x,len(x),replace=True).mean() for _ in range(n)]
    return np.percentile(bs,2.5),np.percentile(bs,97.5)
ci_rows=[]
for vr in VARIANTS:
    rx=real[real.variant==vr].avg_per_trade.to_numpy()
    nx=null[null.variant==vr].avg_per_trade.to_numpy()
    if len(rx)<5: continue
    lo,hi=boot_ci(rx)
    ci_rows.append(dict(variant=vr,mean_net_real=rx.mean(),ci_lo=lo,ci_hi=hi,
                        mean_net_null=nx.mean() if len(nx) else np.nan,
                        real_minus_null=rx.mean()-(nx.mean() if len(nx) else np.nan)))
ci=pd.DataFrame(ci_rows); ci.to_csv(OUT/"bootstrap_ci_net.csv",index=False)
print("\n",ci.to_string(index=False))

# ---- FIG 1: median OOS PF, real vs null, per variant ----
fig,ax=plt.subplots(figsize=(10,5.5))
x=np.arange(len(VARIANTS)); w=0.38
rv=[sr[sr.variant==v].med_oos_pf.iloc[0] if len(sr[sr.variant==v]) else np.nan for v in VARIANTS]
nv=[sn[sn.variant==v].med_oos_pf.iloc[0] if len(sn[sn.variant==v]) else np.nan for v in VARIANTS]
ax.bar(x-w/2,rv,w,label="Real",color="#2c6fbb")
ax.bar(x+w/2,nv,w,label="Bar-shuffle null",color="#c0392b",alpha=.8)
ax.axhline(1.0,ls="--",c="k",lw=1,label="PF=1 (breakeven)")
ax.set_xticks(x); ax.set_xticklabels([LBL[v] for v in VARIANTS],rotation=35,ha="right",fontsize=8)
ax.set_ylabel("Median OOS profit factor (all evaluated coin×family)")
ax.set_title("BTC-regime-conditioned long-only timing on DEX-only alts\nMedian OOS PF: real vs null (daily, per-fill cost, intrabar TP/SL)")
ax.legend(); fig.tight_layout(); fig.savefig(OUT/"figs/fig1_median_pf.pdf"); plt.close(fig)

# ---- FIG 2: mean net per trade with bootstrap CI (real) + null marker ----
fig,ax=plt.subplots(figsize=(10,5.5))
vrs=ci.variant.tolist(); xx=np.arange(len(vrs))
ax.errorbar(xx,ci.mean_net_real,yerr=[ci.mean_net_real-ci.ci_lo,ci.ci_hi-ci.mean_net_real],
            fmt="o",color="#2c6fbb",capsize=4,label="Real (95% bootstrap CI)")
ax.scatter(xx,ci.mean_net_null,marker="x",color="#c0392b",s=60,label="Null mean")
ax.axhline(0,ls="--",c="k",lw=1)
ax.set_xticks(xx); ax.set_xticklabels([LBL[v] for v in vrs],rotation=35,ha="right",fontsize=8)
ax.set_ylabel("Mean net return per trade (after cost)")
ax.set_title("Mean net per-trade by BTC regime, real vs null\n(positive = edge; all variants expected ≤ 0)")
ax.legend(); fig.tight_layout(); fig.savefig(OUT/"figs/fig2_net_per_trade_ci.pdf"); plt.close(fig)

# ---- FIG 3: OOS PF distribution (CDF) real, risk-on vs risk-off vs uncond ----
fig,ax=plt.subplots(figsize=(9,5.5))
for vr,col in [("UNCOND","#444"),("RISK_ON_TREND","#27ae60"),("RISK_OFF_TREND","#c0392b"),("MASTER_ON","#8e44ad")]:
    s=real[real.variant==vr].oos_pf.replace(np.inf,np.nan).dropna()
    s=s.clip(upper=5)
    xs=np.sort(s); ys=np.arange(1,len(xs)+1)/len(xs)
    ax.plot(xs,ys,label=f"{LBL[vr]} (n={len(s)})",color=col,lw=1.8)
ax.axvline(1.0,ls="--",c="k",lw=1)
ax.set_xlabel("OOS profit factor (clipped at 5)"); ax.set_ylabel("Cumulative fraction of coin×family")
ax.set_title("OOS PF distribution by regime (real). Mass left of PF=1 line = losing strategies.")
ax.legend(); fig.tight_layout(); fig.savefig(OUT/"figs/fig3_pf_cdf.pdf"); plt.close(fig)

print("\nfigures + CSVs written to", OUT)
