import pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path('./_gaps/beta_hedge_dex_basket'); F=OUT/'figs'
d=pickle.load(open(OUT/'_series.pkl','rb')); st=d['store']

def ds(s): return pd.to_datetime(s.index.values,unit='s')

# Fig 1: equity curves (equal-weight, real) + buyhold + nulls
fig,ax=plt.subplots(figsize=(9,5.2))
plot=[(('equal','real','unhedged'),'Unhedged long (net)','#888',1.5,'-'),
      (('equal','real','hedge_BTC'),'BTC-beta hedged','#d62728',1.8,'-'),
      (('equal','real','hedge_ETH'),'ETH-beta hedged','#1f77b4',1.8,'-'),
      (('equal','real','buyhold_noskill'),'Buy&hold no-skill','#2ca02c',1.8,'--'),
      (('equal','null','hedge_ETH'),'ETH-hedge NULL (bar-shuffle)','#cc88cc',1.4,':')]
for k,lab,c,lw,ls in plot:
    if k in st:
        s=st[k]; ax.plot(ds(s),(1+s).cumprod(),label=lab,color=c,lw=lw,ls=ls)
ax.axhline(1,color='k',lw=.6,alpha=.5)
ax.set_title('Long DEX basket: beta-hedged vs unhedged vs naive (equal-weight, OOS, net of all costs)')
ax.set_ylabel('Cumulative growth of $1 (OOS)'); ax.set_xlabel('Date')
ax.legend(loc='upper left',fontsize=8.5); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(F/'fig1_equity_curves.pdf')

# Fig 2: real vs null hedged Sharpe distribution
nd=pd.read_csv(OUT/'null_sharpe_dist.csv')
fig,ax=plt.subplots(figsize=(8,5))
for ref,c,real in [('ETH','#1f77b4',1.475),('BTC','#d62728',0.475)]:
    sub=nd[nd.ref==ref].sharpe
    ax.hist(sub,bins=18,alpha=.5,color=c,label=f'NULL {ref}-hedge (60 seeds)')
    ax.axvline(real,color=c,lw=2.2,ls='-',label=f'REAL {ref}-hedge = {real:.2f}')
ax.set_title('Hedged OOS Sharpe: real vs bar-shuffle null\n(real beats null, but null is degraded by turnover-cost on shuffled persistence)')
ax.set_xlabel('Annualized Sharpe (OOS)'); ax.set_ylabel('Null count')
ax.legend(fontsize=8.5); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(F/'fig2_real_vs_null.pdf')

# Fig 3: rebalance sensitivity -- the edge is just trade-less
rdf=pd.read_csv(OUT/'rebal_sensitivity.csv')
fig,ax=plt.subplots(figsize=(8,5))
x=range(len(rdf)); lab=[str(v) for v in rdf.rebal]
ax.plot(x,rdf.unhedged_sharpe,'-o',color='#888',label='Unhedged long')
ax.plot(x,rdf.ethhedge_sharpe,'-o',color='#1f77b4',label='ETH-beta hedged')
ax.axhline(0,color='k',lw=.7)
ax.set_xticks(list(x)); ax.set_xticklabels(lab)
ax.set_xlabel('Rebalance interval (days; "hold"=buy once)'); ax.set_ylabel('OOS Sharpe')
ax.set_title('OOS Sharpe is monotone in rebalance interval:\nthe apparent hedge "edge" is purely lower turnover vs ~164bp per-fill DEX cost')
ax.legend(fontsize=9); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(F/'fig3_rebal_sensitivity.pdf')

# Fig 4: bootstrap CI on headline Sharpe
bt=pd.read_csv(OUT/'bootstrap_sharpe.csv')
fig,ax=plt.subplots(figsize=(8,4.8))
y=range(len(bt))
ax.errorbar(bt.sharpe,y,xerr=[bt.sharpe-bt.ci_lo,bt.ci_hi-bt.sharpe],fmt='o',color='#1f77b4',capsize=4)
ax.axvline(0,color='r',lw=1.2,ls='--')
ax.set_yticks(list(y)); ax.set_yticklabels(bt.variant,fontsize=8.5)
ax.set_xlabel('OOS Sharpe (point + 95% block-bootstrap CI)')
ax.set_title('Every hedged variant\'s 95% CI straddles or sits below zero\n(headline ETH-hedge p(Sharpe<=0)=0.26)')
ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(F/'fig4_bootstrap_ci.pdf')
print('figs written')
