#!/usr/bin/env python3
"""Publication figures for multi-pool dispersion gap."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT='./_gaps/multipool_dispersion'
df=pd.read_csv(f'{OUT}/per_token_dispersion.csv')
clean=df[df.real_med_disp<=0.20].copy()
MEDCOST=clean.real_arb_cost.median()

# FIG 1: dispersion distribution real vs null + cost line
fig,ax=plt.subplots(figsize=(7,4.5))
bins=np.linspace(0,0.10,60)
ax.hist(clean.real_med_disp.clip(0,0.10),bins=bins,alpha=.6,label=f'REAL (median {clean.real_med_disp.median()*1e4:.0f} bp)',color='#2a6f97')
ax.hist(clean.null_med_disp.clip(0,0.10),bins=bins,alpha=.5,label=f'NULL bar-shuffle (median {clean.null_med_disp.median()*1e4:.0f} bp)',color='#c1121f')
ax.axvline(MEDCOST,color='k',ls='--',lw=1.5,label=f'median round-trip arb cost ({MEDCOST*1e4:.0f} bp)')
ax.set_xlabel('Per-token median cross-pool price dispersion  (max-min)/median')
ax.set_ylabel('number of tokens')
ax.set_title('Multi-pool DEX price dispersion vs arb cost (daily, 391 same-chain multi-DEX tokens)')
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f'{OUT}/figs/fig1_dispersion_vs_cost.pdf')

# FIG 2: fraction of bars exceeding cost, real vs null (CDF)
fig,ax=plt.subplots(figsize=(7,4.5))
for col,lab,c in [('real_frac_disp_gt_cost','REAL','#2a6f97'),('null_frac_disp_gt_cost','NULL bar-shuffle','#c1121f')]:
    x=np.sort(clean[col].values); y=np.arange(1,len(x)+1)/len(x)
    ax.plot(x,y,label=f'{lab} (median {np.median(x)*100:.1f}%)',color=c,lw=2)
ax.set_xlabel('fraction of bars where dispersion > round-trip arb cost')
ax.set_ylabel('cumulative fraction of tokens')
ax.set_title('How often is cross-pool dispersion even arbitrageable? (daily)')
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f'{OUT}/figs/fig2_frac_exceed_cost_cdf.pdf')

# FIG 3: costed arb backtest net result real vs null (per-token net_sum, signed-log)
fig,ax=plt.subplots(figsize=(7,4.5))
def slog(v): return np.sign(v)*np.log10(1+np.abs(v))
ax.scatter(slog(clean.real_net_sum),slog(clean.null_net_sum),s=14,alpha=.5,color='#555')
lim=[slog(clean[['real_net_sum','null_net_sum']].values).min(),slog(clean[['real_net_sum','null_net_sum']].values).max()]
ax.plot(lim,lim,'k--',lw=1,alpha=.6)
ax.axhline(0,color='gray',lw=.7); ax.axvline(0,color='gray',lw=.7)
ax.set_xlabel('REAL costed-arb net PnL per token  (signed log10 frac-units)')
ax.set_ylabel('NULL costed-arb net PnL per token  (signed log10)')
ax.set_title(f'Costed "rebalance to consensus" arb: REAL {(clean.real_net_sum>0).mean()*100:.0f}% net+ vs NULL {(clean.null_net_sum>0).mean()*100:.0f}% net+')
fig.tight_layout(); fig.savefig(f'{OUT}/figs/fig3_arb_backtest_real_vs_null.pdf')

# FIG 4: bar chart summary daily + hourly
hs=pd.read_csv(f'{OUT}/summary_hourly_real_vs_null.csv')
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
labels=['median\ndispersion','frac bars\n> cost','tokens\nnet-positive']
dreal=[clean.real_med_disp.median(),clean.real_frac_disp_gt_cost.median(),(clean.real_net_sum>0).mean()]
dnull=[clean.null_med_disp.median(),clean.null_frac_disp_gt_cost.median(),(clean.null_net_sum>0).mean()]
x=np.arange(3); w=.38
axs[0].bar(x-w/2,dreal,w,label='REAL',color='#2a6f97'); axs[0].bar(x+w/2,dnull,w,label='NULL',color='#c1121f')
axs[0].set_xticks(x); axs[0].set_xticklabels(labels); axs[0].set_title('Daily (391 tokens)'); axs[0].legend(); axs[0].set_ylabel('value (frac)')
hreal=[hs.loc[0,'med_disp'],hs.loc[0,'med_frac_gt_cost'],hs.loc[0,'frac_net_pos']]
hnull=[hs.loc[1,'med_disp'],hs.loc[1,'med_frac_gt_cost'],hs.loc[1,'frac_net_pos']]
axs[1].bar(x-w/2,hreal,w,label='REAL',color='#2a6f97'); axs[1].bar(x+w/2,hnull,w,label='NULL',color='#c1121f')
axs[1].set_xticks(x); axs[1].set_xticklabels(labels); axs[1].set_title('Hourly (144 tokens)'); axs[1].legend()
fig.suptitle('REAL vs NULL across timeframes: real prices stay tightly co-arbitraged'); fig.tight_layout()
fig.savefig(f'{OUT}/figs/fig4_summary_bars.pdf')
print('figures written')
