import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D='./_gaps/bsc_orderflow/'

# Fig 1: TEST 0 predictive Spearman heatmap-style (feature x horizon, mean over 5 coins)
c=pd.read_csv(D+'test0_corr.csv')
piv=c.pivot(index='feature',columns='H',values='mean_rho')
fig,ax=plt.subplots(figsize=(7,4))
im=ax.imshow(piv.values,cmap='RdBu_r',vmin=-0.3,vmax=0.3,aspect='auto')
ax.set_xticks(range(len(piv.columns)));ax.set_xticklabels(piv.columns)
ax.set_yticks(range(len(piv.index)));ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j,i,'%.2f'%piv.values[i,j],ha='center',va='center',fontsize=8)
ax.set_xlabel('forward horizon (hours)');ax.set_ylabel('order-flow feature (shift+1)')
ax.set_title('TEST 0: mean Spearman corr, flow feature vs forward return\n(5 BSC DEX pairs, 2026-05-15..29)')
fig.colorbar(im,label='mean rho (5 coins)');fig.tight_layout();fig.savefig(D+'figs/fig1_corr_heatmap.pdf')

# Fig 2: gross vs costed, real vs null
g=pd.read_csv(D+'gross_vs_costed.csv')
fig,ax=plt.subplots(figsize=(6,4))
x=np.arange(2);w=0.35
ax.bar(x-w/2,g.real_mean_net*100,w,label='REAL flow-timed',color='#2c7fb8')
ax.bar(x+w/2,g.null_mean_net*100,w,label='NULL bar-shuffle',color='#bdbdbd')
for i,(r,n,p) in enumerate(zip(g.real_mean_net,g.null_mean_net,g.perm_p)):
    ax.text(i,min(r,n)*100-0.12,'p=%.2f'%p,ha='center',fontsize=9)
ax.set_xticks(x);ax.set_xticklabels(['gross (cost=0)','costed (1.6% RT)'])
ax.axhline(0,color='k',lw=.6);ax.set_ylabel('mean net return per trade (%)')
ax.set_title('TEST A: order-flow timing vs bar-shuffle null\nflow-timed entries never beat random-entry timing')
ax.legend();fig.tight_layout();fig.savefig(D+'figs/fig2_gross_vs_costed.pdf')

# Fig 3: TEST A costed PF distribution real vs null
ra=pd.read_csv(D+'testA_timing_real.csv');na=pd.read_csv(D+'testA_timing_null.csv')
ra=ra[ra.n>=5];na=na[na.n>=5]
fig,ax=plt.subplots(figsize=(6,4))
bins=np.linspace(0,1.2,25)
ax.hist(na.pf.clip(0,1.2),bins=bins,density=True,alpha=.5,label='NULL (50 shuffles)',color='#bdbdbd')
ax.hist(ra.pf.clip(0,1.2),bins=bins,density=True,alpha=.6,label='REAL',color='#2c7fb8')
ax.axvline(1.0,color='r',ls='--',lw=1,label='breakeven PF=1')
ax.set_xlabel('OOS profit factor (costed)');ax.set_ylabel('density')
ax.set_title('TEST A: costed PF distribution, both pile up <<1\n(median PF=0 real & null; cost dominates micro-edge)')
ax.legend();fig.tight_layout();fig.savefig(D+'figs/fig3_pf_dist.pdf')

# Fig 4: TEST B selection real vs null cum return distribution
import json
b=pd.read_csv(D+'testB_selection.csv').iloc[0]
fig,ax=plt.subplots(figsize=(6,4))
# reconstruct null dist not stored; show real vs null summary bars
ax.bar([0,1],[b.real_cum*100,b.null_cum_median*100],color=['#2c7fb8','#bdbdbd'],width=.5)
ax.errorbar([1],[b.null_cum_median*100],yerr=[[ (b.null_cum_median-b.null_cum_p5)*100],[(b.null_cum_p95-b.null_cum_median)*100]],
            fmt='none',ecolor='k',capsize=5)
ax.set_xticks([0,1]);ax.set_xticklabels(['REAL\nlong-top-flow','NULL\nlabel-permuted'])
ax.set_ylabel('cumulative net return (%)')
ax.set_title('TEST B: cross-sectional selection (N=5, UNDERPOWERED)\nreal indistinguishable from label-permuted null (p=%.2f)'%b.perm_p)
ax.axhline(0,color='k',lw=.6);fig.tight_layout();fig.savefig(D+'figs/fig4_selection.pdf')
print('figs written')
