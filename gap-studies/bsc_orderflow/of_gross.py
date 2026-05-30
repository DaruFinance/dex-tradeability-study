"""Isolate the order-flow timing edge BEFORE cost vs the bar-shuffle null, to characterize the
channel: does flow timing beat random-entry timing at all (gross), and how big is the gap to cost?
Same intrabar SL-first sim, cost=0. Reports gross mean-net real vs null."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
rng=np.random.default_rng(7)
df=pd.read_parquet('./_gaps/bsc_orderflow/of_hourly.parquet').sort_values(['pair_address','ts'])

def sim(g,ent,tp,sl,mh,cost):
    o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;n=len(c);i=0;R=[]
    while i<n-1:
        if not ent[i]: i+=1; continue
        ep=o[i+1] if i+1<n else c[i]
        if not np.isfinite(ep) or ep<=0: i+=1; continue
        er=None;j=i+1;end=min(i+1+mh,n)
        while j<end:
            if l[j]<=ep*(1-sl): er=-sl;break
            if h[j]>=ep*(1+tp): er=tp;break
            j+=1
        if er is None: jx=min(end-1,n-1);er=c[jx]/ep-1.0;j=jx
        R.append(er-cost); i=j+1
    return np.array(R)

GRID=[('ofi_usd',0.0),('ofi_usd',0.1),('ofi_cnt',0.0),('ofi_cnt',0.1),('net_usd',0.0),('whale_net_usd',0.0)]
BR=[(0.02,0.02,6),(0.03,0.02,12),(0.03,0.03,24),(0.05,0.04,24)]
def run(shuffle,cost):
    out=[]
    for p,g in df.groupby('pair_address'):
        g=g.sort_values('ts').reset_index(drop=True)
        for f,thr in GRID:
            s=(g[f]>thr).astype(float).values
            for tp,sl,mh in BR:
                ss=s[rng.permutation(len(s))] if shuffle else s
                ent=np.zeros(len(ss),bool); ent[1:]=ss[:-1]>0
                r=sim(g,ent,tp,sl,mh,cost)
                if len(r)>=5: out.append(np.mean(r))
    return np.array(out)

# GROSS (cost=0): does flow-timed entry beat shuffled-entry timing?
g_real=run(False,0.0)
g_null=[np.mean(run(True,0.0)) for _ in range(40)]
print('GROSS (cost=0): real mean-net/trade=%.5f  null mean=%.5f (sd %.5f)'%(np.mean(g_real),np.mean(g_null),np.std(g_null)))
p_gross=(np.sum(np.array(g_null)>=np.mean(g_real))+1)/41
print('  perm p(real>=null) GROSS = %.3f'%p_gross)
# COSTED (1.6%):
c_real=run(False,0.016)
c_null=[np.mean(run(True,0.016)) for _ in range(40)]
print('COSTED (1.6%%): real=%.5f  null=%.5f (sd %.5f)'%(np.mean(c_real),np.mean(c_null),np.std(c_null)))
p_cost=(np.sum(np.array(c_null)>=np.mean(c_real))+1)/41
print('  perm p(real>=null) COSTED = %.3f'%p_cost)
pd.DataFrame({'regime':['gross','costed'],'real_mean_net':[np.mean(g_real),np.mean(c_real)],
              'null_mean_net':[np.mean(g_null),np.mean(c_null)],'perm_p':[p_gross,p_cost]}
            ).to_csv('./_gaps/bsc_orderflow/gross_vs_costed.csv',index=False)
