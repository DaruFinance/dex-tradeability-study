"""Test BSC per-swap order-flow as a predictive channel that price-only timing cannot see.

Coverage (honest): 5 DEX-only BSC pairs (IBS,ODY,Pro,LAX,wkeyDAO2), ~337 hourly bars each
(~14 days, 2026-05-15..05-29), 1683 pair-hours. Dense per-swap tapes (1.27M swaps).

TEST 0  Predictive correlation: does shift(1) order-flow imbalance predict next-bar return?
TEST A  TIMING: long when shift(1) flow signal fires; intrabar TP/SL; per-fill cost.
        NULL = shuffle the flow-signal series within each coin (bar-shuffle), identical pipeline.
TEST B  SELECTION: each hour rank coins by trailing net-buy pressure, long top-K vs equal-weight.
        NULL = permute coin labels on the signal. FLAGGED severely underpowered (N=5 coins).

All signals causal (.shift(1)). All costs per-fill via project model. Seeded.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
from chainscope.costs import round_trip_cost_frac
rng=np.random.default_rng(42)

CHAIN_GAS={'bsc':.2}; CHAIN_NATIVE={'bsc':640}
def coin_cost(reserve):
    size=max(50.,0.0025*(reserve or 0))
    return round_trip_cost_frac(size,reserve,dex='uniswap',chain='bsc',gas_usd=.2,native_usd=640) if reserve else .05

df=pd.read_parquet('./_gaps/bsc_orderflow/of_hourly.parquet')
df=df.sort_values(['pair_address','ts']).reset_index(drop=True)

# forward returns (open-to-open H bars ahead is more tradeable; use close-to-close for corr)
def fwd_ret(c,H): return pd.Series(c).shift(-H)/pd.Series(c)-1.0

# per-coin representative reserve & cost
RES={}; COST={}
for p,g in df.groupby('pair_address'):
    RES[p]=g.reserve_usd.replace(0,np.nan).median()
    COST[p]=coin_cost(RES[p])
print('per-coin round-trip cost frac:',{df[df.pair_address==p].name.iloc[0]:round(COST[p],4) for p in COST})

# ============ TEST 0: predictive correlation (Spearman) of shift(1) features vs fwd return ============
feats=['ofi_usd','ofi_cnt','net_usd','whale_net_usd','n_trades']
rows=[]
for H in [1,3,6,12,24]:
    for f in feats:
        rs=[]; ns=[]
        for p,g in df.groupby('pair_address'):
            g=g.sort_values('ts')
            sig=g[f].shift(1)              # causal: bar-(i-1) flow predicts bar i fwd return
            fr=fwd_ret(g.close.values,H)
            sig=sig.reset_index(drop=True); fr=pd.Series(np.asarray(fr))
            m=sig.notna()&fr.notna()
            if m.sum()<20: continue
            rs.append(sig[m].corr(fr[m],method='spearman'))
            ns.append(int(m.sum()))
        if rs: rows.append({'H':H,'feature':f,'mean_rho':np.nanmean(rs),'median_rho':np.nanmedian(rs),'n_coins':len(rs),'min_n':min(ns)})
corr=pd.DataFrame(rows)
corr.to_csv('./_gaps/bsc_orderflow/test0_corr.csv',index=False)
print('\n=== TEST 0 predictive Spearman corr (shift1 flow vs fwd close ret) ===')
print(corr.round(4).to_string(index=False))

# ============ TEST A: TIMING with intrabar TP/SL, per-fill cost, bar-shuffle null ============
def sim_timing(g, entries, tp, sl, max_hold, cost):
    """Long-only. entries: bool array (already causal/shifted). Enter at next open.
    Intrabar TP/SL using high/low; SL checked first (conservative). Net of round-trip cost."""
    o=g.open.values; h=g.high.values; l=g.low.values; c=g.close.values; n=len(c)
    i=0; rets=[]
    while i<n-1:
        if not entries[i]: i+=1; continue
        ep=o[i+1] if i+1<n else c[i]
        if not np.isfinite(ep) or ep<=0: i+=1; continue
        exit_ret=None; j=i+1
        end=min(i+1+max_hold,n)
        while j<end:
            tph=ep*(1+tp); slp=ep*(1-sl)
            if l[j]<=slp:   exit_ret=-sl; break          # SL first (conservative)
            if h[j]>=tph:   exit_ret=tp;  break
            j+=1
        if exit_ret is None:
            jx=min(end-1,n-1); exit_ret=c[jx]/ep-1.0; j=jx
        rets.append(exit_ret-cost)                        # per-fill round-trip cost
        i=j+1                                             # no overlap
    return np.array(rets)

def pf(rets):
    if len(rets)==0: return np.nan,0
    g=rets[rets>0].sum(); b=-rets[rets<0].sum()
    return (g/b if b>0 else (np.inf if g>0 else np.nan)), len(rets)

# signal: net-buy flip. Enter when shift(1) ofi_usd>thr (positive net buy pressure).
GRID=[('ofi_usd',thr) for thr in [0.0,0.1,0.2]]+[('ofi_cnt',thr) for thr in [0.0,0.1]]+[('whale_net_usd',0.0)]
# brackets scaled to the realised hourly hi-lo range (~1.4%); wide brackets never resolve here
BR=[(0.02,0.02,6),(0.03,0.02,12),(0.03,0.03,24),(0.04,0.03,24),(0.05,0.04,24)]   # tp,sl,max_hold (hourly)
def run_timing(shuffle_null):
    out=[]
    for p,g in df.groupby('pair_address'):
        g=g.sort_values('ts').reset_index(drop=True); cost=COST[p]
        for f,thr in GRID:
            sigraw=(g[f]>thr).astype(float).values
            for (tp,sl,mh) in BR:
                if shuffle_null:
                    perm=rng.permutation(len(sigraw)); sig=sigraw[perm]
                else:
                    sig=sigraw.copy()
                ent=np.zeros(len(sig),bool); ent[1:]=sig[:-1]>0   # shift(1): bar i-1 fires -> act at i
                r=sim_timing(g,ent,tp,sl,mh,cost)
                p_,n=pf(r)
                out.append({'coin':g.name.iloc[0],'feature':f,'thr':thr,'tp':tp,'sl':sl,'mh':mh,
                            'pf':p_,'n':n,'net_mean':np.nanmean(r) if n else np.nan,'tot_net':np.nansum(r) if n else 0.0})
    return pd.DataFrame(out)

real=run_timing(False); real['kind']='real'
# null: 50 reshuffles, pooled distribution
nulls=[]
for s in range(50):
    nd=run_timing(True); nd['seed']=s; nulls.append(nd)
null=pd.concat(nulls,ignore_index=True); null['kind']='null'
real.to_csv('./_gaps/bsc_orderflow/testA_timing_real.csv',index=False)
null.to_csv('./_gaps/bsc_orderflow/testA_timing_null.csv',index=False)

def summ(d):
    dd=d[d.n>=5]
    return dict(strategies=len(dd),median_pf=np.nanmedian(dd.pf.replace(np.inf,np.nan)),
                mean_net=np.nanmean(dd.net_mean),frac_pf_gt1=np.nanmean(dd.pf>1),
                frac_net_pos=np.nanmean(dd.net_mean>0))
print('\n=== TEST A TIMING (per (coin,feature,bracket); n>=5 trades) ===')
print('REAL ',summ(real))
print('NULL ',summ(null))
# headline: median net-mean real vs null and a permutation p-value on pooled median PF
rstat=np.nanmedian(real[real.n>=5].pf.replace(np.inf,np.nan))
nstat_by_seed=[np.nanmedian(null[(null.seed==s)&(null.n>=5)].pf.replace(np.inf,np.nan)) for s in range(50)]
pval=(np.sum(np.array(nstat_by_seed)>=rstat)+1)/(50+1)
print('REAL median PF=%.3f  NULL median PF mean=%.3f (sd %.3f)  perm p(real>=null)=%.3f'%(
      rstat,np.nanmean(nstat_by_seed),np.nanstd(nstat_by_seed),pval))
# robust headline on mean net return (PF can be 0 when no winners)
rnet=np.nanmean(real[real.n>=5].net_mean)
nnet_by_seed=[np.nanmean(null[(null.seed==s)&(null.n>=5)].net_mean) for s in range(50)]
pnet=(np.sum(np.array(nnet_by_seed)>=rnet)+1)/(50+1)
print('REAL mean-net/trade=%.5f  NULL mean-net mean=%.5f (sd %.5f)  perm p(real>=null)=%.3f'%(
      rnet,np.nanmean(nnet_by_seed),np.nanstd(nnet_by_seed),pnet))

# ============ TEST B: SELECTION (cross-sectional), across-coin permutation null. UNDERPOWERED. ============
# Each hour: signal = trailing-K-hour net-buy pressure (shift1). Long the single top coin; benchmark = equal-weight all 5.
piv=df.pivot_table(index='ts',columns='pair_address',values='ofi_usd')
clo=df.pivot_table(index='ts',columns='pair_address',values='close')
ret1=clo.pct_change().shift(-1)                  # next-bar fwd return per coin
costs=np.array([COST[c] for c in piv.columns])
def selection(signal):
    sig=signal.shift(1)                          # causal
    pick=sig.idxmax(axis=1)                       # top coin by net-buy
    # long-top return minus per-trade cost when we switch coins
    rows=[]
    prev=None
    for t in sig.index:
        cpk=pick.loc[t]
        if pd.isna(cpk) or t not in ret1.index: continue
        r=ret1.loc[t,cpk]
        if pd.isna(r): continue
        ci=list(piv.columns).index(cpk)
        c_=costs[ci] if cpk!=prev else 0.0       # cost only on switch (approx)
        rows.append(r-c_); prev=cpk
    return np.array(rows)
trail=piv.rolling(6,min_periods=3).mean()
sel_r=selection(trail)
ew=ret1.mean(axis=1).dropna().values
def cum(r): return float(np.prod(1+r)-1) if len(r) else np.nan
print('\n=== TEST B SELECTION (long top-flow coin vs equal-weight) — UNDERPOWERED N=5 ===')
print('long-top cum net=%.4f  mean=%.5f  n=%d'%(cum(sel_r),np.mean(sel_r) if len(sel_r) else np.nan,len(sel_r)))
print('equal-weight cum (no cost)=%.4f'%cum(ew))
# across-coin permutation null: shuffle column labels of the signal
nullcum=[]
for s in range(200):
    perm=rng.permutation(piv.columns.values)
    tp=trail.copy(); tp.columns=perm; tp=tp[piv.columns]
    nullcum.append(cum(selection(tp)))
nullcum=np.array(nullcum)
pB=(np.sum(nullcum>=cum(sel_r))+1)/(200+1)
print('NULL (label-permuted) cum median=%.4f  p(real>=null)=%.3f'%(np.nanmedian(nullcum),pB))
pd.DataFrame({'real_cum':[cum(sel_r)],'real_mean':[np.mean(sel_r) if len(sel_r) else np.nan],
              'null_cum_median':[np.nanmedian(nullcum)],'null_cum_p5':[np.nanpercentile(nullcum,5)],
              'null_cum_p95':[np.nanpercentile(nullcum,95)],'perm_p':[pB],'n_periods':[len(sel_r)]}
             ).to_csv('./_gaps/bsc_orderflow/testB_selection.csv',index=False)

import json
json.dump({'rstat_medianPF':float(rstat),'null_medianPF_mean':float(np.nanmean(nstat_by_seed)),
           'timing_perm_p_PF':float(pval),'real_mean_net':float(rnet),'null_mean_net':float(np.nanmean(nnet_by_seed)),
           'timing_perm_p_net':float(pnet),'sel_real_cum':float(cum(sel_r)),'sel_null_cum_med':float(np.nanmedian(nullcum)),
           'sel_perm_p':float(pB)},open('./_gaps/bsc_orderflow/headline.json','w'),indent=2)
print('\nDONE')
