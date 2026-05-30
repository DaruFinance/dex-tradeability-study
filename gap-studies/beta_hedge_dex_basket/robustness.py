"""Robustness for the beta-hedge result:
 1. Stationary block bootstrap of the OOS hedged daily-return series -> Sharpe CI.
 2. Window-by-window Sharpe (is the +1.47 from one lucky window?).
 3. Multi-seed null distribution of hedged Sharpe (bar-shuffle).
 4. Rebalance-frequency sensitivity (turnover is the bleed driver).
"""
import pickle, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,'.')
OUT=Path('./_gaps/beta_hedge_dex_basket')
ANN=365.0
rng=np.random.default_rng(7)

d=pickle.load(open(OUT/'_series.pkl','rb'))
st=d['store']

def sharpe(r):
    r=np.asarray(r); r=r[~np.isnan(r)]
    return r.mean()/r.std()*np.sqrt(ANN) if r.std()>0 else np.nan

def block_boot(r, B=2000, L=5):
    r=np.asarray(r); r=r[~np.isnan(r)]; n=len(r)
    out=np.empty(B)
    for b in range(B):
        idx=[]
        while len(idx)<n:
            s=rng.integers(0,n); idx+=list(range(s,min(s+L,n)))
        idx=np.array(idx[:n])
        out[b]=sharpe(r[idx])
    return out

rows=[]
for key in [('equal','real','hedge_ETH'),('equal','real','hedge_BTC'),
            ('equal','real','unhedged'),('liq','real','hedge_ETH'),
            ('liq','real','hedge_BTC')]:
    s=st[key]
    bs=block_boot(s.values)
    rows.append(dict(variant='/'.join(key), n=len(s.dropna()), sharpe=sharpe(s.values),
                     boot_mean=np.nanmean(bs), ci_lo=np.nanpercentile(bs,2.5),
                     ci_hi=np.nanpercentile(bs,97.5), p_sharpe_le0=float(np.mean(bs<=0))))
boot=pd.DataFrame(rows)
boot.to_csv(OUT/'bootstrap_sharpe.csv',index=False)
print('=== Block-bootstrap Sharpe (95% CI) ===')
print(boot.to_string(index=False))

# window-by-window: split OOS into ~21-day chunks, sharpe each
print('\n=== Window-by-window Sharpe (21d chunks) ===')
wrows=[]
for key in [('equal','real','hedge_ETH'),('equal','real','hedge_BTC'),('equal','real','unhedged')]:
    s=st[key].dropna()
    for w in range(0,len(s),21):
        chunk=s.iloc[w:w+21]
        if len(chunk)>=10:
            wrows.append(dict(variant='/'.join(key),win=w//21,n=len(chunk),sharpe=sharpe(chunk.values),
                              tot_ret=(1+chunk).prod()-1))
wdf=pd.DataFrame(wrows)
wdf.to_csv(OUT/'window_sharpe.csv',index=False)
print(wdf.to_string(index=False))

print('\n=== Window stability summary ===')
print(wdf.groupby('variant').sharpe.agg(['mean','std','min','max',lambda x:(x>0).mean()]).rename(columns={'<lambda_0>':'frac_pos'}).to_string())
