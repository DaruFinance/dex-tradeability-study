#!/usr/bin/env python3
"""Robustness: (1) drop scaling-artifact tokens (median dispersion>20% = unit/quote
mismatch, not arb), re-aggregate REAL vs NULL. (2) Hourly cross-check on liquid subset
(hourly bars => more bars, closer to intrabar, where arb is more plausible).
Seeded, reproducible."""
import json, glob
from collections import defaultdict
import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from chainscope.costs import round_trip_cost_frac
OUT='./_gaps/multipool_dispersion'; DATA='./data'
rng=np.random.default_rng(777)
CHAIN_GAS={'bsc':.2,'base':.02,'eth':3.,'arbitrum':.05,'avax':.05,'polygon_pos':.01,'optimism':.05,'sui-network':.01,'solana':.02,'tron':.01}
CHAIN_NATIVE={'bsc':640,'base':3500,'eth':3500,'arbitrum':3500,'avax':35,'polygon_pos':.5,'optimism':3500,'sui-network':3.5,'solana':180,'tron':.3}
def coin_cost(chain,reserve):
    if not reserve: return .05
    size=max(50.,0.0025*reserve)
    return round_trip_cost_frac(size,reserve,dex='uniswap',chain=chain,gas_usd=CHAIN_GAS.get(chain,.05),native_usd=CHAIN_NATIVE.get(chain,100))
def arb_cost(chain,a,b): return 0.5*coin_cost(chain,a)+0.5*coin_cost(chain,b)

# ---- ROBUST re-aggregation on daily, dropping artifacts ----
df=pd.read_csv(f'{OUT}/per_token_dispersion.csv')
clean=df[df.real_med_disp<=0.20].copy()  # legit token-prices stay within 20% across pools
print(f'daily token-groups total={len(df)}  clean(<=20% med disp)={len(clean)}  dropped artifacts={len(df)-len(clean)}')
def agg(d,pfx):
    e=d[d[pfx+'n_events']>0]
    return dict(n=len(d),med_disp=d[pfx+'med_disp'].median(),mean_disp=d[pfx+'med_disp'].mean(),
        med_arb_cost=d[pfx+'arb_cost'].median(),med_frac_gt_cost=d[pfx+'frac_disp_gt_cost'].median(),
        mean_frac_gt_cost=d[pfx+'frac_disp_gt_cost'].mean(),med_disp_ac1=d[pfx+'disp_ac1'].median(),
        med_net_mean=e[pfx+'net_mean'].median() if len(e) else float('nan'),
        frac_tokens_net_pos=float((d[pfx+'net_sum']>0).mean()))
rob=pd.DataFrame([dict(run='REAL_clean',**agg(clean,'real_')),dict(run='NULL_clean',**agg(clean,'null_'))])
rob.to_csv(f'{OUT}/summary_clean_real_vs_null.csv',index=False)
print(rob.to_string()); print()

# ---- HOURLY cross-check on liquid chains (eth,bsc,base,arbitrum) ----
print('loading hourly ohlcv (liquid chains)...')
tok2pools=defaultdict(list)
with open(f'{DATA}/_mega_universe.jsonl') as f:
    for line in f:
        d=json.loads(line); bt=d.get('base_token')
        if bt: tok2pools[bt].append(d)
close_by_pair={}
HCHAINS=['eth','bsc','base','arbitrum']
for ch in HCHAINS:
    fp=f'{DATA}/ohlcv_gt/mega_{ch}_hour.parquet'
    try: hdf=pd.read_parquet(fp,columns=['pair_address','ts','close'])
    except: continue
    for pa,sub in hdf.groupby('pair_address'):
        close_by_pair[pa]=sub.sort_values('ts').drop_duplicates('ts',keep='last').set_index('ts').close
print('hourly pairs',len(close_by_pair))
groups=[]
for tok,pools in tok2pools.items():
    have=[p for p in pools if p['pair'] in close_by_pair]
    if len(have)<2: continue
    for ch in set(p['chain'] for p in have):
        chp=[p for p in have if p['chain']==ch]
        if len(chp)>=2 and len(set(p['dex'] for p in chp))>=2: groups.append((tok,ch,chp)); break
print('hourly same-chain multi-dex groups',len(groups))
def runh(panel,res,chain,shuffle):
    P=panel.values.copy()
    if shuffle:
        for j in range(P.shape[1]): P[:,j]=rng.permutation(P[:,j])
    med=np.median(P,1); disp=(P.max(1)-P.min(1))/med
    sr=sorted([r for r in (res[p] for p in res) if r],reverse=True)
    ac=arb_cost(chain,sr[0],sr[1]) if len(sr)>=2 else arb_cost(chain,sr[0] if sr else None,sr[0] if sr else None)
    sig=disp[:-1]; nxt=disp[1:]; persisted=np.minimum(sig,nxt); fire=sig>ac
    net=(persisted-ac)[fire]
    return dict(n_bars=len(disp),med_disp=float(np.median(disp)),arb_cost=float(ac),
        frac_gt_cost=float((disp>ac).mean()),n_events=int(fire.sum()),
        net_mean=float(net.mean()) if fire.sum() else 0.0,net_sum=float(net.sum()) if fire.sum() else 0.0,
        disp_ac1=float(pd.Series(disp).autocorr(1)) if len(disp)>3 else np.nan)
hrows=[]
for tok,ch,chp in groups:
    ser={p['pair']:close_by_pair[p['pair']] for p in chp if len(close_by_pair[p['pair']])>=24}
    ser={k:v[(v>0)&np.isfinite(v)] for k,v in ser.items()}
    if len(ser)<2: continue
    panel=pd.DataFrame(ser).dropna()
    if len(panel)<24: continue
    res={p['pair']:p.get('reserve_usd') for p in chp}
    r=runh(panel,res,ch,False); rn=runh(panel,res,ch,True)
    rec=dict(token=tok,chain=ch)
    for k,v in r.items(): rec['real_'+k]=v
    for k,v in rn.items(): rec['null_'+k]=v
    hrows.append(rec)
hdf2=pd.DataFrame(hrows)
hdf2.to_csv(f'{OUT}/hourly_per_token.csv',index=False)
hclean=hdf2[hdf2.real_med_disp<=0.20]
print(f'hourly groups analyzed={len(hdf2)} clean={len(hclean)}')
def aggh(d,pfx):
    e=d[d[pfx+'n_events']>0]
    return dict(n=len(d),med_disp=d[pfx+'med_disp'].median(),med_arb_cost=d[pfx+'arb_cost'].median(),
        med_frac_gt_cost=d[pfx+'frac_gt_cost'].median(),med_disp_ac1=d[pfx+'disp_ac1'].median(),
        med_net_mean=e[pfx+'net_mean'].median() if len(e) else float('nan'),
        frac_net_pos=float((d[pfx+'net_sum']>0).mean()))
hsum=pd.DataFrame([dict(run='HOURLY_REAL_clean',**aggh(hclean,'real_')),dict(run='HOURLY_NULL_clean',**aggh(hclean,'null_'))])
hsum.to_csv(f'{OUT}/summary_hourly_real_vs_null.csv',index=False)
print(hsum.to_string())
print('done')
