#!/usr/bin/env python3
"""
GAP: Multi-pool / cross-DEX price dispersion as a DEX arb edge.

Thesis (falsifiable): For DEX-only tokens that live in >=2 pools on the SAME chain
(different DEX), the cross-pool USD price dispersion measured at DAILY bar close
rarely and non-persistently exceeds the realistic round-trip cost of trading BOTH
legs. A causal "rebalance toward consensus" arb backtest is NOT net-positive after
per-fill cost and does not beat its bar-shuffle null.

Method:
- base_token in _mega_universe.jsonl is a chain-prefixed contract address -> exact
  same-token match. We use SAME-CHAIN different-DEX groups (>=2 pools w/ daily OHLCV).
  Cross-chain by base_token is impossible (chain-prefixed); cross-chain by SYMBOL is
  dominated by CEX majors + collision-prone -> excluded from the edge claim (reported
  as a separate feasibility note).
- OHLCV close is USD-denominated (verified: same token ~same USD price across pools
  regardless of USDC/USDT/WETH quote), so cross-pool prices are directly comparable.
- Per-bar dispersion = (max_close - min_close)/median_close across pools at each ts.
- Cost: project model, per pool reserve, applied on BOTH legs (buy cheap pool, sell
  dear pool). RT arb cost = cost(cheap pool) + cost(dear pool).
- Causal backtest: at bar i, using ONLY data <= i-1, if dispersion(i-1) > arb_cost,
  signal an arb; realize PnL = dispersion at the SIGNAL bar (i-1, observable) minus
  cost, but EXECUTED at bar i (we can only act next bar). We measure realized
  convergence: did the dispersion at i-1 actually close by bar i? Net arb PnL per
  event = min(disp_{i-1}, disp_{i-1}) ... see code: we conservatively realize the
  spread that PERSISTS into the executable bar.
- NULL: independently bar-shuffle each pool's close series within token (destroys any
  genuine cross-pool co-movement / mean-reversion timing) and re-run identical pipeline.

CAVEAT (stated up front): OHLCV is bar-aggregated. A daily (or hourly) close does NOT
capture intrabar simultaneity. True atomic arb needs same-block tick/swap data on both
pools. This measures bar-close dispersion persistence, an UPPER-BOUND-ish proxy that
ignores: (a) within-bar the two closes are not simultaneous, (b) price impact of the
arb trade itself beyond the modeled cost, (c) MEV/searcher competition that already
removes atomic arb. So a POSITIVE result here would be necessary-not-sufficient; a
NEGATIVE result is strong (the easy upper bound already fails).
"""
import json, glob, os
from collections import defaultdict
import numpy as np, pandas as pd
import sys
sys.path.insert(0,'.')
from chainscope.costs import round_trip_cost_frac

OUT='./_gaps/multipool_dispersion'
DATA='./data'
SEED=12345
rng=np.random.default_rng(SEED)

CHAIN_GAS={'bsc':.2,'base':.02,'eth':3.,'arbitrum':.05,'avax':.05,'polygon_pos':.01,'optimism':.05,'sui-network':.01,'solana':.02,'tron':.01}
CHAIN_NATIVE={'bsc':640,'base':3500,'eth':3500,'arbitrum':3500,'avax':35,'polygon_pos':.5,'optimism':3500,'sui-network':3.5,'solana':180,'tron':.3}
def coin_cost(chain,reserve):
    if not reserve: return .05
    size=max(50.,0.0025*reserve)
    return round_trip_cost_frac(size,reserve,dex='uniswap',chain=chain,
        gas_usd=CHAIN_GAS.get(chain,.05),native_usd=CHAIN_NATIVE.get(chain,100))

# one-leg cost ~= half of round trip (round_trip_cost_frac already is a round trip = 2 fills).
# For an arb we trade ONE direction on each of two pools = 2 fills total = one round_trip_cost_frac
# split across two pools. We model RT-arb cost = 0.5*coin_cost(cheap)+0.5*coin_cost(dear).
def arb_cost(chain,res_a,res_b):
    return 0.5*coin_cost(chain,res_a)+0.5*coin_cost(chain,res_b)

# ---- build token -> pools (same chain, >=2 pools) ----
tok2pools=defaultdict(list)
with open(f'{DATA}/_mega_universe.jsonl') as f:
    for line in f:
        d=json.loads(line); bt=d.get('base_token')
        if bt: tok2pools[bt].append(d)

# load all daily ohlcv close keyed by pair
print('loading daily ohlcv...')
close_by_pair={}
for fpath in glob.glob(f'{DATA}/ohlcv_gt/mega_*_day.parquet'):
    df=pd.read_parquet(fpath, columns=['pair_address','ts','close'])
    for pa,sub in df.groupby('pair_address'):
        s=sub.sort_values('ts').drop_duplicates('ts',keep='last').set_index('ts').close
        close_by_pair[pa]=s
print('pairs with daily close', len(close_by_pair))

# multi-pool tokens with >=2 pools having data
groups=[]
for tok,pools in tok2pools.items():
    have=[p for p in pools if p['pair'] in close_by_pair]
    if len(have)>=2:
        chains=set(p['chain'] for p in have)
        # same-chain different-dex: require at least 2 distinct dex on one chain
        for ch in chains:
            chp=[p for p in have if p['chain']==ch]
            dexes=set(p['dex'] for p in chp)
            if len(chp)>=2 and len(dexes)>=2:
                groups.append((tok,ch,chp)); break
print('same-chain multi-DEX token-groups with data', len(groups))

def align_panel(chp):
    """align closes on common ts; return df (ts x pool) and reserves list"""
    series={}
    for p in chp:
        s=close_by_pair[p['pair']]
        s=s[(s>0)&np.isfinite(s)]
        if len(s)>=10: series[p['pair']]=s
    if len(series)<2: return None,None
    panel=pd.DataFrame(series)
    panel=panel.dropna()  # require simultaneous bars across all pools
    if len(panel)<10: return None,None
    res={p['pair']:p.get('reserve_usd') for p in chp}
    return panel, res

# ---- per-token dispersion stats + arb backtest (REAL and NULL) ----
def run(panel, res, chain, shuffle=False):
    P=panel.values.copy()  # ts x pools
    if shuffle:
        for j in range(P.shape[1]):
            P[:,j]=rng.permutation(P[:,j])
    n,k=P.shape
    med=np.median(P,axis=1)
    disp=(P.max(1)-P.min(1))/med  # per-bar cross-pool dispersion
    # arb cost: use the two pools that are cheapest/dearest each bar -> use min reserves conservatively?
    # We use the median of pairwise arb costs across the involved pools as representative RT-arb cost.
    pairs=list(res.keys())
    reserves=[res[p] for p in pairs]
    # representative arb cost = arb_cost using the two LARGEST reserves (best case, lowest cost)
    sr=sorted([r for r in reserves if r], reverse=True)
    if len(sr)>=2: ac=arb_cost(chain,sr[0],sr[1])
    else: ac=arb_cost(chain,sr[0] if sr else None, sr[0] if sr else None)
    # causal backtest: signal from disp[i-1]; execute at i; realized = dispersion that
    # PERSISTS = min(disp[i-1], disp[i]) (you capture only the part still open at exec).
    sig=disp[:-1]  # disp at i-1
    nxt=disp[1:]   # disp at i (executable bar)
    persisted=np.minimum(sig,nxt)
    fire=sig>ac
    n_events=int(fire.sum())
    # net pnl per event (executed): realized spread captured minus cost.
    # conservative: you capture the persisted spread (what's still open), pay full arb cost.
    net=np.where(fire, persisted-ac, np.nan)
    net_events=net[fire]
    return dict(
        n_bars=n, n_pools=k, med_disp=float(np.median(disp)), mean_disp=float(np.mean(disp)),
        p90_disp=float(np.percentile(disp,90)), max_disp=float(np.max(disp)),
        arb_cost=float(ac), frac_disp_gt_cost=float((disp>ac).mean()),
        n_events=n_events,
        net_mean=float(np.nanmean(net_events)) if n_events else 0.0,
        net_sum=float(np.nansum(net_events)) if n_events else 0.0,
        net_pos_frac=float((net_events>0).mean()) if n_events else 0.0,
        # persistence: autocorr of disp at lag1
        disp_ac1=float(pd.Series(disp).autocorr(lag=1)) if n>3 else np.nan,
    )

rows=[]
for tok,ch,chp in groups:
    panel,res=align_panel(chp)
    if panel is None: continue
    r=run(panel,res,ch,shuffle=False)
    rn=run(panel,res,ch,shuffle=True)
    rec=dict(token=tok,chain=ch,n_pools_meta=len(chp))
    for kk,vv in r.items(): rec['real_'+kk]=vv
    for kk,vv in rn.items(): rec['null_'+kk]=vv
    rows.append(rec)

res_df=pd.DataFrame(rows)
res_df.to_csv(f'{OUT}/per_token_dispersion.csv',index=False)
print('analyzed token-groups', len(res_df))

# ---- aggregate summary ----
def agg(df):
    return dict(
        n_tokens=len(df),
        med_disp=df.real_med_disp.median(), mean_disp=df.real_med_disp.mean(),
        med_arb_cost=df.real_arb_cost.median(),
        med_frac_gt_cost=df.real_frac_disp_gt_cost.median(),
        mean_frac_gt_cost=df.real_frac_disp_gt_cost.mean(),
        med_disp_ac1=df.real_disp_ac1.median(),
        tokens_w_events=int((df.real_n_events>0).sum()),
        med_net_mean=df.loc[df.real_n_events>0,'real_net_mean'].median() if (df.real_n_events>0).any() else float('nan'),
        frac_tokens_net_pos=float((df.real_net_sum>0).mean()),
    )
A=agg(res_df)
# null aggregate (mirror): build clean frame from null_ cols renamed to real_
ncols={c:c.replace('null_','real_') for c in res_df.columns if c.startswith('null_')}
nulldf=res_df[list(ncols)].rename(columns=ncols).copy()
nulldf['token']=res_df['token']; nulldf['chain']=res_df['chain']
An=agg(nulldf)

summ=pd.DataFrame([dict(run='REAL',**A),dict(run='NULL_barshuffle',**An)])
summ.to_csv(f'{OUT}/summary_real_vs_null.csv',index=False)
print(summ.to_string())

# save dispersion distribution
res_df[['token','chain','real_med_disp','real_arb_cost','real_frac_disp_gt_cost','real_n_events','real_net_mean','real_net_sum','null_net_sum','real_disp_ac1']].to_csv(f'{OUT}/dispersion_distribution.csv',index=False)
print('done')
