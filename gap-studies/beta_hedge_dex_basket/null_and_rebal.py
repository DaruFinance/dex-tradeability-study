"""Multi-seed bar-shuffle NULL distribution for the hedged Sharpe + rebalance
frequency sensitivity. Re-implements the basket+hedge inline (small) to vary seed
and rebal frequency without the full pipeline overhead."""
import glob, json, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,'.')
from chainscope.costs import round_trip_cost_frac
OUT=Path('./_gaps/beta_hedge_dex_basket')
DATA=Path('./data')
ANN=365.0
CHAIN_GAS={'bsc':.2,'base':.02,'eth':3.,'arbitrum':.05,'avax':.05,'polygon_pos':.01,'optimism':.05,'sui-network':.01,'solana':.02,'tron':.01}
CHAIN_NATIVE={'bsc':640,'base':3500,'eth':3500,'arbitrum':3500,'avax':35,'polygon_pos':.5,'optimism':3500,'sui-network':3.5,'solana':180,'tron':.3}
PERP_TAKER=0.0005; FUNDING_DAILY=0.0003
IS_LEN,OOS_LEN=60,21
def coin_cost(chain,reserve):
    if not reserve: return .05
    size=max(50.,0.0025*reserve)
    return round_trip_cost_frac(size,reserve,dex='uniswap',chain=chain,gas_usd=CHAIN_GAS.get(chain,.05),native_usd=CHAIN_NATIVE.get(chain,100))
EXCL=['USDC','USDT','DAI','USD','WBTC','CBBTC','TBTC','BTCB','WETH','WEETH','WSTETH','STETH','RETH','CBETH','EZETH','FRAX','BUSD','TUSD','USDE','SUSDE','WBNB','WMATIC','WAVAX','WFTM','WSOL','PYUSD','LUSD','GUSD','RLUSD','USDD','FDUSD','CRVUSD','GHO','USR','SCRVUSD','WBETH','SFRXETH','OETH','MSOL','JITOSOL']
def isdex(name):
    if not isinstance(name,str): return True
    base=name.upper().split('/')[0].strip().split()[0] if name.split('/')[0].strip() else ''
    return not any(k==base or base.startswith(k) for k in EXCL)
def sharpe(r):
    r=np.asarray(r); r=r[~np.isnan(r)]
    return r.mean()/r.std()*np.sqrt(ANN) if r.std()>0 else np.nan
def maxdd(r):
    eq=np.cumprod(1+np.nan_to_num(r)); return float((eq/np.maximum.accumulate(eq)-1).min())

# build once
files=glob.glob(str(DATA/'ohlcv_gt/mega_*_day.parquet'))
df=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
rows=[json.loads(l) for l in open(DATA/'_mega_universe.jsonl')]
m=pd.DataFrame(rows)[['chain','pair','reserve_usd','name']].rename(columns={'pair':'pair_address','reserve_usd':'res'})
df=df.merge(m,on=['chain','pair_address'],how='left')
df['coinid']=df.chain+'|'+df.pair_address
g=df.groupby('coinid').agg(n=('ts','size'),res=('res','first'),name=('name','first'),chain=('chain','first')).reset_index()
g['ok']=(g.res>=50000)&(g.n>=120)&g.name.map(isdex)
keep=set(g[g.ok].coinid)
sub=df[df.coinid.isin(keep)]
px=sub.pivot_table(index='ts',columns='coinid',values='close',aggfunc='last').sort_index()
ret=px.pct_change().clip(-0.9,1.0)
btc=df[df.pair_address=='0x99ac8ca7087fa4a2a1fb6357269965a2014abc35'].sort_values('ts').set_index('ts')['close'].pct_change().clip(-.5,.5)
eth=df[df.pair_address=='0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'].sort_values('ts').set_index('ts')['close'].pct_change().clip(-.5,.5)
ridx=btc.dropna().index.intersection(eth.dropna().index); lo,hi=ridx.min(),ridx.max()
ret=ret.loc[(ret.index>=lo)&(ret.index<=hi)]
cov=ret.notna().sum(); ret=ret[cov[cov>=120].index]
res=g.set_index('coinid').res.to_dict()
rtc={c:coin_cost(c.split('|')[0],res.get(c)) for c in ret.columns}

def basket_net(R,rebal):
    nT=len(R); gross=np.zeros(nT); tc=np.zeros(nT)
    vals=R.values; cols=R.columns
    rtcv=np.array([rtc[c] for c in cols])
    for i in range(nT):
        row=vals[i]; mk=~np.isnan(row)
        if mk.sum()==0: continue
        w=mk/mk.sum()
        gross[i]=np.nansum(w*np.where(mk,row,0))
        if i%rebal==0: tc[i]=np.sum(w*rtcv)
    return pd.Series(gross,index=R.index), pd.Series(gross-tc,index=R.index)

def hedge(net,refr):
    common=net.index.intersection(refr.index); bg=net.reindex(common); rr=refr.reindex(common)
    n=len(common); h=pd.Series(index=common,dtype=float); i=IS_LEN
    while i<n:
        ib=bg.iloc[i-IS_LEN:i].values; ir=rr.iloc[i-IS_LEN:i].values
        mk=~(np.isnan(ib)|np.isnan(ir))
        beta=np.cov(ib[mk],ir[mk])[0,1]/np.var(ir[mk]) if (mk.sum()>=20 and np.std(ir[mk])>0) else 0.0
        beta=float(np.clip(beta,-3,3))
        for j in range(i,min(i+OOS_LEN,n)):
            mret=rr.iloc[j]; mret=0.0 if np.isnan(mret) else mret
            pf=abs(beta)*FUNDING_DAILY+(abs(beta)*PERP_TAKER if j%7==0 else 0.0)
            h.iloc[j]=bg.iloc[j]-beta*mret-pf
        i+=OOS_LEN
    return h.dropna()

def shuffle_ret(R,seed):
    r=np.random.default_rng(seed); V=R.values.copy()
    for c in range(V.shape[1]):
        col=V[:,c]; idx=np.where(~np.isnan(col))[0]
        if len(idx)>1: col[idx]=col[r.permutation(idx)]; V[:,c]=col
    return pd.DataFrame(V,index=R.index,columns=R.columns)

# 1) NULL distribution: 60 seeds, hedged Sharpe (BTC & ETH)
print('=== Multi-seed NULL hedged Sharpe (60 seeds) ===')
nullrows=[]
for seed in range(60):
    Rs=shuffle_ret(ret,1000+seed)
    _,nets=basket_net(Rs,7)
    for ref,rr in [('BTC',btc),('ETH',eth)]:
        h=hedge(nets,rr); nullrows.append(dict(seed=seed,ref=ref,sharpe=sharpe(h.values)))
ndf=pd.DataFrame(nullrows); ndf.to_csv(OUT/'null_sharpe_dist.csv',index=False)
# real
_,netr=basket_net(ret,7)
real={ref:sharpe(hedge(netr,rr).values) for ref,rr in [('BTC',btc),('ETH',eth)]}
for ref in ['BTC','ETH']:
    nd=ndf[ndf.ref==ref].sharpe.values
    pct=float(np.mean(nd>=real[ref]))
    print(f'{ref}: real Sharpe={real[ref]:.3f} | null mean={nd.mean():.3f} std={nd.std():.3f} '
          f'p95={np.percentile(nd,95):.3f} | P(null>=real)={pct:.3f}')

# 2) rebalance sensitivity (real, unhedged + ETH-hedge)
print('\n=== Rebalance-frequency sensitivity (real) ===')
rrows=[]
for rebal in [1,3,7,14,30,9999]:
    _,nets=basket_net(ret,rebal if rebal<9999 else len(ret)+1)
    oos=nets.iloc[IS_LEN:]
    he=hedge(nets,eth)
    rrows.append(dict(rebal=('hold' if rebal==9999 else rebal),
                      unhedged_sharpe=sharpe(oos.values),unhedged_ret=(1+oos).prod()-1,
                      ethhedge_sharpe=sharpe(he.values),ethhedge_ret=(1+he).prod()-1,
                      ethhedge_maxdd=maxdd(he.values)))
rdf=pd.DataFrame(rrows); rdf.to_csv(OUT/'rebal_sensitivity.csv',index=False)
print(rdf.to_string(index=False))
