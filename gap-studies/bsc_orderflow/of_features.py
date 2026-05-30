"""Build per-hour order-flow features for the 5 BSC pairs with dense per-swap tapes,
and join to hourly OHLCV. Output: of_hourly.parquet (one row per pair-hour).

Causal-safe construction: features describe activity WITHIN bar h. Any predictive use
must .shift(1) so bar-h signal trades at bar h+1 (done in the test scripts, not here).
"""
import glob, pandas as pd, numpy as np

PAIRS = ['0x2a4b99a9c4544d35e8d266111c50b67fea01d53d',  # IBS
         '0x51ff488a8d0303d6be5ff51a820684484ff7b755',  # ODY
         '0x63844bd4bfad910b1643713302a1cc1ed20d50c3',  # Pro
         '0x9996874dbd891c3ecd601eab79d51f92e92d50ee',  # LAX
         '0xb720ea2a201c03578cca05191e8b4ab859704cfe']  # wkeyDAO2
NAME = {PAIRS[0]:'IBS',PAIRS[1]:'ODY',PAIRS[2]:'Pro',PAIRS[3]:'LAX',PAIRS[4]:'wkeyDAO2'}

cols=['pair_address','block_time','side','amount_usd','reserve_usd','maker']
fl=pd.concat([pd.read_parquet(f,columns=cols) for f in glob.glob('./data/flow_trades/chain=bsc/dt=*/*.parquet')],ignore_index=True)
fl=fl[fl.pair_address.isin(PAIRS)].copy()
fl=fl[fl.amount_usd>0]
fl['hour']=fl.block_time.dt.floor('h')
fl['is_buy']=(fl.side=='buy')
fl['buy_usd']=np.where(fl.is_buy,fl.amount_usd,0.0)
fl['sell_usd']=np.where(~fl.is_buy,fl.amount_usd,0.0)

# whale flag: trade in top 5% of that pair's trade-size distribution (computed over full window;
# this is a static threshold, acceptable as it's not a forward-looking *signal* feature but a tag).
fl['whale_thr']=fl.groupby('pair_address').amount_usd.transform(lambda s: s.quantile(0.95))
fl['whale_buy_usd']=np.where(fl.is_buy & (fl.amount_usd>=fl.whale_thr),fl.amount_usd,0.0)
fl['whale_sell_usd']=np.where(~fl.is_buy & (fl.amount_usd>=fl.whale_thr),fl.amount_usd,0.0)

def agg(g):
    n=len(g); nb=g.is_buy.sum(); ns=n-nb
    bu=g.buy_usd.sum(); su=g.sell_usd.sum()
    return pd.Series({
        'n_trades':n,'n_buy':nb,'n_sell':ns,
        'buy_usd':bu,'sell_usd':su,
        'net_usd':bu-su,
        'ofi_usd':(bu-su)/(bu+su) if (bu+su)>0 else 0.0,        # USD order-flow imbalance [-1,1]
        'ofi_cnt':(nb-ns)/n if n>0 else 0.0,                    # count imbalance [-1,1]
        'whale_net_usd':g.whale_buy_usd.sum()-g.whale_sell_usd.sum(),
        'n_makers':g.maker.nunique(),
        'tot_usd':bu+su,
        'reserve_usd':g.reserve_usd.median(),
    })
h=fl.groupby(['pair_address','hour']).apply(agg,include_groups=False).reset_index()
h['ts']=(h.hour.astype('int64')//10**6)   # datetime64[us,UTC] -> unix seconds

# join hourly OHLCV
o=pd.read_parquet('./data/ohlcv_gt/mega_bsc_hour.parquet')
o=o[o.pair_address.isin(PAIRS)][['pair_address','ts','open','high','low','close','volume']].copy()
# OHLCV ts is bar start (unix s). flow hour floor is also bar start. Align directly.
m=o.merge(h,on=['pair_address','ts'],how='left')
# hours with no swaps in tape -> zero flow (legit: no trades that hour)
for c in ['n_trades','n_buy','n_sell','buy_usd','sell_usd','net_usd','whale_net_usd','n_makers','tot_usd']:
    m[c]=m[c].fillna(0.0)
m['ofi_usd']=m['ofi_usd'].fillna(0.0); m['ofi_cnt']=m['ofi_cnt'].fillna(0.0)
m['reserve_usd']=m.groupby('pair_address')['reserve_usd'].ffill().bfill()
m['name']=m.pair_address.map(NAME)
m=m.sort_values(['pair_address','ts']).reset_index(drop=True)

# restrict to the flow-covered window per pair (where we actually have tape)
cov=h.groupby('pair_address').ts.agg(['min','max']).rename(columns={'min':'tmn','max':'tmx'})
m=m.merge(cov,on='pair_address')
m=m[(m.ts>=m.tmn)&(m.ts<=m.tmx)].drop(columns=['tmn','tmx']).reset_index(drop=True)

m.to_parquet('./_gaps/bsc_orderflow/of_hourly.parquet')
print('rows',len(m))
print(m.groupby('name').agg(bars=('ts','size'),covered=('n_trades',lambda s:(s>0).mean()),
      med_ofi=('ofi_usd','median'),med_trades=('n_trades','median')))
print('\nFlow-window hourly bars per pair (this is the usable sample for timing).')
