"""
GAP: Market-beta hedge for a long DEX-only basket.

Thesis (falsifiable): Removing BTC/ETH market beta from a long-only DEX-altcoin
basket (via a CEX-perp short) does NOT produce positive risk-adjusted OOS return,
net of DEX per-fill cost + perp costs. The structural bleed is idiosyncratic /
cost-driven, not market-beta driven.

Method:
  - Build a daily return panel of tradeable DEX coins (reserve>=$50k, enough bars).
  - Exclude wrapped-major / stable-stable reference pairs (those ARE the market /
    not DEX-only coins).
  - BTC & ETH USD reference series from on-disk highest-liquidity WBTC/USDC &
    WETH/USDC pools in mega_eth_day.parquet (close is USD).
  - Equal-weight and liquidity(reserve)-weighted baskets.
  - Walk-forward: estimate beta on IS window (OLS of basket daily ret on BTC ret,
    and on ETH ret), apply hedge on the *next* OOS window. Roll.
  - Costs: DEX per-fill round-trip on each rebalance leg (project cost model);
    perp short: taker fee 0.05% per side + funding 0.01%/8h/leg.
  - Compare:
      (a) unhedged long basket (net DEX cost)
      (b) BTC-beta-hedged basket   (net DEX + perp cost)
      (b2) ETH-beta-hedged basket
      (c) bar-shuffle null (shuffle each coin's returns in time, same pipeline)
      (d) no-skill: buy-and-hold equal-weight (1 rebalance)
  - Metrics OOS only: annualized return, vol, Sharpe, MaxDD, Calmar.

Seeded, reproducible. Daily timeframe (only ~181 bars; hourly not needed for a
beta-hedge portfolio test and would add survivorship noise).
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

OUT = Path("./_gaps/beta_hedge_dex_basket")
DATA = Path("./data")
SEED = 42
rng = np.random.default_rng(SEED)

CHAIN_GAS = {'bsc':.2,'base':.02,'eth':3.,'arbitrum':.05,'avax':.05,'polygon_pos':.01,
             'optimism':.05,'sui-network':.01,'solana':.02,'tron':.01}
CHAIN_NATIVE = {'bsc':640,'base':3500,'eth':3500,'arbitrum':3500,'avax':35,
                'polygon_pos':.5,'optimism':3500,'sui-network':3.5,'solana':180,'tron':.3}
SIZE_FRAC = 0.0025

def coin_cost(chain, reserve):
    if not reserve:
        return .05
    size = max(50., SIZE_FRAC*(reserve or 0))
    return round_trip_cost_frac(size, reserve, dex='uniswap', chain=chain,
                                gas_usd=CHAIN_GAS.get(chain,.05),
                                native_usd=CHAIN_NATIVE.get(chain,100))

# perp short cost model
PERP_TAKER = 0.0005          # 0.05% per side
FUNDING_8H = 0.0001          # 0.01% per 8h per leg
FUNDING_DAILY = FUNDING_8H*3 # 3 funding intervals/day, paid on the perp notional

# reference pools (USD-denominated close), on disk
BTC_POOL = '0x99ac8ca7087fa4a2a1fb6357269965a2014abc35'  # WBTC/USDC 0.3%
ETH_POOL = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'  # WETH/USDC 0.05%

REBAL_DAYS = 7   # weekly rebalance (turnover -> DEX cost)
IS_LEN, OOS_LEN = 60, 21   # ~60d IS beta est / 21d OOS, rolling over the 181-day ref window
MIN_BARS = 120
ANN = 365.0  # daily crypto

# tokens to EXCLUDE from the DEX basket: wrapped majors / stables / liquid-staking
EXCLUDE_KW = ['USDC','USDT','DAI','USD','WBTC','CBBTC','TBTC','BTCB','WETH','WEETH',
              'WSTETH','STETH','RETH','CBETH','EZETH','FRAX','BUSD','TUSD','USDE','SUSDE',
              'WBNB','WMATIC','WAVAX','WFTM','WSOL','PYUSD','LUSD','GUSD','RLUSD','USDD',
              'FDUSD','CRVUSD','GHO','USR','SCRVUSD','WBETH','SFRXETH','OETH','MSOL','JITOSOL']

def load_panel():
    files = glob.glob(str(DATA/'ohlcv_gt/mega_*_day.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = [json.loads(l) for l in open(DATA/'_mega_universe.jsonl')]
    m = pd.DataFrame(rows)[['chain','pair','reserve_usd','vol24','name','dex']]
    m = m.rename(columns={'pair':'pair_address'})
    df = df.merge(m, on=['chain','pair_address'], how='left')
    return df, m

def is_dex_coin(name):
    if not isinstance(name,str): return True
    up = name.upper()
    # a pair "A / B" is excluded if BOTH legs are majors/stables (it's a market ref),
    # OR if it's a stable/stable. Keep alt/major and alt/stable: those carry the alt.
    parts = [p.strip() for p in up.replace('/',' / ').split('/')]
    base = parts[0].split()[0] if parts and parts[0].split() else ''
    # exclude if the BASE token (token of interest) is a major/stable
    return not any(kw==base or base.startswith(kw) for kw in EXCLUDE_KW)

def build_returns(df):
    """Wide daily return matrix for tradeable DEX coins (index=ts, cols=coinid)."""
    g = df.groupby(['chain','pair_address']).agg(n=('ts','size'),res=('reserve_usd','first'),
                                                 name=('name','first')).reset_index()
    g['ok'] = (g.res>=50000)&(g.n>=MIN_BARS)&g.name.map(is_dex_coin)
    keep = g[g.ok].copy()
    keep['coinid'] = keep.chain+'|'+keep.pair_address
    df = df.copy()
    df['coinid'] = df.chain+'|'+df.pair_address
    sub = df[df.coinid.isin(set(keep.coinid))][['coinid','ts','close','chain','reserve_usd']]
    # pivot close
    px = sub.pivot_table(index='ts', columns='coinid', values='close', aggfunc='last').sort_index()
    ret = px.pct_change()
    # clip extreme single-bar returns (data glitches / illiquid prints) at +/-100% per bar
    ret = ret.clip(-0.9, 1.0)
    meta = keep.set_index('coinid')[['chain','res']]
    return ret, meta, keep

def ref_series(df, pool):
    s = df[df.pair_address==pool].sort_values('ts').set_index('ts')['close']
    return s

def metrics(daily_ret):
    """daily_ret: pd.Series of net daily returns (OOS). Return dict."""
    r = daily_ret.dropna()
    if len(r) < 5:
        return dict(n=len(r), ann_ret=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan, calmar=np.nan)
    eq = (1+r).cumprod()
    ann_ret = eq.iloc[-1]**(ANN/len(r)) - 1
    vol = r.std()*np.sqrt(ANN)
    sharpe = (r.mean()/r.std()*np.sqrt(ANN)) if r.std()>0 else np.nan
    peak = eq.cummax()
    dd = (eq/peak - 1).min()
    calmar = (ann_ret/abs(dd)) if dd<0 else np.nan
    return dict(n=len(r), ann_ret=ann_ret, vol=vol, sharpe=sharpe, maxdd=dd, calmar=calmar)

def basket_daily_returns(ret, meta, weight='equal', shuffle=False):
    """Return a daily basket gross-return series with weekly rebalance + turnover cost.
    weight: 'equal' or 'liq'. shuffle: bar-shuffle null (permute each coin's ret in time)."""
    R = ret.copy()
    if shuffle:
        # permute each column's returns in time (breaks any timing structure), seeded
        for c in R.columns:
            v = R[c].values.copy()
            mask = ~np.isnan(v)
            idx = np.where(mask)[0]
            perm = rng.permutation(idx)
            v[idx] = v[perm]
            R[c] = v
    ts = R.index.values
    nT = len(ts)
    chains = meta['chain'].to_dict()
    res = meta['res'].to_dict()
    # per-coin round-trip cost fraction (applied on turnover)
    rtc = {c: coin_cost(chains.get(c.split('|')[0] if '|' in c else c, None) or c.split('|')[0],
                        res.get(c)) for c in R.columns}
    # simpler: coinid = 'chain|pair'
    rtc = {c: coin_cost(c.split('|')[0], res.get(c)) for c in R.columns}

    gross = np.zeros(nT)
    turn_cost = np.zeros(nT)
    active_log = []
    for i in range(nT):
        row = R.iloc[i]
        avail = row.index[row.notna()]
        if len(avail)==0:
            gross[i]=0.0; continue
        if weight=='equal':
            w = pd.Series(1.0/len(avail), index=avail)
        else:
            rr = pd.Series([max(res.get(c,0),1.0) for c in avail], index=avail)
            w = rr/rr.sum()
        gross[i] = float((w*row[avail]).sum())
        active_log.append(len(avail))
        # rebalance cost on rebalance days: assume full turnover of the basket
        if i % REBAL_DAYS == 0:
            # weighted avg round-trip cost across held coins, applied once (one rebalance)
            wc = float((w*pd.Series([rtc[c] for c in avail], index=avail)).sum())
            turn_cost[i] = wc
    g = pd.Series(gross, index=R.index)
    tc = pd.Series(turn_cost, index=R.index)
    net = g - tc
    return g, net, np.median(active_log) if active_log else 0

def wfo_hedge(basket_gross, ref_ret, ref_name):
    """Walk-forward beta hedge. For each OOS window, beta = OLS of basket_gross on
    ref_ret over the prior IS window; hedged_ret = basket - beta*ref_ret - perp_cost.
    Returns (hedged_net_series, beta_log)."""
    idx = basket_gross.index
    common = basket_gross.index.intersection(ref_ret.index)
    bg = basket_gross.reindex(common)
    rr = ref_ret.reindex(common)
    n = len(common)
    hedged = pd.Series(index=common, dtype=float)
    betas = []
    i = IS_LEN
    while i < n:
        is_b = bg.iloc[i-IS_LEN:i].values
        is_r = rr.iloc[i-IS_LEN:i].values
        mask = ~(np.isnan(is_b)|np.isnan(is_r))
        if mask.sum() >= 20 and np.std(is_r[mask])>0:
            beta = np.cov(is_b[mask], is_r[mask])[0,1]/np.var(is_r[mask])
        else:
            beta = 0.0
        beta = float(np.clip(beta, -3, 3))
        oe = min(i+OOS_LEN, n)
        for j in range(i, oe):
            mret = rr.iloc[j]
            if np.isnan(mret): mret = 0.0
            # short beta*notional of perp: pnl from short = -beta*mret
            # perp costs: funding daily on |beta| notional; taker fee on rebalance days
            perp_funding = abs(beta)*FUNDING_DAILY
            perp_fee = abs(beta)*PERP_TAKER if (j % REBAL_DAYS==0) else 0.0
            hedged.iloc[j] = bg.iloc[j] - beta*mret - perp_funding - perp_fee
        betas.append((str(common[i]), beta, ref_name))
        i += OOS_LEN
    return hedged.dropna(), betas

def oos_slice(series):
    """The OOS-covered region = from first IS_LEN onward (matches wfo)."""
    return series.iloc[IS_LEN:]

def main():
    df, m = load_panel()
    ret, meta, keep = build_returns(df)
    print(f"DEX basket universe: {ret.shape[1]} coins, {ret.shape[0]} daily bars")
    keep.to_csv(OUT/'basket_universe.csv', index=False)

    btc = ref_series(df, BTC_POOL); eth = ref_series(df, ETH_POOL)
    btc_ret = btc.pct_change().clip(-0.5,0.5)
    eth_ret = eth.pct_change().clip(-0.5,0.5)

    # CRITICAL: BTC/ETH ref only exist for the last ~181 days. For a FAIR
    # unhedged-vs-hedged comparison, restrict ALL variants to the common window
    # where the reference series exist. Otherwise unhedged runs over 2x more
    # (older, different-regime) days than the hedge can.
    ref_idx = btc_ret.dropna().index.intersection(eth_ret.dropna().index)
    win_lo, win_hi = ref_idx.min(), ref_idx.max()
    ret = ret.loc[(ret.index>=win_lo)&(ret.index<=win_hi)]
    # require coins to actually trade within the window (>=120 of ~180 bars live)
    cov = ret.notna().sum()
    live = cov[cov>=120].index
    ret = ret[live]
    meta = meta.loc[meta.index.intersection(live)]
    print(f"Common ref window: {pd.to_datetime(win_lo,unit='s').date()} .. "
          f"{pd.to_datetime(win_hi,unit='s').date()}  ({len(ret)} bars, {ret.shape[1]} live coins)")

    results = []
    beta_logs = []
    series_store = {}

    for weight in ['equal','liq']:
        for shuffle,tag in [(False,'real'),(True,'null')]:
            gross, net, med_active = basket_daily_returns(ret, meta, weight=weight, shuffle=shuffle)
            # align gross to ref index for hedge; use NET (after dex cost) basket as the long leg
            net_idx = net
            # (a) unhedged long, OOS region
            m_unh = metrics(oos_slice(net_idx)); m_unh.update(weight=weight, data=tag, variant='unhedged', med_active=med_active)
            results.append(m_unh)
            series_store[(weight,tag,'unhedged')] = oos_slice(net_idx)
            # (b) BTC-beta hedged
            h_btc, blog = wfo_hedge(net_idx, btc_ret, 'BTC')
            beta_logs += [(weight,tag)+b for b in blog]
            m_b = metrics(h_btc); m_b.update(weight=weight, data=tag, variant='hedge_BTC', med_active=med_active)
            results.append(m_b); series_store[(weight,tag,'hedge_BTC')]=h_btc
            # (b2) ETH-beta hedged
            h_eth, blog2 = wfo_hedge(net_idx, eth_ret, 'ETH')
            beta_logs += [(weight,tag)+b for b in blog2]
            m_e = metrics(h_eth); m_e.update(weight=weight, data=tag, variant='hedge_ETH', med_active=med_active)
            results.append(m_e); series_store[(weight,tag,'hedge_ETH')]=h_eth

    # (d) no-skill buy&hold equal-weight (1 rebalance, cost once at entry)
    gross_eq, _, _ = basket_daily_returns(ret, meta, weight='equal', shuffle=False)
    bh = oos_slice(gross_eq).copy()
    # single entry cost: weighted-avg rtc applied on first OOS bar
    if len(bh)>0:
        avg_rtc = np.mean([coin_cost(c.split('|')[0], meta['res'].get(c)) for c in ret.columns])
        bh.iloc[0] = bh.iloc[0] - avg_rtc
    m_bh = metrics(bh); m_bh.update(weight='equal', data='real', variant='buyhold_noskill', med_active=np.nan)
    results.append(m_bh); series_store[('equal','real','buyhold_noskill')]=bh

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT/'portfolio_metrics.csv', index=False)
    pd.DataFrame(beta_logs, columns=['weight','data','date','beta','ref']).to_csv(OUT/'beta_log.csv', index=False)

    # save the equity curves used in figs
    eq_rows = []
    for k,s in series_store.items():
        eq = (1+s).cumprod()
        for t,v in eq.items():
            eq_rows.append(dict(weight=k[0],data=k[1],variant=k[2],ts=int(t)//10**9 if t>10**12 else int(t),eq=float(v)))
    pd.DataFrame(eq_rows).to_csv(OUT/'equity_curves.csv', index=False)

    print(res_df.to_string())
    import pickle
    with open(OUT/'_series.pkl','wb') as f:
        pickle.dump({'store':series_store,'btc_ret':btc_ret,'eth_ret':eth_ret,
                     'beta_logs':beta_logs}, f)
    return res_df

if __name__=='__main__':
    main()
