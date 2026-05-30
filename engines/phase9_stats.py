"""Per-pair characterization of the phase-9 simulated strategies (the 'report card').

Re-runs the same per-coin WFO engine but records per-trade LENGTH + counts, and emits a per-pair
table: fee (round-trip cost), median trade length (bars), trade count, coin history (bars), trades
per strategy. Fixed: timeframe=DAILY, WFO IS=90 / OOS=45 / step=45 bars, NS=12 param samples/window.

Output: _phase9_perpair_stats.csv (full per-pair) + per-chain & overall medians printed.
"""
from __future__ import annotations
import glob, json, time
from pathlib import Path
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
NS = 12; MIN_BARS = 80; IS_LEN, OOS_LEN = 90, 45; MIN_IS_TR = 5; SIZE_FRAC = 0.0025
CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05, "polygon_pos": .01,
             "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "sui-network": 3.5, "solana": 180, "tron": .3}
rng = np.random.default_rng(42)

def _sma(x, n): return pd.Series(x).rolling(n).mean().to_numpy()
def _ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()
def _rsi(x, n):
    d = np.diff(x, prepend=x[0]); up = np.clip(d, 0, None); dn = np.clip(-d, 0, None)
    au = pd.Series(up).rolling(n).mean().to_numpy(); ad = pd.Series(dn).rolling(n).mean().to_numpy()
    rs = np.divide(au, ad, out=np.full_like(au, np.nan), where=ad > 0); return 100 - 100 / (1 + rs)
def _rmax(x, n): return pd.Series(x).rolling(n).max().to_numpy()
def _rmin(x, n): return pd.Series(x).rolling(n).min().to_numpy()
def _atr(h, l, c, n):
    pc = np.roll(c, 1); pc[0] = c[0]; tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().to_numpy()
def shift(b):
    b = np.asarray(b, float); o = np.zeros(len(b), bool); o[1:] = np.nan_to_num(b[:-1]) > 0; return o

FAM = {
 "sma_cross": (lambda o,h,l,c,v,p: shift(_sma(c,p['f'])>_sma(c,p['s'])), lambda:{"f":int(rng.choice([3,5,8,10])),"s":int(rng.choice([20,30,50]))}),
 "ema_cross": (lambda o,h,l,c,v,p: shift(_ema(c,p['f'])>_ema(c,p['s'])), lambda:{"f":int(rng.choice([3,5,8,12])),"s":int(rng.choice([21,34,50]))}),
 "price_above_ma": (lambda o,h,l,c,v,p: shift(c>_sma(c,p['L'])), lambda:{"L":int(rng.choice([10,20,30,50]))}),
 "donchian_bo": (lambda o,h,l,c,v,p: shift(c>=_rmax(h,p['L'])), lambda:{"L":int(rng.choice([10,20,30,55]))}),
 "donchian_rev": (lambda o,h,l,c,v,p: shift(c<=_rmin(l,p['L'])), lambda:{"L":int(rng.choice([10,20,30]))}),
 "rsi_os": (lambda o,h,l,c,v,p: shift(_rsi(c,p['L'])<p['thr']), lambda:{"L":int(rng.choice([7,14,21])),"thr":float(rng.choice([20,25,30,35]))}),
 "rsi_mom": (lambda o,h,l,c,v,p: shift(_rsi(c,p['L'])>p['thr']), lambda:{"L":int(rng.choice([7,14,21])),"thr":float(rng.choice([55,60,65,70]))}),
 "boll_rev": (lambda o,h,l,c,v,p: shift(c<_sma(c,p['L'])-p['k']*pd.Series(c).rolling(p['L']).std().to_numpy()), lambda:{"L":int(rng.choice([14,20,30])),"k":float(rng.choice([1.5,2,2.5]))}),
 "boll_bo": (lambda o,h,l,c,v,p: shift(c>_sma(c,p['L'])+p['k']*pd.Series(c).rolling(p['L']).std().to_numpy()), lambda:{"L":int(rng.choice([14,20,30])),"k":float(rng.choice([1,1.5,2]))}),
 "atr_bo": (lambda o,h,l,c,v,p: shift(c>np.roll(c,1)+p['k']*_atr(h,l,c,p['L'])), lambda:{"L":int(rng.choice([10,14,20])),"k":float(rng.choice([1,1.5,2,3]))}),
 "roc_mom": (lambda o,h,l,c,v,p: shift(pd.Series(c).pct_change(p['L']).to_numpy()>p['thr']), lambda:{"L":int(rng.choice([3,5,10,20])),"thr":float(rng.choice([.05,.1,.2,.3]))}),
 "roc_rev": (lambda o,h,l,c,v,p: shift(pd.Series(c).pct_change(p['L']).to_numpy()<-p['thr']), lambda:{"L":int(rng.choice([3,5,10])),"thr":float(rng.choice([.1,.2,.3,.5]))}),
 "vol_bo": (lambda o,h,l,c,v,p: shift((v>p['k']*_sma(v,p['L']))&(c>np.roll(c,1))), lambda:{"L":int(rng.choice([10,20,30])),"k":float(rng.choice([2,3,5]))}),
 "consec_down": (lambda o,h,l,c,v,p: shift(pd.Series((np.diff(c,prepend=c[0])<0).astype(int)).rolling(p['L']).sum().to_numpy()>=p['L']), lambda:{"L":int(rng.choice([2,3,4]))}),
 "consec_up": (lambda o,h,l,c,v,p: shift(pd.Series((np.diff(c,prepend=c[0])>0).astype(int)).rolling(p['L']).sum().to_numpy()>=p['L']), lambda:{"L":int(rng.choice([2,3,4]))}),
 "ma_pullback": (lambda o,h,l,c,v,p: shift((_sma(c,p['L'])>np.roll(_sma(c,p['L']),3))&(c<_sma(c,p['L']))&(c>np.roll(c,1))), lambda:{"L":int(rng.choice([10,20,30]))}),
}

def sim(o,h,l,c,entry,tp,sl,mh,cost):
    """returns list of (net_ret, length_bars)"""
    n=len(c); i=0; out=[]
    while i<n-1:
        if entry[i] and c[i]>0:
            ep=c[i]; tpx=ep*(1+tp); spx=ep*(1-sl); end=min(i+mh,n-1); ret=None; j=i+1
            while j<=end:
                if l[j]<=spx: ret=-sl; break
                if h[j]>=tpx: ret=tp; break
                j+=1
            if ret is None: j=min(j,end); ret=c[j]/ep-1.0
            out.append((ret-cost, j-i)); i=j+1
        else: i+=1
    return out

def pf(rets):
    g=sum(r for r in rets if r>0); ls=-sum(r for r in rets if r<0)
    return g/ls if ls>0 else (np.inf if g>0 else 0.0)

def main():
    univ={}
    for l in open(DATA/"_mega_universe.jsonl"): r=json.loads(l); univ[r["pair"]]=r
    df=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/ohlcv_gt/mega_*_day.parquet")],ignore_index=True)
    df=df.dropna(subset=["close","high","low"]); df=df[df.close>0].sort_values(["pair_address","ts"])
    rows=[]; t0=time.time(); k=0
    for pair,g in df.groupby("pair_address"):
        o,h,l,c,v=(g.open.to_numpy(float),g.high.to_numpy(float),g.low.to_numpy(float),g.close.to_numpy(float),g.volume.fillna(0).to_numpy(float))
        n=len(c)
        if n<MIN_BARS: continue
        u=univ.get(pair,{}); ch=u.get("chain","bsc"); res=u.get("reserve_usd") or 0
        fee=round_trip_cost_frac(max(50,SIZE_FRAC*res),res,dex="uniswap",chain=ch,gas_usd=CHAIN_GAS.get(ch,.05),native_usd=CHAIN_NATIVE.get(ch,100)) if res else .05
        starts=list(range(0,n-IS_LEN-OOS_LEN+1,OOS_LEN))
        if not starts: continue
        per_strat_counts=[]; all_lengths=[]; total=0
        for fname,(fn,sampler) in FAM.items():
            fam_trades=[]
            for st in starts:
                isl=slice(st,st+IS_LEN); osl=slice(st+IS_LEN,st+IS_LEN+OOS_LEN); best=None
                for _ in range(NS):
                    p=sampler(); tp=float(rng.choice([.15,.3,.5,1.])); sl=float(rng.choice([.1,.2,.3])); mh=int(rng.choice([5,10,20,40]))
                    sig=fn(o,h,l,c,v,p); itr=sim(o[isl],h[isl],l[isl],c[isl],sig[isl],tp,sl,mh,fee)
                    if len(itr)<MIN_IS_TR: continue
                    s=pf([r for r,_ in itr])
                    if best is None or s>best[0]: best=(s,p,tp,sl,mh)
                if best is None: continue
                _,p,tp,sl,mh=best; sig=fn(o,h,l,c,v,p)
                otr=sim(o[osl],h[osl],l[osl],c[osl],sig[osl],tp,sl,mh,fee)
                fam_trades+=otr
            if fam_trades:
                per_strat_counts.append(len(fam_trades)); all_lengths+=[ln for _,ln in fam_trades]; total+=len(fam_trades)
        rows.append({"pair":pair,"chain":ch,"reserve_usd":round(res),"fee_bp":round(fee*1e4,1),
                     "history_bars":n,"n_strategies_traded":len(per_strat_counts),
                     "oos_trades_total":total,"oos_trades_per_strategy_med":int(np.median(per_strat_counts)) if per_strat_counts else 0,
                     "trade_len_med_bars":round(float(np.median(all_lengths)),1) if all_lengths else 0})
        k+=1
        if k%300==0: print(f"  {k} coins ({time.time()-t0:.0f}s)",flush=True)
    S=pd.DataFrame(rows); S.to_csv(DATA/"_phase9_perpair_stats.csv",index=False)
    print(f"\n=== {len(S)} pairs characterized -> {DATA}/_phase9_perpair_stats.csv ===")
    print("FIXED: timeframe=DAILY (1d) | WFO IS=90 / OOS=45 / step=45 bars | NS=12 param samples/window | long-only | TP-SL via intrabar high/low")
    def med(col): return S[col].median()
    print(f"\nOVERALL medians across {len(S)} pairs:")
    print(f"  fee (round-trip):           {med('fee_bp'):.0f} bp")
    print(f"  coin history:               {med('history_bars'):.0f} daily bars")
    print(f"  trade length:               {med('trade_len_med_bars'):.1f} bars")
    print(f"  OOS trade count (total/coin):{med('oos_trades_total'):.0f}")
    print(f"  OOS trades per strategy:    {med('oos_trades_per_strategy_med'):.0f}")
    print(f"  strategies traded / coin:   {med('n_strategies_traded'):.0f}")
    print(f"\nPER-CHAIN medians:")
    agg=S.groupby("chain").agg(pairs=("pair","size"), fee_bp=("fee_bp","median"), hist_bars=("history_bars","median"),
        trade_len=("trade_len_med_bars","median"), trades_total=("oos_trades_total","median"),
        trades_per_strat=("oos_trades_per_strategy_med","median"), strats=("n_strategies_traded","median")).round(1)
    print(agg.to_string())

if __name__=="__main__":
    main()
