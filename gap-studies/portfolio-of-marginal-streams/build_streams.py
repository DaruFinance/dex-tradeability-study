"""GAP: build an ACTUAL portfolio from surviving (coin,family) timing streams.

Step 1 (this script): reconstruct TIME-STAMPED OOS per-trade PnL for every (coin,family)
stream, using the SAME phase9 WFO sim (causal, intrabar TP/SL, per-fill cost). Emit one row
per OOS trade: coin, chain, family, entry_ts, exit_ts, net_ret, dollar_size, reserve.
Run for REAL and for NULL (bar-shuffle within coin). Seeded, reproducible.

We deliberately do NOT pre-filter to the survivor corpus here: we emit ALL OOS trades for
every (coin,family) that produced >=MIN_OOS_TR OOS trades, so the portfolio layer can do its
own WALK-FORWARD stream selection without lookahead.
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
OUT = Path("./_gaps/portfolio-of-marginal-streams")
TF = "day"
MIN_BARS, IS_LEN, OOS_LEN = 80, 90, 45
MIN_IS_TR, MIN_OOS_TR = 5, 4
NS = 12
SIZE_FRAC = 0.0025
MODE = sys.argv[1] if len(sys.argv) > 1 else "real"   # real | null
MAXC = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
NULL = (MODE == "null")
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
def _roll_max(x, n): return pd.Series(x).rolling(n).max().to_numpy()
def _roll_min(x, n): return pd.Series(x).rolling(n).min().to_numpy()
def _atr(h, l, c, n):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().to_numpy()
def shift(b):
    b = np.asarray(b, float); out = np.zeros(len(b), bool); out[1:] = np.nan_to_num(b[:-1]) > 0; return out

def fam_sma_cross(o,h,l,c,v,p):  s=_sma(c,p['f']); ls=_sma(c,p['s']); return shift(s>ls)
def fam_ema_cross(o,h,l,c,v,p):  return shift(_ema(c,p['f'])>_ema(c,p['s']))
def fam_price_above_ma(o,h,l,c,v,p): return shift(c>_sma(c,p['L']))
def fam_donchian_bo(o,h,l,c,v,p): return shift(c>=_roll_max(h,p['L']))
def fam_donchian_rev(o,h,l,c,v,p): return shift(c<=_roll_min(l,p['L']))
def fam_rsi_os(o,h,l,c,v,p):     return shift(_rsi(c,p['L'])<p['thr'])
def fam_rsi_mom(o,h,l,c,v,p):    return shift(_rsi(c,p['L'])>p['thr'])
def fam_boll_rev(o,h,l,c,v,p):   m=_sma(c,p['L']); sd=pd.Series(c).rolling(p['L']).std().to_numpy(); return shift(c<m-p['k']*sd)
def fam_boll_bo(o,h,l,c,v,p):    m=_sma(c,p['L']); sd=pd.Series(c).rolling(p['L']).std().to_numpy(); return shift(c>m+p['k']*sd)
def fam_atr_bo(o,h,l,c,v,p):     return shift(c>np.roll(c,1)+p['k']*_atr(h,l,c,p['L']))
def fam_roc_mom(o,h,l,c,v,p):    r=pd.Series(c).pct_change(p['L']).to_numpy(); return shift(r>p['thr'])
def fam_roc_rev(o,h,l,c,v,p):    r=pd.Series(c).pct_change(p['L']).to_numpy(); return shift(r<-p['thr'])
def fam_vol_bo(o,h,l,c,v,p):     vm=_sma(v,p['L']); return shift((v>p['k']*vm)&(c>np.roll(c,1)))
def fam_consec_down(o,h,l,c,v,p):
    dn=(np.diff(c,prepend=c[0])<0).astype(int); run=pd.Series(dn).rolling(p['L']).sum().to_numpy(); return shift(run>=p['L'])
def fam_consec_up(o,h,l,c,v,p):
    upb=(np.diff(c,prepend=c[0])>0).astype(int); run=pd.Series(upb).rolling(p['L']).sum().to_numpy(); return shift(run>=p['L'])
def fam_ma_pullback(o,h,l,c,v,p):
    m=_sma(c,p['L']); rising=m>np.roll(m,3); return shift(rising&(c<m)&(c>np.roll(c,1)))

FAMILIES = {
    "sma_cross": (fam_sma_cross, lambda: {"f": int(rng.choice([3,5,8,10])), "s": int(rng.choice([20,30,50]))}),
    "ema_cross": (fam_ema_cross, lambda: {"f": int(rng.choice([3,5,8,12])), "s": int(rng.choice([21,34,50]))}),
    "price_above_ma": (fam_price_above_ma, lambda: {"L": int(rng.choice([10,20,30,50]))}),
    "donchian_bo": (fam_donchian_bo, lambda: {"L": int(rng.choice([10,20,30,55]))}),
    "donchian_rev": (fam_donchian_rev, lambda: {"L": int(rng.choice([10,20,30]))}),
    "rsi_os": (fam_rsi_os, lambda: {"L": int(rng.choice([7,14,21])), "thr": float(rng.choice([20,25,30,35]))}),
    "rsi_mom": (fam_rsi_mom, lambda: {"L": int(rng.choice([7,14,21])), "thr": float(rng.choice([55,60,65,70]))}),
    "boll_rev": (fam_boll_rev, lambda: {"L": int(rng.choice([14,20,30])), "k": float(rng.choice([1.5,2.0,2.5]))}),
    "boll_bo": (fam_boll_bo, lambda: {"L": int(rng.choice([14,20,30])), "k": float(rng.choice([1.0,1.5,2.0]))}),
    "atr_bo": (fam_atr_bo, lambda: {"L": int(rng.choice([10,14,20])), "k": float(rng.choice([1.0,1.5,2.0,3.0]))}),
    "roc_mom": (fam_roc_mom, lambda: {"L": int(rng.choice([3,5,10,20])), "thr": float(rng.choice([.05,.1,.2,.3]))}),
    "roc_rev": (fam_roc_rev, lambda: {"L": int(rng.choice([3,5,10])), "thr": float(rng.choice([.1,.2,.3,.5]))}),
    "vol_bo": (fam_vol_bo, lambda: {"L": int(rng.choice([10,20,30])), "k": float(rng.choice([2,3,5]))}),
    "consec_down": (fam_consec_down, lambda: {"L": int(rng.choice([2,3,4]))}),
    "consec_up": (fam_consec_up, lambda: {"L": int(rng.choice([2,3,4]))}),
    "ma_pullback": (fam_ma_pullback, lambda: {"L": int(rng.choice([10,20,30]))}),
}

def simulate_ts(o,h,l,c, ts, entry, tp, sl, max_hold, cost):
    """Long-only intrabar TP/SL. Returns list of (entry_ts, exit_ts, net_ret)."""
    n=len(c); i=0; out=[]
    while i < n-1:
        if entry[i] and c[i]>0:
            ep=c[i]; tpx=ep*(1+tp); spx=ep*(1-sl); end=min(i+max_hold, n-1); ret=None
            j=i+1
            while j<=end:
                if l[j]<=spx: ret=-sl; break
                if h[j]>=tpx: ret=tp; break
                j+=1
            if ret is None:
                jj=min(j,end); ret=c[jj]/ep-1.0
            jj=min(j,end)
            out.append((int(ts[i]), int(ts[jj]), ret-cost)); i=j+1
        else: i+=1
    return out

def pf(tr):
    g=sum(t for t in tr if t>0); ls=-sum(t for t in tr if t<0)
    return g/ls if ls>0 else (np.inf if g>0 else 0.0)

def main():
    univ={}
    for line in open(DATA/"_mega_universe.jsonl"):
        r=json.loads(line); univ[r["pair"]]=r
    files=glob.glob(f"{DATA}/ohlcv_gt/mega_*_{TF}.parquet")
    df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["close","high","low"])
    df=df[df.close>0].sort_values(["pair_address","ts"])
    coins=[p for p,_ in df.groupby("pair_address")][:MAXC]
    coinset=set(coins)
    print(f"[{MODE}] {len(coins)} coins, {len(FAMILIES)} families, NS={NS}; IS={IS_LEN}/OOS={OOS_LEN}", flush=True)

    t0=time.time(); trades=[]; stream_meta=[]; nstream=0
    for ci,(pair,g) in enumerate(df.groupby("pair_address")):
        if pair not in coinset: continue
        o,h,l,c,v = (g.open.to_numpy(float),g.high.to_numpy(float),g.low.to_numpy(float),
                     g.close.to_numpy(float),g.volume.fillna(0).to_numpy(float))
        ts=g.ts.to_numpy()
        n=len(c)
        if n<MIN_BARS: continue
        if NULL:
            perm=rng.permutation(n)
            # shuffle OHLCV bars together but KEEP the original time index sorted so
            # exit dates are still real calendar dates (destroys temporal structure only).
            o,h,l,c,v=o[perm],h[perm],l[perm],c[perm],v[perm]
        u=univ.get(pair,{}); res=u.get("reserve_usd") or 0; ch=u.get("chain","bsc")
        size=max(50.,SIZE_FRAC*res)
        cost=round_trip_cost_frac(size,res,dex="uniswap",chain=ch,gas_usd=CHAIN_GAS.get(ch,.05),native_usd=CHAIN_NATIVE.get(ch,100)) if res else .05
        starts=list(range(0, n-IS_LEN-OOS_LEN+1, OOS_LEN))
        if not starts: continue
        for fname,(fn,sampler) in FAMILIES.items():
            oos_trades=[]
            for st in starts:
                isl=slice(st,st+IS_LEN); osl=slice(st+IS_LEN,st+IS_LEN+OOS_LEN)
                best=None
                for _ in range(NS):
                    p=sampler(); tp=float(rng.choice([.15,.3,.5,1.0])); sl=float(rng.choice([.1,.2,.3])); mh=int(rng.choice([5,10,20,40]))
                    sig=fn(o,h,l,c,v,p)
                    is_tr=[t[2] for t in simulate_ts(o[isl],h[isl],l[isl],c[isl], ts[isl], sig[isl], tp,sl,mh, cost)]
                    if len(is_tr)<MIN_IS_TR: continue
                    s=pf(is_tr)
                    if best is None or s>best[0]: best=(s,p,tp,sl,mh)
                if best is None: continue
                _,p,tp,sl,mh=best; sig=fn(o,h,l,c,v,p)
                oos_trades+=simulate_ts(o[osl],h[osl],l[osl],c[osl], ts[osl], sig[osl], tp,sl,mh, cost)
            if len(oos_trades)>=MIN_OOS_TR:
                sid=f"{pair}|{fname}"
                for (ets,xts,r) in oos_trades:
                    trades.append((sid,pair,ch,fname,ets,xts,r,size,res))
                stream_meta.append({"stream":sid,"coin":pair,"chain":ch,"family":fname,
                                    "reserve_usd":res,"dollar_size":size,"n_oos":len(oos_trades)})
                nstream+=1
        if (ci+1)%200==0:
            print(f"  [{ci+1}] streams={nstream} trades={len(trades)} ({time.time()-t0:.0f}s)", flush=True)

    T=pd.DataFrame(trades, columns=["stream","coin","chain","family","entry_ts","exit_ts","net_ret","dollar_size","reserve_usd"])
    M=pd.DataFrame(stream_meta)
    T.to_parquet(OUT/f"trades_{MODE}.parquet", index=False)
    M.to_parquet(OUT/f"streams_{MODE}.parquet", index=False)
    print(f"\n[{MODE}] DONE {time.time()-t0:.0f}s: {nstream} streams, {len(T)} OOS trades")
    print(f"  median net/trade={T.net_ret.median():+.4f} mean={T.net_ret.mean():+.4f}")
    print(f"  saved -> trades_{MODE}.parquet, streams_{MODE}.parquet")

if __name__=="__main__":
    main()
