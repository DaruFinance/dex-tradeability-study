"""Phase 9: PER-COIN multi-strategy walk-forward engine (the real test).

For each coin: ~16 structural long-only strategy FAMILIES (trend/breakout/mean-rev/volatility/
volume), each with IS-tuned params sampled per WFO window. Proper TP/SL brackets using INTRABAR
HIGH/LOW (not close-only). Rolling IS/OOS per coin; keep (coin, family) pairs whose pooled OOS
beats cost. A coin can pass even if its price ended far below launch — we trade the intra-path
peaks, not buy-and-hold.

Unit = (coin, strategy family). ~16 families x 2,300 coins, params sampled per window. Output: the
corpus of surviving (coin, strategy), per-family pass rates, OOS PF distribution, decorrelation.

Usage: python3 _research/phase9_percoin_wfo.py [max_coins] [ns_samples]
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
MAXC = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 12          # param samples per WFO window
NULL = (len(sys.argv) > 3 and sys.argv[3] == "null")        # shuffle bars per coin -> chance baseline
TF = sys.argv[4] if len(sys.argv) > 4 else "day"            # day | hour
# WFO windows scaled per timeframe (hourly has ~24x more bars -> more windows + trades)
if TF == "hour": MIN_BARS, IS_LEN, OOS_LEN = 360, 240, 120  # ~10d IS / 5d OOS, ~6 windows on 1000 bars
else:            MIN_BARS, IS_LEN, OOS_LEN = 80, 90, 45
MIN_IS_TR, MIN_OOS_TR = 5, 4
SIZE_FRAC = 0.0025
CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05, "polygon_pos": .01,
             "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "sui-network": 3.5, "solana": 180, "tron": .3}
rng = np.random.default_rng(42)

# ---- indicator helpers (all causal: computed on bars <= i, used to decide entry at i) ----
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

# ---- 16 structural families: each returns a boolean entry-signal (shifted to be causal) + a param sampler ----
def fam_sma_cross(o,h,l,c,v,p):  s=_sma(c,p['f']); ls=_sma(c,p['s']); return shift(s>ls)
def fam_ema_cross(o,h,l,c,v,p):  return shift(_ema(c,p['f'])>_ema(c,p['s']))
def fam_price_above_ma(o,h,l,c,v,p): return shift(c>_sma(c,p['L']))
def fam_donchian_bo(o,h,l,c,v,p): return shift(c>=_roll_max(h,p['L']))            # breakout (high)
def fam_donchian_rev(o,h,l,c,v,p): return shift(c<=_roll_min(l,p['L']))           # buy channel low
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

def shift(b):
    b = np.asarray(b, float); out = np.zeros(len(b), bool); out[1:] = np.nan_to_num(b[:-1]) > 0; return out

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

def simulate(o,h,l,c, entry, tp, sl, max_hold, cost):
    """Long-only with INTRABAR TP/SL via high/low. Returns list of net trade returns."""
    n=len(c); i=0; out=[]
    while i < n-1:
        if entry[i] and c[i]>0:
            ep=c[i]; tpx=ep*(1+tp); spx=ep*(1-sl); end=min(i+max_hold, n-1); ret=None
            j=i+1
            while j<=end:
                if l[j]<=spx: ret=-sl; break          # SL hit intrabar (conservative: SL before TP)
                if h[j]>=tpx: ret=tp; break            # TP hit intrabar
                j+=1
            if ret is None: ret=c[min(j,end)]/ep-1.0   # time exit at close
            out.append(ret-cost); i=j+1
        else: i+=1
    return out

def pf(tr):
    g=sum(t for t in tr if t>0); ls=-sum(t for t in tr if t<0)
    return g/ls if ls>0 else (np.inf if g>0 else 0.0)

def main():
    univ={}
    for l in open(DATA/"_mega_universe.jsonl"): r=json.loads(l); univ[r["pair"]]=r
    files=glob.glob(f"{DATA}/ohlcv_gt/mega_*_{TF}.parquet")
    df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["close","high","low"])
    df=df[df.close>0].sort_values(["pair_address","ts"])
    coins=[p for p,_ in df.groupby("pair_address")][:MAXC]
    print(f"{len(coins)} coins, {len(FAMILIES)} families, NS={NS}/window; tf={TF}: IS={IS_LEN}/OOS={OOS_LEN}", flush=True)

    t0=time.time(); corpus=[]; all_evals=[]; fam_pass={k:0 for k in FAMILIES}; fam_tot={k:0 for k in FAMILIES}
    coins_with_pass=0; n_eval=0
    for ci,(pair,g) in enumerate(df.groupby("pair_address")):
        if pair not in set(coins): continue
        o,h,l,c,v = (g.open.to_numpy(float),g.high.to_numpy(float),g.low.to_numpy(float),
                     g.close.to_numpy(float),g.volume.fillna(0).to_numpy(float))
        n=len(c)
        if n<MIN_BARS: continue
        if NULL:                                  # destroy temporal structure (bars permuted together)
            perm=rng.permutation(n); o,h,l,c,v=o[perm],h[perm],l[perm],c[perm],v[perm]
        u=univ.get(pair,{}); res=u.get("reserve_usd") or 0; ch=u.get("chain","bsc")
        size=max(50.,SIZE_FRAC*res)
        cost=round_trip_cost_frac(size,res,dex="uniswap",chain=ch,gas_usd=CHAIN_GAS.get(ch,.05),native_usd=CHAIN_NATIVE.get(ch,100)) if res else .05
        # WFO window starts
        starts=list(range(0, n-IS_LEN-OOS_LEN+1, OOS_LEN))
        if not starts: continue
        coin_pass=False
        for fname,(fn,sampler) in FAMILIES.items():
            fam_tot[fname]+=1; n_eval+=1
            oos_all=[]
            for st in starts:
                isl=slice(st,st+IS_LEN); osl=slice(st+IS_LEN,st+IS_LEN+OOS_LEN)
                best=None
                for _ in range(NS):
                    p=sampler(); tp=float(rng.choice([.15,.3,.5,1.0])); sl=float(rng.choice([.1,.2,.3])); mh=int(rng.choice([5,10,20,40]))
                    sig=fn(o,h,l,c,v,p)
                    is_tr=simulate(o[isl],h[isl],l[isl],c[isl], sig[isl], tp,sl,mh, cost)
                    if len(is_tr)<MIN_IS_TR: continue
                    s=pf(is_tr)
                    if best is None or s>best[0]: best=(s,p,tp,sl,mh)
                if best is None: continue
                _,p,tp,sl,mh=best; sig=fn(o,h,l,c,v,p)
                oos_all+=simulate(o[osl],h[osl],l[osl],c[osl], sig[osl], tp,sl,mh, cost)
            if len(oos_all)>=MIN_OOS_TR:
                P=pf(oos_all); net=sum(oos_all)
                all_evals.append({"oos_pf":P, "oos_net":net, "oos_n":len(oos_all), "avg_per_trade":net/len(oos_all)})
                if P>1.0 and net>0:
                    corpus.append({"coin":pair,"chain":ch,"family":fname,"oos_pf":P,"oos_n":len(oos_all),"oos_net":net})
                    fam_pass[fname]+=1; coin_pass=True
        if coin_pass: coins_with_pass+=1
        if (ci+1)%200==0:
            print(f"  [{ci+1}] coins_with_pass={coins_with_pass} corpus={len(corpus)} ({time.time()-t0:.0f}s)", flush=True)
    C=pd.DataFrame(corpus); A=pd.DataFrame(all_evals)
    print(f"\n=== RESULTS ({n_eval} (coin,family) evaluations, {time.time()-t0:.0f}s) ===")
    # FULL distribution (all evaluated, not just winners) — the unbiased measure
    if len(A):
        apf=A.oos_pf.replace(np.inf,np.nan)
        print(f"ALL {len(A)} evaluated (>= {MIN_OOS_TR} OOS trades):")
        print(f"  median OOS PF (ALL)        : {apf.median():.3f}")
        print(f"  mean   OOS PF (ALL, capped): {apf.clip(upper=10).mean():.3f}")
        print(f"  %% with OOS PF>1            : {100*(A.oos_pf>1).mean():.1f}%")
        print(f"  median avg net return/trade: {A.avg_per_trade.median():+.4f}  (mean {A.avg_per_trade.mean():+.4f})")
    print(f"survivors (PF>1 & net>0): {len(C)}  | coins with >=1 survivor: {coins_with_pass}")
    if len(C):
        print(f"OOS PF of survivors: median={C.oos_pf.replace(np.inf,np.nan).median():.2f}")
        print("per-family pass rate (passing / coins-evaluated):")
        for k in FAMILIES:
            if fam_tot[k]: print(f"  {k:16} {fam_pass[k]:5}/{fam_tot[k]:<5} = {100*fam_pass[k]/fam_tot[k]:4.0f}%")
        C.to_parquet(DATA/"_phase9_corpus.parquet", index=False)
        print(f"\nsaved corpus -> {DATA}/_phase9_corpus.parquet")
    print("\nNOTE: in-sample param selection per window then OOS eval = honest WFO. A coin passes by hosting "
          "timing strategies that survive OOS net of cost, regardless of its overall price trend.")

if __name__=="__main__":
    main()
