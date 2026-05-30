"""
(C) STRATEGY 2, "AVOID FIRST N DAYS" age filter on the phase9 16-family timing test.

Question: does excising the early launch/bleed window lift OOS PF vs the unfiltered
baseline AND vs a bar-shuffle null? Run the SAME multi-family per-coin WFO (adapted
from phase9_percoin_wfo.py) on coins with enough bars, under two regimes:
   baseline : use all bars
   skipN    : drop the first N bars of each coin before any windowing
Each regime is also run as a NULL (bars permuted within coin). Report full
distribution (median + mean across all coin x family evals) real-vs-null per N.

Long-only, intrabar SL-first TP/SL, per-fill cost, causal .shift, disjoint WFO.
Seeded. Uses LAUNCH-CAPTURED young pools (age<150, first-bar==launch) so that the
"first N days" is a genuine launch offset, plus enough-bars filter.
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,".")
from chainscope.costs import round_trip_cost_frac

DATA=Path("./data"); OUT=Path("./_gaps/launch-cohort")
SEED=42; rng=np.random.default_rng(SEED)
CHAIN_GAS={"bsc":.2,"base":.02,"eth":3.,"arbitrum":.05,"avax":.05,"polygon_pos":.01,"optimism":.05,"sui-network":.01,"solana":.02,"tron":.01}
CHAIN_NATIVE={"bsc":640,"base":3500,"eth":3500,"arbitrum":3500,"avax":35,"polygon_pos":.5,"optimism":3500,"sui-network":3.5,"solana":180,"tron":.3}
SIZE_FRAC=0.0025
MIN_BARS,IS_LEN,OOS_LEN=70,45,25; MIN_IS_TR,MIN_OOS_TR=5,4; NS=10

def _sma(x,n): return pd.Series(x).rolling(n).mean().to_numpy()
def _ema(x,n): return pd.Series(x).ewm(span=n,adjust=False).mean().to_numpy()
def _rsi(x,n):
    d=np.diff(x,prepend=x[0]); up=np.clip(d,0,None); dn=np.clip(-d,0,None)
    au=pd.Series(up).rolling(n).mean().to_numpy(); ad=pd.Series(dn).rolling(n).mean().to_numpy()
    rs=np.divide(au,ad,out=np.full_like(au,np.nan),where=ad>0); return 100-100/(1+rs)
def _rmax(x,n): return pd.Series(x).rolling(n).max().to_numpy()
def _rmin(x,n): return pd.Series(x).rolling(n).min().to_numpy()
def _atr(h,l,c,n):
    pc=np.roll(c,1); pc[0]=c[0]; tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).rolling(n).mean().to_numpy()
def shift(b):
    b=np.asarray(b,float); out=np.zeros(len(b),bool); out[1:]=np.nan_to_num(b[:-1])>0; return out

def f_sma(o,h,l,c,v,p): return shift(_sma(c,p['f'])>_sma(c,p['s']))
def f_ema(o,h,l,c,v,p): return shift(_ema(c,p['f'])>_ema(c,p['s']))
def f_pma(o,h,l,c,v,p): return shift(c>_sma(c,p['L']))
def f_dbo(o,h,l,c,v,p): return shift(c>=_rmax(h,p['L']))
def f_drev(o,h,l,c,v,p): return shift(c<=_rmin(l,p['L']))
def f_rsios(o,h,l,c,v,p): return shift(_rsi(c,p['L'])<p['thr'])
def f_rsimom(o,h,l,c,v,p): return shift(_rsi(c,p['L'])>p['thr'])
def f_brev(o,h,l,c,v,p): m=_sma(c,p['L']); sd=pd.Series(c).rolling(p['L']).std().to_numpy(); return shift(c<m-p['k']*sd)
def f_bbo(o,h,l,c,v,p): m=_sma(c,p['L']); sd=pd.Series(c).rolling(p['L']).std().to_numpy(); return shift(c>m+p['k']*sd)
def f_abo(o,h,l,c,v,p): return shift(c>np.roll(c,1)+p['k']*_atr(h,l,c,p['L']))
def f_rocm(o,h,l,c,v,p): r=pd.Series(c).pct_change(p['L']).to_numpy(); return shift(r>p['thr'])
def f_rocr(o,h,l,c,v,p): r=pd.Series(c).pct_change(p['L']).to_numpy(); return shift(r<-p['thr'])
def f_volbo(o,h,l,c,v,p): vm=_sma(v,p['L']); return shift((v>p['k']*vm)&(c>np.roll(c,1)))
def f_cdn(o,h,l,c,v,p): dn=(np.diff(c,prepend=c[0])<0).astype(int); return shift(pd.Series(dn).rolling(p['L']).sum().to_numpy()>=p['L'])
def f_cup(o,h,l,c,v,p): up=(np.diff(c,prepend=c[0])>0).astype(int); return shift(pd.Series(up).rolling(p['L']).sum().to_numpy()>=p['L'])
def f_pull(o,h,l,c,v,p): m=_sma(c,p['L']); rising=m>np.roll(m,3); return shift(rising&(c<m)&(c>np.roll(c,1)))

FAMS={"sma_cross":(f_sma,lambda:{"f":int(rng.choice([3,5,8,10])),"s":int(rng.choice([20,30,50]))}),
"ema_cross":(f_ema,lambda:{"f":int(rng.choice([3,5,8,12])),"s":int(rng.choice([21,34,50]))}),
"price_above_ma":(f_pma,lambda:{"L":int(rng.choice([10,20,30,50]))}),
"donchian_bo":(f_dbo,lambda:{"L":int(rng.choice([10,20,30,55]))}),
"donchian_rev":(f_drev,lambda:{"L":int(rng.choice([10,20,30]))}),
"rsi_os":(f_rsios,lambda:{"L":int(rng.choice([7,14,21])),"thr":float(rng.choice([20,25,30,35]))}),
"rsi_mom":(f_rsimom,lambda:{"L":int(rng.choice([7,14,21])),"thr":float(rng.choice([55,60,65,70]))}),
"boll_rev":(f_brev,lambda:{"L":int(rng.choice([14,20,30])),"k":float(rng.choice([1.5,2.0,2.5]))}),
"boll_bo":(f_bbo,lambda:{"L":int(rng.choice([14,20,30])),"k":float(rng.choice([1.0,1.5,2.0]))}),
"atr_bo":(f_abo,lambda:{"L":int(rng.choice([10,14,20])),"k":float(rng.choice([1.0,1.5,2.0,3.0]))}),
"roc_mom":(f_rocm,lambda:{"L":int(rng.choice([3,5,10,20])),"thr":float(rng.choice([.05,.1,.2,.3]))}),
"roc_rev":(f_rocr,lambda:{"L":int(rng.choice([3,5,10])),"thr":float(rng.choice([.1,.2,.3,.5]))}),
"vol_bo":(f_volbo,lambda:{"L":int(rng.choice([10,20,30])),"k":float(rng.choice([2,3,5]))}),
"consec_down":(f_cdn,lambda:{"L":int(rng.choice([2,3,4]))}),
"consec_up":(f_cup,lambda:{"L":int(rng.choice([2,3,4]))}),
"ma_pullback":(f_pull,lambda:{"L":int(rng.choice([10,20,30]))})}

def simulate(o,h,l,c,entry,tp,sl,mh,cost):
    n=len(c); i=0; out=[]
    while i<n-1:
        if entry[i] and c[i]>0:
            ep=c[i]; tpx=ep*(1+tp); spx=ep*(1-sl); end=min(i+mh,n-1); ret=None; j=i+1
            while j<=end:
                if l[j]<=spx: ret=-sl; break
                if h[j]>=tpx: ret=tp; break
                j+=1
            if ret is None: ret=c[min(j,end)]/ep-1.0
            out.append(ret-cost); i=j+1
        else: i+=1
    return out
def pf(tr):
    g=sum(t for t in tr if t>0); ls=-sum(t for t in tr if t<0)
    return g/ls if ls>0 else (np.inf if g>0 else 0.0)

def run_regime(cohort, skipN, null):
    """Run per-coin WFO over all families; drop first skipN bars; return list of per-eval dicts."""
    evals=[]
    for pair,(o,h,l,c,v,ch,res) in cohort.items():
        if skipN>0:
            if len(c)<=skipN+MIN_BARS: continue
            o,h,l,c,v=o[skipN:],h[skipN:],l[skipN:],c[skipN:],v[skipN:]
        n=len(c)
        if n<MIN_BARS: continue
        if null:
            perm=rng.permutation(n); o,h,l,c,v=o[perm],h[perm],l[perm],c[perm],v[perm]
        cost=coin_cost(ch,res)
        starts=list(range(0,n-IS_LEN-OOS_LEN+1,OOS_LEN))
        if not starts: continue
        for fname,(fn,samp) in FAMS.items():
            oos_all=[]
            for st in starts:
                isl=slice(st,st+IS_LEN); osl=slice(st+IS_LEN,st+IS_LEN+OOS_LEN); best=None
                for _ in range(NS):
                    p=samp(); tp=float(rng.choice([.15,.3,.5,1.0])); sl=float(rng.choice([.1,.2,.3])); mh=int(rng.choice([5,10,20,40]))
                    sig=fn(o,h,l,c,v,p)
                    is_tr=simulate(o[isl],h[isl],l[isl],c[isl],sig[isl],tp,sl,mh,cost)
                    if len(is_tr)<MIN_IS_TR: continue
                    s=pf(is_tr)
                    if best is None or s>best[0]: best=(s,p,tp,sl,mh)
                if best is None: continue
                _,p,tp,sl,mh=best; sig=fn(o,h,l,c,v,p)
                oos_all+=simulate(o[osl],h[osl],l[osl],c[osl],sig[osl],tp,sl,mh,cost)
            if len(oos_all)>=MIN_OOS_TR:
                evals.append({"coin":pair,"family":fname,"oos_pf":pf(oos_all),
                              "oos_net":sum(oos_all),"oos_n":len(oos_all),
                              "avg":sum(oos_all)/len(oos_all)})
    return pd.DataFrame(evals)

def coin_cost(ch,res):
    if not res: return .05
    size=max(50.,SIZE_FRAC*res)
    return round_trip_cost_frac(size,res,dex="uniswap",chain=ch,gas_usd=CHAIN_GAS.get(ch,.05),native_usd=CHAIN_NATIVE.get(ch,100))

def load_cohort(max_age=150,diff_tol=15):
    univ={}
    for l in open(DATA/"_mega_universe.jsonl"):
        r=json.loads(l); univ[r["pair"]]=r
    files=glob.glob(f"{DATA}/ohlcv_gt/mega_*_day.parquet")
    df=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True).dropna(subset=["close","high","low","open"])
    df=df[(df.close>0)&(df.high>0)&(df.low>0)].sort_values(["pair_address","ts"])
    maxts=df.ts.max(); out={}
    for pair,g in df.groupby("pair_address"):
        u=univ.get(pair)
        if u is None: continue
        a=u.get("age_days")
        if a is None or a<0 or a>=max_age: continue
        first=g.ts.min(); impl=(maxts-first)/86400.0
        if abs(impl-a)>diff_tol: continue
        if len(g)<MIN_BARS: continue
        out[pair]=(g.open.to_numpy(float),g.high.to_numpy(float),g.low.to_numpy(float),
                   g.close.to_numpy(float),g.volume.fillna(0).to_numpy(float),
                   u.get("chain","bsc"),u.get("reserve_usd") or 0.0)
    return out

if __name__=="__main__":
    t0=time.time()
    cohort=load_cohort()
    print(f"cohort (launch-captured, >= {MIN_BARS} bars): {len(cohort)} pools",flush=True)
    rows=[]
    for skipN in [0,3,7,14,21]:
        for null in [False,True]:
            rng2=np.random.default_rng(SEED)   # reset for comparability across regimes
            globals()['rng']=rng2
            df=run_regime(cohort,skipN,null)
            if len(df)==0:
                print(f"skipN={skipN} null={null}: no evals"); continue
            apf=df.oos_pf.replace(np.inf,np.nan)
            row={"skipN":skipN,"null":null,"n_eval":len(df),
                 "median_oos_pf":apf.median(),"mean_oos_pf_cap10":apf.clip(upper=10).mean(),
                 "pct_pf_gt1":100*(df.oos_pf>1).mean(),
                 "median_avg_ret":df.avg.median(),"mean_avg_ret":df.avg.mean()}
            rows.append(row)
            print(f"skipN={skipN:2d} null={int(null)}: n={len(df):5d} medPF={row['median_oos_pf']:.3f} "
                  f"meanPF={row['mean_oos_pf_cap10']:.3f} %PF>1={row['pct_pf_gt1']:.1f}% "
                  f"medAvgRet={row['median_avg_ret']:+.4f} meanAvgRet={row['mean_avg_ret']:+.4f}",flush=True)
    R=pd.DataFrame(rows); R.to_csv(OUT/"age_filter_results.csv",index=False)
    print(f"\nsaved -> {OUT}/age_filter_results.csv  ({time.time()-t0:.0f}s)")
