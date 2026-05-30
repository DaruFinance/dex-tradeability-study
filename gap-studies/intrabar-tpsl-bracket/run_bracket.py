"""GAP: intrabar TP/SL resolution convention. phase9's simulate() assumes SL-first when a
bar's range spans BOTH the TP and SL levels. For these huge-intrabar-range DEX coins that is a
strongly conservative choice that could mask a breakout/trend edge.

We re-run the phase9 per-coin WFO under THREE intrabar resolution rules for spanning bars:
  (a) sl_first    -- current/conservative: assume SL hit first  (lower bound on returns)
  (b) tp_first    -- optimistic: assume TP hit first            (upper bound on returns)
  (c) split       -- proportional/coin-flip: 50/50 expectation over the two outcomes
We ALSO track the FRACTION of resolved trades that were "ambiguous" (bar spanned both),
which is exactly the lever the SL-first convention pulls.

Real vs NULL (bar-shuffle within coin) under the IDENTICAL pipeline for every rule.
Long-only AMM spot. Per-fill cost via project model. Causal signals (.shift). Disjoint WFO.

Usage: python3 run_bracket.py <max_coins> <ns> <tf> [null]
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
OUT  = Path("./_gaps/intrabar-tpsl-bracket")
MAXC = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
NS   = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TF   = sys.argv[3] if len(sys.argv) > 3 else "day"
NULL = (len(sys.argv) > 4 and sys.argv[4] == "null")
if TF == "hour": MIN_BARS, IS_LEN, OOS_LEN = 360, 240, 120
else:            MIN_BARS, IS_LEN, OOS_LEN = 80, 90, 45
MIN_IS_TR, MIN_OOS_TR = 5, 4
SIZE_FRAC = 0.0025
CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05, "polygon_pos": .01,
             "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "sui-network": 3.5, "solana": 180, "tron": .3}
RULES = ["sl_first", "tp_first", "split"]
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

def fam_sma_cross(o,h,l,c,v,p):  return shift(_sma(c,p['f'])>_sma(c,p['s']))
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

def simulate(o,h,l,c, entry, tp, sl, max_hold, cost, rule):
    """Long-only intrabar TP/SL via high/low, with explicit spanning-bar resolution rule.
    Returns (net_trade_returns, n_spanning_bars_that_decided_an_exit).
    For 'split' the spanning-bar trade contributes the EXPECTED return 0.5*tp+0.5*(-sl)
    (mean over the coin-flip), so the full distribution mean is exact."""
    n=len(c); i=0; out=[]; span=0
    while i < n-1:
        if entry[i] and c[i]>0:
            ep=c[i]; tpx=ep*(1+tp); spx=ep*(1-sl); end=min(i+max_hold, n-1); ret=None
            j=i+1
            while j<=end:
                hit_sl = l[j]<=spx; hit_tp = h[j]>=tpx
                if hit_sl and hit_tp:               # AMBIGUOUS bar spans both
                    span+=1
                    if rule=="sl_first":   ret=-sl
                    elif rule=="tp_first": ret=tp
                    else:                  ret=0.5*tp+0.5*(-sl)   # split / E[outcome]
                    break
                if hit_sl: ret=-sl; break
                if hit_tp: ret=tp;  break
                j+=1
            if ret is None: ret=c[min(j,end)]/ep-1.0
            out.append(ret-cost); i=j+1
        else: i+=1
    return out, span

def pf(tr):
    g=sum(t for t in tr if t>0); ls=-sum(t for t in tr if t<0)
    return g/ls if ls>0 else (np.inf if g>0 else 0.0)

def main():
    univ={}
    for ln in open(DATA/"_mega_universe.jsonl"): r=json.loads(ln); univ[r["pair"]]=r
    files=glob.glob(f"{DATA}/ohlcv_gt/mega_*_{TF}.parquet")
    df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["close","high","low"])
    df=df[df.close>0].sort_values(["pair_address","ts"])
    coins=set([p for p,_ in df.groupby("pair_address")][:MAXC])
    print(f"{len(coins)} coins, {len(FAMILIES)} fams, NS={NS}, tf={TF}, NULL={NULL}, rules={RULES}", flush=True)

    t0=time.time()
    # per-rule accumulators of full-distribution evals
    evals={r:[] for r in RULES}
    span_tot=0; trade_tot=0; n_eval=0
    for ci,(pair,g) in enumerate(df.groupby("pair_address")):
        if pair not in coins: continue
        o,h,l,c,v = (g.open.to_numpy(float),g.high.to_numpy(float),g.low.to_numpy(float),
                     g.close.to_numpy(float),g.volume.fillna(0).to_numpy(float))
        n=len(c)
        if n<MIN_BARS: continue
        if NULL:
            perm=rng.permutation(n); o,h,l,c,v=o[perm],h[perm],l[perm],c[perm],v[perm]
        u=univ.get(pair,{}); res=u.get("reserve_usd") or 0; ch=u.get("chain","bsc")
        size=max(50.,SIZE_FRAC*res)
        cost=round_trip_cost_frac(size,res,dex="uniswap",chain=ch,gas_usd=CHAIN_GAS.get(ch,.05),native_usd=CHAIN_NATIVE.get(ch,100)) if res else .05
        starts=list(range(0, n-IS_LEN-OOS_LEN+1, OOS_LEN))
        if not starts: continue
        for fname,(fn,sampler) in FAMILIES.items():
            n_eval+=1
            # IS selection done ONCE (rule does not change which params look best much; to keep the
            # three rules strictly comparable we select IS params with the SAME conservative sl_first
            # rule, then evaluate OOS under each of the three rules). This isolates the OOS effect of
            # the convention without letting selection chase the optimistic bound (which would be a leak
            # of the rule into selection).
            oos={r:[] for r in RULES}
            for st in starts:
                isl=slice(st,st+IS_LEN); osl=slice(st+IS_LEN,st+IS_LEN+OOS_LEN)
                best=None
                for _ in range(NS):
                    p=sampler(); tp=float(rng.choice([.15,.3,.5,1.0])); sl=float(rng.choice([.1,.2,.3])); mh=int(rng.choice([5,10,20,40]))
                    sig=fn(o,h,l,c,v,p)
                    is_tr,_=simulate(o[isl],h[isl],l[isl],c[isl], sig[isl], tp,sl,mh, cost, "sl_first")
                    if len(is_tr)<MIN_IS_TR: continue
                    s=pf(is_tr)
                    if best is None or s>best[0]: best=(s,p,tp,sl,mh)
                if best is None: continue
                _,p,tp,sl,mh=best; sig=fn(o,h,l,c,v,p)
                for r in RULES:
                    tr,sp=simulate(o[osl],h[osl],l[osl],c[osl], sig[osl], tp,sl,mh, cost, r)
                    oos[r]+=tr
                    if r=="sl_first": span_tot+=sp; trade_tot+=len(tr)
            if len(oos["sl_first"])>=MIN_OOS_TR:
                for r in RULES:
                    tr=oos[r]; P=pf(tr); net=sum(tr)
                    evals[r].append({"coin":pair,"chain":ch,"family":fname,"oos_pf":P,
                                     "oos_net":net,"oos_n":len(tr),"avg_per_trade":net/len(tr) if tr else 0.})
        if (ci+1)%300==0:
            print(f"  [{ci+1}] n_eval={n_eval} span/trade={span_tot}/{trade_tot} ({time.time()-t0:.0f}s)", flush=True)

    tag = "null" if NULL else "real"
    rows=[]
    for r in RULES:
        A=pd.DataFrame(evals[r])
        A.to_csv(OUT/f"evals_{tag}_{TF}_{r}.csv", index=False)
        apf=A.oos_pf.replace(np.inf,np.nan)
        nsurv=int(((A.oos_pf>1)&(A.oos_net>0)).sum())
        rows.append({"tag":tag,"tf":TF,"rule":r,"n_eval":len(A),
                     "median_oos_pf":apf.median(),"mean_oos_pf_cap10":apf.clip(upper=10).mean(),
                     "pct_pf_gt1":100*(A.oos_pf>1).mean(),
                     "median_net_per_trade":A.avg_per_trade.median(),
                     "mean_net_per_trade":A.avg_per_trade.mean(),
                     "n_survivors":nsurv})
    S=pd.DataFrame(rows)
    S["span_frac"]=span_tot/trade_tot if trade_tot else np.nan
    S.to_csv(OUT/f"summary_{tag}_{TF}.csv", index=False)
    print(f"\n=== {tag} {TF} ===  spanning-bar fraction (of exits) = {span_tot}/{trade_tot} = {span_tot/max(1,trade_tot):.3f}")
    print(S.to_string(index=False))

if __name__=="__main__":
    main()
