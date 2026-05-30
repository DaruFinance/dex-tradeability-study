"""
GAP: Is "no edge OOS" partly an artifact of NOISE-FITTING in phase9's IS selection?

phase9 selects best-of-NS=12 params on as few as MIN_IS_TR=5 IS trades, then evaluates OOS.
Selecting a max-PF param set on 5 trades is selecting on pure noise => OOS degradation is
guaranteed by construction. This sweep asks: if we make IS selection LESS noisy (longer IS,
higher MIN_IS_TR, an objective that penalizes low trade count), does OOS edge appear?

If longer/stricter IS does NOT lift OOS above null, the negative result is robust.
If it DOES, we quantify by how much.

Methodology (identical pipeline for REAL and NULL):
  - Same 16 long-only families, same intrabar SL-first TP/SL sim, same per-fill cost model.
  - Causal signals (shift(1)). Disjoint IS/OOS WFO windows. Params chosen IS, evaluated OOS.
  - NULL = permute bars within each coin (destroys timing structure) through identical pipeline.
  - Sweep grid: IS_LEN x MIN_IS_TR x NS x objective.
  - Report FULL distribution of OOS PF and mean net-per-trade (all evals, not just winners),
    REAL vs NULL side by side, per setting.

Seeded, reproducible. Hourly timeframe (median ~866 bars => room to extend IS).
"""
from __future__ import annotations
import glob, json, sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
OUT  = Path("./_gaps/is-noise-fitting")
TF   = "hour"
MAXC_PER_CHAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 250   # cap coins/chain for runtime
SEED = 42
OOS_LEN = 120
MIN_OOS_TR = 4

CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05, "polygon_pos": .01,
             "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "sui-network": 3.5, "solana": 180, "tron": .3}
SIZE_FRAC = 0.0025

# ---- indicator helpers (causal) ----
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

def make_families(rng):
    return {
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

@njit(cache=True)
def _sim(o,h,l,c, entry, tp, sl, max_hold, cost):
    n=len(c); i=0; out=np.empty(n); k=0
    while i < n-1:
        if entry[i] and c[i]>0:
            ep=c[i]; tpx=ep*(1+tp); spx=ep*(1-sl); end=i+max_hold
            if end>n-1: end=n-1
            ret=2.0; j=i+1
            while j<=end:
                if l[j]<=spx: ret=-sl; break
                if h[j]>=tpx: ret=tp; break
                j+=1
            if ret==2.0:
                jj=j if j<end else end
                ret=c[jj]/ep-1.0
            out[k]=ret-cost; k+=1; i=j+1
        else: i+=1
    return out[:k]

def pf(tr):
    g=tr[tr>0].sum(); ls=-tr[tr<0].sum()
    return g/ls if ls>0 else (np.inf if g>0 else 0.0)

def is_objective(tr, kind):
    """IS selection score. tr = net trade returns (numpy)."""
    n=len(tr)
    P=pf(tr)
    if kind=="pf":
        return P
    if kind=="pf_pen":
        # shrink toward 1.0 by trade count: more trades => trust PF more. penalizes low-n picks.
        Pc = min(P, 10.0)
        w = n/(n+10.0)            # n=5 -> .33 weight, n=30 -> .75, n=120 -> .92
        return 1.0 + w*(Pc-1.0)
    if kind=="netpt":             # mean net per trade (a different, count-aware objective)
        return tr.mean()
    return P

def run_setting(df, coins_set, univ, IS_LEN, MIN_IS_TR, NS, obj, null):
    rng = np.random.default_rng(SEED)
    FAMILIES = make_families(rng)
    evals=[]   # one row per (coin,family) with OOS pooled stats
    for pair, g in df.groupby("pair_address"):
        if pair not in coins_set: continue
        o=g.open.to_numpy(float); h=g.high.to_numpy(float); l=g.low.to_numpy(float)
        c=g.close.to_numpy(float); v=g.volume.fillna(0).to_numpy(float)
        n=len(c)
        if n < IS_LEN+OOS_LEN: continue
        if null:
            perm=rng.permutation(n); o,h,l,c,v=o[perm],h[perm],l[perm],c[perm],v[perm]
        u=univ.get(pair,{}); res=u.get("reserve_usd") or 0; ch=u.get("chain","bsc")
        size=max(50.,SIZE_FRAC*res)
        cost=round_trip_cost_frac(size,res,dex="uniswap",chain=ch,gas_usd=CHAIN_GAS.get(ch,.05),
                                  native_usd=CHAIN_NATIVE.get(ch,100)) if res else .05
        starts=list(range(0, n-IS_LEN-OOS_LEN+1, OOS_LEN))
        if not starts: continue
        for fname,(fn,sampler) in FAMILIES.items():
            oos_all=[]
            for st in starts:
                isl=slice(st,st+IS_LEN); osl=slice(st+IS_LEN,st+IS_LEN+OOS_LEN)
                best=None
                for _ in range(NS):
                    p=sampler(); tp=float(rng.choice([.15,.3,.5,1.0])); sl=float(rng.choice([.1,.2,.3])); mh=int(rng.choice([5,10,20,40]))
                    sig=fn(o,h,l,c,v,p)
                    is_tr=_sim(o[isl],h[isl],l[isl],c[isl], sig[isl], tp,sl,mh, cost)
                    if len(is_tr)<MIN_IS_TR: continue
                    s=is_objective(is_tr, obj)
                    if best is None or s>best[0]: best=(s,p,tp,sl,mh)
                if best is None: continue
                _,p,tp,sl,mh=best; sig=fn(o,h,l,c,v,p)
                oos_all.append(_sim(o[osl],h[osl],l[osl],c[osl], sig[osl], tp,sl,mh, cost))
            if oos_all:
                oa=np.concatenate(oos_all)
                if len(oa)>=MIN_OOS_TR:
                    P=pf(oa); net=oa.sum()
                    evals.append({"coin":pair,"chain":ch,"family":fname,"oos_pf":P,
                                  "oos_n":len(oa),"oos_net":net,"avg_per_trade":net/len(oa)})
    return pd.DataFrame(evals)

def summarize(A, tag):
    if not len(A):
        return {"setting":tag,"n_eval":0}
    apf=A.oos_pf.replace(np.inf,np.nan)
    return {
        "setting":tag,
        "n_eval":len(A),
        "median_oos_pf":apf.median(),
        "mean_oos_pf_cap10":apf.clip(upper=10).mean(),
        "pct_pf_gt1":100*(A.oos_pf>1).mean(),
        "median_net_pt":A.avg_per_trade.median(),
        "mean_net_pt":A.avg_per_trade.mean(),
        "n_survivors":int(((A.oos_pf>1)&(A.oos_net>0)).sum()),
        "median_oos_n":A.oos_n.median(),
    }

def load_df(univ):
    files=glob.glob(f"{DATA}/ohlcv_gt/mega_*_{TF}.parquet")
    df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["close","high","low"])
    df=df[df.close>0].sort_values(["pair_address","ts"])
    LONGEST_IS=480
    sizes=df.groupby("pair_address").size()
    eligible=sizes[sizes>=LONGEST_IS+OOS_LEN].index
    df=df[df.pair_address.isin(eligible)]
    keep=[]
    for ch,gg in df.groupby("chain"):
        ps=sorted(gg.pair_address.unique())[:MAXC_PER_CHAIN]
        keep+=ps
    coins_set=set(keep)
    df=df[df.pair_address.isin(coins_set)].copy()
    return df, coins_set

def run_one(args):
    """Run ONE (setting, null) -> returns (summary_row, detail_df). For process pool."""
    name,ISL,MTR,NSx,obj,null = args
    univ={}
    for ln in open(DATA/"_mega_universe.jsonl"):
        r=json.loads(ln); univ[r["pair"]]=r
    df,coins_set=load_df(univ)
    A=run_setting(df, coins_set, univ, ISL, MTR, NSx, obj, null)
    s=summarize(A, name); s["kind"]="null" if null else "real"
    s["IS_LEN"]=ISL; s["MIN_IS_TR"]=MTR; s["NS"]=NSx; s["obj"]=obj
    tag=f"{name}{'_NULL' if null else '_REAL'}"
    if len(A): A.assign(setting=name, kind=("null" if null else "real")).to_parquet(OUT/f"_detail_{tag}.parquet", index=False)
    return s, tag

def main():
    t0=time.time()
    univ={}
    for ln in open(DATA/"_mega_universe.jsonl"):
        r=json.loads(ln); univ[r["pair"]]=r
    df,coins_set=load_df(univ)
    print(f"{len(coins_set)} coins (>= 600 bars, cap {MAXC_PER_CHAIN}/chain), tf={TF}", flush=True)
    print("chains:", df.groupby('chain').pair_address.nunique().to_dict(), flush=True)
    del df

    # ---- SWEEP GRID ----
    # baseline phase9-equivalent: IS=240, MIN_IS_TR=5, NS=12, obj=pf
    IS_grid    = [240, 360, 480]
    MINTR_grid = [5, 15, 30]
    NS_grid    = [12, 40]
    OBJ_grid   = ["pf", "pf_pen", "netpt"]

    rows=[]
    # full factorial is large; do a structured set:
    #  (A) baseline + each axis varied one-at-a-time, REAL and NULL
    #  (B) the "strictest" combo (long IS + high min_tr + high NS + pf_pen)
    settings=[]
    settings.append(("IS240_TR5_NS12_pf", 240,5,12,"pf"))          # baseline
    for x in IS_grid:    settings.append((f"IS{x}_TR5_NS12_pf", x,5,12,"pf"))
    for x in MINTR_grid: settings.append((f"IS240_TR{x}_NS12_pf", 240,x,12,"pf"))
    for x in NS_grid:    settings.append((f"IS240_TR5_NS{x}_pf", 240,5,x,"pf"))
    for x in OBJ_grid:   settings.append((f"IS240_TR5_NS12_{x}", 240,5,12,x))
    settings.append(("STRICT_IS480_TR30_NS40_pfpen", 480,30,40,"pf_pen"))
    settings.append(("STRICT_IS360_TR15_NS40_pfpen", 360,15,40,"pf_pen"))
    # dedup by name
    seen=set(); uniq=[]
    for s in settings:
        if s[0] in seen: continue
        seen.add(s[0]); uniq.append(s)

    jobs=[]
    for name,ISL,MTR,NSx,obj in uniq:
        for null in (False, True):
            jobs.append((name,ISL,MTR,NSx,obj,null))

    from multiprocessing import Pool
    with Pool(processes=min(24, len(jobs))) as pool:
        for s,tag in pool.imap_unordered(run_one, jobs):
            rows.append(s)
            print(f"  {tag:42} n={s.get('n_eval',0):5} medPF={s.get('median_oos_pf',float('nan')):.3f} "
                  f"%>1={s.get('pct_pf_gt1',float('nan')):.1f} netpt={s.get('median_net_pt',float('nan')):+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    R=pd.DataFrame(rows)
    R.to_csv(OUT/"sweep_results.csv", index=False)
    print("\nsaved", OUT/"sweep_results.csv", f"  total {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
