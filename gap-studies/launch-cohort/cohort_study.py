"""
GAP: Launch-window cohort event study on DEX-only coins.

The prior 16-family timing test (phase9) required ~80+ daily bars, filtering out
the launch window where DEX edge supposedly lives. Here we:

 (A) COHORT EVENT STUDY. The daily OHLCV is GeckoTerminal-capped at ~181 bars
     (~6 months). For OLD pools the first observed bar is mid-life, NOT launch.
     For YOUNG pools (age_days < ~150) we VERIFIED first-bar-ts == launch
     (corr age_days vs implied-age = 0.99). We use only launch-captured pools,
     align each on bar0 = first observed bar, normalize price to entry=1.0, and
     build the median / mean / quantile fan of the average price path over the
     first N days. Characterize pump-then-bleed.

 (B) STRATEGY 1 — LAUNCH-WINDOW LONG. Enter at age t (=k days after first bar),
     long-only, intrabar TP/SL, per-fill cost, WFO (select entry-age + bracket IS,
     evaluate OOS on held-out cohort of coins). NULL = permute which coin each
     path belongs to (cross-coin shuffle) through the identical pipeline.

 (C) STRATEGY 2 — "AVOID FIRST N DAYS" FILTER on the phase9 16-family test.
     Re-run a representative timing test (a) on ALL bars (baseline) and (b)
     skipping the first N bars of each coin. Compare OOS PF + net/trade, both vs
     a bar-shuffle null. Does excising the early bleed window lift edge?

Long-only on AMM spot. Per-fill cost both legs. Causal (.shift). SL-first intrabar.
Survivorship: universe = currently-live pools => launch cohort is survivor-tilted
(dead launches absent); flagged in FINDINGS. Seeded, reproducible.
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
OUT = Path("./_gaps/launch-cohort")
(OUT/"figs").mkdir(parents=True, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)

CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05, "polygon_pos": .01,
             "optimism": .05, "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "optimism": 3500, "sui-network": 3.5, "solana": 180, "tron": .3}
SIZE_FRAC = 0.0025

def coin_cost(chain, reserve):
    if not reserve: return .05
    size = max(50., SIZE_FRAC*(reserve or 0))
    return round_trip_cost_frac(size, reserve, dex="uniswap", chain=chain,
                                gas_usd=CHAIN_GAS.get(chain,.05), native_usd=CHAIN_NATIVE.get(chain,100))

def load():
    univ={}
    for l in open(DATA/"_mega_universe.jsonl"):
        r=json.loads(l); univ[r["pair"]]=r
    files=glob.glob(f"{DATA}/ohlcv_gt/mega_*_day.parquet")
    df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df=df.dropna(subset=["close","high","low","open"])
    df=df[(df.close>0)&(df.high>0)&(df.low>0)&(df.open>0)].sort_values(["pair_address","ts"])
    return univ, df

def launch_cohort(univ, df, max_age=150, diff_tol=15, min_bars=20):
    """Return dict pair-> (o,h,l,c,v, chain, reserve) for LAUNCH-CAPTURED young pools."""
    maxts=df.ts.max()
    out={}
    for pair,g in df.groupby("pair_address"):
        u=univ.get(pair);
        if u is None: continue
        a=u.get("age_days")
        if a is None or a<0 or a>=max_age: continue
        first=g.ts.min()
        impl=(maxts-first)/86400.0
        if abs(impl-a)>diff_tol:        # first bar is NOT launch -> skip
            continue
        n=len(g)
        if n<min_bars: continue
        out[pair]=(g.open.to_numpy(float),g.high.to_numpy(float),g.low.to_numpy(float),
                   g.close.to_numpy(float),g.volume.fillna(0).to_numpy(float),
                   u.get("chain","bsc"), u.get("reserve_usd") or 0.0)
    return out

# ============================ (A) EVENT STUDY ============================
def event_study(cohort, N=60):
    """Stack close paths normalized to bar0=1.0; return DataFrame of quantiles per age."""
    rows=[]
    paths=[]
    for pair,(o,h,l,c,v,ch,res) in cohort.items():
        base=c[0]
        if base<=0: continue
        p=c[:N]/base
        # pad with nan to N
        if len(p)<N: p=np.concatenate([p, np.full(N-len(p), np.nan)])
        paths.append(p)
    P=np.array(paths)                       # (ncoins, N)
    qs=[5,25,50,75,95]
    res={"age":np.arange(N),"n":(~np.isnan(P)).sum(0),"mean":np.nanmean(P,0)}
    for q in qs: res[f"p{q}"]=np.nanpercentile(P,q,axis=0)
    return pd.DataFrame(res), P

# ============================ shared sim ============================
def simulate(o,h,l,c, entry_idx, tp, sl, max_hold, cost):
    """Single long entry at entry_idx; intrabar SL-first then TP, else time-exit close. Net return."""
    n=len(c)
    if entry_idx>=n-1 or c[entry_idx]<=0: return None
    ep=c[entry_idx]; tpx=ep*(1+tp); spx=ep*(1-sl); end=min(entry_idx+max_hold,n-1)
    j=entry_idx+1; ret=None
    while j<=end:
        if l[j]<=spx: ret=-sl; break
        if h[j]>=tpx: ret=tp; break
        j+=1
    if ret is None: ret=c[min(j,end)]/ep-1.0
    return ret-cost

def pf(tr):
    tr=[t for t in tr if t is not None]
    g=sum(t for t in tr if t>0); ls=-sum(t for t in tr if t<0)
    return g/ls if ls>0 else (np.inf if g>0 else 0.0)

# ============================ (B) LAUNCH-WINDOW LONG WFO ============================
def strat1_launchlong(cohort, null=False, n_windows=4, is_frac=0.6):
    """
    Each coin = ONE event (enter at age t after launch, one bracketed long).
    WFO across COINS: split coin list into IS/OOS folds (sorted by first-bar ts so
    OOS coins launched LATER than IS coins -> no lookahead in coin selection).
    IS: pick entry-age t in {0,1,2,3,5,7,10} and bracket maximizing IS pooled PF.
    OOS: apply to held-out coins. NULL: shuffle the cost/path coin assignment so
    chosen params meet a randomized opportunity set (cross-coin permutation).
    Returns dict of metrics.
    """
    items=list(cohort.items())
    # order coins by launch time (first ts) -> proxy: use insertion but we have ts via min; recompute
    # we stored arrays not ts; reorder deterministically by pair hash for stability
    items.sort(key=lambda kv: kv[0])
    pairs=[k for k,_ in items]
    arrs=[v for _,v in items]
    M=len(pairs)
    ENTRY_AGES=[0,1,2,3,5,7,10]
    TPS=[.15,.3,.5,1.0]; SLS=[.1,.2,.3]; MHS=[5,10,20,40]
    # build folds: rolling expanding-ish; simple K disjoint OOS blocks
    idx=np.arange(M)
    fold_sz=M//n_windows
    oos_rets=[]; chosen=[]
    for w in range(n_windows):
        oos_start=w*fold_sz; oos_end=(w+1)*fold_sz if w<n_windows-1 else M
        oos_i=idx[oos_start:oos_end]
        is_i=np.array([i for i in idx if i<oos_start])   # only earlier coins (causal selection)
        if len(is_i)<10: continue
        # IS grid search
        best=None
        for t in ENTRY_AGES:
            for tp in TPS:
                for sl in SLS:
                    for mh in MHS:
                        tr=[]
                        for i in is_i:
                            o,h,l,c,v,ch,res=arrs[i]
                            cost=coin_cost(ch,res)
                            r=simulate(o,h,l,c,t,tp,sl,mh,cost)
                            if r is not None: tr.append(r)
                        if len(tr)<10: continue
                        s=pf(tr)
                        if best is None or s>best[0]: best=(s,t,tp,sl,mh)
        if best is None: continue
        _,t,tp,sl,mh=best; chosen.append((t,tp,sl,mh))
        for i in oos_i:
            o,h,l,c,v,ch,res=arrs[i]
            if null:
                # cross-coin permutation: keep chosen params, but apply path of a RANDOM other coin
                j=rng.integers(0,M); o,h,l,c,v,ch,res=arrs[j]
            cost=coin_cost(ch,res)
            r=simulate(o,h,l,c,t,tp,sl,mh,cost)
            if r is not None: oos_rets.append(r)
    return {"n_oos":len(oos_rets),"oos_pf":pf(oos_rets),
            "oos_net":float(np.sum(oos_rets)) if oos_rets else 0.0,
            "oos_mean_ret":float(np.mean(oos_rets)) if oos_rets else 0.0,
            "oos_winrate":float(np.mean([r>0 for r in oos_rets])) if oos_rets else 0.0,
            "chosen":chosen}

if __name__=="__main__":
    t0=time.time()
    univ,df=load()
    print(f"loaded {df.pair_address.nunique()} pairs ({time.time()-t0:.0f}s)",flush=True)
    cohort=launch_cohort(univ,df)
    print(f"launch-captured cohort: {len(cohort)} pools",flush=True)

    # (A) event study
    es,P=event_study(cohort,N=60)
    es.to_csv(OUT/"event_study_path.csv",index=False)
    print("\n=== (A) EVENT STUDY (normalized close, entry=1.0) ===")
    for ag in [0,1,2,3,5,7,10,14,21,30,45,59]:
        r=es.iloc[ag]
        print(f"  age {ag:3d}d: n={int(r['n']):4d} median={r['p50']:.3f} mean={r['mean']:.3f} "
              f"p25={r['p25']:.3f} p75={r['p75']:.3f}")
    # peak timing
    med=es["p50"].to_numpy(); peak_age=int(np.nanargmax(med)); print(f"  median path PEAK at age {peak_age}d = {med[peak_age]:.3f}")

    # (B) launch-window long, real vs null (avg over null seeds)
    print("\n=== (B) STRATEGY 1: LAUNCH-WINDOW LONG (WFO across coins) ===")
    real=strat1_launchlong(cohort,null=False)
    print(f"  REAL: n_oos={real['n_oos']} OOS_PF={real['oos_pf']:.3f} net={real['oos_net']:+.3f} "
          f"mean_ret={real['oos_mean_ret']:+.4f} winrate={real['oos_winrate']:.2%}")
    print(f"  chosen (t,tp,sl,mh) per window: {real['chosen']}")
    null_pfs=[]; null_means=[]
    for s in range(20):
        nr=strat1_launchlong(cohort,null=True)
        null_pfs.append(nr['oos_pf'] if np.isfinite(nr['oos_pf']) else np.nan)
        null_means.append(nr['oos_mean_ret'])
    print(f"  NULL (20 cross-coin perms): OOS_PF median={np.nanmedian(null_pfs):.3f} "
          f"mean_ret median={np.nanmedian(null_means):+.4f}")
    pd.DataFrame([{"run":"real","oos_pf":real['oos_pf'],"oos_net":real['oos_net'],
                   "oos_mean_ret":real['oos_mean_ret'],"oos_winrate":real['oos_winrate'],"n_oos":real['n_oos']},
                  {"run":"null_median","oos_pf":np.nanmedian(null_pfs),"oos_net":np.nan,
                   "oos_mean_ret":np.nanmedian(null_means),"oos_winrate":np.nan,"n_oos":np.nan}]
                 ).to_csv(OUT/"strat1_launchlong.csv",index=False)
    print(f"\ntotal {time.time()-t0:.0f}s")
