"""
GAP: Cost-model depth-assumption sensitivity for the T_DEX timing study.

The phase9 engine prices each trade with a SINGLE scalar round-trip cost computed
under the v2 constant-product assumption that executable single-sided depth = reserve_usd/2.
For Uniswap/Pancake v3 (concentrated liquidity), depth NEAR MID is a *fraction* f of TVL,
so true slippage can be far larger. Also, the bar close is assumed executable at mid,
which is false for thin coins with a single stale closing swap (a "stale-close" penalty).

We re-run the IDENTICAL phase9 WFO pipeline (16 long-only timing families, intrabar TP/SL,
disjoint IS/OOS, in-sample param selection) under a grid of cost models:
  - depth fraction f in {0.05, 0.20, 0.50(=v2 baseline), 1.00}  (one_sided = f*reserve)
  - stale-close penalty in {0, 30, 100} bp per leg (additive)
  - PLUS a continuous cost-MULTIPLIER sweep to locate the break-even cost where the
    full-distribution median edge (and buy&hold) flips sign.

Real vs NULL (bars permuted within coin) reported side by side at every grid point.
Long-only AMM spot. Per-fill cost on both legs. Seeded, reproducible.

Decomposition of the baseline cost (derived analytically, verified numerically):
  size S = SIZE_FRAC*R = 0.0025*R ; v2 single-sided depth = R/2
  buy impact/leg  = S/(R/2) = 2*SIZE_FRAC = 0.005 (50 bp)  -- CONSTANT in R
  Under v3 depth = f*R : buy impact/leg = S/(f*R) = SIZE_FRAC/f -> scales by (0.5/f)
  fee = swap_fee_bps (uniswap=30bp) per leg ; gas amortized over S (hurts tiny pools).
"""
from __future__ import annotations
import glob, json, sys, time, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import swap_fee_bps, gas_cost_usd

DATA = Path("./data")
OUT  = Path("./_gaps/cost_depth_sensitivity")
TF   = "day"
MAXC = int(os.environ.get("MAXC", "100000"))
NS   = 10                      # param samples per WFO window
MIN_BARS, IS_LEN, OOS_LEN = 80, 90, 45
MIN_IS_TR, MIN_OOS_TR = 5, 4
SIZE_FRAC = 0.0025
CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05, "polygon_pos": .01,
             "optimism": .05, "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "optimism": 3500, "sui-network": 3.5, "solana": 180, "tron": .3}
SEED = 42

# ---------- parameterized cost model ----------
def round_trip_cost_param(size_usd, reserve_usd, dex, chain, gas_usd, native_usd,
                          depth_frac=0.5, stale_bp_per_leg=0.0):
    """Round-trip fractional cost with a tunable single-sided depth fraction (v3 active-tick)
    and an additive per-leg stale-close penalty (bp). depth_frac=0.5 reproduces the v2 baseline."""
    if not reserve_usd or reserve_usd <= 0:
        return 0.05
    fee = swap_fee_bps(dex) / 1e4
    gas = gas_usd if gas_usd is not None else gas_cost_usd(chain, native_usd, dex)
    one_sided = depth_frac * reserve_usd
    ratio = size_usd / one_sided
    buy_impact = ratio
    sell_impact = ratio / (1.0 + ratio)
    stale = stale_bp_per_leg / 1e4
    buy_leg = fee + max(buy_impact, 2.0/1e4) + stale
    sell_leg = fee + max(sell_impact, 2.0/1e4) + stale
    gas_frac = (2 * gas / size_usd) if size_usd > 0 else 0.0
    return buy_leg + sell_leg + gas_frac

# ---------- indicators / families (verbatim from phase9) ----------
def _sma(x,n): return pd.Series(x).rolling(n).mean().to_numpy()
def _ema(x,n): return pd.Series(x).ewm(span=n,adjust=False).mean().to_numpy()
def _rsi(x,n):
    d=np.diff(x,prepend=x[0]); up=np.clip(d,0,None); dn=np.clip(-d,0,None)
    au=pd.Series(up).rolling(n).mean().to_numpy(); ad=pd.Series(dn).rolling(n).mean().to_numpy()
    rs=np.divide(au,ad,out=np.full_like(au,np.nan),where=ad>0); return 100-100/(1+rs)
def _roll_max(x,n): return pd.Series(x).rolling(n).max().to_numpy()
def _roll_min(x,n): return pd.Series(x).rolling(n).min().to_numpy()
def _atr(h,l,c,n):
    pc=np.roll(c,1); pc[0]=c[0]; tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).rolling(n).mean().to_numpy()
def shift(b):
    b=np.asarray(b,float); out=np.zeros(len(b),bool); out[1:]=np.nan_to_num(b[:-1])>0; return out

def fam_sma_cross(o,h,l,c,v,p): return shift(_sma(c,p['f'])>_sma(c,p['s']))
def fam_ema_cross(o,h,l,c,v,p): return shift(_ema(c,p['f'])>_ema(c,p['s']))
def fam_price_above_ma(o,h,l,c,v,p): return shift(c>_sma(c,p['L']))
def fam_donchian_bo(o,h,l,c,v,p): return shift(c>=_roll_max(h,p['L']))
def fam_donchian_rev(o,h,l,c,v,p): return shift(c<=_roll_min(l,p['L']))
def fam_rsi_os(o,h,l,c,v,p): return shift(_rsi(c,p['L'])<p['thr'])
def fam_rsi_mom(o,h,l,c,v,p): return shift(_rsi(c,p['L'])>p['thr'])
def fam_boll_rev(o,h,l,c,v,p): m=_sma(c,p['L']); sd=pd.Series(c).rolling(p['L']).std().to_numpy(); return shift(c<m-p['k']*sd)
def fam_boll_bo(o,h,l,c,v,p): m=_sma(c,p['L']); sd=pd.Series(c).rolling(p['L']).std().to_numpy(); return shift(c>m+p['k']*sd)
def fam_atr_bo(o,h,l,c,v,p): return shift(c>np.roll(c,1)+p['k']*_atr(h,l,c,p['L']))
def fam_roc_mom(o,h,l,c,v,p): r=pd.Series(c).pct_change(p['L']).to_numpy(); return shift(r>p['thr'])
def fam_roc_rev(o,h,l,c,v,p): r=pd.Series(c).pct_change(p['L']).to_numpy(); return shift(r<-p['thr'])
def fam_vol_bo(o,h,l,c,v,p): vm=_sma(v,p['L']); return shift((v>p['k']*vm)&(c>np.roll(c,1)))
def fam_consec_down(o,h,l,c,v,p):
    dn=(np.diff(c,prepend=c[0])<0).astype(int); run=pd.Series(dn).rolling(p['L']).sum().to_numpy(); return shift(run>=p['L'])
def fam_consec_up(o,h,l,c,v,p):
    upb=(np.diff(c,prepend=c[0])>0).astype(int); run=pd.Series(upb).rolling(p['L']).sum().to_numpy(); return shift(run>=p['L'])
def fam_ma_pullback(o,h,l,c,v,p):
    m=_sma(c,p['L']); rising=m>np.roll(m,3); return shift(rising&(c<m)&(c>np.roll(c,1)))

def make_families(rng):
    return {
        "sma_cross":(fam_sma_cross,lambda:{"f":int(rng.choice([3,5,8,10])),"s":int(rng.choice([20,30,50]))}),
        "ema_cross":(fam_ema_cross,lambda:{"f":int(rng.choice([3,5,8,12])),"s":int(rng.choice([21,34,50]))}),
        "price_above_ma":(fam_price_above_ma,lambda:{"L":int(rng.choice([10,20,30,50]))}),
        "donchian_bo":(fam_donchian_bo,lambda:{"L":int(rng.choice([10,20,30,55]))}),
        "donchian_rev":(fam_donchian_rev,lambda:{"L":int(rng.choice([10,20,30]))}),
        "rsi_os":(fam_rsi_os,lambda:{"L":int(rng.choice([7,14,21])),"thr":float(rng.choice([20,25,30,35]))}),
        "rsi_mom":(fam_rsi_mom,lambda:{"L":int(rng.choice([7,14,21])),"thr":float(rng.choice([55,60,65,70]))}),
        "boll_rev":(fam_boll_rev,lambda:{"L":int(rng.choice([14,20,30])),"k":float(rng.choice([1.5,2.0,2.5]))}),
        "boll_bo":(fam_boll_bo,lambda:{"L":int(rng.choice([14,20,30])),"k":float(rng.choice([1.0,1.5,2.0]))}),
        "atr_bo":(fam_atr_bo,lambda:{"L":int(rng.choice([10,14,20])),"k":float(rng.choice([1.0,1.5,2.0,3.0]))}),
        "roc_mom":(fam_roc_mom,lambda:{"L":int(rng.choice([3,5,10,20])),"thr":float(rng.choice([.05,.1,.2,.3]))}),
        "roc_rev":(fam_roc_rev,lambda:{"L":int(rng.choice([3,5,10])),"thr":float(rng.choice([.1,.2,.3,.5]))}),
        "vol_bo":(fam_vol_bo,lambda:{"L":int(rng.choice([10,20,30])),"k":float(rng.choice([2,3,5]))}),
        "consec_down":(fam_consec_down,lambda:{"L":int(rng.choice([2,3,4]))}),
        "consec_up":(fam_consec_up,lambda:{"L":int(rng.choice([2,3,4]))}),
        "ma_pullback":(fam_ma_pullback,lambda:{"L":int(rng.choice([10,20,30]))}),
    }

def simulate_gross(o,h,l,c,entry,tp,sl,max_hold):
    """Long-only, intrabar TP/SL (SL-first). Returns list of GROSS trade returns (no cost)."""
    n=len(c); i=0; out=[]
    while i<n-1:
        if entry[i] and c[i]>0:
            ep=c[i]; tpx=ep*(1+tp); spx=ep*(1-sl); end=min(i+max_hold,n-1); ret=None; j=i+1
            while j<=end:
                if l[j]<=spx: ret=-sl; break
                if h[j]>=tpx: ret=tp; break
                j+=1
            if ret is None: ret=c[min(j,end)]/ep-1.0
            out.append(ret); i=j+1
        else: i+=1
    return out

def pf(tr):
    g=sum(t for t in tr if t>0); ls=-sum(t for t in tr if t<0)
    return g/ls if ls>0 else (np.inf if g>0 else 0.0)

def main():
    null_mode = (len(sys.argv) > 1 and sys.argv[1] == "null")
    tag = "null" if null_mode else "real"
    rng = np.random.default_rng(SEED)
    families = make_families(rng)

    univ={}
    for l in open(DATA/"_mega_universe.jsonl"):
        r=json.loads(l); univ[r["pair"]]=r
    files=glob.glob(f"{DATA}/ohlcv_gt/mega_*_{TF}.parquet")
    df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["close","high","low"])
    df=df[df.close>0].sort_values(["pair_address","ts"])

    # For each (coin,family,window): collect GROSS oos trades + the per-coin baseline cost
    # components (so we can re-price under ANY cost model WITHOUT re-running the WFO).
    # KEY: param selection (IS) is done GROSS here. This is a deliberate, documented choice:
    # to isolate the COST sensitivity, the trade SET is held fixed across cost models, and we
    # vary only the per-trade cost subtracted. (Param selection is cost-insensitive at the
    # ~160bp level for the families that survive; verified separately for the baseline.)
    rows=[]                      # one row per (coin,family) with concatenated gross oos returns
    coins_done=0; t0=time.time()
    groups = list(df.groupby("pair_address"))
    for ci,(pair,g) in enumerate(groups):
        if coins_done>=MAXC: break
        o,h,l,c,v=(g.open.to_numpy(float),g.high.to_numpy(float),g.low.to_numpy(float),
                   g.close.to_numpy(float),g.volume.fillna(0).to_numpy(float))
        n=len(c)
        if n<MIN_BARS: continue
        if null_mode:
            perm=rng.permutation(n); o,h,l,c,v=o[perm],h[perm],l[perm],c[perm],v[perm]
        u=univ.get(pair,{}); res=u.get("reserve_usd") or 0; ch=u.get("chain","bsc")
        if not res: continue
        size=max(50.,SIZE_FRAC*res)
        gas=CHAIN_GAS.get(ch,.05); nat=CHAIN_NATIVE.get(ch,100)
        # buy&hold gross return over full history (path-independent reference)
        bh = c[-1]/c[0]-1.0 if c[0]>0 else np.nan
        starts=list(range(0,n-IS_LEN-OOS_LEN+1,OOS_LEN))
        if not starts: continue
        coins_done+=1
        # baseline cost (v2) used ONLY for IS param selection so the chosen trade set is realistic
        base_cost=round_trip_cost_param(size,res,"uniswap",ch,gas,nat,depth_frac=0.5,stale_bp_per_leg=0.0)
        for fname,(fn,sampler) in families.items():
            oos_gross=[]
            for st in starts:
                isl=slice(st,st+IS_LEN); osl=slice(st+IS_LEN,st+IS_LEN+OOS_LEN)
                best=None
                for _ in range(NS):
                    p=sampler(); tp=float(rng.choice([.15,.3,.5,1.0])); sl=float(rng.choice([.1,.2,.3])); mh=int(rng.choice([5,10,20,40]))
                    sig=fn(o,h,l,c,v,p)
                    is_g=simulate_gross(o[isl],h[isl],l[isl],c[isl],sig[isl],tp,sl,mh)
                    if len(is_g)<MIN_IS_TR: continue
                    is_net=[r-base_cost for r in is_g]
                    s=pf(is_net)
                    if best is None or s>best[0]: best=(s,p,tp,sl,mh)
                if best is None: continue
                _,p,tp,sl,mh=best; sig=fn(o,h,l,c,v,p)
                oos_gross+=simulate_gross(o[osl],h[osl],l[osl],c[osl],sig[osl],tp,sl,mh)
            if len(oos_gross)>=MIN_OOS_TR:
                rows.append({"coin":pair,"chain":ch,"family":fname,"reserve_usd":res,
                             "size_usd":size,"gas_usd":gas,"native_usd":nat,
                             "oos_gross":np.array(oos_gross),"n":len(oos_gross),"bh":bh})
        if (ci+1)%500==0:
            print(f"[{tag}] {ci+1}/{len(groups)} coins_done={coins_done} rows={len(rows)} ({time.time()-t0:.0f}s)",flush=True)
    print(f"[{tag}] collected {len(rows)} (coin,family) gross trade sets over {coins_done} coins ({time.time()-t0:.0f}s)",flush=True)

    # ---- cost grid: depth_frac x stale penalty ----
    depth_fracs=[0.05,0.20,0.50,1.00]
    stale_bps=[0.0,30.0,100.0]
    grid=[]
    for dfrac in depth_fracs:
        for stale in stale_bps:
            costs=np.array([round_trip_cost_param(r["size_usd"],r["reserve_usd"],"uniswap",
                              r["chain"],r["gas_usd"],r["native_usd"],depth_frac=dfrac,
                              stale_bp_per_leg=stale) for r in rows])
            pfs=[]; nets=[]; avg=[]
            for r,cst in zip(rows,costs):
                net=r["oos_gross"]-cst
                pfs.append(pf(net)); nets.append(net.sum()); avg.append(net.mean())
            pfs=np.array(pfs); nets=np.array(nets); avg=np.array(avg)
            apf=pd.Series(pfs).replace(np.inf,np.nan)
            grid.append({"tag":tag,"depth_frac":dfrac,"stale_bp":stale,
                         "median_cost_bp":np.median(costs)*1e4,"n":len(rows),
                         "median_oos_pf":apf.median(),"mean_oos_pf_cap10":apf.clip(upper=10).mean(),
                         "pct_pf_gt1":100*(pfs>1).mean(),"median_avg_ret_bp":np.median(avg)*1e4,
                         "mean_avg_ret_bp":np.mean(avg)*1e4,"n_survivors":int(((pfs>1)&(nets>0)).sum())})
    pd.DataFrame(grid).to_csv(OUT/f"grid_{tag}.csv",index=False)
    print(f"[{tag}] wrote grid_{tag}.csv")

    # ---- continuous cost-MULTIPLIER sweep (scale the v2 baseline cost) for break-even ----
    base_costs=np.array([round_trip_cost_param(r["size_usd"],r["reserve_usd"],"uniswap",
                          r["chain"],r["gas_usd"],r["native_usd"],depth_frac=0.5,stale_bp_per_leg=0.0)
                         for r in rows])
    mults=np.concatenate([np.array([0.0,0.1,0.25]),np.linspace(0.5,5.0,19)])
    sweep=[]
    for m in mults:
        pfs=[]; avg=[]
        for r,bc in zip(rows,base_costs):
            net=r["oos_gross"]-m*bc
            pfs.append(pf(net)); avg.append(net.mean())
        pfs=np.array(pfs); avg=np.array(avg)
        apf=pd.Series(pfs).replace(np.inf,np.nan)
        sweep.append({"tag":tag,"cost_mult":m,"median_cost_bp":np.median(m*base_costs)*1e4,
                      "median_oos_pf":apf.median(),"pct_pf_gt1":100*(pfs>1).mean(),
                      "median_avg_ret_bp":np.median(avg)*1e4,"mean_avg_ret_bp":np.mean(avg)*1e4})
    pd.DataFrame(sweep).to_csv(OUT/f"sweep_{tag}.csv",index=False)
    print(f"[{tag}] wrote sweep_{tag}.csv")

    # ---- buy&hold reference (gross; B&H pays cost once round-trip, cost-mult applies) ----
    bh_g=np.array([r["bh"] for r in rows if np.isfinite(r["bh"])])
    # collapse to per-coin (B&H is per-coin not per-family); dedupe by coin
    seen={};
    for r in rows:
        if np.isfinite(r["bh"]): seen[r["coin"]]=(r["bh"],round_trip_cost_param(r["size_usd"],r["reserve_usd"],"uniswap",r["chain"],r["gas_usd"],r["native_usd"],depth_frac=0.5))
    bh_arr=np.array([x[0] for x in seen.values()]); bh_cost=np.array([x[1] for x in seen.values()])
    bh_rows=[]
    for m in mults:
        net=bh_arr-m*bh_cost
        bh_rows.append({"tag":tag,"cost_mult":m,"n_coins":len(net),
                        "median_bh_net":np.median(net),"mean_bh_net":np.mean(net),
                        "pct_bh_net_pos":100*(net>0).mean()})
    pd.DataFrame(bh_rows).to_csv(OUT/f"buyhold_{tag}.csv",index=False)
    print(f"[{tag}] wrote buyhold_{tag}.csv  (median gross B&H={np.median(bh_arr):+.3f})")

if __name__=="__main__":
    main()
