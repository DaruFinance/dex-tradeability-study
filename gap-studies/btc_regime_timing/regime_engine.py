"""
GAP: BTC risk-on/off REGIME-conditioned long-only timing on DEX-only alts.

Prior result: unconditional long-only price/volume timing has NO edge on DEX-only coins
(real worse than bar-shuffle null). This script asks: does that no-edge result HIDE a
regime (BTC uptrend / low-vol "risk-on") where long-only timing IS net-positive, masked by
averaging over risk-off bleed?

Method (adapts _research/phase9_percoin_wfo.py, daily timeframe):
  1. Build a CAUSAL BTC USD regime from the highest-liquidity WBTC/USDC pool close in
     mega_eth_day.parquet (pair 0x99ac...). Regime at bar i uses data <= i-1 (shifted).
       - trend regime:  BTC close[i-1] > SMA50(BTC)[i-1]    -> risk-ON  (uptrend)
       - vol   regime:  realized vol20(BTC)[i-1] < rolling-median(vol20)  -> risk-ON (low vol)
       - combined master switch: ON = uptrend AND lowvol
  2. Map each alt daily bar (by unix-day timestamp) to BTC regime at that day.
  3. Re-run 16 long-only families with INTRABAR TP/SL, per-fill cost, rolling WFO.
     Entry signal is ANDed with the regime mask -> we only take longs in the chosen regime.
  4. Variants run end-to-end through identical pipeline:
       UNCOND (baseline), RISK_ON_TREND, RISK_OFF_TREND, RISK_ON_LOWVOL, RISK_OFF_LOWVOL,
       MASTER_ON (uptrend & lowvol), MASTER_OFF.
  5. NULL: bars permuted within each coin (destroys timing) BUT regime mask is held on the
     real calendar then permuted with the bars too (so the null sees the same regime balance,
     just decorrelated from price path). Real edge must beat its own null.

Long-only on AMM spot. Per-fill cost both legs. SL-before-TP intrabar (conservative).
Seeded, reproducible.
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
OUT = Path("./_gaps/btc_regime_timing")
SEED = 42
MAXC = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
DO_NULL = (len(sys.argv) > 3 and sys.argv[3] == "null")
TF = "day"
MIN_BARS, IS_LEN, OOS_LEN = 80, 90, 45
MIN_IS_TR, MIN_OOS_TR = 5, 4
SIZE_FRAC = 0.0025
WBTC_POOL = "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35"  # WBTC/USDC 0.3% on eth, $28M reserve
CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05, "polygon_pos": .01,
             "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "sui-network": 3.5, "solana": 180, "tron": .3}
rng = np.random.default_rng(SEED)

# ---------- causal indicators ----------
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

def simulate(o,h,l,c, entry, tp, sl, max_hold, cost):
    n=len(c); i=0; out=[]
    while i < n-1:
        if entry[i] and c[i]>0:
            ep=c[i]; tpx=ep*(1+tp); spx=ep*(1-sl); end=min(i+max_hold, n-1); ret=None
            j=i+1
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

# ---------- BTC regime ----------
def build_btc_regimes():
    eth = pd.read_parquet(DATA/"ohlcv_gt/mega_eth_day.parquet")
    b = eth[eth.pair_address==WBTC_POOL].sort_values("ts")
    c = b.close.to_numpy(float); ts = b.ts.to_numpy()
    day = (ts // 86400).astype(np.int64)
    s = pd.Series(c)
    sma50 = s.rolling(50).mean().to_numpy()
    vol20 = s.pct_change().rolling(20).std().to_numpy()
    # causal: shift everything by 1 (regime for day d uses info up to d-1)
    c_sh = np.roll(c, 1); sma_sh = np.roll(sma50, 1); vol_sh = np.roll(vol20, 1)
    c_sh[0]=np.nan; sma_sh[0]=np.nan; vol_sh[0]=np.nan
    vmed = np.nanmedian(vol_sh)  # threshold from full BTC window (regime label, not a tradable param)
    uptrend = c_sh > sma_sh
    lowvol  = vol_sh < vmed
    valid   = ~np.isnan(c_sh) & ~np.isnan(sma_sh) & ~np.isnan(vol_sh)
    reg = {}
    for k,d in enumerate(day):
        if valid[k]:
            reg[int(d)] = {"uptrend": bool(uptrend[k]), "lowvol": bool(lowvol[k])}
    return reg, vmed, (uptrend[valid].mean(), lowvol[valid].mean())

VARIANTS = ["UNCOND","RISK_ON_TREND","RISK_OFF_TREND","RISK_ON_LOWVOL","RISK_OFF_LOWVOL","MASTER_ON","MASTER_OFF"]

def regime_mask(days, reg, variant):
    """Boolean mask over alt bars: is this bar's BTC-day in the desired regime?
    Bars with no BTC regime info (outside window or warmup) are False (no trade)."""
    if variant == "UNCOND":
        return np.ones(len(days), bool)
    m = np.zeros(len(days), bool)
    for i,d in enumerate(days):
        r = reg.get(int(d))
        if r is None: continue
        if variant=="RISK_ON_TREND":    m[i]=r["uptrend"]
        elif variant=="RISK_OFF_TREND": m[i]=not r["uptrend"]
        elif variant=="RISK_ON_LOWVOL": m[i]=r["lowvol"]
        elif variant=="RISK_OFF_LOWVOL":m[i]=not r["lowvol"]
        elif variant=="MASTER_ON":      m[i]=r["uptrend"] and r["lowvol"]
        elif variant=="MASTER_OFF":     m[i]=not (r["uptrend"] and r["lowvol"])
    return m

def main():
    reg, vmed, (fup, flv) = build_btc_regimes()
    print(f"BTC regime built: {len(reg)} regime-days, vol-thr={vmed:.4f}, frac uptrend={fup:.2f}, frac lowvol={flv:.2f}", flush=True)

    univ={}
    for ln in open(DATA/"_mega_universe.jsonl"):
        r=json.loads(ln); univ[r["pair"]]=r
    files=glob.glob(f"{DATA}/ohlcv_gt/mega_*_{TF}.parquet")
    df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["close","high","low"])
    df=df[df.close>0].sort_values(["pair_address","ts"])
    # exclude the BTC pool itself
    coins=[p for p,_ in df.groupby("pair_address") if p!=WBTC_POOL][:MAXC]
    coinset=set(coins)
    print(f"{len(coins)} alt coins, {len(FAMILIES)} families, NS={NS}, NULL={DO_NULL}", flush=True)

    t0=time.time()
    rows=[]  # per (coin,family,variant) OOS eval
    n_eval=0
    for ci,(pair,g) in enumerate(df.groupby("pair_address")):
        if pair not in coinset: continue
        o,h,l,c,v = (g.open.to_numpy(float),g.high.to_numpy(float),g.low.to_numpy(float),
                     g.close.to_numpy(float),g.volume.fillna(0).to_numpy(float))
        days=(g.ts.to_numpy()//86400).astype(np.int64)
        n=len(c)
        if n<MIN_BARS: continue
        # how much of this coin overlaps the BTC regime window?
        overlap = np.mean([int(d) in reg for d in days])
        if overlap < 0.3:  # need meaningful regime coverage to condition
            continue
        if DO_NULL:
            perm=rng.permutation(n)
            o,h,l,c,v=o[perm],h[perm],l[perm],c[perm],v[perm]
            days=days[perm]  # carry regime label with the shuffled bar -> same regime balance, decorrelated
        u=univ.get(pair,{}); res=u.get("reserve_usd") or 0; ch=u.get("chain","bsc")
        size=max(50.,SIZE_FRAC*res)
        cost=round_trip_cost_frac(size,res,dex="uniswap",chain=ch,gas_usd=CHAIN_GAS.get(ch,.05),native_usd=CHAIN_NATIVE.get(ch,100)) if res else .05
        # precompute regime masks per variant (over full coin series)
        masks={vr:regime_mask(days,reg,vr) for vr in VARIANTS}
        starts=list(range(0, n-IS_LEN-OOS_LEN+1, OOS_LEN))
        if not starts: continue
        for fname,(fn,sampler) in FAMILIES.items():
            # IS-tune ONCE per window on the UNCOND signal (regime is an overlay, not a tuned param);
            # then evaluate OOS for EACH variant with the same tuned params -> apples-to-apples.
            oos={vr:[] for vr in VARIANTS}
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
                for vr in VARIANTS:
                    e=sig & masks[vr]
                    oos[vr]+=simulate(o[osl],h[osl],l[osl],c[osl], e[osl], tp,sl,mh, cost)
            for vr in VARIANTS:
                tr=oos[vr]
                if len(tr)>=MIN_OOS_TR:
                    rows.append({"coin":pair,"chain":ch,"family":fname,"variant":vr,
                                 "oos_pf":pf(tr),"oos_n":len(tr),"oos_net":sum(tr),
                                 "avg_per_trade":sum(tr)/len(tr)})
                    n_eval+=1
        if (ci+1)%300==0:
            print(f"  [{ci+1}] rows={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    R=pd.DataFrame(rows)
    tag="null" if DO_NULL else "real"
    R.to_parquet(OUT/f"evals_{tag}.parquet", index=False)
    print(f"\nsaved {len(R)} rows -> evals_{tag}.parquet ({time.time()-t0:.0f}s)")
    # quick summary
    for vr in VARIANTS:
        s=R[R.variant==vr]
        if len(s):
            apf=s.oos_pf.replace(np.inf,np.nan)
            print(f"  {vr:16} n={len(s):5} medPF={apf.median():.3f} %PF>1={100*(s.oos_pf>1).mean():4.1f}% "
                  f"medNet/tr={s.avg_per_trade.median():+.4f} meanNet/tr={s.avg_per_trade.mean():+.4f}")

if __name__=="__main__":
    main()
