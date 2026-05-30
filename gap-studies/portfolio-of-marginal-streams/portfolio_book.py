"""GAP step 2b: ECONOMICALLY HONEST book-level portfolio + capacity.

Fixes the sparse-day renormalization that inflated Sharpe in portfolio.py. Here we run a
real DOLLAR book:
  - Book has AUM dollars. Target per-stream notional = w_i * AUM, capped at the stream's
    dollar capacity (0.25% of reserve). Capital not deployable sits in CASH (0 return).
  - Each trade i,t deploys notional_{i} on its entry date, realizes notional_i * net_ret on
    exit date. Daily book PnL = sum of realized trade PnL that day. Daily book return =
    PnL / AUM.  This is the honest, capacity-constrained return on capital.
  - WALK-FORWARD selection identical to portfolio.py (trailing-positive, monthly rebal,
    weights from trailing-only data).
  - Capacity sweep: AUM in {1e3 ... 1e8}; report Sharpe / ann_ret / deployed-fraction vs AUM.
  - REAL vs NULL side by side.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path("./_gaps/portfolio-of-marginal-streams")
ANN = 365.0
MIN_TRAIL_TR = 4
TRAIL_DAYS = 90
rng = np.random.default_rng(42)

def load(mode):
    T = pd.read_parquet(OUT/f"trades_{mode}.parquet")
    T["exit_date"] = pd.to_datetime(T.exit_ts, unit="s").dt.normalize()
    T["entry_date"] = pd.to_datetime(T.entry_ts, unit="s").dt.normalize()
    # dollar capacity per trade = the stream's dollar_size (0.25% reserve, floored at $50)
    T["cap"] = T.dollar_size.clip(lower=50.0)
    return T

def metrics(daily_ret):
    r = daily_ret.dropna()
    if len(r) < 5 or r.std() == 0:
        return dict(n=int(len(r)), ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, maxdd=np.nan, calmar=np.nan)
    mu = r.mean(); sd = r.std()
    ann_ret = mu*ANN; ann_vol = sd*np.sqrt(ANN); sharpe = (mu/sd)*np.sqrt(ANN)
    eq = (1+r).cumprod(); dd = eq/eq.cummax()-1; maxdd = dd.min()
    calmar = ann_ret/abs(maxdd) if maxdd < 0 else np.nan
    return dict(n=int(len(r)), ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, maxdd=maxdd, calmar=calmar)

def select_weights(T, weighting="ew", select_positive=True):
    """Walk-forward: for each monthly block, assign target weight per stream from trailing data.
    Returns DataFrame [stream, block_start, weight] of *intended* weights (pre-capacity)."""
    SD = T.groupby(["stream","exit_date"]).net_ret.mean().reset_index()
    wide = SD.pivot_table(index="exit_date", columns="stream", values="net_ret").sort_index()
    dates = wide.index
    if len(dates) < 30: return pd.DataFrame(columns=["stream","block_start","block_end","weight"])
    rebs = pd.date_range(dates.min(), dates.max(), freq="MS")
    rebs = [d for d in rebs if d > dates.min()+pd.Timedelta(days=TRAIL_DAYS)]
    rows = []
    for bi, rd in enumerate(rebs):
        blk_end = rebs[bi+1] if bi+1 < len(rebs) else dates.max()+pd.Timedelta(days=1)
        trail = wide.loc[(wide.index < rd) & (wide.index >= rd - pd.Timedelta(days=TRAIL_DAYS*4))]
        cnt = trail.notna().sum(); elig = cnt[cnt >= MIN_TRAIL_TR].index
        if select_positive and len(elig):
            tmean = trail[elig].mean(); elig = tmean[tmean > 0].index
        if len(elig) == 0: continue
        sub = trail[elig].fillna(0.0)
        if weighting == "ew":
            w = pd.Series(1.0/len(elig), index=elig)
        elif weighting == "invvol":
            vol = sub.std().replace(0, np.nan); iv = (1.0/vol).fillna(0.0)
            w = iv/iv.sum() if iv.sum() > 0 else pd.Series(1.0/len(elig), index=elig)
        else:
            w = pd.Series(1.0/len(elig), index=elig)
        for s, wv in w.items():
            rows.append((s, rd, blk_end, wv))
    return pd.DataFrame(rows, columns=["stream","block_start","block_end","weight"])

def book_returns(T, W, aum):
    """Given intended weights W and AUM, deploy capacity-capped notional per trade, realize PnL
    on exit_date. Daily book return = daily realized PnL / AUM."""
    if len(W) == 0: return pd.Series(dtype=float), np.nan
    # map each trade to the block whose [block_start, block_end) contains its ENTRY date
    # and to the stream's intended weight in that block.
    W = W.sort_values("block_start")
    # build interval index per stream
    pnl_by_day = {}
    deployed_sum = 0.0; intended_sum = 0.0
    Wg = {s: g.sort_values("block_start") for s, g in W.groupby("stream")}
    for s, g in T.groupby("stream"):
        if s not in Wg: continue
        blocks = Wg[s]
        bs = blocks.block_start.to_numpy(); be = blocks.block_end.to_numpy(); ww = blocks.weight.to_numpy()
        ent = g.entry_date.to_numpy(); ex = g.exit_date.to_numpy()
        ret = g.net_ret.to_numpy(); cap = g.cap.to_numpy()
        # for each trade find its block by entry date
        idx = np.searchsorted(bs, ent, side="right") - 1
        for k in range(len(g)):
            bi = idx[k]
            if bi < 0: continue
            if ent[k] < bs[bi] or ent[k] >= be[bi]: continue
            target = ww[bi]*aum
            notional = min(target, cap[k])      # capacity cap
            intended_sum += target; deployed_sum += notional
            pnl = notional*ret[k]
            d = ex[k]
            pnl_by_day[d] = pnl_by_day.get(d, 0.0)+pnl
    if not pnl_by_day: return pd.Series(dtype=float), np.nan
    s = pd.Series(pnl_by_day).sort_index()
    # fill calendar gaps with 0 PnL (cash days)
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full, fill_value=0.0)
    daily_ret = s/aum
    deployed_frac = deployed_sum/intended_sum if intended_sum > 0 else np.nan
    return daily_ret, deployed_frac

def aggregate_capacity(T, W):
    """Max deployable capital if every stream sized to its cap with EW intended weights:
    at the EW weight 1/N, AUM saturates when AUM/N == median cap; aggregate cap = sum of caps."""
    caps = T.groupby("stream").cap.first()
    return dict(n_streams=len(caps), total_cap=caps.sum(), median_cap=caps.median(), mean_cap=caps.mean())

def main():
    summary = []; cap_rows = []; series_store = {}
    aums = [1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8]
    for mode in ["real","null"]:
        T = load(mode)
        agg = aggregate_capacity(T, None)
        print(f"[{mode}] aggregate capacity: n_streams={agg['n_streams']} total_cap=${agg['total_cap']:,.0f} "
              f"median_cap=${agg['median_cap']:,.0f} mean_cap=${agg['mean_cap']:,.0f}")
        for wt, sp, tag in [("ew",True,"ew_pos"),("ew",False,"ew_all"),("invvol",True,"invvol_pos")]:
            W = select_weights(T, weighting=wt, select_positive=sp)
            # reference AUM small enough to almost never hit caps:
            for aum in aums:
                dr, dep = book_returns(T, W, aum)
                m = metrics(dr); m.update(dict(mode=mode, config=tag, aum=aum, deployed_frac=dep))
                cap_rows.append(m)
                if abs(aum-1e4) < 1: series_store[f"{mode}|{tag}"] = dr   # store $10k book curve
    C = pd.DataFrame(cap_rows)
    keep = ["mode","config","aum","n","ann_ret","ann_vol","sharpe","maxdd","calmar","deployed_frac"]
    C = C[keep]
    C.to_csv(OUT/"capacity_sweep.csv", index=False)
    pd.set_option("display.width",240); pd.set_option("display.max_columns",20); pd.set_option("display.max_rows",200)
    print("\n=== CAPACITY SWEEP (book-level, capacity-capped, net of cost) ===")
    print(C.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sdf = pd.DataFrame(series_store); sdf.to_parquet(OUT/"book_daily_10k.parquet")
    print("\nsaved capacity_sweep.csv + book_daily_10k.parquet")

if __name__ == "__main__":
    main()
