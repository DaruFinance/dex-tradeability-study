"""GAP step 2: WALK-FORWARD portfolio construction + capacity from the per-trade streams.

Builds a daily portfolio from the time-stamped OOS trades of every (coin,family) stream.
Methodology (no lookahead):
  - Each trade's net return is realized on its exit date. A stream's daily return =
    mean net return of trades exiting that day (capital allocated to that stream that day).
  - Calendar split into monthly rebalance blocks. At rebalance t, eligible streams = those
    with >=MIN_TRAIL_TR trades whose EXIT date < t (strictly past). Weights computed ONLY
    from trailing data. Applied to the next block's realized returns.
  - Selection rule (walk-forward): keep streams with trailing PF>1 (positive past), else
    'all-eligible' variant. Compare to NULL run identically.
  - Weighting: equal-weight, inverse-vol, equal-risk-contribution (ERC), portfolio vol-target.
  - Metrics: ann return, ann vol, Sharpe (sqrt(365)), MaxDD, Calmar, turnover. Net of per-fill cost
    (already in net_ret). Compare REAL vs NULL side by side.
  - CAPACITY: each trade can deploy <= dollar_size (0.25% of reserve). Aggregate deployable $ and
    show Sharpe vs AUM as you scale notional past per-stream caps (excess capital sits in cash).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path("./_gaps/portfolio-of-marginal-streams")
ANN = 365.0
MIN_TRAIL_TR = 4         # need this many past trades to size a stream
SEED = 42
rng = np.random.default_rng(SEED)

def load(mode):
    T = pd.read_parquet(OUT/f"trades_{mode}.parquet")
    M = pd.read_parquet(OUT/f"streams_{mode}.parquet")
    T["exit_date"] = pd.to_datetime(T.exit_ts, unit="s").dt.normalize()
    T["entry_date"] = pd.to_datetime(T.entry_ts, unit="s").dt.normalize()
    return T, M

def stream_daily(T):
    """daily realized return per stream = mean net_ret of trades exiting that day."""
    g = T.groupby(["stream","exit_date"]).net_ret.mean().reset_index()
    return g

def metrics(daily_ret):
    """daily_ret: pd.Series indexed by date (portfolio daily return, fraction)."""
    r = daily_ret.dropna()
    if len(r) < 5 or r.std() == 0:
        return dict(n=len(r), ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, maxdd=np.nan, calmar=np.nan)
    mu = r.mean(); sd = r.std()
    ann_ret = mu * ANN
    ann_vol = sd * np.sqrt(ANN)
    sharpe = (mu / sd) * np.sqrt(ANN) if sd > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    calmar = ann_ret / abs(maxdd) if maxdd < 0 else np.nan
    return dict(n=len(r), ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, maxdd=maxdd, calmar=calmar)

def erc_weights(cov, iters=200):
    """equal risk contribution via simple fixed-point on long-only weights."""
    n = cov.shape[0]
    w = np.ones(n)/n
    for _ in range(iters):
        mrc = cov @ w
        mrc = np.where(mrc <= 1e-12, 1e-12, mrc)
        w_new = 1.0 / mrc
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < 1e-8: w = w_new; break
        w = w_new
    return w

def build_portfolio(SD, weighting="ew", select_positive=True, vol_target=None,
                    rebal="MS", trail_days=90):
    """SD: long stream-daily frame [stream, exit_date, net_ret].
    Returns (portfolio daily return series, turnover, avg_n_streams)."""
    wide = SD.pivot_table(index="exit_date", columns="stream", values="net_ret")
    wide = wide.sort_index()
    dates = wide.index
    if len(dates) < 30:
        return pd.Series(dtype=float), np.nan, np.nan
    # rebalance dates
    rebs = pd.date_range(dates.min(), dates.max(), freq=rebal)
    rebs = [d for d in rebs if d > dates.min() + pd.Timedelta(days=trail_days)]
    port = pd.Series(0.0, index=dates)
    prev_w = None
    turn = []
    nstreams_log = []
    # assign each date to the most recent rebalance block
    for bi, rd in enumerate(rebs):
        blk_end = rebs[bi+1] if bi+1 < len(rebs) else dates.max() + pd.Timedelta(days=1)
        trail = wide.loc[(wide.index < rd) & (wide.index >= rd - pd.Timedelta(days=trail_days*4))]
        # eligible streams: enough trailing observations
        cnt = trail.notna().sum()
        elig = cnt[cnt >= MIN_TRAIL_TR].index
        if select_positive and len(elig):
            # trailing PF>1 == trailing sum>0 proxy: use mean>0 of realized returns
            tmean = trail[elig].mean()
            elig = tmean[tmean > 0].index
        if len(elig) == 0:
            continue
        sub = trail[elig].fillna(0.0)
        if weighting == "ew":
            w = np.ones(len(elig))/len(elig)
        elif weighting == "invvol":
            vol = sub.std().replace(0, np.nan)
            iv = (1.0/vol).fillna(0.0).to_numpy()
            w = iv/iv.sum() if iv.sum() > 0 else np.ones(len(elig))/len(elig)
        elif weighting == "erc":
            cov = sub.cov().to_numpy()
            cov = cov + np.eye(len(elig))*1e-9
            w = erc_weights(cov)
        else:
            raise ValueError(weighting)
        w = pd.Series(w, index=elig)
        # apply to block: portfolio daily return = sum_i w_i * stream_i daily realized return
        blk = wide.loc[(wide.index >= rd) & (wide.index < blk_end), elig]
        if len(blk):
            # weight only streams that traded that day, renormalize within active set per day
            blk_filled = blk.fillna(0.0)
            active = blk.notna()
            wmat = active.mul(w, axis=1)
            wsum = wmat.sum(axis=1).replace(0, np.nan)
            wnorm = wmat.div(wsum, axis=0).fillna(0.0)
            pr = (blk_filled * wnorm).sum(axis=1)
            port.loc[pr.index] = pr.values
        # turnover vs prev weights
        if prev_w is not None:
            allk = prev_w.index.union(w.index)
            t = (w.reindex(allk).fillna(0) - prev_w.reindex(allk).fillna(0)).abs().sum()
            turn.append(t)
        prev_w = w
        nstreams_log.append(len(elig))
    # vol target rescale (ex-ante, using trailing realized vol of the portfolio)
    if vol_target is not None:
        # scale each day by target/trailing_vol(prev 30d), shifted to avoid lookahead
        rv = port.rolling(30).std().shift(1)
        scale = (vol_target/np.sqrt(ANN)) / rv
        scale = scale.clip(upper=5.0).fillna(1.0)
        port = port * scale
    avg_n = np.mean(nstreams_log) if nstreams_log else np.nan
    avg_turn = np.mean(turn) if turn else np.nan
    return port, avg_turn, avg_n

def run_all(mode):
    T, M = load(mode)
    SD = stream_daily(T)
    rows = []
    configs = [
        ("ew_pos", "ew", True, None),
        ("ew_all", "ew", False, None),
        ("invvol_pos", "invvol", True, None),
        ("erc_pos", "erc", True, None),
        ("ew_pos_voltgt20", "ew", True, 0.20),
        ("invvol_pos_voltgt20", "invvol", True, 0.20),
    ]
    series = {}
    for name, wt, sp, vt in configs:
        port, turn, navg = build_portfolio(SD, weighting=wt, select_positive=sp, vol_target=vt)
        m = metrics(port)
        m.update(dict(config=name, mode=mode, turnover=turn, avg_streams=navg))
        rows.append(m); series[name] = port
    return pd.DataFrame(rows), series, T, M

if __name__ == "__main__":
    out_rows = []
    all_series = {}
    bundle = {}
    for mode in ["real","null"]:
        df, series, T, M = run_all(mode)
        out_rows.append(df)
        for k,v in series.items(): all_series[f"{mode}|{k}"] = v
        bundle[mode] = (T,M)
    R = pd.concat(out_rows, ignore_index=True)
    cols = ["mode","config","n","ann_ret","ann_vol","sharpe","maxdd","calmar","turnover","avg_streams"]
    R = R[cols]
    R.to_csv(OUT/"portfolio_metrics.csv", index=False)
    pd.set_option("display.width",200); pd.set_option("display.max_columns",20)
    print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # save the equity series for figures
    sdf = pd.DataFrame(all_series)
    sdf.to_parquet(OUT/"portfolio_daily_returns.parquet")
    print("\nsaved portfolio_metrics.csv + portfolio_daily_returns.parquet")
