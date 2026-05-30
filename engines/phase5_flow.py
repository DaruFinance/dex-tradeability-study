"""Phase 5: FLOW archetype test, does on-chain taker buy/sell imbalance have OOS edge where
price/volume TA does not? Uses raw trade tapes (flow_trades dataset, with sides + reserves) that
GeckoTerminal cannot provide. This is the actual thesis (the D-series flow archetype).

Builds hourly bars per coin from trades: close (last price), buy_usd/sell_usd, imbalance =
(buy-sell)/total. Causal (.shift(1)). Archetypes: flow_follow (imbalance>>0 -> long, momentum of
flow) and flow_fade (imbalance<<0 -> long, reversal). Long-only, per-fill cost from per-bar reserve,
pooled calendar IS/OOS, per-coin strategies + decorrelation. Same honest WFO discipline as phase4.

Usage: python3 _research/phase5_flow.py <is_split> <min_is_trades_per_coin>
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
IS_SPLIT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
MIN_IS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SIZE_FRAC = 0.0025


def bars_from_trades():
    """Hourly bars per pair from flow_trades: close, buy/sell usd, reserve."""
    q = """SELECT pair_address, chain,
                  CAST(epoch(block_time) AS BIGINT)//3600 AS hr,
                  last(price_usd ORDER BY block_number, log_index) AS close,
                  sum(CASE WHEN side='buy' THEN amount_usd ELSE 0 END) AS buy_usd,
                  sum(CASE WHEN side='sell' THEN amount_usd ELSE 0 END) AS sell_usd,
                  last(reserve_usd ORDER BY block_number, log_index) AS reserve,
                  count(*) AS n
           FROM read_parquet(?, union_by_name=true)
           WHERE price_usd IS NOT NULL AND price_usd>0
           GROUP BY 1,2,3 ORDER BY 1,3"""
    return duckdb.execute(q, [f"{DATA}/flow_trades/**/*.parquet"]).df()


def simulate_t(c, ts, sig, mh, st, cost):
    n = len(c); i = 0; out = []
    while i < n - 1:
        if sig[i] and c[i] > 0:
            ep = c[i]; ex = min(i + mh, n - 1); sp = ep * (1 - st); j = i + 1
            while j <= ex and c[j] > sp: j += 1
            j = min(j, ex); out.append((ts[j], c[j] / ep - 1.0 - cost)); i = j + 1
        else: i += 1
    return out


def main():
    if not glob.glob(f"{DATA}/flow_trades/**/*.parquet", recursive=True):
        print("no flow_trades yet, raw pull still running."); return
    df = bars_from_trades()
    df["total"] = df.buy_usd + df.sell_usd
    df["imb"] = np.where(df.total > 0, (df.buy_usd - df.sell_usd) / df.total, 0.0)
    pools = []
    for pair, g in df.groupby("pair_address"):
        g = g.sort_values("hr")
        if len(g) < 80: continue
        reserve = g.reserve.dropna().median()
        size = max(50.0, SIZE_FRAC * (reserve or 0))
        cost = round_trip_cost_frac(size, reserve, dex="pancakeswap", chain=g.chain.iloc[0],
                                    gas_usd=0.20, native_usd=640.0) if reserve else 0.05
        pools.append({"pair": pair, "c": g.close.to_numpy(), "imb": g.imb.to_numpy(),
                      "ts": (g.hr.to_numpy() * 3600), "cost": cost})
    print(f"flow universe: {len(pools)} coins with >=80 hourly bars from raw tape")
    if len(pools) < 4:
        print("too few coins with flow data yet; rerun when raw pull has more."); return

    tmin = min(p["ts"].min() for p in pools); tmax = max(p["ts"].max() for p in pools)
    split = tmin + IS_SPLIT * (tmax - tmin)
    print(f"median rt_cost={np.median([p['cost'] for p in pools])*1e4:.0f}bp; IS/OOS split {IS_SPLIT:.0%}")

    # flow archetypes: imbalance z-score, follow vs fade
    def feat_z(imb, L):
        s = pd.Series(imb).shift(1)
        return ((s - s.rolling(L).mean()) / s.rolling(L).std()).to_numpy()
    combos = []
    for L in (6, 12, 24, 48):
        for thr in (0.5, 1.0, 1.5):
            for mode in ("follow", "fade"):
                for mh in (6, 12, 24):
                    for st in (0.15, 0.30):
                        combos.append((L, thr, mode, mh, st))
    print(f"{len(combos)} flow combos")

    strategies, allc = [], []
    for (L, thr, mode, mh, st) in combos:
        for p in pools:
            z = feat_z(p["imb"], L)
            sig = (z > thr) if mode == "follow" else (z < -thr)
            sig = sig & ~np.isnan(z)
            is_tr = simulate_t(p["c"], p["ts"], sig & (p["ts"] < split), mh, st, p["cost"])
            oos_tr = simulate_t(p["c"], p["ts"], sig & (p["ts"] >= split), mh, st, p["cost"])
            def pf(tr):
                g = sum(r for _, r in tr if r > 0); l = -sum(r for _, r in tr if r < 0)
                return g / l if l > 0 else (np.inf if g > 0 else 0)
            allc.append({"mode": mode, "is_n": len(is_tr), "oos_n": len(oos_tr),
                         "is_pf": pf(is_tr), "oos_pf": pf(oos_tr),
                         "oos_net": sum(r for _, r in oos_tr)})
            if len(is_tr) >= MIN_IS and pf(is_tr) > 1.0 and oos_tr:
                strategies.append({"is_pf": pf(is_tr), "oos_pf": pf(oos_tr),
                                   "oos_net": sum(r for _, r in oos_tr), "oos_n": len(oos_tr), "mode": mode})
    A = pd.DataFrame(allc)
    print(f"\n## ALL {len(A)} (combo,coin): med_oos_pf={A.oos_pf.replace(np.inf,np.nan).median():.2f} "
          f"%oos_pf>1={100*(A.oos_pf>1).mean():.0f}% %oos_net>0={100*(A.oos_net>0).mean():.0f}%")
    for m in ("follow", "fade"):
        d = A[A["mode"] == m]
        print(f"  {m}: med_oos_pf={d.oos_pf.replace(np.inf,np.nan).median():.2f} %oos_pf>1={100*(d.oos_pf>1).mean():.0f}%")
    S = pd.DataFrame(strategies)
    print(f"\n## IS-positive flow strategies (IS PF>1 & >={MIN_IS} trades): {len(S)}")
    if len(S):
        print(f"  -> OOS: med_oos_pf={S.oos_pf.replace(np.inf,np.nan).median():.2f}  "
              f"%oos_pf>1={100*(S.oos_pf>1).mean():.0f}%  %oos_net>0={100*(S.oos_net>0).mean():.0f}%")
        print(f"  mode mix: {S['mode'].value_counts().to_dict()}")
        print("  (vs price-TA: IS-positive -> OOS PF 0.38, 0% positive. Does FLOW beat that?)")
    print(f"\n[done] flow test on {len(pools)} coins")


if __name__ == "__main__":
    main()
