"""Phase 10: FLOW-BASED SELECTION test — does smart-money / net-buy flow predict WHICH coins pump?

The one hypothesis price-timing can't answer. Joins Bitquery daily flow (buy/sell USD, unique
buyers) to our daily OHLCV (forward returns). Builds causal selection signals (net-buy pressure,
buyer growth, net-buy USD trend) and tests: each day, rank coins by the flow signal, long top-K,
forward return vs equal-weight universe. IS/OOS calendar split. NULL = permute the flow signal
ACROSS coins each day (destroys the flow->return link) -> chance baseline. Real >> null = flow edge.

Usage: python3 _research/phase10_flow_selection.py [is_split]
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path("./data")
IS_SPLIT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
COST = 0.0164   # ~164 bp round-trip per rebalanced position
rng = np.random.default_rng(7)

def main():
    flow = pd.read_parquet(DATA / "_bitquery_flow_daily.parquet")
    for c in ["buy_usd","sell_usd","buy_traders","sell_traders","buy_n","sell_n"]:
        if c not in flow: flow[c] = 0.0
        flow[c] = flow[c].fillna(0.0)
    # map (chain, token) -> pair via universe
    tok2pair = {}
    for l in open(DATA / "_mega_universe.jsonl"):
        r = json.loads(l); bt = r.get("base_token") or ""
        if "_" in bt: tok2pair[(r["chain"], bt.split("_",1)[1].lower())] = r["pair"]
    flow["pair"] = [tok2pair.get((c, t.lower())) for c, t in zip(flow.chain, flow.token)]
    flow = flow.dropna(subset=["pair"])
    # daily OHLCV close per pair
    px = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/ohlcv_gt/mega_*_day.parquet")], ignore_index=True)
    px = px.dropna(subset=["close"]); px = px[px.close>0]
    px["date"] = pd.to_datetime(px.ts, unit="s").dt.strftime("%Y-%m-%d")
    px = px.groupby(["pair_address","date"]).close.last().reset_index().rename(columns={"pair_address":"pair"})
    # merge flow + price
    m = flow.merge(px, on=["pair","date"], how="inner").sort_values(["pair","date"])
    print(f"flow∩price: {m.pair.nunique()} coins, {len(m)} coin-days")
    if m.pair.nunique() < 20:
        print("too few coins joined yet (flow pull still running?)"); return

    # build panels: close, and flow features (causal -> shift later)
    close = m.pivot_table(index="date", columns="pair", values="close", aggfunc="last").sort_index()
    buy = m.pivot_table(index="date", columns="pair", values="buy_usd", aggfunc="sum").reindex_like(close).fillna(0)
    sell = m.pivot_table(index="date", columns="pair", values="sell_usd", aggfunc="sum").reindex_like(close).fillna(0)
    btr = m.pivot_table(index="date", columns="pair", values="buy_traders", aggfunc="sum").reindex_like(close).fillna(0)
    dates = close.index.to_numpy()
    # FLOW SIGNALS (all shifted +1 day = causal: yesterday's flow ranks today's buy)
    netpress = ((buy - sell) / (buy + sell).replace(0, np.nan)).shift(1)            # net-buy pressure
    buyusd3 = buy.rolling(3).sum().shift(1)                                          # 3d buy USD
    buyergrow = (btr.rolling(3).mean() / btr.rolling(10).mean().replace(0,np.nan)).shift(1)  # buyer accel
    SIGS = {"net_buy_pressure": netpress, "buy_usd_3d": buyusd3, "buyer_growth": buyergrow}

    def fwd(R): return close.shift(-R) / close - 1.0
    split = dates[int(IS_SPLIT*len(dates))]

    def run(sig, K, R, shuffle):
        f = fwd(R); si=so=bi=bo=0.0; ni=no=0
        for i in range(0, len(dates)-R, R):
            s = sig.iloc[i].dropna(); ff = f.iloc[i]
            alive = ff.dropna().index; cand = s.index.intersection(alive)
            if len(cand) < 3*K: continue
            sv = s.loc[cand]
            if shuffle: sv = pd.Series(rng.permutation(sv.values), index=sv.index)  # break flow->coin link
            picks = sv.sort_values(ascending=False).index[:K]
            strat = ff.loc[picks].mean() - COST; bench = ff.loc[alive].mean()
            if dates[i] < split: si+=strat; bi+=bench; ni+=1
            else: so+=strat; bo+=bench; no+=1
        return (so-bo, ni, no)

    print(f"IS/OOS split {IS_SPLIT:.0%}; cost {COST*1e4:.0f}bp; long top-K by flow signal vs equal-weight universe\n")
    print(f"{'signal':18} {'K':>3} {'R':>3} | {'REAL oos edge':>13} {'NULL oos edge':>13} {'real>null?':>10}")
    wins=0; tot=0
    for name, sig in SIGS.items():
        for K in (5,10,20):
            for R in (3,7,14):
                real,ni,no = run(sig,K,R,False)
                null,_,_   = run(sig,K,R,True)
                if ni<3 or no<2: continue
                tot+=1; wins+= real>null
                flag = "YES" if real>null else "no"
                print(f"{name:18} {K:>3} {R:>3} | {real:>+13.3f} {null:>+13.3f} {flag:>10}")
    print(f"\nreal>null in {wins}/{tot} configs. (flow has selection edge only if real consistently & materially > null AND real>0)")

if __name__ == "__main__":
    main()
