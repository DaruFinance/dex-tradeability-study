"""Phase 8: cross-sectional momentum on the full mega corpus (~2,300 coins, 7 chains), with a
SURVIVORSHIP diagnostic. Tests the make-or-break: does long-top-K-by-momentum beat the equal-weight
universe OOS, net of cost, on a BROAD panel that includes coins which crashed/died in-window?

Survivorship note: the mega universe is GT-list-sourced (current-live pools) so NOT fully
survivorship-free, but it's far broader than the earlier top-200 and includes a long tail of
declining/near-dead coins. The diagnostic below quantifies how many losers the panel actually
contains, so we know how much to trust the edge.

Usage: python3 _research/phase8_cross_mega.py [is_split]
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
IS_SPLIT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
SIZE_FRAC = 0.0025
CHAIN_GAS = {"bsc": 0.20, "base": 0.02, "eth": 3.0, "arbitrum": 0.05, "avax": 0.05,
             "polygon_pos": 0.01, "sui-network": 0.01, "solana": 0.02, "tron": 0.01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": 0.5, "sui-network": 3.5, "solana": 180, "tron": 0.3}


def load():
    univ = {}
    for l in open(DATA / "_mega_universe.jsonl"):
        r = json.loads(l); univ[r["pair"]] = r
    frames = [pd.read_parquet(f) for f in glob.glob(f"{DATA}/ohlcv_gt/mega_*_day.parquet")]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    df = df[df.close > 0]
    df["day"] = (df.ts // 86400).astype(int)
    return df, univ


def main():
    df, univ = load()
    # per-coin cost from reserve (sizing 0.25% reserve, per-chain gas)
    cost = {}
    for pair, u in univ.items():
        res = u.get("reserve_usd") or 0; ch = u.get("chain", "bsc")
        size = max(50.0, SIZE_FRAC * res)
        cost[pair] = round_trip_cost_frac(size, res, dex="uniswap", chain=ch,
                                          gas_usd=CHAIN_GAS.get(ch, 0.05), native_usd=CHAIN_NATIVE.get(ch, 100)) if res else 0.05

    # --- survivorship diagnostic ---
    dd, finals, vols, lifes = [], [], [], []
    for pair, g in df.groupby("pair_address"):
        c = g.sort_values("day").close.to_numpy()
        if len(c) < 10: continue
        peak = np.maximum.accumulate(c)
        dd.append((c[-1] / peak[-1]) - 1.0)          # drawdown from peak at end
        finals.append(c[-1] / c[0] - 1.0)            # total return
        vols.append(np.std(np.diff(np.log(c))))
        lifes.append(len(c))
    dd, finals, vols = np.array(dd), np.array(finals), np.array(vols)
    print(f"=== SURVIVORSHIP diagnostic ({len(dd)} coins, daily) ===")
    print(f"  total return: median={np.median(finals):+.2f} mean={np.mean(finals):+.2f} %>0={100*(finals>0).mean():.0f}%")
    print(f"  drawdown-from-peak-at-end: median={np.median(dd):.2f}  %coins down>50% from peak={100*(dd<-0.5).mean():.0f}%  >80%={100*(dd<-0.8).mean():.0f}%")
    print(f"  -> panel DOES contain losers: {100*(finals<-0.5).mean():.0f}% of coins ended down >50%, {100*(finals<-0.9).mean():.0f}% down >90% (rug-like).")
    print(f"  (so cross-sectional momentum CAN pick coins that then crash, a meaningful survivorship test, though GT-list sourcing means truly-delisted coins are still under-counted.)")

    # exclude stablecoins (near-zero vol) from the tradeable signal universe
    keep = {pair for pair, g in df.groupby("pair_address")
            if g.close.pct_change().std() > 0.01 and len(g) >= 40}
    df = df[df.pair_address.isin(keep)]
    px = df.pivot_table(index="day", columns="pair_address", values="close", aggfunc="last").sort_index()
    print(f"\ncross-sectional panel: {px.shape[1]} coins x {px.shape[0]} days (stablecoins/short excluded)")

    days = px.index.to_numpy(); split_day = days[int(IS_SPLIT * len(days))]

    def run(Lsig, K, R, mode):
        sig = px.pct_change(Lsig, fill_method=None).shift(1)
        fwd = px.shift(-R) / px - 1.0
        si = so = bi = bo = 0.0; ni = no = 0
        for i in range(0, len(days) - R, R):
            s = sig.iloc[i].dropna(); f = fwd.iloc[i]
            alive = f.dropna().index; cand = s.index.intersection(alive)
            if len(cand) < 3 * K: continue
            ranked = s.loc[cand].sort_values(ascending=(mode == "rev"))
            picks = ranked.index[:K]
            strat = np.mean([f[p] - cost.get(p, 0.05) for p in picks])
            bench = f.loc[alive].mean()
            if days[i] < split_day: si += strat; bi += bench; ni += 1
            else: so += strat; bo += bench; no += 1
        return si, bi, ni, so, bo, no

    print(f"\n{'sig':6} {'K':>3} {'R':>3} {'mode':4} | {'IS edge':>8} {'OOS edge':>9} | {'OOS strat':>9} {'OOS bench':>9}")
    rows = []
    for L in (5, 10, 20, 30):
        for K in (3, 5, 10, 20):
            for R in (3, 7, 14):
                for mode in ("mom", "rev"):
                    si, bi, ni, so, bo, no = run(L, K, R, mode)
                    if ni < 3 or no < 2: continue
                    rows.append({"L": L, "K": K, "R": R, "mode": mode, "is_edge": si - bi,
                                 "oos_edge": so - bo, "oos_strat": so, "oos_bench": bo})
    Rdf = pd.DataFrame(rows)
    isp = Rdf[Rdf.is_edge > 0].sort_values("is_edge", ascending=False)
    print(f"\ncombos with positive IS edge: {len(isp)}/{len(Rdf)}")
    if len(isp):
        print(f"  -> OOS edge>0 (persists): {100*(isp.oos_edge>0).mean():.0f}%   median OOS edge {isp.oos_edge.median():+.3f}")
        for m in ("mom", "rev"):
            d = isp[isp["mode"] == m]
            if len(d): print(f"    {m}: {len(d)} IS-pos, OOS-persist {100*(d.oos_edge>0).mean():.0f}%, median OOS edge {d.oos_edge.median():+.3f}")
        print("\n  top IS-edge combos -> OOS:")
        for _, r in isp.head(10).iterrows():
            print(f"    mom{int(r['L'])} K{int(r['K'])} R{int(r['R'])} {r['mode']} | IS={r['is_edge']:+.3f} OOS={r['oos_edge']:+.3f} (strat {r['oos_strat']:+.3f} vs bench {r['oos_bench']:+.3f})")
    Rdf.to_parquet(DATA / "_phase8_cross_mega.parquet", index=False)
    print("\n  edge = top-K return minus equal-weight-universe return, net of per-coin reserve-based cost.")


if __name__ == "__main__":
    main()
