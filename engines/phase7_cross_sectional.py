"""Phase 7: CROSS-SECTIONAL strategies on existing daily OHLCV (zero new data).

Tests RELATIVE ranking instead of per-coin timing: each rebalance, long the top-K DEX-only coins
by a signal vs the equal-weight universe benchmark. This removes the market up/down-regime confound
(it's relative), is a single rule (no per-coin overfit), and is long-only. Per-rebalance cost.
IS/OOS calendar split. Tests momentum (long winners) and reversion (long losers).

Usage: python3 _research/phase7_cross_sectional.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path("./data")
COST = 0.0151   # ~151 bp round-trip per rebalanced position (0.25%-reserve sizing, from phase1)
IS_SPLIT = 0.6


def load_panel():
    cexv = {json.loads(l)["pair"]: json.loads(l)["status"] for l in open(DATA / "_cex_verdicts_universe.jsonl")}
    frames = []
    for f in glob.glob(f"{DATA}/ohlcv_gt/*day*.parquet"):
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    df = df[df.pair_address.map(lambda p: cexv.get(p) == "dex_only")]
    df["day"] = (df.ts // 86400).astype(int)
    px = df.pivot_table(index="day", columns="pair_address", values="close", aggfunc="last").sort_index()
    return px


def run(px, signal_fn, K, R, mode, is_split):
    """Long top-K (mode=mom) or bottom-K (mode=rev) by signal, rebalance every R days, eq-weight.
    Returns (strat_cum_is, bench_cum_is, strat_cum_oos, bench_cum_oos)."""
    days = px.index.to_numpy()
    sig = signal_fn(px)                  # causal signal (already shifted)
    fwd = px.shift(-R) / px - 1.0        # R-day forward return per coin
    split_day = days[int(is_split * len(days))]
    s_is = s_oos = b_is = b_oos = 0.0; n_is = n_oos = 0
    for i in range(0, len(days) - R, R):
        d = days[i]
        s = sig.iloc[i].dropna(); f = fwd.iloc[i]
        alive = f.dropna().index
        cand = s.index.intersection(alive)
        if len(cand) < 2 * K:
            continue
        ranked = s.loc[cand].sort_values(ascending=(mode == "rev"))
        picks = ranked.index[:K]
        strat = f.loc[picks].mean() - COST       # rebalanced positions pay cost
        bench = f.loc[alive].mean()              # equal-weight hold-all (no per-rebalance cost)
        if d < split_day:
            s_is += strat; b_is += bench; n_is += 1
        else:
            s_oos += strat; b_oos += bench; n_oos += 1
    return s_is, b_is, n_is, s_oos, b_oos, n_oos


def main():
    px = load_panel()
    print(f"panel: {px.shape[1]} dex_only coins x {px.shape[0]} days")
    if px.shape[1] < 10 or px.shape[0] < 30:
        print("panel too small"); return

    def mom(L): return lambda px: px.pct_change(L).shift(1)
    def volnorm(L): return lambda px: (px.pct_change(L) / px.pct_change().rolling(L).std()).shift(1)
    sigs = {f"mom{L}": mom(L) for L in (5, 10, 20, 30)}
    sigs.update({f"volnorm{L}": volnorm(L) for L in (10, 20)})

    print(f"\n{'signal':10} {'K':>3} {'R':>3} {'mode':4} | {'IS strat':>9} {'IS bench':>9} {'IS edge':>8} | {'OOS strat':>9} {'OOS bench':>9} {'OOS edge':>8}")
    rows = []
    for name, fn in sigs.items():
        for K in (3, 5, 10):
            for R in (3, 7, 14):
                for mode in ("mom", "rev"):
                    si, bi, ni, so, bo, no = run(px, fn, K, R, mode, IS_SPLIT)
                    if ni < 3 or no < 2:
                        continue
                    edge_is = si - bi; edge_oos = so - bo
                    rows.append({"sig": name, "K": K, "R": R, "mode": mode, "is_edge": edge_is,
                                 "oos_edge": edge_oos, "oos_strat": so, "oos_bench": bo})
    Rdf = pd.DataFrame(rows)
    # show combos with positive IS edge -> do they persist OOS? (the real test)
    isp = Rdf[Rdf.is_edge > 0].sort_values("is_edge", ascending=False)
    print(f"\ncombos with POSITIVE IS edge (strat beats universe in-sample): {len(isp)}/{len(Rdf)}")
    if len(isp):
        print(f"  of those -> OOS edge>0 (persists): {100*(isp.oos_edge>0).mean():.0f}%   median OOS edge: {isp.oos_edge.median():+.3f}")
        for m in ("mom", "rev"):
            d = isp[isp["mode"] == m]
            if len(d):
                print(f"    mode={m}: {len(d)} IS-positive, OOS-persist {100*(d.oos_edge>0).mean():.0f}%, median OOS edge {d.oos_edge.median():+.3f}")
        print("\n  top IS-edge combos and their OOS:")
        for _, r in isp.head(12).iterrows():
            print(f"    {r['sig']:9} K{int(r['K']):<2} R{int(r['R']):<2} {r['mode']:3} | IS_edge={r['is_edge']:+.3f} | OOS_edge={r['oos_edge']:+.3f} (strat={r['oos_strat']:+.3f} bench={r['oos_bench']:+.3f})")
    print(f"\n  (edge = cumulative top-K return minus equal-weight-universe return, net of {COST*1e4:.0f}bp/rebalance.")
    print("   positive & IS->OOS-persistent edge = cross-sectional signal works where per-coin timing didn't.)")
    Rdf.to_parquet(DATA / "_phase7_cross_sectional.parquet", index=False)


if __name__ == "__main__":
    main()
