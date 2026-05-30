"""
GAP: bitquery count-based flow selection.

Two deliverables:
(1) Diagnose WHY buy_usd/sell_usd are zero in _bitquery_flow_daily.parquet (read bitquery_flow.py),
    document the exact fix (NO Bitquery quota spend).
(2) Count-based flow selection test using the POPULATED count fields:
    - trader imbalance: (buy_traders - sell_traders)/(buy_traders + sell_traders)
    - tx-count imbalance: (buy_n - sell_n)/(buy_n + sell_n)
    - buyer growth (buy_traders trend) -- limited by short panel
    Cross-sectional: rank coins by signal at day t (causal, uses <= t-1), long top-K vs
    equal-weight universe forward return. Costed per project model. Across-coin permutation null.

CRITICAL DATA LIMITATION: flow panel is only 5 calendar days (2026-05-25..05-29). A calendar
IS/OOS WFO (phase10 design) is impossible. We run a single-horizon cross-sectional selection
test with a strong across-coin permutation null + bootstrap, and report it AS exploratory.

Seeded, reproducible.
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, '.')
from chainscope.costs import round_trip_cost_frac

DATA = Path("./data")
OUT = Path("./_gaps/bitquery-count-flow")
SEED = 7
rng = np.random.default_rng(SEED)

CHAIN_GAS = {'bsc':.2,'base':.02,'eth':3.,'arbitrum':.05,'avax':.05,'polygon_pos':.01,
             'optimism':.05,'sui-network':.01,'solana':.02,'tron':.01}
CHAIN_NATIVE = {'bsc':640,'base':3500,'eth':3500,'arbitrum':3500,'avax':35,'polygon_pos':.5,
                'optimism':3500,'sui-network':3.5,'solana':180,'tron':.3}

def coin_cost(chain, reserve):
    size = max(50., 0.0025*(reserve or 0))
    if not reserve:
        return .05
    return round_trip_cost_frac(size, reserve, dex='uniswap', chain=chain,
                                gas_usd=CHAIN_GAS.get(chain,.05),
                                native_usd=CHAIN_NATIVE.get(chain,100))

def load_panel():
    flow = pd.read_parquet(DATA / "_bitquery_flow_daily.parquet")
    for c in ["buy_usd","sell_usd","buy_traders","sell_traders","buy_n","sell_n"]:
        flow[c] = flow[c].fillna(0.0)
    # map (chain, token) -> pair + reserve + chain via universe
    tok2pair, pair2reserve, pair2chain = {}, {}, {}
    for l in open(DATA / "_mega_universe.jsonl"):
        r = json.loads(l); bt = r.get("base_token") or ""
        if "_" in bt:
            key = (r["chain"], bt.split("_",1)[1].lower())
            tok2pair[key] = r["pair"]
            pair2reserve[r["pair"]] = r.get("reserve_usd") or 0
            pair2chain[r["pair"]] = r["chain"]
    flow["pair"] = [tok2pair.get((c, t.lower())) for c, t in zip(flow.chain, flow.token)]
    flow = flow.dropna(subset=["pair"]).copy()
    flow["reserve"] = flow.pair.map(pair2reserve)
    # daily OHLCV close per pair
    px = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/ohlcv_gt/mega_*_day.parquet")],
                   ignore_index=True)
    px = px.dropna(subset=["close"]); px = px[px.close > 0]
    px["date"] = pd.to_datetime(px.ts, unit="s").dt.strftime("%Y-%m-%d")
    px = px.groupby(["pair_address","date"]).close.last().reset_index().rename(
        columns={"pair_address":"pair"})
    return flow, px, pair2reserve, pair2chain

def build_signals(flow, px):
    m = flow.merge(px, on=["pair","date"], how="inner").sort_values(["pair","date"])
    close = m.pivot_table(index="date", columns="pair", values="close", aggfunc="last").sort_index()
    def piv(col): return m.pivot_table(index="date", columns="pair", values=col,
                                        aggfunc="sum").reindex_like(close).fillna(0)
    btr, str_ = piv("buy_traders"), piv("sell_traders")
    bn, sn = piv("buy_n"), piv("sell_n")
    # CAUSAL: signal at day t uses flow observed on day t-1 (shift(1))
    eps = 1e-9
    trader_imb = ((btr - str_) / (btr + str_ + eps)).shift(1)
    tx_imb     = ((bn - sn) / (bn + sn + eps)).shift(1)
    # buyer growth: today's buyers vs prior day's (needs >=2 days history -> short panel limits)
    buyer_grow = (btr / (btr.shift(1) + eps)).shift(1)
    # raw activity (buyers) as a non-directional control
    buyers_lvl = btr.shift(1)
    SIGS = {"trader_imbalance": trader_imb, "tx_imbalance": tx_imb,
            "buyer_growth": buyer_grow, "buyers_level": buyers_lvl}
    return close, SIGS

def run_horizon(close, sig, K, R, shuffle, pair2reserve, pair2chain, rng_local):
    """One-shot cross-sectional: at each rebal day, rank by sig, long top-K, forward R-day return.
    Returns list of per-rebal (strat_net - bench) edges."""
    dates = close.index.to_numpy()
    fwd = close.shift(-R) / close - 1.0
    edges = []
    for i in range(0, len(dates) - R):
        s = sig.iloc[i].dropna()
        ff = fwd.iloc[i]
        alive = ff.dropna().index
        cand = s.index.intersection(alive)
        if len(cand) < 3*K:
            continue
        sv = s.loc[cand]
        if shuffle:
            sv = pd.Series(rng_local.permutation(sv.values), index=sv.index)
        picks = sv.sort_values(ascending=False).index[:K]
        # per-fill cost on each picked coin (both legs handled by round_trip_cost_frac)
        costs = np.array([coin_cost(pair2chain.get(p), pair2reserve.get(p)) for p in picks])
        strat_net = (ff.loc[picks].values - costs).mean()
        bench = ff.loc[alive].mean()
        edges.append(strat_net - bench)
    return edges

def main():
    flow, px, pair2reserve, pair2chain = load_panel()
    close, SIGS = build_signals(flow, px)
    dates = list(close.index)
    print(f"panel: {close.shape[1]} coins x {len(dates)} days {dates}", flush=True)
    med_cost = np.median([coin_cost(pair2chain.get(p), pair2reserve.get(p)) for p in close.columns])
    print(f"median per-fill round-trip cost: {med_cost*1e4:.0f} bp", flush=True)

    rows = []
    NPERM = 2000
    for name, sig in SIGS.items():
        for K in (5, 10, 20):
            for R in (1, 2, 3):
                if R >= len(dates):
                    continue
                real_edges = run_horizon(close, sig, K, R, False, pair2reserve, pair2chain, rng)
                if len(real_edges) == 0:
                    continue
                real_mean = float(np.mean(real_edges))
                real_med = float(np.median(real_edges))
                # across-coin permutation null distribution
                null_means = []
                for _ in range(NPERM):
                    rng_l = np.random.default_rng(rng.integers(1<<31))
                    ne = run_horizon(close, sig, K, R, True, pair2reserve, pair2chain, rng_l)
                    if ne:
                        null_means.append(np.mean(ne))
                null_means = np.array(null_means)
                null_mean = float(null_means.mean()) if len(null_means) else np.nan
                # empirical one-sided p: P(null >= real)
                pval = float((null_means >= real_mean).mean()) if len(null_means) else np.nan
                rows.append(dict(signal=name, K=K, R=R, n_rebal=len(real_edges),
                                 real_mean_edge=real_mean, real_med_edge=real_med,
                                 null_mean_edge=null_mean,
                                 null_p5=float(np.percentile(null_means,5)) if len(null_means) else np.nan,
                                 null_p95=float(np.percentile(null_means,95)) if len(null_means) else np.nan,
                                 real_beats_null=real_mean > null_mean,
                                 real_net_positive=real_mean > 0,
                                 perm_pvalue=pval))
                print(f"  {name:16} K={K:>2} R={R} | real {real_mean*1e4:>+7.0f}bp "
                      f"null {null_mean*1e4:>+7.0f}bp [{np.percentile(null_means,5)*1e4:>+6.0f},"
                      f"{np.percentile(null_means,95)*1e4:>+6.0f}] p={pval:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "count_flow_selection_results.csv", index=False)
    print(f"\nwrote {OUT/'count_flow_selection_results.csv'}", flush=True)

    # summary
    valid = res[res.n_rebal >= 1]
    print("\n=== SUMMARY (all configs) ===")
    print(f"configs: {len(res)}")
    print(f"real beats null: {res.real_beats_null.sum()}/{len(res)}")
    print(f"real net-positive: {res.real_net_positive.sum()}/{len(res)}")
    print(f"real net-positive AND beats null: {(res.real_net_positive & res.real_beats_null).sum()}/{len(res)}")
    print(f"perm p<0.05 (real significantly > null): {(res.perm_pvalue < 0.05).sum()}/{len(res)}")
    print(f"median real edge across configs: {res.real_mean_edge.median()*1e4:+.0f} bp")
    print(f"median null edge across configs:  {res.null_mean_edge.median()*1e4:+.0f} bp")

    # figure: real vs null edge per config
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,6))
    lab = res.signal + " K" + res.K.astype(str) + " R" + res.R.astype(str)
    x = np.arange(len(res))
    ax.bar(x-0.2, res.real_mean_edge*1e4, width=0.4, label="real edge", color="#2c7fb8")
    ax.bar(x+0.2, res.null_mean_edge*1e4, width=0.4, label="null edge (across-coin perm mean)", color="#bbbbbb")
    # null 5-95 band markers
    ax.errorbar(x+0.2, res.null_mean_edge*1e4,
                yerr=[(res.null_mean_edge-res.null_p5)*1e4, (res.null_p95-res.null_mean_edge)*1e4],
                fmt="none", ecolor="#888888", capsize=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=90, fontsize=6)
    ax.set_ylabel("OOS-style edge vs equal-weight (bp, net of per-fill cost)")
    ax.set_title("Count-based DEX flow selection: REAL vs across-coin NULL\n"
                 "(DEX-only coins, 5-day Bitquery flow panel, exploratory)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "real_vs_null_edge.pdf")
    print(f"wrote {OUT/'figs'/'real_vs_null_edge.pdf'}")

    # second figure: distribution of real edge vs null for the single best-looking config
    best = res.loc[res.real_mean_edge.idxmax()]
    print(f"\nbest real config: {best.signal} K={best.K} R={best.R} "
          f"real={best.real_mean_edge*1e4:+.0f}bp p={best.perm_pvalue:.3f}")

if __name__ == "__main__":
    main()
