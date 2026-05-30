"""
GAP: Passive LP / market-making on DEX-only coins.

Thesis (falsifiable): Passive liquidity provision (earning swap fees minus impermanent
loss minus gas) on DEX-only AMM pools is NOT net-profitable on a risk-adjusted, WFO,
cost-honest basis, and does NOT beat a null that shuffles which pools you LP into.
i.e. the fee spread we PAY as takers elsewhere is not harvestable as passive LPs here,
because IL + gas + volume decay eat it.

Model:
  - Pool fee tier from dex name via chainscope.costs.swap_fee_bps.
  - LP position size = max(50, 0.0025 * reserve) (same sizing as taker model).
  - Daily fee income (full-range v2) = volume_bar * fee_frac * lp_share,
        lp_share = P / (R + P).
  - Daily fee income (concentrated) = volume_bar * fee_frac * lp_share * conc_mult
        ONLY while price in range; 0 when out-of-range. conc_mult = capital efficiency
        of a +/-band around current price (v3 math).
  - Impermanent loss from the realized OHLCV close-to-close price path:
        v2 full-range: IL(r) = 2*sqrt(r)/(1+r) - 1   (vs HODL)
        concentrated [pa,pb]: position value vs HODL with v3 value function.
  - Gas: round-trip enter+exit (2x per-swap gas) at WFO-window boundaries; concentrated
        also pays a rebalance gas each time price exits the band (re-center).

WFO: rolling IS/OOS per coin. IS: rank pools by a CAUSAL liquidity/volume signal
(mean fee-yield proxy = vol/reserve over IS window). Select top-K. Evaluate realized LP
net APR over the disjoint OOS window. NULL: shuffle which pool's OOS path each selected
slot gets (permute pool->OOS assignment across coins) -> destroys any selection skill.

Long-only-compatible: LP is a long-both-tokens position, not a short. No shorting.
"""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from chainscope.costs import swap_fee_bps, gas_cost_usd

DATA = Path("./data")
OUT = Path("./_gaps/lp_market_making")
TF = "day"
CHAIN_GAS = {"bsc": .2, "base": .02, "eth": 3., "arbitrum": .05, "avax": .05, "polygon_pos": .01,
             "optimism": .05, "sui-network": .01, "solana": .02, "tron": .01}
CHAIN_NATIVE = {"bsc": 640, "base": 3500, "eth": 3500, "arbitrum": 3500, "avax": 35,
                "polygon_pos": .5, "optimism": 3500, "sui-network": 3.5, "solana": 180, "tron": .3}
SIZE_FRAC = 0.0025
MIN_BARS = 80
IS_LEN, OOS_LEN = 90, 45        # ~3mo IS / 1.5mo OOS daily
TOPK = 5                         # pools selected per window
SEED = 42
rng = np.random.default_rng(SEED)

# concentrated-range half-width as fractional band around price (e.g. +/-30%)
CONC_BAND = 0.30


def load_universe():
    meta = {}
    for line in open(DATA / "_mega_universe.jsonl"):
        d = json.loads(line)
        meta[(d["chain"], d["pair"])] = d
    return meta


def gas_per_swap(chain):
    nat = CHAIN_NATIVE.get(chain, 100)
    return gas_cost_usd(chain, native_usd=nat, dex=None) if chain == "bsc" else \
        (CHAIN_GAS.get(chain, .05))  # flat per-swap gas estimate per chain in USD


# ---------- LP PnL math ----------
def il_v2(r):
    """Impermanent loss fraction (<=0) vs HODL for full-range x*y=k, price ratio r=Pend/Pstart."""
    r = np.asarray(r, float)
    val = 2.0 * np.sqrt(r) / (1.0 + r) - 1.0
    return val  # <=0


def conc_multiplier(band):
    """Capital-efficiency multiplier of a concentrated +/-band vs full range.
    For a symmetric range [P/(1+b), P*(1+b)] the v3 boost ~ 1/(1 - 1/sqrt(1+b)).
    Larger boost for tighter band."""
    # v3 liquidity concentration: L_conc / L_full = 1/(1 - (pa/P)^0.25 ...) -> use std approx
    # boost = 1 / (1 - 1/ (1+b)^0.5 ) for upper-only; use symmetric average form
    lo = 1.0 / (1.0 + band)
    boost = 1.0 / (1.0 - np.sqrt(lo))
    return boost


def lp_simulate(o, h, l, c, v, reserve, fee_frac, chain, mode):
    """Simulate an LP position held over the whole bar array (one OOS window slice).
    Returns dict with net_pnl_frac (PnL as fraction of deposited capital), fee_frac_total,
    il_frac, gas_frac, n_bars, apr.
    All causal: uses realized volume + price path within the window only. Entry at bar 0
    price, exit at last bar price.
    """
    c = np.asarray(c, float); v = np.asarray(v, float)
    n = len(c)
    if n < 2 or reserve is None or reserve <= 0:
        return None
    P = max(50.0, SIZE_FRAC * reserve)
    lp_share = P / (reserve + P)        # diluted by our own deposit
    p0 = c[0]
    if p0 <= 0:
        return None
    gas1 = gas_per_swap(chain)

    if mode == "full":
        # fee income each bar = volume * fee_frac * lp_share  (full range always in range)
        fees_usd = np.nansum(v[1:]) * fee_frac * lp_share   # bar0 is entry; accrue from bar1
        r = c[-1] / p0
        il = il_v2(r)                                       # fraction <=0 of position value
        # position value (excl fees) = HODL_value * (1+il); but we measure PnL vs the USD
        # capital deposited. HODL of 50/50 deposit: value_factor = (1 + r)/2 ... we measure
        # LP PnL relative to just holding the deposited USD as 50/50, then add fees.
        hodl_factor = (1.0 + r) / 2.0
        lp_value_factor = hodl_factor * (1.0 + il)
        # PnL vs deposited USD = (lp_value_factor - 1) + fees/P  ; but the (lp_value_factor-1)
        # includes the directional move of the volatile leg, which an LP is exposed to.
        # To isolate the MARKET-MAKING edge we report BOTH: total and IL-vs-fees.
        price_pnl = lp_value_factor - 1.0                  # directional + IL
        fee_pnl = fees_usd / P
        gas_pnl = -(2.0 * gas1) / P                        # enter + exit
        net = fee_pnl + il + gas_pnl                       # market-neutral view: fees + IL + gas
        net_total = price_pnl + fee_pnl + gas_pnl          # includes directional exposure
        rebal_gas = 0.0
    else:  # concentrated
        boost = conc_multiplier(CONC_BAND)
        pa, pb = p0 / (1.0 + CONC_BAND), p0 * (1.0 + CONC_BAND)
        # track in-range bars + rebalances (re-center when price exits band)
        in_range = np.zeros(n, bool)
        rebalances = 0
        lo, hi = pa, pb
        center = p0
        for i in range(1, n):
            px = c[i]
            if lo <= px <= hi:
                in_range[i] = True
            else:
                # price exited -> rebalance: re-center band on current price
                rebalances += 1
                center = px
                lo, hi = center / (1.0 + CONC_BAND), center * (1.0 + CONC_BAND)
                in_range[i] = True  # after rebalance back in range
        # fees: only accrue when in range, but boosted by concentration multiplier
        fee_bars = v[1:] * in_range[1:]
        fees_usd = np.nansum(fee_bars) * fee_frac * lp_share * boost
        # IL for concentrated is amplified by ~boost while in range; approximate IL on the
        # realized end/start ratio, scaled by boost (concentrated positions suffer more IL
        # per unit price move because liquidity is denser).
        r = c[-1] / p0
        il_base = il_v2(r)
        il = il_base * boost                               # amplified divergence loss
        il = np.clip(il, -1.0, 0.0)                        # cannot lose >100%
        hodl_factor = (1.0 + r) / 2.0
        price_pnl = hodl_factor * (1.0 + il) - 1.0
        fee_pnl = fees_usd / P
        # gas: enter + exit + one swap per rebalance
        rebal_gas = rebalances * gas1
        gas_pnl = -(2.0 * gas1 + rebal_gas) / P
        net = fee_pnl + il + gas_pnl
        net_total = price_pnl + fee_pnl + gas_pnl

    apr = net * (365.0 / n)
    apr_total = net_total * (365.0 / n)
    return dict(net=net, net_total=net_total, fee_pnl=fee_pnl, il=il if mode == "full" else il,
                gas_pnl=gas_pnl, apr=apr, apr_total=apr_total, n=n, P=P, lp_share=lp_share,
                fee_frac=fee_frac, rebal=(rebalances if mode == "conc" else 0),
                vol_sum=float(np.nansum(v[1:])), r=c[-1] / p0)


def main():
    t0 = time.time()
    meta = load_universe()
    files = sorted(glob.glob(str(DATA / f"ohlcv_gt/mega_*_{TF}.parquet")))
    rows_full, rows_conc = [], []
    perpool = []   # per-pool aggregated for selection
    # Build a long table of (chain,pair) -> sorted bars
    npool = 0
    for f in files:
        df = pd.read_parquet(f)
        for (chain, pair), g in df.groupby(["chain", "pair_address"]):
            g = g.sort_values("ts")
            if len(g) < MIN_BARS:
                continue
            m = meta.get((chain, pair))
            if m is None:
                continue
            reserve = m.get("reserve_usd")
            dex = m.get("dex")
            fee_frac = swap_fee_bps(dex) / 1e4
            o = g.open.to_numpy(); h = g.high.to_numpy(); l = g.low.to_numpy()
            c = g.close.to_numpy(); v = g.volume.to_numpy()
            if np.any(~np.isfinite(c)) or np.nanmin(c) <= 0:
                continue
            npool += 1
            nb = len(c)
            # rolling WFO windows
            wins = []
            start = 0
            while start + IS_LEN + OOS_LEN <= nb:
                is_sl = slice(start, start + IS_LEN)
                oos_sl = slice(start + IS_LEN, start + IS_LEN + OOS_LEN)
                # CAUSAL selection signal computed on IS only: mean fee-yield proxy
                is_vol = np.nansum(v[is_sl])
                is_yield = (is_vol * fee_frac) / (reserve if reserve else np.nan) / IS_LEN  # daily yield proxy
                # realized OOS LP outcomes
                for mode, sink in (("full", rows_full), ("conc", rows_conc)):
                    res = lp_simulate(o[oos_sl], h[oos_sl], l[oos_sl], c[oos_sl], v[oos_sl],
                                      reserve, fee_frac, chain, mode)
                    if res is None:
                        continue
                    res.update(chain=chain, pair=pair, dex=dex, reserve=reserve, mode=mode,
                               win=len(wins), is_yield=is_yield, fee_bps=fee_frac * 1e4)
                    sink.append(res)
                wins.append(1)
                start += OOS_LEN  # disjoint OOS windows
    df_full = pd.DataFrame(rows_full)
    df_conc = pd.DataFrame(rows_conc)
    df_full.to_csv(OUT / "lp_full_allwindows.csv", index=False)
    df_conc.to_csv(OUT / "lp_conc_allwindows.csv", index=False)
    print(f"pools used: {npool}  full rows {len(df_full)}  conc rows {len(df_conc)}  "
          f"t={time.time()-t0:.1f}s")
    return df_full, df_conc, meta


if __name__ == "__main__":
    main()
