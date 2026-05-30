"""DEX trade cost model: so backtests are never costless.

A realistic DEX fill has three cost components:
  1. swap fee , the pool's protocol fee (bps of notional)
  2. price impact, constant-product slippage from finite pool depth
  3. gas      , fixed per-swap network cost

Price impact uses the constant-product (x*y=k) approximation. Given total pool
liquidity `reserve_usd`, the relevant single-sided reserve is ~reserve_usd/2.
For a buy of size S into quote reserve R: avg execution price = mid*(1 + S/R).
For a sell of size S against base reserve R: avg price = mid/(1 + S/R).

All defaults are overridable. Tune them to your venue/fee tier before trusting a backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chains import Chain

# Default swap fee (bps of notional) by DEX-name substring. PancakeSwap/Uniswap v3
# have multiple tiers; these are the common defaults, override per pool if known.
DEX_FEE_BPS: dict[str, float] = {
    "pancakeswap-v3": 25.0,
    "pancakeswap_v3": 25.0,
    "pancakeswap": 25.0,
    "uniswap_v3": 30.0,
    "uniswap-v3": 30.0,
    "uniswap": 30.0,
    "biswap": 10.0,
    "sushiswap": 30.0,
    "raydium": 25.0,
    "orca": 30.0,
    "whirlpool": 30.0,
    "meteora": 20.0,
    "pumpswap": 25.0,
    "fluxbeam": 30.0,
}
DEFAULT_FEE_BPS = 30.0

# Fixed gas/priority cost per swap, in USD, by chain.
DEFAULT_GAS_USD: dict[str, float] = {
    Chain.BSC.value: 0.30,
    Chain.SOLANA.value: 0.02,
}

# Conservative minimum slippage floor (bps) applied even when modeled impact is tiny.
DEFAULT_SLIP_FLOOR_BPS = 2.0

# EVM gas units per swap (approx). Combined with gas price (gwei) and the native-token
# USD price to get historical gas cost. Solana uses the exact tx fee (gas_native) instead.
GAS_UNITS = {
    "pancakeswap-v2": 110_000, "pancakeswap-v3": 140_000, "uniswap-v3": 140_000,
    "default_evm": 120_000,
}
DEFAULT_GAS_PRICE_GWEI = {Chain.BSC.value: 1.0}   # ~1 gwei base fee on BSC (override per run)


def gas_cost_usd(chain: str | Chain, native_usd: float | None = None, dex: str | None = None,
                 gas_native: float | None = None, gas_units: float | None = None,
                 gas_price_gwei: float | None = None) -> float:
    """Per-swap gas in USD. Exact when gas_native (native-token fee paid, e.g. a Solana tx
    fee) + native_usd are given; else modeled for EVM as units x gwei x native price; else
    a flat fallback. Gas is a ~fixed cost per tx, so it dominates small trades."""
    chain_v = Chain.parse(chain).value
    if gas_native is not None and native_usd is not None:
        return gas_native * native_usd
    if native_usd is not None and chain_v == Chain.BSC.value:
        units = gas_units if gas_units is not None else GAS_UNITS.get(dex or "", GAS_UNITS["default_evm"])
        gwei = gas_price_gwei if gas_price_gwei is not None else DEFAULT_GAS_PRICE_GWEI.get(chain_v, 1.0)
        return units * gwei * 1e-9 * native_usd
    return DEFAULT_GAS_USD.get(chain_v, 0.10)


def swap_fee_bps(dex: str | None) -> float:
    if not dex:
        return DEFAULT_FEE_BPS
    d = dex.lower()
    for key, bps in DEX_FEE_BPS.items():
        if key in d:
            return bps
    return DEFAULT_FEE_BPS


def price_impact_frac(side: str, size_usd: float, reserve_usd: float | None) -> float:
    """Constant-product price-impact cost as a fraction of notional (>=0).
    `reserve_usd` is total pool TVL; single-sided reserve is taken as half."""
    if not reserve_usd or reserve_usd <= 0 or size_usd <= 0:
        return 1.0  # unknown/empty liquidity -> treat as maximally costly
    one_sided = reserve_usd / 2.0
    ratio = size_usd / one_sided
    if side == "buy":
        return ratio                      # mid*(1+ratio) -> cost fraction = ratio
    return ratio / (1.0 + ratio)          # sell: 1 - 1/(1+ratio)


@dataclass
class CostBreakdown:
    side: str
    mid_price: float
    exec_price: float
    fee_frac: float
    impact_frac: float
    gas_usd: float
    total_frac: float        # fee + impact (notional fraction); gas is separate USD

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "mid_price": self.mid_price,
            "exec_price": self.exec_price,
            "fee_frac": self.fee_frac,
            "impact_frac": self.impact_frac,
            "gas_usd": self.gas_usd,
            "total_frac": self.total_frac,
        }


def apply_costs(side: str, mid_price: float, size_usd: float, reserve_usd: float | None,
                dex: str | None = None, chain: str | Chain = Chain.SOLANA,
                fee_bps: float | None = None, slip_floor_bps: float = DEFAULT_SLIP_FLOOR_BPS,
                gas_usd: float | None = None, native_usd: float | None = None,
                gas_native: float | None = None) -> CostBreakdown:
    side = "buy" if str(side).lower().startswith("b") else "sell"
    fee_frac = (swap_fee_bps(dex) if fee_bps is None else fee_bps) / 1e4
    impact = max(price_impact_frac(side, size_usd, reserve_usd), slip_floor_bps / 1e4)
    gas = gas_usd if gas_usd is not None else gas_cost_usd(chain, native_usd, dex, gas_native)
    total = fee_frac + impact
    exec_price = mid_price * (1 + total) if side == "buy" else mid_price * (1 - total)
    return CostBreakdown(side, mid_price, exec_price, fee_frac, impact, gas, total)


def round_trip_cost_frac(size_usd: float, reserve_usd: float | None, dex: str | None = None,
                         chain: str | Chain = Chain.SOLANA, fee_bps: float | None = None,
                         gas_usd: float | None = None, native_usd: float | None = None,
                         gas_native: float | None = None) -> float:
    """Total cost of entering and exiting a position, as a fraction of notional
    (fees + slippage both legs + 2x gas amortized over size). Pass native_usd (BNB/SOL
    price) for real gas; pass the pool's exact fee_bps when known. Use to floor strategy edge."""
    buy = apply_costs("buy", 1.0, size_usd, reserve_usd, dex, chain, fee_bps,
                      gas_usd=gas_usd, native_usd=native_usd, gas_native=gas_native)
    sell = apply_costs("sell", 1.0, size_usd, reserve_usd, dex, chain, fee_bps,
                       gas_usd=gas_usd, native_usd=native_usd, gas_native=gas_native)
    gas_frac = (2 * buy.gas_usd / size_usd) if size_usd > 0 else 0.0
    return buy.total_frac + sell.total_frac + gas_frac


def revert_cost_usd(chain: str | Chain, native_usd: float | None = None, dex: str | None = None,
                    gas_native: float | None = None) -> float:
    """Cost of a reverted swap: gas paid, no fill. Model failed txs (slippage-tolerance
    exceeded, sandwich-induced) as this gas-lost cost so a strategy's revert rate is priced in."""
    return gas_cost_usd(chain, native_usd, dex, gas_native)
