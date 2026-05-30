"""Turn raw OHLCV into a backtest-ready frame with realistic DEX costs attached.

This does NOT run a backtest — it produces clean, point-in-time price bars plus a
per-bar round-trip cost floor (fees + slippage + gas), so any strategy built on top
is costed from the start. Feed the frame into your own WFO engine.

No-lookahead note: each bar's cost uses that bar's own `reserve_usd` when the source
provides it (Bitquery). Free OHLCV (GeckoTerminal) has no per-bar liquidity, so pass
`reserve_usd=` (e.g. the pool's current liquidity from `client.pools(...)`); be aware
that applying *current* depth to *historical* bars is an approximation — flagged, not hidden.
"""
from __future__ import annotations

from .chains import Chain
from .costs import round_trip_cost_frac
from .models import OHLCV


def ohlcv_to_df(bars: list[OHLCV], reserve_usd: float | None = None):
    import pandas as pd

    rows = []
    for b in bars:
        rows.append({
            "timestamp": b.timestamp,
            "open": b.open, "high": b.high, "low": b.low, "close": b.close,
            "volume": b.volume,
            "reserve_usd": b.reserve_usd if b.reserve_usd is not None else reserve_usd,
            "source": b.source,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


async def backtest_frame(client, chain: str | Chain, pair_address: str, timeframe: str = "1h",
                         limit: int = 1000, size_usd: float = 1000.0,
                         dex: str | None = None, reserve_usd: float | None = None):
    """Fetch OHLCV for a pool and attach a per-bar round-trip cost floor.

    Adds column `rt_cost_frac`: the fraction of notional lost entering+exiting a
    `size_usd` position at that bar's liquidity — i.e. the minimum price move a
    trade must capture just to break even. Returns a pandas DataFrame.
    """
    import pandas as pd  # noqa: F401

    c = Chain.parse(chain)
    bars = await client.ohlcv(c, pair_address, timeframe, limit)
    df = ohlcv_to_df(bars, reserve_usd=reserve_usd)
    if df.empty:
        return df
    df["rt_cost_frac"] = df["reserve_usd"].apply(
        lambda r: round_trip_cost_frac(size_usd, r, dex=dex, chain=c)
    )
    df.attrs["chain"] = c.value
    df.attrs["pair_address"] = pair_address
    df.attrs["timeframe"] = timeframe
    df.attrs["size_usd"] = size_usd
    return df
