"""chainscope: multi-chain on-chain token search & research data source.

Quick start:
    import asyncio
    from chainscope import Client

    async def main():
        async with Client() as cs:
            tok = await cs.token("bsc", "0x...")
            print(tok)
    asyncio.run(main())
"""
from __future__ import annotations

from .chains import Chain
from .models import (
    ExchangePresence,
    Launch,
    OHLCV,
    Pool,
    RugReport,
    Ticker,
    Token,
)

__version__ = "0.1.0"

__all__ = [
    "Chain",
    "Client",
    "Token",
    "Pool",
    "OHLCV",
    "RugReport",
    "Launch",
    "Ticker",
    "ExchangePresence",
    "__version__",
]


def __getattr__(name):  # lazy import to avoid pulling all providers at package import
    if name == "Client":
        from .aggregate import Client
        return Client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
