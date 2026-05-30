"""Per-host async rate limiting. A simple, predictable min-interval limiter:
calls to the same host are serialized with at least 1/rate seconds between them.
"""
from __future__ import annotations

import asyncio


class RateLimiter:
    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            wait = self._next_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_at = max(now, self._next_at) + self.min_interval


# Conservative default request rates (per second) keyed by host substring.
# Kept under each provider's documented free-tier ceiling.
DEFAULT_RATES: dict[str, float] = {
    "api.dexscreener.com": 4.0,        # docs: 300/min
    "api.geckoterminal.com": 0.45,     # docs: 30/min
    "pro-api.coingecko.com": 8.0,
    "api.coingecko.com": 0.45,         # demo: 30/min
    "api.gopluslabs.io": 1.0,
    "api.honeypot.is": 2.0,
    "api.rugcheck.xyz": 2.0,
    "public-api.birdeye.so": 0.9,      # free standard ~1 rps
    "solana-gateway.moralis.io": 2.0,
    "deep-index.moralis.io": 2.0,
    "streaming.bitquery.io": 1.0,
    "api.helius.xyz": 2.0,
    "mainnet.helius-rpc.com": 8.0,
    "api.mainnet-beta.solana.com": 4.0,
    "bsc-dataseed.binance.org": 8.0,
    "api.etherscan.io": 4.0,
}
DEFAULT_RATE = 3.0


class RateLimiterRegistry:
    def __init__(self):
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()

    async def get(self, host: str) -> RateLimiter:
        async with self._lock:
            lim = self._limiters.get(host)
            if lim is None:
                rate = DEFAULT_RATES.get(host, DEFAULT_RATE)
                lim = RateLimiter(rate)
                self._limiters[host] = lim
            return lim
