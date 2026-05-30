"""Provider contract. Every data source subclasses Provider, declares which
chains and capabilities it supports, and overrides only the methods it can
serve. Unsupported methods raise NotSupported; the aggregator routes around them.

Implementing a provider:
  1. Set `name`, `supported_chains`, `capabilities`.
  2. If it needs an API key, set `requires_key = True` and `key_env = "FOO_API_KEY"`.
  3. Override the capability methods you declared, returning canonical models.
  4. Fetch only via `self.http.get_json(...)` / `self.http.post_json(...)`.
  5. Catch `HttpError` for 404/not-found and return None / [] rather than raising.
"""
from __future__ import annotations

from ..chains import Chain
from ..config import Settings
from ..http import HttpClient
from ..models import ExchangePresence, Launch, OHLCV, Pool, RugReport, Token, Trade

# Capability identifiers
CAP_SEARCH = "search"
CAP_TOKEN = "token"
CAP_POOLS = "pools"
CAP_OHLCV = "ohlcv"
CAP_TRADES = "trades"
CAP_RUG = "rug"
CAP_LISTINGS = "listings"
CAP_NEW_PAIRS = "new_pairs"
CAP_LAUNCH = "launch"
CAP_STREAM_LAUNCHES = "stream_launches"

ALL_CAPS = {
    CAP_SEARCH, CAP_TOKEN, CAP_POOLS, CAP_OHLCV, CAP_TRADES, CAP_RUG,
    CAP_LISTINGS, CAP_NEW_PAIRS, CAP_LAUNCH, CAP_STREAM_LAUNCHES,
}


class NotSupported(Exception):
    pass


class Provider:
    name: str = "base"
    supported_chains: frozenset[Chain] = frozenset()
    capabilities: frozenset[str] = frozenset()
    requires_key: bool = False
    key_env: str | None = None
    onchain: bool = False   # True = reconstructs from raw RPC only (no gated 3rd-party API)

    def __init__(self, http: HttpClient, settings: Settings):
        self.http = http
        self.settings = settings

    @property
    def enabled(self) -> bool:
        if self.requires_key:
            return bool(self.settings.get_key(self.key_env))
        return True

    def supports(self, chain: Chain, cap: str) -> bool:
        return (
            self.enabled
            and Chain.parse(chain) in self.supported_chains
            and cap in self.capabilities
        )

    # ---- capability methods (override the ones you declare) ----

    async def search(self, query: str, chain: Chain | None = None) -> list[Token]:
        raise NotSupported

    async def get_token(self, chain: Chain, address: str) -> Token | None:
        raise NotSupported

    async def get_pools(self, chain: Chain, address: str) -> list[Pool]:
        raise NotSupported

    async def get_ohlcv(self, chain: Chain, pair_address: str, timeframe: str = "1h",
                        limit: int = 1000, before: int | None = None) -> list[OHLCV]:
        raise NotSupported

    async def get_trades(self, chain: Chain, pair_address: str,
                         since: int | None = None, until: int | None = None,
                         limit: int = 1000) -> list[Trade]:
        """Trade tape for a pool. `since`/`until` are unix seconds. Newest-first
        unless otherwise documented by the provider."""
        raise NotSupported

    async def get_rug(self, chain: Chain, address: str) -> RugReport | None:
        raise NotSupported

    async def get_listings(self, chain: Chain | None, address: str | None,
                           symbol: str | None = None) -> ExchangePresence | None:
        raise NotSupported

    async def get_new_pairs(self, chain: Chain) -> list[Pool]:
        raise NotSupported

    async def get_launch(self, chain: Chain, address: str) -> Launch | None:
        raise NotSupported

    async def stream_launches(self, chain: Chain):
        """Async generator yielding Launch records in real time."""
        raise NotSupported
        yield  # pragma: no cover  (marks this as a generator)
