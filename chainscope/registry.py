"""Assembles the provider set and routes capabilities to providers in priority order.

Design principle: each provider is an INDEPENDENT datapoint, not a redundant copy.
For single-value capabilities (token, ohlcv, ...) we fall back down the priority
list until one returns data. For `rug` we MERGE every supporting provider, because
each computes risk differently and we want all the independent signals.
"""
from __future__ import annotations

from .chains import Chain
from .config import Settings, get_settings
from .http import HttpClient
from .providers.base import (
    CAP_LAUNCH, CAP_LISTINGS, CAP_NEW_PAIRS, CAP_OHLCV, CAP_POOLS, CAP_RUG,
    CAP_SEARCH, CAP_STREAM_LAUNCHES, CAP_TOKEN, CAP_TRADES, Provider,
)
from .providers.birdeye import BirdeyeProvider
from .providers.bitquery import BitqueryProvider
from .providers.bsc_chain import BscChainProvider
from .providers.bsc_indexer import BscIndexerProvider
from .providers.bsc_rpc import BscRpcProvider
from .providers.coingecko import CoinGeckoProvider
from .providers.dexscreener import DexScreenerProvider
from .providers.geckoterminal import GeckoTerminalProvider
from .providers.goplus import GoPlusProvider
from .providers.honeypot import HoneypotProvider
from .providers.moralis import MoralisProvider
from .providers.pumpportal import PumpPortalProvider
from .providers.rugcheck import RugCheckProvider
from .providers.solana_chain import SolanaChainProvider
from .providers.solana_indexer import SolanaIndexerProvider
from .providers.solana_rpc import SolanaRpcProvider

PROVIDER_CLASSES = [
    DexScreenerProvider,
    GeckoTerminalProvider,
    CoinGeckoProvider,
    GoPlusProvider,
    HoneypotProvider,
    RugCheckProvider,
    SolanaRpcProvider,
    BscRpcProvider,
    PumpPortalProvider,
    BirdeyeProvider,
    BitqueryProvider,
    MoralisProvider,
    BscIndexerProvider,
    SolanaIndexerProvider,
    SolanaChainProvider,
    BscChainProvider,
]

# Priority order per capability (best/preferred first). Paid providers are listed
# ahead of free ones for OHLCV/trades so DEEP history is used when a key is present,
# automatically falling back to the free recent source otherwise.
# On-chain providers reconstruct purely from RPC (no gated 3rd-party API). When
# settings.onchain_only is set, only these are used. On-chain sources are listed
# FIRST for token/pools/rug so they're preferred even in mixed mode. (OHLCV/trades
# keep the fast/deep sources first, with the indexers as the on-chain fallback.)
ONCHAIN_NAMES = {
    "bsc_rpc", "solana_rpc", "bsc_indexer", "solana_indexer", "bsc_chain", "solana_chain",
}

PRIORITY: dict[str, list[str]] = {
    CAP_SEARCH:    ["dexscreener", "geckoterminal"],     # no on-chain name index
    CAP_TOKEN:     ["bsc_chain", "solana_chain", "bsc_rpc", "solana_rpc", "dexscreener", "geckoterminal", "birdeye"],
    CAP_POOLS:     ["bsc_chain", "solana_chain", "dexscreener", "geckoterminal"],
    CAP_OHLCV:     ["birdeye", "bitquery", "geckoterminal", "bsc_indexer", "solana_indexer"],
    CAP_TRADES:    ["bitquery", "birdeye", "geckoterminal", "bsc_indexer", "solana_indexer"],
    CAP_RUG:       ["bsc_chain", "solana_chain", "solana_rpc", "bsc_rpc", "rugcheck", "honeypot", "goplus", "birdeye"],
    CAP_LISTINGS:  [],                                   # dropped: CEX listings have no on-chain source
    CAP_NEW_PAIRS: ["dexscreener", "geckoterminal"],
    CAP_LAUNCH:    ["bitquery", "moralis"],
    CAP_STREAM_LAUNCHES: ["pumpportal"],
}


def _is_onchain(p) -> bool:
    return getattr(p, "onchain", False) or p.name in ONCHAIN_NAMES


class Registry:
    def __init__(self, http: HttpClient, settings: Settings):
        self.http = http
        self.settings = settings
        self.by_name: dict[str, Provider] = {
            cls.name: cls(http, settings) for cls in PROVIDER_CLASSES
        }

    def get(self, name: str) -> Provider | None:
        return self.by_name.get(name)

    def providers_for(self, cap: str, chain: Chain | None = None) -> list[Provider]:
        order = PRIORITY.get(cap, list(self.by_name))
        out: list[Provider] = []
        for name in order:
            p = self.by_name.get(name)
            if p is None or not p.enabled or cap not in p.capabilities:
                continue
            if chain is not None and Chain.parse(chain) not in p.supported_chains:
                continue
            if self.settings.onchain_only and not _is_onchain(p):
                continue   # on-chain-only mode: skip gated 3rd-party providers
            out.append(p)
        return out

    def status(self) -> list[dict]:
        rows = []
        for cls in PROVIDER_CLASSES:
            p = self.by_name[cls.name]
            rows.append({
                "name": p.name,
                "enabled": p.enabled,
                "requires_key": p.requires_key,
                "key_env": p.key_env,
                "chains": sorted(c.value for c in p.supported_chains),
                "capabilities": sorted(p.capabilities),
            })
        return rows


def build_registry(settings: Settings | None = None) -> tuple[HttpClient, Registry]:
    settings = settings or get_settings()
    http = HttpClient(timeout=settings.http_timeout, max_retries=settings.max_retries)
    return http, Registry(http, settings)
