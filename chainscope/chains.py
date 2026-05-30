"""Chain definitions, per-chain identifiers used by each provider, and address validation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Chain(str, Enum):
    SOLANA = "solana"
    BSC = "bsc"

    @classmethod
    def parse(cls, value: "str | Chain") -> "Chain":
        if isinstance(value, Chain):
            return value
        v = str(value).strip().lower()
        aliases = {
            "sol": cls.SOLANA,
            "solana": cls.SOLANA,
            "bsc": cls.BSC,
            "bnb": cls.BSC,
            "binance-smart-chain": cls.BSC,
            "binancecoin": cls.BSC,
            "bnb-smart-chain": cls.BSC,
        }
        if v in aliases:
            return aliases[v]
        raise ValueError(f"unknown chain: {value!r} (supported: solana, bsc)")


@dataclass(frozen=True)
class ChainSpec:
    chain: Chain
    is_evm: bool
    native_symbol: str
    # provider-specific network identifiers
    dexscreener_id: str          # chainId on DexScreener
    geckoterminal_network: str   # network slug on GeckoTerminal
    coingecko_platform: str      # asset_platform id on CoinGecko
    goplus_chain_id: str | None  # numeric chain id for GoPlus EVM endpoint; None => dedicated endpoint
    default_rpc: str


_SOLANA = ChainSpec(
    chain=Chain.SOLANA,
    is_evm=False,
    native_symbol="SOL",
    dexscreener_id="solana",
    geckoterminal_network="solana",
    coingecko_platform="solana",
    goplus_chain_id=None,  # GoPlus uses /solana/token_security
    default_rpc="https://api.mainnet-beta.solana.com",
)

_BSC = ChainSpec(
    chain=Chain.BSC,
    is_evm=True,
    native_symbol="BNB",
    dexscreener_id="bsc",
    geckoterminal_network="bsc",
    coingecko_platform="binance-smart-chain",
    goplus_chain_id="56",
    # publicnode serves eth_getLogs + archival reads; the official bsc-dataseed
    # silently returns empty getLogs, which breaks event scanning.
    default_rpc="https://bsc.publicnode.com",
)

SPECS: dict[Chain, ChainSpec] = {Chain.SOLANA: _SOLANA, Chain.BSC: _BSC}


def spec(chain: "str | Chain") -> ChainSpec:
    return SPECS[Chain.parse(chain)]


_EVM_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_BASE58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def is_valid_address(chain: "str | Chain", address: str) -> bool:
    c = Chain.parse(chain)
    if c == Chain.BSC:
        return bool(_EVM_RE.match(address or ""))
    return bool(_BASE58_RE.match(address or ""))


def normalize_address(chain: "str | Chain", address: str) -> str:
    """Canonicalize an address for use as a storage/cache key (lowercase EVM, raw base58 Solana)."""
    c = Chain.parse(chain)
    a = (address or "").strip()
    return a.lower() if c == Chain.BSC else a


def detect_chain(address: str) -> Chain | None:
    """Best-effort guess of which chain an address belongs to, by format."""
    if _EVM_RE.match(address or ""):
        return Chain.BSC
    if _BASE58_RE.match(address or ""):
        return Chain.SOLANA
    return None
