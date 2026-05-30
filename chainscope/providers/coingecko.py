"""CoinGecko — exchange-listing presence (CEX + DEX venues) for a token.

Works keyless at a low rate; a demo or pro key raises the limit. The
contract-address endpoint resolves a chain+address to a CoinGecko coin and
carries an inline `tickers` array; we also page /tickers to capture the full
venue set, then classify each venue as CEX or DEX.

Capabilities: listings. Covers both Solana and BSC.
Docs: https://docs.coingecko.com/reference/coins-contract-address
"""
from __future__ import annotations

from ..chains import Chain, spec
from ..http import HttpError
from ..models import ExchangePresence, Ticker
from .base import CAP_LISTINGS, Provider

# Identifiers (CoinGecko `market.identifier`) we treat as decentralized exchanges.
_DEX_IDENTIFIERS = {
    "uniswap_v2", "uniswap_v3", "uniswap-v4",
    "pancakeswap_new", "pancakeswap-v3-bsc", "pancakeswap_v3_bsc",
    "sushiswap", "raydium", "raydium2", "orca", "meteora",
    "curve", "balancer", "jupiter", "gmx", "trader_joe",
    "quickswap", "aerodrome-base",
}
# Substrings that strongly imply a DEX/AMM venue.
_DEX_SUBSTRINGS = (
    "swap", "dex", "_v2", "_v3", "amm",
    "raydium", "orca", "meteora", "uniswap", "pancake", "sushi", "curve",
)


def _f(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _is_dex(identifier: str | None) -> bool:
    if not identifier:
        return False
    ident = identifier.lower()
    if ident in _DEX_IDENTIFIERS:
        return True
    return any(s in ident for s in _DEX_SUBSTRINGS)


def _exchange_type(identifier: str | None) -> str:
    return "dex" if _is_dex(identifier) else "cex"


def _ticker_from_entry(t: dict) -> Ticker:
    market = t.get("market") or {}
    conv_last = t.get("converted_last") or {}
    conv_vol = t.get("converted_volume") or {}
    return Ticker(
        source="coingecko",
        exchange=market.get("name") or market.get("identifier") or "",
        exchange_type=_exchange_type(market.get("identifier")),
        base=t.get("base"),
        target=t.get("target"),
        price_usd=_f(conv_last.get("usd")),
        volume_usd=_f(conv_vol.get("usd")),
        trust_score=t.get("trust_score"),
        trade_url=t.get("trade_url"),
        raw=t,
    )


class CoinGeckoProvider(Provider):
    name = "coingecko"
    supported_chains = frozenset({Chain.SOLANA, Chain.BSC})
    capabilities = frozenset({CAP_LISTINGS})
    requires_key = False
    key_env = "COINGECKO_API_KEY"

    def _base_and_headers(self) -> tuple[str, dict | None]:
        key = self.settings.get_key(self.key_env)
        tier = (self.settings.coingecko_tier or "demo").lower()
        if tier == "pro" and key:
            return "https://pro-api.coingecko.com/api/v3", {"x-cg-pro-api-key": key}
        if key:
            return "https://api.coingecko.com/api/v3", {"x-cg-demo-api-key": key}
        return "https://api.coingecko.com/api/v3", None

    async def get_listings(self, chain: Chain | None, address: str | None,
                           symbol: str | None = None) -> ExchangePresence | None:
        chain = Chain.parse(chain)
        if not address:
            return None
        base, headers = self._base_and_headers()
        platform = spec(chain).coingecko_platform

        try:
            coin = await self.http.get_json(
                f"{base}/coins/{platform}/contract/{address}",
                headers=headers,
                cache_ttl=300,
            )
        except HttpError as exc:
            if exc.status == 404:
                return None  # token not tracked by CoinGecko (common for new memecoins)
            raise

        coin_id = coin.get("id")
        coin_symbol = coin.get("symbol")

        # Collect tickers from the contract response, then page /tickers to merge
        # the full venue set, de-duplicating by (identifier, base, target).
        seen: set[tuple] = set()
        tickers: list[Ticker] = []

        def _add(entries: list[dict]) -> None:
            for t in entries or []:
                if not isinstance(t, dict):
                    continue
                market = t.get("market") or {}
                key = (market.get("identifier"), t.get("base"), t.get("target"))
                if key in seen:
                    continue
                seen.add(key)
                tickers.append(_ticker_from_entry(t))

        _add(coin.get("tickers") or [])

        if coin_id:
            for page in range(1, 6):
                try:
                    data = await self.http.get_json(
                        f"{base}/coins/{coin_id}/tickers",
                        params={"page": page},
                        headers=headers,
                        cache_ttl=300,
                    )
                except HttpError as exc:
                    if exc.status == 404:
                        break
                    raise
                page_tickers = (data or {}).get("tickers") or []
                if not page_tickers:
                    break
                _add(page_tickers)

        cex = sorted({t.exchange for t in tickers
                      if t.exchange_type == "cex" and t.exchange})
        dex = sorted({t.exchange for t in tickers
                      if t.exchange_type == "dex" and t.exchange})

        return ExchangePresence(
            source="coingecko",
            chain=chain,
            address=address,
            coingecko_id=coin_id,
            symbol=symbol or coin_symbol,
            cex_exchanges=cex,
            dex_exchanges=dex,
            tickers=tickers,
            raw=coin,
        )
