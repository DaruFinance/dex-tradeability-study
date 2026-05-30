"""DexScreener: free, no API key. Reference implementation other providers mirror.

Capabilities: search, token, pools, new_pairs.
Covers both Solana and BSC. No OHLCV, no holders (use GeckoTerminal / Birdeye for those).
Docs: https://docs.dexscreener.com/api/reference
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..chains import Chain, normalize_address, spec
from ..http import HttpError
from ..models import Pool, Token
from .base import CAP_NEW_PAIRS, CAP_POOLS, CAP_SEARCH, CAP_TOKEN, Provider

BASE = "https://api.dexscreener.com"


def _f(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _i(x) -> int | None:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _ms_to_dt(ms) -> datetime | None:
    v = _i(ms)
    if not v:
        return None
    return datetime.fromtimestamp(v / 1000, tz=timezone.utc)


def _pair_to_pool(chain: Chain, p: dict) -> Pool:
    liq = p.get("liquidity") or {}
    vol = p.get("volume") or {}
    chg = p.get("priceChange") or {}
    txns = (p.get("txns") or {}).get("h24") or {}
    base = p.get("baseToken") or {}
    quote = p.get("quoteToken") or {}
    return Pool(
        source="dexscreener",
        chain=chain,
        dex=p.get("dexId"),
        pair_address=p.get("pairAddress") or "",
        base_address=base.get("address"),
        base_symbol=base.get("symbol"),
        quote_address=quote.get("address"),
        quote_symbol=quote.get("symbol"),
        price_usd=_f(p.get("priceUsd")),
        price_native=_f(p.get("priceNative")),
        liquidity_usd=_f(liq.get("usd")),
        liquidity_base=_f(liq.get("base")),
        liquidity_quote=_f(liq.get("quote")),
        fdv=_f(p.get("fdv")),
        market_cap=_f(p.get("marketCap")),
        volume_5m=_f(vol.get("m5")),
        volume_1h=_f(vol.get("h1")),
        volume_24h=_f(vol.get("h24")),
        price_change_24h=_f(chg.get("h24")),
        txns_24h_buys=_i(txns.get("buys")),
        txns_24h_sells=_i(txns.get("sells")),
        created_at=_ms_to_dt(p.get("pairCreatedAt")),
        url=p.get("url"),
        raw=p,
    )


def _aggregate_token(chain: Chain, address: str, pairs: list[dict]) -> Token | None:
    if not pairs:
        return None
    key = normalize_address(chain, address)
    mine = [p for p in pairs
            if normalize_address(chain, (p.get("baseToken") or {}).get("address", "")) == key]
    if not mine:
        mine = pairs
    best = max(mine, key=lambda p: _f((p.get("liquidity") or {}).get("usd")) or 0.0)
    base = best.get("baseToken") or {}
    info = best.get("info") or {}
    vol = best.get("volume") or {}
    chg = best.get("priceChange") or {}
    txns = (best.get("txns") or {}).get("h24") or {}

    def _sum(field: str, window: str) -> float:
        return sum(_f((p.get(field) or {}).get(window)) or 0.0 for p in mine)

    created = [c for c in (_ms_to_dt(p.get("pairCreatedAt")) for p in mine) if c]
    return Token(
        source="dexscreener",
        chain=chain,
        address=address,
        symbol=base.get("symbol"),
        name=base.get("name"),
        price_usd=_f(best.get("priceUsd")),
        price_native=_f(best.get("priceNative")),
        market_cap=_f(best.get("marketCap")),
        fdv=_f(best.get("fdv")),
        liquidity_usd=_sum("liquidity", "usd"),
        volume_5m=_sum("volume", "m5"),
        volume_1h=_sum("volume", "h1"),
        volume_24h=_sum("volume", "h24"),
        price_change_5m=_f(chg.get("m5")),
        price_change_1h=_f(chg.get("h1")),
        price_change_24h=_f(chg.get("h24")),
        txns_24h_buys=_i(txns.get("buys")),
        txns_24h_sells=_i(txns.get("sells")),
        pair_count=len(mine),
        created_at=min(created) if created else None,
        image_url=info.get("imageUrl"),
        websites=[w.get("url") for w in (info.get("websites") or []) if w.get("url")],
        socials=[s.get("url") for s in (info.get("socials") or []) if s.get("url")],
        raw=best,
    )


class DexScreenerProvider(Provider):
    name = "dexscreener"
    supported_chains = frozenset({Chain.SOLANA, Chain.BSC})
    capabilities = frozenset({CAP_SEARCH, CAP_TOKEN, CAP_POOLS, CAP_NEW_PAIRS})

    async def search(self, query: str, chain: Chain | None = None) -> list[Token]:
        data = await self.http.get_json(
            f"{BASE}/latest/dex/search", params={"q": query}, cache_ttl=20
        )
        pairs = data.get("pairs") or []
        # group pairs by (chainId, baseToken.address)
        groups: dict[tuple[str, str], list[dict]] = {}
        for p in pairs:
            cid = p.get("chainId")
            try:
                pc = Chain.parse(cid)
            except ValueError:
                continue
            if chain is not None and pc != Chain.parse(chain):
                continue
            addr = (p.get("baseToken") or {}).get("address")
            if not addr:
                continue
            groups.setdefault((pc.value, normalize_address(pc, addr)), []).append(p)
        out = []
        for (cid, _addr), grp in groups.items():
            tok = _aggregate_token(Chain.parse(cid), (grp[0].get("baseToken") or {}).get("address"), grp)
            if tok:
                out.append(tok)
        out.sort(key=lambda t: t.liquidity_usd or 0.0, reverse=True)
        return out

    async def get_token(self, chain: Chain, address: str) -> Token | None:
        chain = Chain.parse(chain)
        try:
            pairs = await self.http.get_json(
                f"{BASE}/token-pairs/v1/{spec(chain).dexscreener_id}/{address}", cache_ttl=20
            )
        except HttpError as exc:
            if exc.status == 404:
                return None
            raise
        if isinstance(pairs, dict):
            pairs = pairs.get("pairs") or []
        return _aggregate_token(chain, address, pairs or [])

    async def get_pools(self, chain: Chain, address: str) -> list[Pool]:
        chain = Chain.parse(chain)
        try:
            pairs = await self.http.get_json(
                f"{BASE}/token-pairs/v1/{spec(chain).dexscreener_id}/{address}", cache_ttl=20
            )
        except HttpError as exc:
            if exc.status == 404:
                return []
            raise
        if isinstance(pairs, dict):
            pairs = pairs.get("pairs") or []
        return [_pair_to_pool(chain, p) for p in (pairs or []) if p.get("pairAddress")]

    async def get_new_pairs(self, chain: Chain) -> list[Pool]:
        """Approximate new-pair discovery via boosted/profiled tokens."""
        chain = Chain.parse(chain)
        data = await self.http.get_json(f"{BASE}/token-profiles/latest/v1", cache_ttl=30)
        profiles = data if isinstance(data, list) else (data.get("profiles") or [])
        out: list[Pool] = []
        for prof in profiles:
            if prof.get("chainId") != spec(chain).dexscreener_id:
                continue
            addr = prof.get("tokenAddress")
            if not addr:
                continue
            out.extend(await self.get_pools(chain, addr))
        return out
