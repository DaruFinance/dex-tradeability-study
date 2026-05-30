"""High-level client. Merges INDEPENDENT signals from many providers into one
rich record, with free-first fallback for single-value capabilities and full
multi-source merging for risk signals.
"""
from __future__ import annotations

import asyncio
import logging

from .chains import Chain
from .config import Settings, get_settings
from .models import ExchangePresence, Launch, OHLCV, Pool, RugReport, Token, Trade
from .providers.base import (
    CAP_LAUNCH, CAP_LISTINGS, CAP_OHLCV, CAP_POOLS, CAP_RUG, CAP_SEARCH,
    CAP_STREAM_LAUNCHES, CAP_TOKEN, CAP_TRADES, NotSupported,
)
from .registry import Registry, build_registry

log = logging.getLogger("chainscope")


def _empty(v) -> bool:
    return v is None or v == [] or v == ""


def _merge_models(models: list, cls):
    """Field-wise merge of same-type models in priority order: each field takes
    the first non-empty value. Records provenance under raw['sources']."""
    models = [m for m in models if m is not None]
    if not models:
        return None
    merged: dict = {}
    for m in models:
        d = m.model_dump(exclude={"raw", "observed_at", "source"})
        for k, v in d.items():
            if (k not in merged or _empty(merged.get(k))) and not _empty(v):
                merged[k] = v
    # backfill any required fields from the highest-priority model
    for k, v in models[0].model_dump(exclude={"raw"}).items():
        merged.setdefault(k, v)
    merged["source"] = "+".join(dict.fromkeys(m.source for m in models))
    obj = cls(**{k: v for k, v in merged.items() if k in cls.model_fields})
    obj.raw = {"sources": {m.source: m.model_dump(exclude={"raw"}) for m in models}}
    return obj


def _merge_rug(reports: list[RugReport], chain: Chain, address: str) -> RugReport | None:
    reports = [r for r in reports if r is not None]
    if not reports:
        return None

    def first(attr):
        for r in reports:
            v = getattr(r, attr)
            if not _empty(v):
                return v
        return None

    def any_true(attr):
        vals = [getattr(r, attr) for r in reports if getattr(r, attr) is not None]
        return (True in vals) if vals else None

    def max_num(attr):
        vals = [getattr(r, attr) for r in reports if getattr(r, attr) is not None]
        return max(vals) if vals else None

    flags = sorted({f for r in reports for f in (r.flags or [])})
    merged = RugReport(
        chain=chain, address=address,
        source="+".join(dict.fromkeys(r.source for r in reports)),
        risk_score=max_num("risk_score"),
        rugged=any_true("rugged"),
        is_honeypot=any_true("is_honeypot"),
        buy_tax=max_num("buy_tax"), sell_tax=max_num("sell_tax"),
        transfer_tax=max_num("transfer_tax"),
        mint_authority_active=any_true("mint_authority_active"),
        freeze_authority_active=any_true("freeze_authority_active"),
        mintable=any_true("mintable"),
        owner_address=first("owner_address"),
        ownership_renounced=first("ownership_renounced"),
        can_take_back_ownership=any_true("can_take_back_ownership"),
        hidden_owner=any_true("hidden_owner"),
        lp_locked_pct=max_num("lp_locked_pct"),
        lp_burned_pct=max_num("lp_burned_pct"),
        top10_holder_pct=max_num("top10_holder_pct"),
        creator_pct=max_num("creator_pct"),
        holder_count=max_num("holder_count"),
        is_open_source=first("is_open_source"),
        is_proxy=any_true("is_proxy"),
        transfer_pausable=any_true("transfer_pausable"),
        has_blacklist=any_true("has_blacklist"),
        has_whitelist=any_true("has_whitelist"),
        anti_whale=any_true("anti_whale"),
        flags=flags,
    )
    merged.raw = {"sources": {r.source: r.model_dump(exclude={"raw"}) for r in reports}}
    return merged


class Client:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.http, self.registry = build_registry(self.settings)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        await self.http.aclose()

    async def _safe(self, coro):
        try:
            return await coro
        except NotSupported:
            return None
        except Exception as exc:  # one bad provider shouldn't sink the aggregate
            log.debug("provider call failed: %s", exc)
            return None

    # ---- single-value capabilities (free-first fallback) ----

    async def search(self, query: str, chain: str | Chain | None = None,
                     limit: int = 30) -> list[Token]:
        c = Chain.parse(chain) if chain else None
        for p in self.registry.providers_for(CAP_SEARCH, c):
            res = await self._safe(p.search(query, c))
            if res:
                return res[:limit]
        return []

    async def pools(self, chain: str | Chain, address: str) -> list[Pool]:
        c = Chain.parse(chain)
        for p in self.registry.providers_for(CAP_POOLS, c):
            res = await self._safe(p.get_pools(c, address))
            if res:
                return res
        return []

    async def ohlcv(self, chain: str | Chain, pair_address: str, timeframe: str = "1h",
                    limit: int = 1000, before: int | None = None) -> list[OHLCV]:
        c = Chain.parse(chain)
        for p in self.registry.providers_for(CAP_OHLCV, c):
            res = await self._safe(p.get_ohlcv(c, pair_address, timeframe, limit, before))
            if res:
                return res
        return []

    async def trades(self, chain: str | Chain, pair_address: str, since: int | None = None,
                     until: int | None = None, limit: int = 1000) -> list[Trade]:
        c = Chain.parse(chain)
        for p in self.registry.providers_for(CAP_TRADES, c):
            res = await self._safe(p.get_trades(c, pair_address, since, until, limit))
            if res:
                return res
        return []

    async def listings(self, chain: str | Chain, address: str,
                       symbol: str | None = None) -> ExchangePresence | None:
        c = Chain.parse(chain)
        for p in self.registry.providers_for(CAP_LISTINGS, c):
            res = await self._safe(p.get_listings(c, address, symbol))
            if res:
                return res
        return None

    async def launch(self, chain: str | Chain, address: str) -> Launch | None:
        c = Chain.parse(chain)
        for p in self.registry.providers_for(CAP_LAUNCH, c):
            res = await self._safe(p.get_launch(c, address))
            if res:
                return res
        return None

    # ---- merged / multi-source capabilities ----

    async def token(self, chain: str | Chain, address: str) -> Token | None:
        """Merge token fields across all token providers (independent datapoints)."""
        c = Chain.parse(chain)
        providers = self.registry.providers_for(CAP_TOKEN, c)
        results = await asyncio.gather(*(self._safe(p.get_token(c, address)) for p in providers))
        return _merge_models([r for r in results if r], Token)

    async def rug(self, chain: str | Chain, address: str) -> RugReport | None:
        """Collect EVERY rug provider's verdict and merge (worst-case) into one report."""
        c = Chain.parse(chain)
        providers = self.registry.providers_for(CAP_RUG, c)
        results = await asyncio.gather(*(self._safe(p.get_rug(c, address)) for p in providers))
        return _merge_rug([r for r in results if r], c, address)

    async def profile(self, chain: str | Chain, address: str) -> dict:
        """One rich, point-in-time record combining every independent datapoint."""
        c = Chain.parse(chain)
        token, pools, rug, listings, launch = await asyncio.gather(
            self.token(c, address),
            self.pools(c, address),
            self.rug(c, address),
            self._safe(self.listings(c, address)),
            self._safe(self.launch(c, address)),
        )
        top_pool = max(pools, key=lambda p: p.liquidity_usd or 0.0) if pools else None
        return {
            "chain": c.value,
            "address": address,
            "token": token,
            "top_pool": top_pool,
            "pool_count": len(pools),
            "rug": rug,
            "listings": listings,
            "launch": launch,
        }

    # ---- launch event feeds (provider-specific) ----

    async def recent_launches(self, chain: str | Chain = Chain.SOLANA, limit: int = 100) -> list[Launch]:
        return await self._launch_feed("recent_launches", chain, limit)

    async def recent_graduated(self, chain: str | Chain = Chain.SOLANA, limit: int = 100) -> list[Launch]:
        return await self._launch_feed("recent_graduated", chain, limit)

    async def recent_bonding(self, chain: str | Chain = Chain.SOLANA, limit: int = 100) -> list[Launch]:
        return await self._launch_feed("recent_bonding", chain, limit)

    async def _launch_feed(self, method: str, chain, limit) -> list[Launch]:
        c = Chain.parse(chain)
        m = self.registry.get("moralis")
        if m and m.enabled and hasattr(m, method) and c in m.supported_chains:
            res = await self._safe(getattr(m, method)(c, limit))
            return res or []
        return []

    # ---- live stream ----

    async def stream_launches(self, chain: str | Chain = Chain.SOLANA):
        c = Chain.parse(chain)
        providers = self.registry.providers_for(CAP_STREAM_LAUNCHES, c)
        if not providers:
            raise NotSupported(f"no launch stream provider for {c.value}")
        gen = providers[0].stream_launches(c)
        try:
            async for launch in gen:
                yield launch
        finally:
            await gen.aclose()  # close the inner ws while the loop is still alive
