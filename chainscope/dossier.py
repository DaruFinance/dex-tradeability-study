"""Per-coin full dossier: one call assembles everything we can dig on a token:
metadata, every pool, price/volume, recent trades, multi-pool depth + routing,
a cost/slippage curve by trade size, rug/safety, launch/graduation, MEV exposure,
and creator. Each section is independent and degrades gracefully, so a partial
chain still returns a useful report.

This is the "give me everything on this coin" entry point. Holder analytics
(holders_evm / holders_sol) slot in once those modules land.
"""
from __future__ import annotations

import asyncio

from . import costs
from .chains import Chain


async def build_dossier(client, chain: str | Chain, address: str, *,
                        ohlcv_tf: str = "1h", ohlcv_limit: int = 200,
                        trade_limit: int = 200,
                        sizes: tuple[float, ...] = (1_000, 50_000, 250_000)) -> dict:
    c = Chain.parse(chain)

    async def safe(coro, default=None):
        try:
            return await coro
        except Exception:
            return default

    token, pools, rug, launch = await asyncio.gather(
        safe(client.token(c, address)),
        safe(client.pools(c, address), []),
        safe(client.rug(c, address)),
        safe(client.launch(c, address)),
    )
    pools = list(pools or [])
    pools.sort(key=lambda p: p.liquidity_usd or 0.0, reverse=True)
    dominant = pools[0] if pools else None

    ohlcv: list = []
    trades: list = []
    if dominant:
        # prefer OUR on-chain indexer tape (carries block_number/log_index/maker/reserves
        # that MEV detection + cost modeling need); fall back to whatever client.trades finds.
        idx = client.registry.get("bsc_indexer" if c == Chain.BSC else "solana_indexer")
        trade_coro = (idx.get_trades(c, dominant.pair_address, limit=trade_limit)
                      if idx and idx.enabled
                      else client.trades(c, dominant.pair_address, limit=trade_limit))
        ohlcv, trades = await asyncio.gather(
            safe(client.ohlcv(c, dominant.pair_address, ohlcv_tf, ohlcv_limit), []),
            safe(trade_coro, []),
        )
        if not trades:
            trades = await safe(client.trades(c, dominant.pair_address, limit=trade_limit), [])

    # multi-pool depth + routing + cost/slippage curve by trade size
    depth = None
    cost_curve: list[dict] = []
    pool_dicts = [{
        "pair_address": p.pair_address, "dex": p.dex,
        "reserve_usd": p.liquidity_usd or 0.0,
        "fee_bps": costs.swap_fee_bps(p.dex),
        "price_usd": p.price_usd,
    } for p in pools if (p.liquidity_usd or 0) > 0]
    if pool_dicts and dominant:
        try:
            from .routing import liquidity_graph, split_route
            depth = liquidity_graph(pool_dicts)
            for s in sizes:
                routed = {}
                try:
                    routed = split_route(pool_dicts, s, "buy") or {}
                except Exception:
                    pass
                cost_curve.append({
                    "size_usd": s,
                    "routed_impact_frac": routed.get("blended_impact_frac"),
                    "vs_single_pool_frac": routed.get("vs_single_pool"),
                    "single_pool_round_trip_frac": costs.round_trip_cost_frac(
                        s, dominant.liquidity_usd, dex=dominant.dex, chain=c),
                })
        except Exception:
            pass

    # MEV exposure on the recent trade tape
    mev_summary = None
    if trades:
        try:
            from .mev import detect_sandwiches
            summ = detect_sandwiches(list(trades))
            mev_summary = {k: v for k, v in summ.items() if k != "sandwiched"}
            mev_summary["total_trades"] = len(trades)
        except Exception:
            pass

    vol24 = (token.volume_24h if token and token.volume_24h
             else (sum((p.volume_24h or 0) for p in pools) or None))
    liq = (token.liquidity_usd if token and token.liquidity_usd
           else (sum((p.liquidity_usd or 0) for p in pools) or None))
    created = (launch.created_at if launch and launch.created_at
               else min((p.created_at for p in pools if p.created_at), default=None))

    return {
        "chain": c.value, "address": address,
        "symbol": token.symbol if token else None,
        "name": token.name if token else None,
        "decimals": token.decimals if token else None,
        "price_usd": token.price_usd if token else (dominant.price_usd if dominant else None),
        "market_cap": token.market_cap if token else None,
        "fdv": token.fdv if token else None,
        "liquidity_usd": liq,
        "volume_24h": vol24,
        "created_at": created,
        "pool_count": len(pools),
        "pools": pools,
        "dominant_pool": dominant,
        "ohlcv": ohlcv or [],
        "trades": trades or [],
        "depth": depth,
        "cost_curve": cost_curve,
        "rug": rug,
        "mev": mev_summary,
        "launch": launch,
        "creator": launch.creator if launch else None,
        "token_obj": token,
        # holders: populated once holders_evm / holders_sol are wired in
        "holders": None,
    }
