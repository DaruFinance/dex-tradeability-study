"""PumpPortal — real-time pump.fun launch + graduation stream (Solana only).

A single websocket carries new-token creation events and migration
(graduation) events. We subscribe to both and yield canonical Launch records
as they arrive, reconnecting transparently on connection drops.

Capabilities: stream_launches. Solana only.
Unlike the HTTP providers this connects the `websockets` library directly —
the data is a persistent stream, not a request/response fetch.
Docs: https://pumpportal.fun/data-api/real-time
"""
from __future__ import annotations

import asyncio
import json
import logging

import websockets

from ..chains import Chain
from ..models import Launch
from .base import CAP_STREAM_LAUNCHES, NotSupported, Provider

log = logging.getLogger("chainscope.pumpportal")

WS_URL = "wss://pumpportal.fun/api/data"
_SUBSCRIPTIONS = (
    {"method": "subscribeNewToken"},
    {"method": "subscribeMigration"},
)
_RECONNECT_BACKOFF = 2.0


def _f(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _parse_message(msg: dict) -> Launch | None:
    """Map a raw PumpPortal message to a Launch, or None to skip it."""
    if not isinstance(msg, dict):
        return None

    # Subscription acks look like {"message": "Successfully subscribed ..."}.
    if "message" in msg and "mint" not in msg:
        return None

    mint = msg.get("mint")
    if not mint:
        return None

    tx_type = msg.get("txType")
    name = msg.get("name")
    symbol = msg.get("symbol")

    is_migration = tx_type == "migrate" or (
        msg.get("pool") is not None and not name and tx_type != "create"
    )
    if is_migration:
        return Launch(
            source="pumpportal",
            chain=Chain.SOLANA,
            launchpad="pumpfun",
            address=mint,
            name=name,
            symbol=symbol,
            complete=True,
            migrated_pool=msg.get("pool"),
            target_dex="pumpswap",
            raw=msg,
        )

    is_new_token = tx_type == "create" or (
        (name or symbol) is not None
        and (msg.get("vSolInBondingCurve") is not None
             or msg.get("vTokensInBondingCurve") is not None
             or msg.get("bondingCurveKey") is not None)
    )
    if is_new_token:
        return Launch(
            source="pumpportal",
            chain=Chain.SOLANA,
            launchpad="pumpfun",
            address=mint,
            name=name,
            symbol=symbol,
            creator=msg.get("traderPublicKey"),
            market_cap_usd=None,  # marketCapSol is denominated in SOL, not USD
            virtual_sol_reserves=_f(msg.get("vSolInBondingCurve")),
            virtual_token_reserves=_f(msg.get("vTokensInBondingCurve")),
            complete=False,
            raw=msg,
        )

    return None


class PumpPortalProvider(Provider):
    name = "pumpportal"
    supported_chains = frozenset({Chain.SOLANA})
    capabilities = frozenset({CAP_STREAM_LAUNCHES})
    requires_key = False

    async def stream_launches(self, chain: Chain):
        """Async generator yielding pump.fun Launch records in real time.

        Reconnects on transport errors / closed connections with a brief
        backoff. Runs until the consumer stops iterating.
        """
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported

        while True:
            try:
                # Explicit recv() (not `async for raw in ws`) so that when the
                # consumer stops iterating, this generator is suspended at `yield`
                # with no pending recv and the `async with` tears down cleanly.
                async with websockets.connect(WS_URL) as ws:
                    for sub in _SUBSCRIPTIONS:
                        await ws.send(json.dumps(sub))
                    while True:
                        raw = await ws.recv()
                        try:
                            msg = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        launch = _parse_message(msg)
                        if launch is not None:
                            yield launch
            except (websockets.ConnectionClosed, OSError) as exc:
                log.debug("pumpportal ws reconnecting after %r", exc)
                await asyncio.sleep(_RECONNECT_BACKOFF)
                continue
            except (GeneratorExit, asyncio.CancelledError):
                # Consumer stopped; let the async-with above close the socket.
                raise
