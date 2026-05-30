"""Birdeye — deep historical price/trade source for backtests. Requires BIRDEYE_API_KEY.

Capabilities: token, ohlcv, trades, rug. Covers Solana and BSC.

Birdeye prices/keys data by *token* address, so we pass `pair_address` through as
the Birdeye `address` param. The free model fields it fills (multi-year OHLCV bars
and the per-swap trade tape) are exactly what feeds DEX-trade fill simulation.

Verified endpoints (docs.birdeye.so, May 2026, v3 current):
  OHLCV    GET /defi/v3/ohlcv                  https://docs.birdeye.so/reference/get-defi-v3-ohlcv
  Trades   GET /defi/txs/token/seek_by_time    https://docs.birdeye.so/reference/get-defi-txs-token-seek_by_time
  Overview GET /defi/token_overview            https://docs.birdeye.so/reference/get-defi-token_overview
  Security GET /defi/token_security            https://docs.birdeye.so/reference/get-defi-token_security

Every request carries headers {"X-API-KEY": <key>, "x-chain": "solana"|"bsc"}.
Responses are wrapped as {"success": bool, "data": ...}; we guard data extraction.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from ..chains import Chain
from ..http import HttpError
from ..models import OHLCV, RugReport, Token, Trade
from .base import CAP_OHLCV, CAP_RUG, CAP_TOKEN, CAP_TRADES, Provider

BASE = "https://public-api.birdeye.so"

# How Birdeye labels each chain in the x-chain header.
_X_CHAIN = {Chain.SOLANA: "solana", Chain.BSC: "bsc"}

# Canonical chainscope timeframe -> Birdeye `type`. Birdeye supports:
# 1s 15s 30s 1m 3m 5m 15m 30m 1H 2H 4H 6H 8H 12H 1D 3D 1W 1M
_TF_TO_TYPE = {
    "1s": "1s", "15s": "15s", "30s": "30s",
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "8h": "8H", "12h": "12H",
    "1d": "1D", "3d": "3D", "1w": "1W", "1mo": "1M", "1M": "1M",
}

# Approx seconds per bar, to compute a paging time window for OHLCV.
_TF_SECONDS = {
    "1s": 1, "15s": 15, "30s": 30,
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800, "1mo": 2592000, "1M": 2592000,
}

_OHLCV_MAX = 1000      # records we request per OHLCV page (API hard cap is 5000)
_TXS_MAX = 100         # /defi/txs/token/seek_by_time hard cap on `limit`


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


def _b(x) -> bool | None:
    """Truthy/falsy interpretation that preserves None for absent fields."""
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "yes", "y"):
            return True
        if s in ("", "0", "false", "no", "n", "null", "none"):
            return False
        return True            # non-empty descriptive string => present/active
    return bool(x)


def _s_to_dt(s) -> datetime | None:
    v = _i(s)
    if v is None:
        return None
    return datetime.fromtimestamp(v, tz=timezone.utc)


def _first(d: dict, *keys):
    """First present (not-None) value among keys; tolerates absent keys."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _frac(x) -> float | None:
    """Normalize a holder-percentage to a 0..1 fraction. Birdeye usually returns a
    fraction already; if a value >1 slips through it is treated as a percent."""
    v = _f(x)
    if v is None:
        return None
    return v / 100.0 if v > 1.0 else v


class BirdeyeProvider(Provider):
    name = "birdeye"
    supported_chains = frozenset({Chain.SOLANA, Chain.BSC})
    capabilities = frozenset({CAP_TOKEN, CAP_OHLCV, CAP_TRADES, CAP_RUG})
    requires_key = True
    key_env = "BIRDEYE_API_KEY"

    # ---- helpers -------------------------------------------------------

    def _headers(self, chain: Chain) -> dict:
        return {
            "X-API-KEY": self.settings.get_key(self.key_env) or "",
            "x-chain": _X_CHAIN[Chain.parse(chain)],
        }

    @staticmethod
    def _data(resp) -> dict:
        """Unwrap {"success": bool, "data": ...}; always hand back a dict."""
        if not isinstance(resp, dict):
            return {}
        data = resp.get("data")
        return data if isinstance(data, dict) else {}

    # ---- ohlcv ---------------------------------------------------------

    async def get_ohlcv(self, chain: Chain, pair_address: str, timeframe: str = "1h",
                        limit: int = 1000, before: int | None = None) -> list[OHLCV]:
        chain = Chain.parse(chain)
        bird_type = _TF_TO_TYPE.get(timeframe, _TF_TO_TYPE.get(timeframe.lower()))
        if bird_type is None:
            return []
        bar_sec = _TF_SECONDS.get(timeframe, _TF_SECONDS.get(timeframe.lower(), 3600))
        headers = self._headers(chain)

        time_to = int(before) if before else int(time.time())
        remaining = max(int(limit), 0)
        bars: list[OHLCV] = []
        seen: set[int] = set()

        # Page backwards in time until we have `limit` bars or the API runs dry.
        while remaining > 0:
            page = min(remaining, _OHLCV_MAX)
            # Pad the window so the API has room to return a full page.
            time_from = time_to - bar_sec * (page + 2)
            if time_from < 0:
                time_from = 0
            try:
                resp = await self.http.get_json(
                    f"{BASE}/defi/v3/ohlcv",
                    params={
                        "address": pair_address,
                        "type": bird_type,
                        "time_from": time_from,
                        "time_to": time_to,
                        "currency": "usd",
                    },
                    headers=headers,
                    cache_ttl=60,
                )
            except HttpError as exc:
                if exc.status in (401, 403, 404):
                    break
                raise
            items = self._data(resp).get("items") or []
            if not items:
                break

            new_oldest = time_to
            added = 0
            for it in items:
                ut = _i(_first(it, "unixTime", "unix_time", "t"))
                if ut is None or ut in seen:
                    continue
                seen.add(ut)
                ts = _s_to_dt(ut)
                if ts is None:
                    continue
                bars.append(OHLCV(
                    source="birdeye",
                    chain=chain,
                    pair_address=pair_address,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=_f(_first(it, "o", "open")) or 0.0,
                    high=_f(_first(it, "h", "high")) or 0.0,
                    low=_f(_first(it, "l", "low")) or 0.0,
                    close=_f(_first(it, "c", "close")) or 0.0,
                    volume=_f(_first(it, "v", "volume", "vUSD", "v_usd")),
                    reserve_usd=None,
                    trade_count=_i(_first(it, "trade", "trades", "txns", "nTrade")),
                    raw=it,
                ))
                added += 1
                if ut < new_oldest:
                    new_oldest = ut

            remaining -= added
            # Advance the window strictly before the oldest bar we just got.
            if added == 0 or new_oldest >= time_to:
                break
            time_to = new_oldest - 1

        bars.sort(key=lambda o: o.timestamp)
        return bars[-int(limit):] if limit and len(bars) > limit else bars

    # ---- trades --------------------------------------------------------

    async def get_trades(self, chain: Chain, pair_address: str,
                         since: int | None = None, until: int | None = None,
                         limit: int = 1000) -> list[Trade]:
        chain = Chain.parse(chain)
        headers = self._headers(chain)
        remaining = max(int(limit), 0)
        offset = 0
        out: list[Trade] = []

        while remaining > 0 and offset < 10000:
            page = min(remaining, _TXS_MAX)
            params: dict = {
                "address": pair_address,
                "offset": offset,
                "limit": page,
                "tx_type": "swap",
            }
            if since is not None:
                params["after_time"] = int(since)
            if until is not None:
                params["before_time"] = int(until)
            try:
                resp = await self.http.get_json(
                    f"{BASE}/defi/txs/token/seek_by_time",
                    params=params,
                    headers=headers,
                    # trades: no cache
                )
            except HttpError as exc:
                if exc.status in (401, 403, 404):
                    break
                raise
            data = self._data(resp)
            items = data.get("items") or []
            if not items:
                break
            for it in items:
                out.append(self._to_trade(chain, pair_address, it))
            got = len(items)
            remaining -= got
            offset += got
            if got < page:
                break

        return out[:int(limit)] if limit else out

    @staticmethod
    def _to_trade(chain: Chain, pair_address: str, it: dict) -> Trade:
        # Birdeye swaps carry a base/quote pair as `from`/`to` legs plus a `side`.
        frm = it.get("from") or {}
        to = it.get("to") or {}
        side = it.get("side")
        if side is None:
            tt = (it.get("txType") or it.get("type") or "").lower()
            side = "buy" if tt in ("buy", "add") else ("sell" if tt in ("sell", "remove") else None)
        side = side.lower() if isinstance(side, str) else side

        # base = the token we're querying; pick the leg that matches its address.
        base_leg, quote_leg = frm, to
        f_addr = (frm.get("address") or "").lower()
        t_addr = (to.get("address") or "").lower()
        pa = (pair_address or "").lower()
        if t_addr == pa and f_addr != pa:
            base_leg, quote_leg = to, frm
        elif f_addr == pa:
            base_leg, quote_leg = frm, to

        amount_base = _f(_first(base_leg, "uiAmount", "ui_amount", "amount"))
        amount_quote = _f(_first(quote_leg, "uiAmount", "ui_amount", "amount"))

        return Trade(
            source="birdeye",
            chain=chain,
            pair_address=pair_address,
            block_time=_s_to_dt(_first(it, "blockUnixTime", "block_unix_time", "unixTime")),
            tx_hash=_first(it, "txHash", "tx_hash"),
            side=side,
            dex=_first(it, "source", "dex", "poolId", "exchange"),
            price_usd=_f(_first(it, "priceUSD", "price_usd", "price",
                                "pricePair", "tokenPrice")),
            price_native=_f(_first(it, "priceNative", "price_native")),
            amount_base=amount_base,
            amount_quote=amount_quote,
            amount_usd=_f(_first(it, "volumeUSD", "volume_usd", "volumeUsd", "valueUsd")),
            reserve_usd=None,
            maker=_first(it, "owner", "maker", "wallet", "txFrom"),
            raw=it,
        )

    # ---- token ---------------------------------------------------------

    async def get_token(self, chain: Chain, address: str) -> Token | None:
        chain = Chain.parse(chain)
        try:
            resp = await self.http.get_json(
                f"{BASE}/defi/token_overview",
                params={"address": address},
                headers=self._headers(chain),
                cache_ttl=20,
            )
        except HttpError as exc:
            if exc.status in (401, 403, 404):
                return None
            raise
        d = self._data(resp)
        if not d:
            return None
        return Token(
            source="birdeye",
            chain=chain,
            address=address,
            symbol=_first(d, "symbol"),
            name=_first(d, "name"),
            decimals=_i(_first(d, "decimals")),
            total_supply=_f(_first(d, "totalSupply", "supply", "circulatingSupply")),
            price_usd=_f(_first(d, "price")),
            market_cap=_f(_first(d, "marketCap", "mc")),
            fdv=_f(_first(d, "fdv", "realMc")),
            liquidity_usd=_f(_first(d, "liquidity")),
            volume_1h=_f(_first(d, "v1hUSD", "v1h", "volume1h")),
            volume_24h=_f(_first(d, "v24hUSD", "v24h", "volume24h")),
            price_change_1h=_f(_first(d, "priceChange1hPercent")),
            price_change_24h=_f(_first(d, "priceChange24hPercent")),
            holder_count=_i(_first(d, "holder", "holders", "holderCount")),
            raw=d,
        )

    # ---- rug -----------------------------------------------------------

    async def get_rug(self, chain: Chain, address: str) -> RugReport | None:
        chain = Chain.parse(chain)
        try:
            resp = await self.http.get_json(
                f"{BASE}/defi/token_security",
                params={"address": address},
                headers=self._headers(chain),
                cache_ttl=300,
            )
        except HttpError as exc:
            if exc.status in (401, 403, 404):
                return None
            raise
        d = self._data(resp)
        if not d:
            return None

        mint_auth = _first(d, "mintAuthority", "mintable")
        freeze_auth = _first(d, "freezeAuthority", "freezeable", "freezable")
        mutable = _b(_first(d, "mutableMetadata", "metadataMutable", "mutable"))
        transfer_fee = _f(_first(d, "transferFeeData", "transferFee"))
        # transferFeeData may be a nested object carrying the bps fee.
        if transfer_fee is None:
            tfd = d.get("transferFeeData")
            if isinstance(tfd, dict):
                transfer_fee = _frac(_first(tfd, "feePercent", "transferFeePercent")) or \
                    (_f(_first(tfd, "maximumFee", "newerTransferFee")))

        flags: list[str] = []
        if _b(mint_auth):
            flags.append("mint_authority_active")
        if _b(freeze_auth):
            flags.append("freeze_authority_active")
        if mutable:
            flags.append("mutable_metadata")
        if _b(_first(d, "transferFeeEnable")):
            flags.append("transfer_fee_enabled")
        if transfer_fee:
            flags.append("transfer_fee")

        return RugReport(
            source="birdeye",
            chain=chain,
            address=address,
            mint_authority_active=_b(mint_auth),
            freeze_authority_active=_b(freeze_auth),
            mintable=_b(_first(d, "mintable")),
            transfer_tax=transfer_fee,
            top10_holder_pct=_frac(_first(d, "top10HolderPercent",
                                          "top10HoldersPercent", "top10UserPercent")),
            creator_pct=_frac(_first(d, "creatorPercentage", "creatorPercent")),
            owner_address=_first(d, "ownerAddress", "owner"),
            is_open_source=_b(_first(d, "isOpenSource")),
            holder_count=_i(_first(d, "holderCount", "holder")),
            flags=flags,
            raw=d,
        )
