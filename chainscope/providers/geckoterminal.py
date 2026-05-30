"""GeckoTerminal — free, no API key (~30 req/min, rate limited by HttpClient).

Capabilities: search, token, pools, ohlcv, new_pairs.
Covers both Solana and BSC. Adds OHLCV on top of the DexScreener baseline.
Docs: https://apiguide.geckoterminal.com/
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..chains import Chain, normalize_address, spec
from ..http import HttpError
from ..models import OHLCV, Pool, Token, Trade
from .base import (
    CAP_NEW_PAIRS,
    CAP_OHLCV,
    CAP_POOLS,
    CAP_SEARCH,
    CAP_TOKEN,
    CAP_TRADES,
    Provider,
)

BASE = "https://api.geckoterminal.com/api/v2"
ACCEPT_HEADER = {"Accept": "application/json;version=20230302"}

# canonical timeframe -> (gt_timeframe, aggregate)
_TIMEFRAMES = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "15m": ("minute", 15),
    "1h": ("hour", 1),
    "4h": ("hour", 4),
    "12h": ("hour", 12),
    "1d": ("day", 1),
}


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


def _iso_to_dt(s) -> datetime | None:
    if not s:
        return None
    try:
        # GeckoTerminal uses trailing "Z" for UTC, which fromisoformat
        # rejects on older Pythons -> normalize to +00:00.
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sec_to_dt(ts) -> datetime | None:
    v = _i(ts)
    if v is None:
        return None
    return datetime.fromtimestamp(v, tz=timezone.utc)


def _strip_net(net: str, token_id) -> str | None:
    """relationships token id is "{net}_{address}" -> the address."""
    if not token_id:
        return None
    prefix = f"{net}_"
    s = str(token_id)
    return s[len(prefix):] if s.startswith(prefix) else s


def _rel_token_address(net: str, rel: dict, side: str) -> str | None:
    data = ((rel or {}).get(side) or {}).get("data") or {}
    return _strip_net(net, data.get("id"))


def _pool_to_model(chain: Chain, net: str, p: dict) -> Pool:
    attrs = p.get("attributes") or {}
    rels = p.get("relationships") or {}
    vol = attrs.get("volume_usd") or {}
    chg = attrs.get("price_change_percentage") or {}
    txns_24h = (attrs.get("transactions") or {}).get("h24") or {}
    dex = ((rels.get("dex") or {}).get("data") or {}).get("id")
    return Pool(
        source="geckoterminal",
        chain=chain,
        dex=dex,
        pair_address=attrs.get("address") or "",
        base_address=_rel_token_address(net, rels, "base_token"),
        quote_address=_rel_token_address(net, rels, "quote_token"),
        price_usd=_f(attrs.get("base_token_price_usd")),
        price_native=_f(attrs.get("base_token_price_native_currency")),
        liquidity_usd=_f(attrs.get("reserve_in_usd")),
        fdv=_f(attrs.get("fdv_usd")),
        market_cap=_f(attrs.get("market_cap_usd")),
        volume_5m=_f(vol.get("m5")),
        volume_1h=_f(vol.get("h1")),
        volume_24h=_f(vol.get("h24")),
        price_change_24h=_f(chg.get("h24")),
        txns_24h_buys=_i(txns_24h.get("buys")),
        txns_24h_sells=_i(txns_24h.get("sells")),
        created_at=_iso_to_dt(attrs.get("pool_created_at")),
        raw=p,
    )


def _token_from_attrs(chain: Chain, address: str, attrs: dict, raw: dict) -> Token:
    vol = attrs.get("volume_usd") or {}
    return Token(
        source="geckoterminal",
        chain=chain,
        address=attrs.get("address") or address,
        symbol=attrs.get("symbol"),
        name=attrs.get("name"),
        decimals=_i(attrs.get("decimals")),
        total_supply=_f(attrs.get("total_supply")),
        price_usd=_f(attrs.get("price_usd")),
        market_cap=_f(attrs.get("market_cap_usd")),
        fdv=_f(attrs.get("fdv_usd")),
        liquidity_usd=_f(attrs.get("total_reserve_in_usd")),
        volume_24h=_f(vol.get("h24")),
        image_url=attrs.get("image_url"),
        raw=raw,
    )


class GeckoTerminalProvider(Provider):
    name = "geckoterminal"
    supported_chains = frozenset({Chain.SOLANA, Chain.BSC})
    capabilities = frozenset(
        {CAP_SEARCH, CAP_TOKEN, CAP_POOLS, CAP_OHLCV, CAP_TRADES, CAP_NEW_PAIRS}
    )
    requires_key = False

    async def get_token(self, chain: Chain, address: str) -> Token | None:
        chain = Chain.parse(chain)
        net = spec(chain).geckoterminal_network
        try:
            data = await self.http.get_json(
                f"{BASE}/networks/{net}/tokens/{address}",
                headers=ACCEPT_HEADER,
                cache_ttl=20,
            )
        except HttpError as exc:
            if exc.status == 404:
                return None
            raise
        node = data.get("data") or {}
        attrs = node.get("attributes") or {}
        if not attrs:
            return None
        return _token_from_attrs(chain, address, attrs, node)

    async def get_pools(self, chain: Chain, address: str) -> list[Pool]:
        chain = Chain.parse(chain)
        net = spec(chain).geckoterminal_network
        try:
            data = await self.http.get_json(
                f"{BASE}/networks/{net}/tokens/{address}/pools",
                headers=ACCEPT_HEADER,
                cache_ttl=20,
            )
        except HttpError as exc:
            if exc.status == 404:
                return []
            raise
        pools = data.get("data") or []
        return [
            _pool_to_model(chain, net, p)
            for p in pools
            if (p.get("attributes") or {}).get("address")
        ]

    async def search(self, query: str, chain: Chain | None = None) -> list[Token]:
        params: dict = {"query": query}
        if chain is not None:
            chain = Chain.parse(chain)
            params["network"] = spec(chain).geckoterminal_network
        data = await self.http.get_json(
            f"{BASE}/search/pools",
            params=params,
            headers=ACCEPT_HEADER,
            cache_ttl=20,
        )
        pools = data.get("data") or []

        # Group pools by (chain, base token address). The base token id encodes
        # the network ("{net}_{address}"), so we can derive the chain per pool.
        groups: dict[tuple[str, str], list[dict]] = {}
        for p in pools:
            rels = p.get("relationships") or {}
            base_id = (((rels.get("base_token") or {}).get("data") or {}).get("id"))
            if not base_id:
                continue
            net_prefix = str(base_id).split("_", 1)[0]
            try:
                pc = Chain.parse(net_prefix)
            except ValueError:
                continue
            if chain is not None and pc != chain:
                continue
            net = spec(pc).geckoterminal_network
            addr = _strip_net(net, base_id)
            if not addr:
                continue
            groups.setdefault((pc.value, normalize_address(pc, addr)), []).append(p)

        out: list[Token] = []
        for (cid, _addr), grp in groups.items():
            pc = Chain.parse(cid)
            net = spec(pc).geckoterminal_network
            # canonical pool = max reserve_in_usd
            best = max(
                grp,
                key=lambda p: _f((p.get("attributes") or {}).get("reserve_in_usd")) or 0.0,
            )
            battrs = best.get("attributes") or {}
            brels = best.get("relationships") or {}
            base_addr = _rel_token_address(net, brels, "base_token") or _addr
            vol_24h = sum(
                _f((p.get("attributes") or {}).get("volume_usd", {}).get("h24")) or 0.0
                for p in grp
            )
            liq = sum(
                _f((p.get("attributes") or {}).get("reserve_in_usd")) or 0.0
                for p in grp
            )
            out.append(
                Token(
                    source="geckoterminal",
                    chain=pc,
                    address=base_addr,
                    price_usd=_f(battrs.get("base_token_price_usd")),
                    price_native=_f(battrs.get("base_token_price_native_currency")),
                    fdv=_f(battrs.get("fdv_usd")),
                    market_cap=_f(battrs.get("market_cap_usd")),
                    liquidity_usd=liq,
                    volume_24h=vol_24h,
                    pair_count=len(grp),
                    raw=best,
                )
            )
        out.sort(key=lambda t: t.liquidity_usd or 0.0, reverse=True)
        return out

    async def get_new_pairs(self, chain: Chain) -> list[Pool]:
        chain = Chain.parse(chain)
        net = spec(chain).geckoterminal_network
        data = await self.http.get_json(
            f"{BASE}/networks/{net}/new_pools",
            headers=ACCEPT_HEADER,
            cache_ttl=30,
        )
        pools = data.get("data") or []
        return [
            _pool_to_model(chain, net, p)
            for p in pools
            if (p.get("attributes") or {}).get("address")
        ]

    async def get_trades(self, chain: Chain, pair_address: str,
                         since: int | None = None, until: int | None = None,
                         limit: int = 1000) -> list[Trade]:
        """Recent trade tape (~last 300 swaps). GeckoTerminal has no time-range
        params, so since/until are applied client-side. For deep history use Bitquery."""
        chain = Chain.parse(chain)
        net = spec(chain).geckoterminal_network
        try:
            data = await self.http.get_json(
                f"{BASE}/networks/{net}/pools/{pair_address}/trades",
                headers=ACCEPT_HEADER,
                cache_ttl=10,
            )
        except HttpError as exc:
            if exc.status == 404:
                return []
            raise
        out: list[Trade] = []
        for t in (data.get("data") or []):
            a = t.get("attributes") or {}
            ts = _iso_to_dt(a.get("block_timestamp"))
            if ts is None:
                continue
            epoch = int(ts.timestamp())
            if since is not None and epoch < since:
                continue
            if until is not None and epoch > until:
                continue
            kind = a.get("kind")  # "buy" | "sell" (from the base token's view)
            if kind == "buy":
                base_amt, quote_amt = a.get("to_token_amount"), a.get("from_token_amount")
                price_usd = _f(a.get("price_to_in_usd"))
            else:
                base_amt, quote_amt = a.get("from_token_amount"), a.get("to_token_amount")
                price_usd = _f(a.get("price_from_in_usd"))
            out.append(
                Trade(
                    source="geckoterminal",
                    chain=chain,
                    pair_address=pair_address,
                    block_time=ts,
                    tx_hash=a.get("tx_hash"),
                    side=kind,
                    price_usd=price_usd,
                    amount_base=_f(base_amt),
                    amount_quote=_f(quote_amt),
                    amount_usd=_f(a.get("volume_in_usd")),
                    maker=a.get("tx_from_address"),
                    raw=t,
                )
            )
        out.sort(key=lambda x: x.block_time, reverse=True)
        return out[:limit]

    async def get_ohlcv(self, chain: Chain, pair_address: str, timeframe: str = "1h",
                        limit: int = 1000, before: int | None = None) -> list[OHLCV]:
        chain = Chain.parse(chain)
        net = spec(chain).geckoterminal_network
        gt_tf, agg = _TIMEFRAMES.get(timeframe, ("hour", 1))
        params: dict = {"aggregate": agg, "limit": limit, "currency": "usd"}
        if before is not None:
            params["before_timestamp"] = before
        try:
            data = await self.http.get_json(
                f"{BASE}/networks/{net}/pools/{pair_address}/ohlcv/{gt_tf}",
                params=params,
                headers=ACCEPT_HEADER,
                cache_ttl=60,
            )
        except HttpError as exc:
            if exc.status == 404:
                return []
            raise
        attrs = (data.get("data") or {}).get("attributes") or {}
        rows = attrs.get("ohlcv_list") or []
        out: list[OHLCV] = []
        for row in rows:
            if not row or len(row) < 6:
                continue
            ts = _sec_to_dt(row[0])
            o, h, l, c = _f(row[1]), _f(row[2]), _f(row[3]), _f(row[4])
            if ts is None or o is None or h is None or l is None or c is None:
                continue
            out.append(
                OHLCV(
                    source="geckoterminal",
                    chain=chain,
                    pair_address=pair_address,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=_f(row[5]),
                )
            )
        return out
