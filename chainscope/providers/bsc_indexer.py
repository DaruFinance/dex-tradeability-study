"""BSC chain indexer: reconstructs the trade tape, OHLCV, and per-block LIQUIDITY
directly from raw PancakeSwap-V2 logs via eth_getLogs. No third-party index, no key.

This is the data the paid platforms charge to serve; here we dig it from the chain:
  - per-block reserves  <- V2 `Sync(uint112,uint112)` events  (the slippage input)
  - trades              <- V2 `Swap(...)` events
  - OHLCV               <- bucketed from the above
Deep history is fetched in chunked block ranges, because free public RPCs cap the
getLogs span. Point at a local archive node (BSC_ARCHIVE_RPC_URL) for full speed.

Pricing is in USD when the pool is quoted in a known stablecoin (USDT/USDC/BUSD);
otherwise price is in the quote token (native) and USD fields are left None.

Handles BOTH PancakeSwap V2 (Sync/Swap; reserves = real token balances) and V3
(concentrated liquidity: price from sqrtPriceX96, local depth from per-swap virtual
reserves L/sqrtP and L*sqrtP). The pool version is auto-detected via fee().
Note: V3 reserve_usd is a *local* (in-tick) depth proxy, exact for trades that
don't cross ticks; very large trades would walk multiple ticks (not modeled).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from ..chains import Chain
from ..http import HttpError
from ..models import OHLCV, Trade
from .base import CAP_OHLCV, CAP_TRADES, NotSupported, Provider

log = logging.getLogger("chainscope.bsc_indexer")

# event topic0 hashes (PancakeSwap V2 == Uniswap V2 ABI)
SYNC_TOPIC = "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"
SWAP_V2_TOPIC = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
SWAP_V3_TOPIC = "0x19b47279256b2a23a1665c810c8d55a1758940ee09377d4f8d26497a3577dc83"

# function selectors
SEL_TOKEN0 = "0x0dfe1681"
SEL_TOKEN1 = "0xd21220a7"
SEL_DECIMALS = "0x313ce567"
SEL_FEE = "0xddca3f43"      # fee(), present on V3 pools (uint24, units of 1e-6), absent on V2

STABLES = {
    "0x55d398326f99059ff775485246999027b3197955",  # USDT (BSC)
    "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",  # USDC (BSC)
    "0xe9e7cea3dedca5984780bafc599bd69add087d56",  # BUSD (BSC)
}
WBNB = "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"
WBNB_USDT_PAIR = "0x16b9a82891338f9ba80e2d6970fdda79d1eb0dae"  # PancakeSwap V2 USDT/WBNB
SEL_GET_RESERVES = "0x0902f1ac"

_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600,
               "4h": 14400, "12h": 43200, "1d": 86400}

CHUNK = 5000          # blocks per getLogs call (safe on free archival RPCs)
MAX_CHUNKS = 60       # cap per request; beyond this, run a dedicated backfill / local node


def _u(hexstr: str, start: int, end: int) -> int:
    return int(hexstr[start:end], 16)


def _i(hexstr: str, start: int, end: int) -> int:
    """Signed two's-complement decode of a 32-byte ABI word (int256/int24 sign-extended)."""
    v = int(hexstr[start:end], 16)
    return v - (1 << 256) if v >= (1 << 255) else v


class BscIndexerProvider(Provider):
    name = "bsc_indexer"
    supported_chains = frozenset({Chain.BSC})
    capabilities = frozenset({CAP_OHLCV, CAP_TRADES})
    requires_key = False

    def __init__(self, http, settings):
        super().__init__(http, settings)
        # rotate across free archival endpoints that serve wide getLogs ranges;
        # set BSC_ARCHIVE_RPC_URL (or a comma list) to pin your own / a local node.
        env = os.environ.get("BSC_ARCHIVE_RPC_URL")
        self.rpcs = [u.strip() for u in env.split(",")] if env else [
            "https://bsc.publicnode.com",
            "https://binance.llamarpc.com",
            "https://bsc-rpc.publicnode.com",
        ]
        self._rpc_i = 0
        self._meta_cache: dict[str, dict] = {}
        self._clock: tuple[int, int, float] | None = None  # (block, ts, sec_per_block)
        self._bnb: float | None = None  # cached BNB/USD

    # ---- low-level RPC (rotates endpoints on failure) ----

    async def _rpc(self, method: str, params: list):
        last: Exception | None = None
        for _ in range(len(self.rpcs)):
            url = self.rpcs[self._rpc_i]
            try:
                resp = await self.http.post_json(
                    url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                )
                if "error" in resp:
                    raise HttpError(0, url, str(resp["error"]))
                return resp.get("result")
            except HttpError as exc:
                last = exc
                self._rpc_i = (self._rpc_i + 1) % len(self.rpcs)  # rotate, retry next host
        raise last  # type: ignore

    async def head_block(self) -> int:
        return await self._latest_block()

    async def finalized_head(self) -> int:
        """Finalized block number: the reorg-safe end for a historical backfill
        (recent blocks can revert; the historical set should use finalized only)."""
        try:
            b = await self._rpc("eth_getBlockByNumber", ["finalized", False])
            if b and b.get("number"):
                return int(b["number"], 16)
        except HttpError:
            pass
        return max(1, await self._latest_block() - 15)  # fallback: lag head ~15 blocks

    async def enrich_gas(self, trades: list[Trade], limit: int | None = None) -> list[Trade]:
        """Populate exact gas_native (BNB paid) per trade from tx receipts, one RPC call
        per unique tx. Off the hot path; call only when you need real historical gas
        (otherwise the cost model uses a modeled estimate)."""
        cache: dict[str, float] = {}
        for t in (trades[:limit] if limit else trades):
            h = t.tx_hash
            if not h:
                continue
            if h in cache:
                t.gas_native = cache[h]
                continue
            try:
                r = await self._rpc("eth_getTransactionReceipt", [h])
            except HttpError:
                continue
            if r and r.get("gasUsed") and r.get("effectiveGasPrice"):
                g = int(r["gasUsed"], 16) * int(r["effectiveGasPrice"], 16) / 1e18
                t.gas_native = g
                cache[h] = g
        return trades

    async def _eth_call(self, to: str, data: str) -> str | None:
        try:
            return await self._rpc("eth_call", [{"to": to, "data": data}, "latest"])
        except HttpError:
            return None

    async def _latest_block(self) -> int:
        return int(await self._rpc("eth_blockNumber", []), 16)

    async def _bnb_usd(self) -> float | None:
        """BNB/USD from the WBNB/USDT V2 pool reserves (on-chain, cached). Anchors USD
        pricing for the many coins quoted in WBNB rather than a stablecoin."""
        if self._bnb is not None:
            return self._bnb
        res = await self._eth_call(WBNB_USDT_PAIR, SEL_GET_RESERVES)
        if res and len(res) >= 130:
            r0 = int(res[2:66], 16) / 1e18   # USDT reserve
            r1 = int(res[66:130], 16) / 1e18  # WBNB reserve
            self._bnb = (r0 / r1) if r1 else None
        return self._bnb

    async def _block_ts(self, block: int) -> int | None:
        b = await self._rpc("eth_getBlockByNumber", [hex(block), False])
        if not b or not b.get("timestamp"):
            return None
        return int(b["timestamp"], 16)

    async def _clock_calibrate(self):
        """Derive seconds/block by sampling two recent CONFIRMED blocks. The exact tip
        can return null on some providers (e.g. Alchemy), so we sample below it."""
        if self._clock is not None:
            return self._clock
        latest = await self._latest_block()
        hi = max(1, latest - 16)          # confirmed block (avoid the null-tip race)
        lo = max(1, hi - 20000)
        t_hi = await self._block_ts(hi)
        t_lo = await self._block_ts(lo)
        if t_hi is None:
            import time as _t
            self._clock = (latest, int(_t.time()), 0.75)
            return self._clock
        spb = ((t_hi - t_lo) / max(1, hi - lo)) if t_lo is not None else 0.75
        # anchor at the confirmed block; _ts_of_block / _block_at_time are relative to it
        self._clock = (hi, t_hi, spb if spb > 0 else 0.75)
        return self._clock

    def _ts_of_block(self, block: int) -> datetime:
        latest, t_latest, spb = self._clock  # type: ignore
        secs = t_latest - (latest - block) * spb
        return datetime.fromtimestamp(secs, tz=timezone.utc)

    async def _block_at_time(self, ts_unix: float) -> int:
        latest, t_latest, spb = await self._clock_calibrate()
        return max(1, int(latest - (t_latest - ts_unix) / spb))

    async def _token_meta(self, pair: str) -> dict:
        pair = pair.lower()
        if pair in self._meta_cache:
            return self._meta_cache[pair]
        t0 = await self._eth_call(pair, SEL_TOKEN0)
        t1 = await self._eth_call(pair, SEL_TOKEN1)
        if not t0 or not t1:
            raise NotSupported(f"{pair} is not a V2 pair (no token0/token1)")
        t0 = "0x" + t0[-40:]
        t1 = "0x" + t1[-40:]
        d0 = await self._eth_call(t0, SEL_DECIMALS)
        d1 = await self._eth_call(t1, SEL_DECIMALS)
        meta = {
            "token0": t0, "token1": t1,
            "dec0": int(d0, 16) if d0 and d0 != "0x" else 18,
            "dec1": int(d1, 16) if d1 and d1 != "0x" else 18,
        }
        # resolve the quote (USD-denominating) side + the USD value of one quote unit.
        # Stablecoins = $1; WBNB-quoted pools are anchored via the live BNB/USD price, so
        # the (very common) WBNB-paired new coins still get USD price/volume/liquidity.
        if t0 in STABLES:
            meta.update(stable_idx=0, base_idx=1, quote_idx=0, quote_usd=1.0)
        elif t1 in STABLES:
            meta.update(stable_idx=1, base_idx=0, quote_idx=1, quote_usd=1.0)
        elif t0 == WBNB:
            meta.update(stable_idx=None, base_idx=1, quote_idx=0, quote_usd=await self._bnb_usd())
        elif t1 == WBNB:
            meta.update(stable_idx=None, base_idx=0, quote_idx=1, quote_usd=await self._bnb_usd())
        else:
            meta.update(stable_idx=None, base_idx=0, quote_idx=1, quote_usd=None)
        # detect V2 vs V3: only V3 pools expose fee()
        fee_hex = await self._eth_call(pair, SEL_FEE)
        if fee_hex and fee_hex != "0x" and int(fee_hex, 16) > 0:
            meta["kind"] = "v3"
            meta["fee_bps"] = int(fee_hex, 16) / 100.0   # fee units are 1e-6 -> bps
        else:
            meta["kind"] = "v2"
            meta["fee_bps"] = 25.0
        self._meta_cache[pair] = meta
        return meta

    async def _get_logs(self, pair: str, topic: str, from_b: int, to_b: int) -> list[dict]:
        out: list[dict] = []
        b = from_b
        chunks = 0
        while b <= to_b and chunks < MAX_CHUNKS:
            hi = min(b + CHUNK - 1, to_b)
            try:
                res = await self._rpc("eth_getLogs", [{
                    "address": pair, "topics": [topic],
                    "fromBlock": hex(b), "toBlock": hex(hi),
                }])
                out.extend(res or [])
            except HttpError as exc:
                log.debug("getLogs %s-%s failed: %s", b, hi, exc)
            b = hi + 1
            chunks += 1
        if b <= to_b:
            log.warning("bsc_indexer: truncated at %d blocks (MAX_CHUNKS=%d); "
                        "use a local archive node for deeper history", b - from_b, MAX_CHUNKS)
        return sorted(out, key=lambda lg: int(lg["blockNumber"], 16))

    # ---- decoding ----

    @staticmethod
    def _decode_sync(data: str, dec0: int, dec1: int) -> tuple[float, float]:
        d = data[2:]
        return _u(d, 0, 64) / 10 ** dec0, _u(d, 64, 128) / 10 ** dec1

    @staticmethod
    def _decode_swap_v2(data: str, dec0: int, dec1: int) -> tuple[float, float, float, float]:
        d = data[2:]
        return (_u(d, 0, 64) / 10 ** dec0, _u(d, 64, 128) / 10 ** dec1,
                _u(d, 128, 192) / 10 ** dec0, _u(d, 192, 256) / 10 ** dec1)

    @staticmethod
    def _decode_swap_v3(data: str) -> tuple[int, int, int, int, int]:
        """V3 Swap(amount0,amount1,sqrtPriceX96,liquidity,tick), raw (un-scaled) ints."""
        d = data[2:]
        return (_i(d, 0, 64), _i(d, 64, 128),     # amount0, amount1 (signed, pool's perspective)
                _u(d, 128, 192), _u(d, 192, 256),  # sqrtPriceX96, liquidity
                _i(d, 256, 320))                   # tick (signed)

    def _price_usd(self, meta: dict, r0: float, r1: float) -> tuple[float | None, float | None]:
        """V2: (price_of_base_in_usd, pool_liquidity_usd) from real reserves, scaled by the
        quote's USD value (1 for stables, BNB/USD for WBNB); (native, None) if quote has no USD anchor."""
        qi, qusd = meta["quote_idx"], meta.get("quote_usd")
        q = r1 if qi == 1 else r0   # quote reserve (decimal-adjusted)
        b = r0 if qi == 1 else r1   # base reserve
        if not b:
            return None, None
        price = q / b               # base priced in quote token
        if qusd is None:
            return price, None
        return price * qusd, 2 * q * qusd

    def _v3_price_reserve(self, meta: dict, sqrt_price_x96: int,
                          liquidity: int) -> tuple[float | None, float | None]:
        """V3: price of base (USD if stable-quoted) + local virtual-reserve depth in USD."""
        if sqrt_price_x96 <= 0:
            return None, None
        dec0, dec1 = meta["dec0"], meta["dec1"]
        qi, qusd = meta["quote_idx"], meta.get("quote_usd")
        sqrt_p = sqrt_price_x96 / 2 ** 96
        price0_in_1 = (sqrt_p * sqrt_p) * 10 ** (dec0 - dec1)   # token1 per token0, human
        x0 = (liquidity / sqrt_p) / 10 ** dec0                  # virtual reserve token0
        x1 = (liquidity * sqrt_p) / 10 ** dec1                  # virtual reserve token1
        if qi == 1:        # token1 is quote, base = token0
            price, q = price0_in_1, x1
        else:              # token0 is quote, base = token1
            price, q = (1 / price0_in_1 if price0_in_1 else None), x0
        if price is None:
            return None, None
        if qusd is None:
            return price, None
        return price * qusd, 2 * q * qusd

    # ---- capabilities ----

    def _build_trades(self, pair_address: str, meta: dict,
                      sync: list[dict], swaps: list[dict]) -> list[Trade]:
        if not swaps:
            return []  # likely a V3 pool (no V2 Swap events), not decoded yet
        # reserve_usd at each block from the nearest preceding Sync
        reserve_by_block = []
        for lg in sync:
            r0, r1 = self._decode_sync(lg["data"], meta["dec0"], meta["dec1"])
            _, liq = self._price_usd(meta, r0, r1)
            reserve_by_block.append((int(lg["blockNumber"], 16), liq))
        reserve_by_block.sort()

        def reserve_at(blk: int) -> float | None:
            best = None
            for b, liq in reserve_by_block:
                if b <= blk:
                    best = liq
                else:
                    break
            return best

        base_idx, qusd = meta["base_idx"], meta.get("quote_usd")
        out: list[Trade] = []
        for lg in swaps:
            blk = int(lg["blockNumber"], 16)
            a0i, a1i, a0o, a1o = self._decode_swap_v2(lg["data"], meta["dec0"], meta["dec1"])
            base_in, base_out = (a0i, a0o) if base_idx == 0 else (a1i, a1o)
            quote_in, quote_out = (a1i, a1o) if base_idx == 0 else (a0i, a0o)
            if base_out > 0:
                side, amt_base, amt_quote = "buy", base_out, quote_in
            else:
                side, amt_base, amt_quote = "sell", base_in, quote_out
            amt_usd = amt_quote * qusd if qusd is not None else None
            price_usd = (amt_quote / amt_base) * qusd if (qusd is not None and amt_base) else None
            out.append(Trade(
                source="bsc_indexer", chain=Chain.BSC, pair_address=pair_address,
                block_time=self._ts_of_block(blk), side=side, dex="pancakeswap-v2",
                price_usd=price_usd, amount_base=amt_base, amount_quote=amt_quote,
                amount_usd=amt_usd, reserve_usd=reserve_at(blk), tx_hash=lg.get("transactionHash"),
                block_number=blk, log_index=int(lg["logIndex"], 16) if lg.get("logIndex") else None,
                fee_bps=meta.get("fee_bps"),
                maker=("0x" + lg["topics"][2][-40:]) if len(lg.get("topics", [])) > 2 else None,
                raw=lg,
            ))
        out.sort(key=lambda t: t.block_time, reverse=True)
        return out

    def _build_trades_v3(self, pair_address: str, meta: dict, swaps: list[dict]) -> list[Trade]:
        base_idx, qusd = meta["base_idx"], meta.get("quote_usd")
        out: list[Trade] = []
        for lg in swaps:
            blk = int(lg["blockNumber"], 16)
            a0_raw, a1_raw, sqrt_p, liq, _tick = self._decode_swap_v3(lg["data"])
            a0 = a0_raw / 10 ** meta["dec0"]
            a1 = a1_raw / 10 ** meta["dec1"]
            base_amt = a0 if base_idx == 0 else a1
            quote_amt = a1 if base_idx == 0 else a0
            side = "buy" if base_amt < 0 else "sell"   # base out of pool => trader bought
            price, reserve = self._v3_price_reserve(meta, sqrt_p, liq)
            amt_quote = abs(quote_amt)
            out.append(Trade(
                source="bsc_indexer", chain=Chain.BSC, pair_address=pair_address,
                block_time=self._ts_of_block(blk), side=side, dex="pancakeswap-v3",
                price_usd=price if qusd is not None else None,
                price_native=price if qusd is None else None,
                amount_base=abs(base_amt), amount_quote=amt_quote,
                amount_usd=amt_quote * qusd if qusd is not None else None,
                reserve_usd=reserve, tx_hash=lg.get("transactionHash"),
                block_number=blk, log_index=int(lg["logIndex"], 16) if lg.get("logIndex") else None,
                fee_bps=meta.get("fee_bps"),
                maker=("0x" + lg["topics"][2][-40:]) if len(lg.get("topics", [])) > 2 else None,
                raw=lg,
            ))
        out.sort(key=lambda t: t.block_time, reverse=True)
        return out

    def _bars_to_ohlcv(self, pair_address: str, timeframe: str, bars: dict) -> list[OHLCV]:
        out = []
        for bucket in sorted(bars):
            b = bars[bucket]
            out.append(OHLCV(
                source="bsc_indexer", chain=Chain.BSC, pair_address=pair_address,
                timeframe=timeframe, timestamp=datetime.fromtimestamp(bucket, tz=timezone.utc),
                open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                volume=b["v"] or None, reserve_usd=b["res"], trade_count=b["n"] or None,
            ))
        return out

    async def get_trades(self, chain: Chain, pair_address: str, since: int | None = None,
                         until: int | None = None, limit: int = 1000) -> list[Trade]:
        if Chain.parse(chain) != Chain.BSC:
            raise NotSupported
        meta = await self._token_meta(pair_address)
        await self._clock_calibrate()
        latest, t_latest, _ = self._clock  # type: ignore
        to_b = await self._block_at_time(until) if until else latest
        from_b = await self._block_at_time(since) if since else max(1, to_b - CHUNK * 4)
        return (await self.fetch_trades_range(pair_address, from_b, to_b))[:limit]

    async def fetch_trades_range(self, pair_address: str, from_b: int, to_b: int) -> list[Trade]:
        """Decode all trades in a block range: the unit the backfill engine drives.
        Branches on the pool version detected in _token_meta (V2 Sync/Swap vs V3 Swap)."""
        meta = await self._token_meta(pair_address)
        await self._clock_calibrate()
        if meta["kind"] == "v3":
            swaps = await self._get_logs(pair_address, SWAP_V3_TOPIC, from_b, to_b)
            return self._build_trades_v3(pair_address, meta, swaps)
        sync = await self._get_logs(pair_address, SYNC_TOPIC, from_b, to_b)
        swaps = await self._get_logs(pair_address, SWAP_V2_TOPIC, from_b, to_b)
        return self._build_trades(pair_address, meta, sync, swaps)

    async def get_ohlcv(self, chain: Chain, pair_address: str, timeframe: str = "1h",
                        limit: int = 1000, before: int | None = None) -> list[OHLCV]:
        if Chain.parse(chain) != Chain.BSC:
            raise NotSupported
        tf_sec = _TF_SECONDS.get(timeframe, 3600)
        meta = await self._token_meta(pair_address)
        latest, t_latest, _ = await self._clock_calibrate()
        end_ts = before if before else t_latest
        start_ts = end_ts - tf_sec * limit
        from_b = await self._block_at_time(start_ts)
        to_b = await self._block_at_time(end_ts)

        qusd, base_idx = meta.get("quote_usd"), meta["base_idx"]
        bars: dict[int, dict] = {}

        if meta["kind"] == "v3":
            for lg in await self._get_logs(pair_address, SWAP_V3_TOPIC, from_b, to_b):
                ts = int(self._ts_of_block(int(lg["blockNumber"], 16)).timestamp())
                bucket = ts - (ts % tf_sec)
                a0_raw, a1_raw, sqrt_p, liq, _t = self._decode_swap_v3(lg["data"])
                price, reserve = self._v3_price_reserve(meta, sqrt_p, liq)
                if price is None:
                    continue
                q_raw, q_dec = (a1_raw, meta["dec1"]) if base_idx == 0 else (a0_raw, meta["dec0"])
                b = bars.setdefault(bucket, {"o": price, "h": price, "l": price, "c": price,
                                             "v": 0.0, "res": reserve, "n": 0})
                b["h"] = max(b["h"], price); b["l"] = min(b["l"], price)
                b["c"] = price; b["res"] = reserve
                b["v"] += (abs(q_raw) / 10 ** q_dec) * qusd if qusd is not None else 0.0
                b["n"] += 1
            return self._bars_to_ohlcv(pair_address, timeframe, bars)

        # V2: price + reserve from Sync (spot, post-trade); volume from Swap
        sync = await self._get_logs(pair_address, SYNC_TOPIC, from_b, to_b)
        swaps = await self._get_logs(pair_address, SWAP_V2_TOPIC, from_b, to_b)
        for lg in sync:
            ts = int(self._ts_of_block(int(lg["blockNumber"], 16)).timestamp())
            bucket = ts - (ts % tf_sec)
            r0, r1 = self._decode_sync(lg["data"], meta["dec0"], meta["dec1"])
            price, liq = self._price_usd(meta, r0, r1)
            if price is None:
                continue
            b = bars.setdefault(bucket, {"o": price, "h": price, "l": price, "c": price,
                                         "v": 0.0, "res": liq, "n": 0})
            b["h"] = max(b["h"], price); b["l"] = min(b["l"], price)
            b["c"] = price; b["res"] = liq
        for lg in swaps:
            ts = int(self._ts_of_block(int(lg["blockNumber"], 16)).timestamp())
            bucket = ts - (ts % tf_sec)
            if bucket not in bars:
                continue
            a0i, a1i, a0o, a1o = self._decode_swap_v2(lg["data"], meta["dec0"], meta["dec1"])
            quote_in, quote_out = (a1i, a1o) if base_idx == 0 else (a0i, a0o)
            bars[bucket]["v"] += (quote_in + quote_out) * qusd if qusd is not None else 0.0
            bars[bucket]["n"] += 1
        return self._bars_to_ohlcv(pair_address, timeframe, bars)
