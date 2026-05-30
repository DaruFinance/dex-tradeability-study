"""Concentrated-liquidity TICK LIQUIDITY MAP reconstructor for Uniswap-V3 / PancakeSwap-V3
pools, pure on-chain (eth_getLogs + eth_call). No third-party index, no key.

In a V3 pool liquidity is *not* uniform across price: each LP deposits into a [tickLower,
tickUpper] range, so the depth available to a trade depends on how many ticks the trade
crosses. The active-tick virtual-reserve approximation used in `bsc_indexer._v3_price_reserve`
is exact only for trades small enough to stay inside the current tick range; anything larger
walks across initialized ticks where the in-range liquidity L jumps.

This module rebuilds the distribution of L across ticks from the raw liquidity events:

  Mint(sender, owner indexed, tickLower indexed, tickUpper indexed, amount, amount0, amount1)
  Burn(owner indexed, tickLower indexed, tickUpper indexed, amount, amount0, amount1)

For each Mint of `amount` (= L) over [tickLower, tickUpper], the active in-range liquidity is
higher by `amount` for every tick in that band. Encoded as net-liquidity deltas (the same
`liquidityNet` convention the pool stores per tick): +amount at tickLower, -amount at tickUpper.
Burn is the reverse. Accumulating these into a {tick: net_delta} map and taking the running
prefix-sum from the left gives the active L at any tick, i.e. the full liquidity curve.

`slippage()` then walks that curve outward from the current tick using exact V3 swap math
(within a tick the pool is a CPMM with virtual reserves x = L/sqrtP, y = L*sqrtP; crossing an
initialized tick adds/removes its net delta). That is the correct cross-tick price impact the
in-tick approximation misses.

Tick / price math (Uniswap V3 whitepaper + dev docs):
  sqrtP(tick) = 1.0001^(tick/2),  sqrtPriceX96 = sqrtP * 2^96
  human price of token0 in token1 = 1.0001^tick * 10^(dec0 - dec1)
  moving L from sqrtPa up to sqrtPb consumes dy = L*(sqrtPb - sqrtPa) of token1
                                       and dx = L*(1/sqrtPa - 1/sqrtPb) of token0

Scope: EVM V3 pools (BSC PancakeSwap-V3 / Uniswap-V3 ABI is identical for Mint/Burn). The map
is bounded by the scanned block window, not full pool history (see caveats in build()).
"""
from __future__ import annotations

import logging
import math
import os

from .chains import Chain
from .http import HttpError

log = logging.getLogger("chainscope.ticks")

# --- verified event topic0 hashes (keccak256 of the canonical event signatures) ---
# Mint(address,address,int24,int24,uint128,uint256,uint256)
MINT_TOPIC = "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
# Burn(address,int24,int24,uint128,uint256,uint256)
BURN_TOPIC = "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"
# In BOTH events the indexed topics are [topic0, owner, tickLower, tickUpper]; the
# non-indexed `amount` (L), amount0, amount1 live in `data`. (sender is also non-indexed,
# and only present in Mint: leading the data word for Mint.)

# --- function selectors (verified: first 4 bytes of keccak256 of the signature) ---
SEL_SLOT0 = "0x3850c7bd"        # slot0() -> (sqrtPriceX96, tick, ...)
SEL_LIQUIDITY = "0x1a686502"    # liquidity() -> uint128 (active in-range L)
SEL_TICK_SPACING = "0xd0c93a7c"  # tickSpacing() -> int24
SEL_TOKEN0 = "0x0dfe1681"
SEL_TOKEN1 = "0xd21220a7"
SEL_DECIMALS = "0x313ce567"
SEL_FEE = "0xddca3f43"           # fee() -> uint24 (1e-6 units)

# token0 stable => price of base (token1) is 1/price0; token1 stable => price0 is the USD price
STABLES = {
    "0x55d398326f99059ff775485246999027b3197955",  # USDT (BSC)
    "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",  # USDC (BSC)
    "0xe9e7cea3dedca5984780bafc599bd69add087d56",  # BUSD (BSC)
}

CHUNK = 5000          # blocks per getLogs call (safe on free public RPCs)
MAX_CHUNKS = 80       # cap per build(); beyond this the map is truncated (logged)

MIN_TICK = -887272
MAX_TICK = 887272
Q96 = 1 << 96


def _u(hexstr: str, start: int, end: int) -> int:
    return int(hexstr[start:end], 16)


def _i(hexstr: str, start: int, end: int) -> int:
    """Signed two's-complement decode of a 32-byte ABI word (int24/int256 sign-extended)."""
    v = int(hexstr[start:end], 16)
    return v - (1 << 256) if v >= (1 << 255) else v


def tick_to_sqrt_price(tick: int) -> float:
    """sqrtP = 1.0001^(tick/2), the real square-root price (not the X96 fixed-point form)."""
    return math.pow(1.0001, tick / 2.0)


def sqrt_price_x96_to_sqrt_price(sqrt_price_x96: int) -> float:
    return sqrt_price_x96 / Q96


def sqrt_price_to_tick(sqrt_price: float) -> int:
    """Inverse of tick_to_sqrt_price; tick = log_1.0001(sqrtP^2) = 2*ln(sqrtP)/ln(1.0001)."""
    return int(math.floor(2.0 * math.log(sqrt_price) / math.log(1.0001)))


class TickMap:
    """Reconstructs and queries the per-tick net-liquidity map of a single V3 pool.

    Not a Provider: a standalone on-chain reader with its own endpoint-rotating `_rpc`,
    mirroring `bsc_indexer` so it works with the same free public RPCs.
    """

    def __init__(self, settings=None, http=None):
        # `http` (an HttpClient) is optional; if absent we lazily build one so this class
        # can be used stand-alone in a script. settings is unused beyond optional RPC env.
        self._http = http
        self._own_http = False
        env = os.environ.get("BSC_ARCHIVE_RPC_URL")
        self.rpcs = [u.strip() for u in env.split(",")] if env else [
            "https://bsc.publicnode.com",
            "https://binance.llamarpc.com",
            "https://bsc-rpc.publicnode.com",
        ]
        self._rpc_i = 0
        self._meta_cache: dict[str, dict] = {}

    async def _client(self):
        if self._http is None:
            from .http import HttpClient
            self._http = HttpClient()
            self._own_http = True
        return self._http

    async def aclose(self) -> None:
        if self._own_http and self._http is not None:
            await self._http.aclose()
            self._http = None
            self._own_http = False

    # ---- low-level RPC (rotates endpoints on failure), same pattern as bsc_indexer ----

    async def _rpc(self, method: str, params: list):
        http = await self._client()
        last: Exception | None = None
        for _ in range(len(self.rpcs)):
            url = self.rpcs[self._rpc_i]
            try:
                resp = await http.post_json(
                    url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                )
                if "error" in resp:
                    raise HttpError(0, url, str(resp["error"]))
                return resp.get("result")
            except HttpError as exc:
                last = exc
                self._rpc_i = (self._rpc_i + 1) % len(self.rpcs)
        raise last  # type: ignore

    async def _eth_call(self, to: str, data: str) -> str | None:
        try:
            return await self._rpc("eth_call", [{"to": to, "data": data}, "latest"])
        except HttpError:
            return None

    async def _latest_block(self) -> int:
        return int(await self._rpc("eth_blockNumber", []), 16)

    # ---- pool metadata ----

    async def pool_meta(self, pool: str) -> dict:
        pool = pool.lower()
        if pool in self._meta_cache:
            return self._meta_cache[pool]
        t0 = await self._eth_call(pool, SEL_TOKEN0)
        t1 = await self._eth_call(pool, SEL_TOKEN1)
        if not t0 or not t1 or t0 == "0x" or t1 == "0x":
            raise ValueError(f"{pool} has no token0/token1, not a V3 pool")
        t0 = "0x" + t0[-40:]
        t1 = "0x" + t1[-40:]
        d0 = await self._eth_call(t0, SEL_DECIMALS)
        d1 = await self._eth_call(t1, SEL_DECIMALS)
        spacing_hex = await self._eth_call(pool, SEL_TICK_SPACING)
        fee_hex = await self._eth_call(pool, SEL_FEE)
        meta = {
            "pool": pool,
            "token0": t0,
            "token1": t1,
            "dec0": int(d0, 16) if d0 and d0 != "0x" else 18,
            "dec1": int(d1, 16) if d1 and d1 != "0x" else 18,
            "tick_spacing": _i(spacing_hex, 2, 66) if spacing_hex and spacing_hex != "0x" else 1,
            "fee_bps": (int(fee_hex, 16) / 100.0) if fee_hex and fee_hex != "0x" else 0.0,
        }
        if t0 in STABLES:
            meta.update(stable_idx=0, base_idx=1)
        elif t1 in STABLES:
            meta.update(stable_idx=1, base_idx=0)
        else:
            meta.update(stable_idx=None, base_idx=0)
        self._meta_cache[pool] = meta
        return meta

    # ---- price scaling helpers ----

    def _price0_in_1(self, meta: dict, sqrt_price: float) -> float:
        """Human price of token0 expressed in token1 (one whole token0 -> N whole token1)."""
        return (sqrt_price * sqrt_price) * 10 ** (meta["dec0"] - meta["dec1"])

    def _base_price_usd(self, meta: dict, sqrt_price: float) -> float | None:
        """Price of the *base* token in USD if the pool is stable-quoted, else None."""
        p0 = self._price0_in_1(meta, sqrt_price)
        if meta["stable_idx"] == 1:        # token1 is the stable -> base is token0
            return p0
        if meta["stable_idx"] == 0:        # token0 is the stable -> base is token1
            return (1.0 / p0) if p0 else None
        return None

    # ---- current pool state ----

    async def current_state(self, pool: str) -> dict:
        """slot0() + liquidity(): the live price, tick, and active in-range liquidity."""
        meta = await self.pool_meta(pool)
        slot0 = await self._eth_call(pool, SEL_SLOT0)
        liq_hex = await self._eth_call(pool, SEL_LIQUIDITY)
        if not slot0 or slot0 == "0x":
            raise ValueError(f"{pool}: slot0() returned nothing")
        d = slot0[2:]
        sqrt_price_x96 = _u(d, 0, 64)              # uint160, word 0
        current_tick = _i(d, 64, 128)              # int24, word 1 (sign-extended)
        active_liquidity = int(liq_hex, 16) if liq_hex and liq_hex != "0x" else 0
        sqrt_price = sqrt_price_x96_to_sqrt_price(sqrt_price_x96)
        return {
            "sqrt_price_x96": sqrt_price_x96,
            "sqrt_price": sqrt_price,
            "current_tick": current_tick,
            "active_liquidity": active_liquidity,
            "price0_in_1": self._price0_in_1(meta, sqrt_price),
            "base_price_usd": self._base_price_usd(meta, sqrt_price),
            "tick_spacing": meta["tick_spacing"],
            "meta": meta,
        }

    # ---- tick-map reconstruction from Mint/Burn logs ----

    async def _get_logs(self, pool: str, topic: str, from_b: int, to_b: int) -> list[dict]:
        out: list[dict] = []
        b = from_b
        chunks = 0
        while b <= to_b and chunks < MAX_CHUNKS:
            hi = min(b + CHUNK - 1, to_b)
            try:
                res = await self._rpc("eth_getLogs", [{
                    "address": pool, "topics": [topic],
                    "fromBlock": hex(b), "toBlock": hex(hi),
                }])
                out.extend(res or [])
            except HttpError as exc:
                log.debug("getLogs %s %s-%s failed: %s", topic[:10], b, hi, exc)
            b = hi + 1
            chunks += 1
        truncated = b <= to_b
        if truncated:
            log.warning("ticks: %s scan truncated at +%d blocks (MAX_CHUNKS=%d); "
                        "tick map covers only the scanned window",
                        topic[:10], b - from_b, MAX_CHUNKS)
        return out

    @staticmethod
    def _decode_mint_burn(lg: dict, is_mint: bool) -> tuple[int, int, int]:
        """Return (tickLower, tickUpper, amountL). Ticks come from indexed topics[2],[3];
        the liquidity `amount` (uint128) is the first 32-byte data word for Burn, and the
        SECOND data word for Mint (the first Mint data word is the non-indexed `sender`)."""
        topics = lg["topics"]
        tick_lower = _i(topics[2], 2, 66)
        tick_upper = _i(topics[3], 2, 66)
        d = lg["data"][2:]
        # Mint data layout: [sender(addr), amount(uint128), amount0, amount1]
        # Burn data layout: [amount(uint128), amount0, amount1]
        amount = _u(d, 64, 128) if is_mint else _u(d, 0, 64)
        return tick_lower, tick_upper, amount

    async def build(self, pool: str, from_block: int | None = None,
                    to_block: int | None = None) -> dict[int, int]:
        """Reconstruct the {tick: net_liquidity_delta} map from Mint/Burn over a block range.

        Convention (matches the pool's own per-tick `liquidityNet`): a Mint of L over
        [lower, upper] contributes +L at `lower` and -L at `upper`; Burn contributes the
        reverse. The running prefix-sum of these deltas (scanned left to right) yields the
        active in-range liquidity L at any tick.

        NOTE: the map is bounded by [from_block, to_block]. Liquidity minted before
        from_block (and not burned within the window) is invisible here, so the prefix-sum
        is a *relative* curve over the window unless you anchor it to the live active L from
        current_state(), which is exactly what slippage() does.
        """
        pool = pool.lower()
        await self.pool_meta(pool)
        latest = await self._latest_block()
        to_b = to_block if to_block is not None else latest
        from_b = from_block if from_block is not None else max(1, to_b - CHUNK * MAX_CHUNKS)

        deltas: dict[int, int] = {}
        for lg in await self._get_logs(pool, MINT_TOPIC, from_b, to_b):
            lo, hi, amt = self._decode_mint_burn(lg, is_mint=True)
            deltas[lo] = deltas.get(lo, 0) + amt
            deltas[hi] = deltas.get(hi, 0) - amt
        for lg in await self._get_logs(pool, BURN_TOPIC, from_b, to_b):
            lo, hi, amt = self._decode_mint_burn(lg, is_mint=False)
            deltas[lo] = deltas.get(lo, 0) - amt
            deltas[hi] = deltas.get(hi, 0) + amt
        # drop ticks whose net delta cancelled to zero
        deltas = {t: v for t, v in deltas.items() if v != 0}
        log.info("ticks: built map for %s over blocks %d-%d -> %d initialized ticks",
                 pool, from_b, to_b, len(deltas))
        return deltas

    # ---- cross-tick slippage ----

    async def slippage(self, pool: str, size_usd: float, side: str,
                       tick_map: dict[int, int] | None = None,
                       from_block: int | None = None) -> dict:
        """Average fill price and price-impact fraction for a `size_usd` trade.

        side == "buy"  -> buying the *base* token with the quote (price moves UP);
        side == "sell" -> selling the base token for quote (price moves DOWN).

        Walks the liquidity curve from the current tick outward, applying exact V3 swap math
        per tick band and re-anchoring active L by the tick's net delta whenever an initialized
        tick is crossed. Returns {avg_price, mid_price, impact_frac, ticks_crossed,
        filled_usd, requested_usd}. Prices are USD (stable-quoted pool) else token1-per-token0.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        meta = await self.pool_meta(pool)
        state = await self.current_state(pool)
        if tick_map is None:
            tick_map = await self.build(pool, from_block=from_block)

        dec0, dec1 = meta["dec0"], meta["dec1"]
        spacing = meta["tick_spacing"] or 1
        stable_idx = meta["stable_idx"]

        sqrt_p = state["sqrt_price"]                 # real sqrt price (not X96)
        L = float(state["active_liquidity"])         # live active in-range liquidity (anchor)
        cur_tick = state["current_tick"]
        mid_price = state["base_price_usd"] if stable_idx is not None else state["price0_in_1"]
        if not L or sqrt_p <= 0 or not mid_price:
            return {"avg_price": None, "mid_price": mid_price, "impact_frac": None,
                    "ticks_crossed": 0, "filled_usd": 0.0, "requested_usd": size_usd}

        # Convert the USD budget to a budget in the token the trader *pays*.
        # buy base: pay quote token; sell base: pay base token.
        base_idx = meta["base_idx"]
        # which raw token index does the trader spend?
        if side == "buy":
            pay_idx = 1 - base_idx           # pays the quote token
        else:
            pay_idx = base_idx               # pays the base token
        pay_dec = dec0 if pay_idx == 0 else dec1

        # price of the pay-token in USD, to turn size_usd into a token budget
        if stable_idx is not None:
            if pay_idx == stable_idx:
                pay_usd_price = 1.0
            else:
                pay_usd_price = state["base_price_usd"]   # the non-stable side is the base
        else:
            pay_usd_price = None   # no USD anchor: treat size_usd as already in pay-token units
        budget_token = (size_usd / pay_usd_price) if pay_usd_price else size_usd

        # In V3, a buy of the base raises price when base==token0? It depends which token is
        # token0. Selling token0 (adding token0) lowers price; selling token1 raises it.
        # Trader spends pay_idx token => that token enters the pool. If token0 enters (pay_idx==0)
        # price (token0-in-token1) goes DOWN (sqrtP decreases). If token1 enters, sqrtP increases.
        direction_up = (pay_idx == 1)        # True => sqrtP increases, crossing UPPER ticks

        # We integrate exact V3 math. Within a band of constant L between sqrtPa and sqrtPb:
        #   amount of token0 = L*(1/sqrtPa - 1/sqrtPb)   (token0 leaving as price rises)
        #   amount of token1 = L*(sqrtPb - sqrtPa)       (token1 leaving as price falls)
        # The trader's *input* token is the one entering; output is the other.
        spent_raw = 0.0           # in pay-token raw (un-decimaled) units
        recv_raw = 0.0            # in receive-token raw units
        recv_idx = 1 - pay_idx
        ticks_crossed = 0
        budget_raw = budget_token * 10 ** pay_dec

        # ordered list of initialized ticks we may cross, in the direction of travel
        if direction_up:
            boundaries = sorted(t for t in tick_map if t > cur_tick)
        else:
            boundaries = sorted((t for t in tick_map if t <= cur_tick), reverse=True)

        cur_sqrt = sqrt_p
        cur_L = L
        bi = 0
        MAX_STEPS = 200000   # hard safety bound
        steps = 0
        while spent_raw < budget_raw and steps < MAX_STEPS:
            steps += 1
            # determine the next tick boundary and its sqrt price
            if bi < len(boundaries):
                next_tick = boundaries[bi]
                next_sqrt = tick_to_sqrt_price(next_tick)
            else:
                # no more initialized ticks in this direction; fill the rest at the edge
                next_tick = MAX_TICK if direction_up else MIN_TICK
                next_sqrt = tick_to_sqrt_price(next_tick)

            if cur_L <= 0:
                # liquidity gap: jump price to the next boundary without filling, then re-anchor
                cur_sqrt = next_sqrt
                if bi < len(boundaries):
                    cur_L += tick_map[next_tick] if direction_up else -tick_map[next_tick]
                    bi += 1
                    ticks_crossed += 1
                    continue
                break

            remaining_raw = budget_raw - spent_raw
            # max input this band can absorb if price travels all the way to next_sqrt
            if direction_up:
                # token1 enters; full-band token1 input = L*(next_sqrt - cur_sqrt)
                band_in = cur_L * (next_sqrt - cur_sqrt)
            else:
                # token0 enters; full-band token0 input = L*(1/next_sqrt - 1/cur_sqrt)
                band_in = cur_L * (1.0 / next_sqrt - 1.0 / cur_sqrt)
            band_in = abs(band_in)

            if band_in >= remaining_raw and band_in > 0:
                # trade terminates inside this band: solve for the partial target sqrt price
                if direction_up:
                    target_sqrt = cur_sqrt + remaining_raw / cur_L
                    out_amt = cur_L * (1.0 / cur_sqrt - 1.0 / target_sqrt)   # token0 out
                else:
                    # token0 in = L*(1/target - 1/cur) = remaining  => 1/target = 1/cur + rem/L
                    inv_target = (1.0 / cur_sqrt) + remaining_raw / cur_L
                    target_sqrt = 1.0 / inv_target
                    out_amt = cur_L * (cur_sqrt - target_sqrt)              # token1 out
                spent_raw += remaining_raw
                recv_raw += abs(out_amt)
                cur_sqrt = target_sqrt
                break
            else:
                # consume the whole band, cross the boundary, re-anchor L by the net delta
                if direction_up:
                    out_amt = cur_L * (1.0 / cur_sqrt - 1.0 / next_sqrt)    # token0 out
                else:
                    out_amt = cur_L * (cur_sqrt - next_sqrt)               # token1 out
                spent_raw += band_in
                recv_raw += abs(out_amt)
                cur_sqrt = next_sqrt
                if bi < len(boundaries):
                    net = tick_map[next_tick]
                    # crossing up: L += liquidityNet; crossing down: L -= liquidityNet
                    cur_L += net if direction_up else -net
                    bi += 1
                    ticks_crossed += 1
                else:
                    # ran off the end of the known map with budget remaining
                    break

        # convert raw token flows to human units and to base/quote semantics
        spent_human = spent_raw / 10 ** pay_dec
        recv_human = recv_raw / 10 ** (dec0 if recv_idx == 0 else dec1)

        # base & quote amounts of the trade
        if side == "buy":
            base_amt = recv_human          # received base
            quote_amt = spent_human        # spent quote
        else:
            base_amt = spent_human         # sold base
            quote_amt = recv_human         # received quote

        if base_amt <= 0:
            return {"avg_price": None, "mid_price": mid_price, "impact_frac": None,
                    "ticks_crossed": ticks_crossed, "filled_usd": 0.0,
                    "requested_usd": size_usd}

        # average execution price of the base token, in quote (token1-per-token0 convention,
        # already USD when token1 is the stable side; if token0 is the stable side it is
        # USD-per-base because quote IS the stable; native pools => token-units).
        if stable_idx == 0:
            # token0 is the stable -> price of base(token1) in USD = quote(token0)/base(token1)
            avg_price = quote_amt / base_amt
        elif stable_idx == 1:
            # token1 is the stable -> quote IS token1 -> USD per base(token0)
            avg_price = quote_amt / base_amt
        else:
            avg_price = quote_amt / base_amt   # native: token-per-base

        impact_frac = (avg_price - mid_price) / mid_price if mid_price else None
        if side == "sell" and impact_frac is not None:
            impact_frac = -impact_frac   # a sell fills BELOW mid; report magnitude as negative

        # USD notional actually filled before any run-off the end of the scanned map.
        # When stable-quoted, avg_price is base-in-USD, so base_filled*avg_price is exact;
        # for a native (no stable) pool we have no USD anchor and leave it None.
        filled_usd = base_amt * avg_price if stable_idx is not None else None

        return {
            "avg_price": avg_price,
            "mid_price": mid_price,
            "impact_frac": impact_frac,
            "ticks_crossed": ticks_crossed,
            "base_filled": base_amt,
            "quote_filled": quote_amt,
            "filled_usd": filled_usd,
            "requested_usd": size_usd,
            "side": side,
            "fully_filled": spent_raw >= budget_raw * 0.999,
        }
