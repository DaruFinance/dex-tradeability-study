"""Multi-pool routing / liquidity-graph: split an order across the pools a token
trades in, instead of mispricing it against a single pool.

A token rarely lives in one pool. It trades across several fee tiers and DEXs at
once (e.g. CAKE in a PancakeSwap-V2 CAKE/WBNB pool *and* two or three V3 fee-tier
pools). Real routers (1inch, PancakeSwap smart-router, Uniswap auto-router) split a
large fill across all of them, because each pool's constant-product depth is
finite: pricing the whole order against one pool overstates impact and understates
the achievable blended price. This module models that split with pure on-chain
inputs and the same CPMM impact math as `costs.py`.

Algorithm (marginal-price equalization, greedy):
  For a buy of cumulative notional `x` already routed into a pool with one-sided
  quote reserve R and fee f, the *marginal* price of the next dollar (relative to
  the pool mid) is approximately

      m_buy(x)  = mid * (1 + x / R) / (1 - f)      # impact linear in x/R, fee multiplies
      m_sell(x) = mid * (1 - x / R) * (1 - f)      # sell: price falls as you sell more

  Optimal routing equalizes the marginal price across pools (water-filling): the
  marginal cost of the last dollar must be identical everywhere, else you'd move it
  to a cheaper pool. We reach that allocation by handing the order out in small
  increments, always to the pool with the best current marginal price. Deeper pools
  (larger R) and cheaper pools (smaller f) climb their marginal curve slower, so
  they naturally receive more. This is robust, monotone, and converges to the
  equal-marginal optimum without solving a Lagrangian.

The realized average execution price per pool then uses the *integrated* CPMM cost
(the same `price_impact_frac` model in costs.py, applied to that pool's allocation),
so the blended price is a true average-fill, not a marginal one.

On-chain only: `gather_pools_onchain` reads each pool's live reserves via eth_call
(V2 getReserves(); V3 virtual reserves from slot0()/liquidity()), mirroring the
pattern in providers/bsc_indexer.py, no third-party index, no API key.

Caveats:
  * Reserves are a static snapshot at call time, not per-bar; for a backtest pull
    reserves per bar (Sync/Swap events) rather than a single `latest` read.
  * V3 depth is the *local* in-tick virtual reserve (L, sqrtP); exact only for fills
    that don't cross ticks. Large fills walk multiple ticks, combine with a tick
    map (ticks.py) for the full curve. Here it is a single-tick approximation.
  * Cross-DEX gas (one swap per pool touched) is NOT included in the split cost;
    add `len(allocations) * gas_usd` from costs.py if routing across many venues.
"""
from __future__ import annotations

from .chains import Chain
from .costs import price_impact_frac, swap_fee_bps

# PancakeSwap-V2 / Uniswap-V2 getReserves() -> (uint112 reserve0, uint112 reserve1, uint32 ts)
SEL_GET_RESERVES = "0x0902f1ac"
# V3 slot0() -> (uint160 sqrtPriceX96, int24 tick, ...); liquidity() -> uint128
SEL_SLOT0 = "0x3850c7bd"
SEL_LIQUIDITY = "0x1a686502"
# ERC-20 / pool views (same selectors used across the BSC providers)
SEL_TOKEN0 = "0x0dfe1681"
SEL_TOKEN1 = "0xd21220a7"
SEL_DECIMALS = "0x313ce567"
SEL_FEE = "0xddca3f43"      # fee() on V3 pools (uint24, units of 1e-6); absent on V2

STABLES = {
    "0x55d398326f99059ff775485246999027b3197955",  # USDT (BSC)
    "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",  # USDC (BSC)
    "0xe9e7cea3dedca5984780bafc599bd69add087d56",  # BUSD (BSC)
}
# WBNB: common non-stable quote; price quoted in native unless a USD anchor is given
WBNB = "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"

_DEFAULT_STEPS = 2000   # greedy increments; 2000 gives sub-bp allocation granularity


# --------------------------------------------------------------------------- #
# split_route: the core allocator                                            #
# --------------------------------------------------------------------------- #

def _norm_side(side: str) -> str:
    return "buy" if str(side).lower().startswith("b") else "sell"


def _pool_reserve_usd(p: dict) -> float:
    """Total pool TVL in USD from a routing dict. Accepts `reserve_usd` (total) or
    `reserve_base_usd` (single-sided value of the base leg -> doubled to a total)."""
    if p.get("reserve_usd") is not None:
        return float(p["reserve_usd"])
    if p.get("reserve_base_usd") is not None:
        return 2.0 * float(p["reserve_base_usd"])
    return 0.0


def _pool_fee_frac(p: dict) -> float:
    bps = p.get("fee_bps")
    if bps is None:
        bps = swap_fee_bps(p.get("dex"))
    return float(bps) / 1e4


def _marginal_price(side: str, mid: float, fee: float, one_sided_r: float,
                    spent: float) -> float:
    """Marginal (next-dollar) execution price in this pool after `spent` already routed.
    Lower is better for a buy; higher is better for a sell."""
    if one_sided_r <= 0:
        return float("inf") if side == "buy" else 0.0
    ratio = spent / one_sided_r
    if side == "buy":
        return mid * (1.0 + ratio) / (1.0 - fee)
    return mid * (1.0 - ratio) * (1.0 - fee)


def _pool_exec(side: str, mid: float, fee: float, reserve_usd: float,
               size_usd: float) -> tuple[float, float]:
    """Average execution price + impact fraction for `size_usd` filled in one pool,
    using the integrated CPMM model from costs.py (true average, not marginal)."""
    if size_usd <= 0:
        return mid, 0.0
    impact = price_impact_frac(side, size_usd, reserve_usd)
    if side == "buy":
        exec_price = mid * (1.0 + impact) / (1.0 - fee)
    else:
        exec_price = mid * (1.0 - impact) * (1.0 - fee)
    return exec_price, impact


def split_route(pools: list[dict], size_usd: float, side: str = "buy",
                steps: int = _DEFAULT_STEPS) -> dict:
    """Optimally split a `size_usd` order across `pools` to minimize total cost.

    Each pool dict needs: pair_address, dex, fee_bps (or inferred), price_usd, and
    depth as `reserve_usd` (total TVL) or `reserve_base_usd` (single-sided base leg).

    Routing equalizes the marginal execution price across pools (CPMM water-filling),
    implemented greedily: the order is handed out in `steps` increments, each to the
    pool with the best current marginal price. Returns allocations, the size-weighted
    blended price/impact, the total cost fraction vs the best single pool's mid, and
    `vs_single_pool`: the fractional cost saving over routing 100% through the
    deepest pool.
    """
    side = _norm_side(side)
    usable = []
    for p in pools:
        r = _pool_reserve_usd(p)
        mid = p.get("price_usd")
        if r <= 0 or mid is None or mid <= 0:
            continue  # skip empty / unpriced pools
        usable.append({
            "pair": p.get("pair_address"),
            "dex": p.get("dex"),
            "mid": float(mid),
            "fee": _pool_fee_frac(p),
            "reserve_usd": r,
            "one_sided": r / 2.0,
            "spent": 0.0,
        })

    if not usable or size_usd <= 0:
        return {
            "allocations": [], "blended_price": None, "blended_impact_frac": None,
            "total_cost_frac": None, "vs_single_pool": None,
        }

    # Reference mid for cost-fraction reporting: the deepest pool's mid price.
    deepest = max(usable, key=lambda q: q["reserve_usd"])
    ref_mid = deepest["mid"]

    # ---- greedy marginal allocation ----
    inc = size_usd / steps
    for _ in range(steps):
        best = None
        best_mp = float("inf") if side == "buy" else float("-inf")
        for q in usable:
            mp = _marginal_price(side, q["mid"], q["fee"], q["one_sided"], q["spent"])
            if (side == "buy" and mp < best_mp) or (side == "sell" and mp > best_mp):
                best_mp, best = mp, q
        best["spent"] += inc

    # ---- realized per-pool fills (integrated CPMM) ----
    allocations = []
    total_base = 0.0      # base tokens received (buy) or sold (sell)
    weighted_impact = 0.0
    routed = 0.0
    for q in usable:
        amt = q["spent"]
        if amt <= 0:
            continue
        exec_price, impact = _pool_exec(side, q["mid"], q["fee"], q["reserve_usd"], amt)
        total_base += amt / exec_price
        weighted_impact += impact * amt
        routed += amt
        allocations.append({
            "pair": q["pair"], "dex": q["dex"], "amount_usd": amt,
            "exec_price": exec_price, "impact_frac": impact,
        })
    allocations.sort(key=lambda a: a["amount_usd"], reverse=True)

    blended_price = (routed / total_base) if total_base > 0 else None
    blended_impact = (weighted_impact / routed) if routed > 0 else None
    # total cost fraction vs the reference (deepest-pool) mid
    total_cost_frac = ((blended_price - ref_mid) / ref_mid) if (
        blended_price is not None and side == "buy"
    ) else ((ref_mid - blended_price) / ref_mid) if blended_price is not None else None

    # ---- baseline: slam 100% through the single deepest pool ----
    single_price, _ = _pool_exec(side, deepest["mid"], deepest["fee"],
                                 deepest["reserve_usd"], size_usd)
    if side == "buy":
        single_cost = (single_price - ref_mid) / ref_mid
    else:
        single_cost = (ref_mid - single_price) / ref_mid
    vs_single = (single_cost - total_cost_frac) if (
        total_cost_frac is not None
    ) else None

    return {
        "allocations": allocations,
        "blended_price": blended_price,
        "blended_impact_frac": blended_impact,
        "total_cost_frac": total_cost_frac,
        "vs_single_pool": vs_single,          # cost-fraction saved vs single deepest pool
        "single_pool_price": single_price,
        "single_pool_cost_frac": single_cost,
        "size_usd": size_usd,
        "side": side,
    }


# --------------------------------------------------------------------------- #
# gather_pools_onchain: read live reserves for each pool                      #
# --------------------------------------------------------------------------- #

def _u(hexstr: str, start: int, end: int) -> int:
    return int(hexstr[start:end], 16)


class _OnchainReader:
    """Minimal eth_call client that mirrors bsc_indexer's _rpc style (own RPC,
    endpoint rotation), so routing stays self-contained and on-chain independent."""

    def __init__(self, http, settings):
        self.http = http
        self.rpcs = [settings.rpc_url(Chain.BSC),
                     "https://bsc.publicnode.com",
                     "https://binance.llamarpc.com"]
        self._i = 0

    async def _rpc(self, method: str, params: list):
        last: Exception | None = None
        for _ in range(len(self.rpcs)):
            url = self.rpcs[self._i]
            try:
                resp = await self.http.post_json(
                    url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                )
                if isinstance(resp, dict) and resp.get("error"):
                    raise RuntimeError(str(resp["error"]))
                return resp.get("result") if isinstance(resp, dict) else None
            except Exception as exc:  # transport / RPC error -> rotate
                last = exc
                self._i = (self._i + 1) % len(self.rpcs)
        if last:
            raise last
        return None

    async def call(self, to: str, data: str) -> str | None:
        try:
            res = await self._rpc("eth_call", [{"to": to, "data": data}, "latest"])
            return res if isinstance(res, str) and res not in ("", "0x") else None
        except Exception:
            return None


async def _pool_meta(reader: _OnchainReader, pair: str) -> dict | None:
    """token0/token1 addresses + decimals, stable-quote side, and V2-vs-V3 fee."""
    t0 = await reader.call(pair, SEL_TOKEN0)
    t1 = await reader.call(pair, SEL_TOKEN1)
    if not t0 or not t1:
        return None
    t0 = ("0x" + t0[-40:]).lower()
    t1 = ("0x" + t1[-40:]).lower()
    d0 = await reader.call(t0, SEL_DECIMALS)
    d1 = await reader.call(t1, SEL_DECIMALS)
    meta = {
        "token0": t0, "token1": t1,
        "dec0": int(d0, 16) if d0 else 18,
        "dec1": int(d1, 16) if d1 else 18,
    }
    if t0 in STABLES:
        meta.update(stable_idx=0, base_idx=1)
    elif t1 in STABLES:
        meta.update(stable_idx=1, base_idx=0)
    else:
        meta.update(stable_idx=None, base_idx=0)
    fee_hex = await reader.call(pair, SEL_FEE)
    if fee_hex and int(fee_hex, 16) > 0:
        meta["kind"] = "v3"
        meta["fee_bps"] = int(fee_hex, 16) / 100.0   # 1e-6 units -> bps
    else:
        meta["kind"] = "v2"
        meta["fee_bps"] = 25.0
    return meta


async def _v2_depth(reader: _OnchainReader, pair: str, meta: dict
                    ) -> tuple[float | None, float | None]:
    """V2 getReserves() -> (base_price_in_quote, single-leg quote reserve in token units).
    Returns (price, quote_reserve_tokens) where price is base priced in the quote token."""
    res = await reader.call(pair, SEL_GET_RESERVES)
    if not res:
        return None, None
    d = res[2:]
    r0 = _u(d, 0, 64) / 10 ** meta["dec0"]
    r1 = _u(d, 64, 128) / 10 ** meta["dec1"]
    base_idx = meta["base_idx"]
    base_r, quote_r = (r0, r1) if base_idx == 0 else (r1, r0)
    price = (quote_r / base_r) if base_r else None      # quote per base
    return price, quote_r


async def _v3_depth(reader: _OnchainReader, pair: str, meta: dict
                    ) -> tuple[float | None, float | None]:
    """V3 virtual reserves from slot0() sqrtPriceX96 + liquidity() L.
    x = L/sqrtP (token0), y = L*sqrtP (token1). Returns (base_price_in_quote,
    single-leg quote virtual reserve in token units)."""
    slot0 = await reader.call(pair, SEL_SLOT0)
    liq_hex = await reader.call(pair, SEL_LIQUIDITY)
    if not slot0 or not liq_hex:
        return None, None
    sqrt_x96 = _u(slot0[2:], 0, 64)
    liquidity = int(liq_hex, 16)
    if sqrt_x96 <= 0 or liquidity <= 0:
        return None, None
    sqrt_p = sqrt_x96 / 2 ** 96
    dec0, dec1 = meta["dec0"], meta["dec1"]
    x0 = (liquidity / sqrt_p) / 10 ** dec0                       # virtual reserve token0
    x1 = (liquidity * sqrt_p) / 10 ** dec1                       # virtual reserve token1
    price0_in_1 = (sqrt_p * sqrt_p) * 10 ** (dec0 - dec1)         # token1 per token0
    base_idx = meta["base_idx"]
    if base_idx == 0:
        price = price0_in_1                  # base=token0, quote=token1
        quote_r = x1
    else:
        price = (1.0 / price0_in_1) if price0_in_1 else None      # base=token1, quote=token0
        quote_r = x0
    return price, quote_r


async def gather_pools_onchain(http, settings, chain, pool_addresses: list[str],
                               quote_price_usd: float | None = None) -> list[dict]:
    """Read live reserves for each pool on-chain and return dicts `split_route` consumes.

    For each pool: detect V2 vs V3, read reserves (V2 getReserves / V3 virtual reserves
    from slot0+liquidity), and compute the base price in the quote token. If the quote
    is a stablecoin the price is already in USD and reserve_usd is exact. For a native
    quote (WBNB) pass `quote_price_usd` (the BNB/USD price) to anchor both the price and
    the depth in USD: otherwise price/reserve are left in quote-token units and that
    pool will only be routable alongside same-quote pools (the caller should pass the
    anchor). Mirrors bsc_indexer's eth_call decoding; no API key, no third-party index.
    """
    if Chain.parse(chain) != Chain.BSC:
        raise ValueError("gather_pools_onchain currently supports BSC pools only")
    reader = _OnchainReader(http, settings)
    out: list[dict] = []
    for pair in pool_addresses:
        pair = pair.lower()
        meta = await _pool_meta(reader, pair)
        if meta is None:
            continue
        if meta["kind"] == "v3":
            price_q, quote_r = await _v3_depth(reader, pair, meta)
            dex = "pancakeswap-v3"
        else:
            price_q, quote_r = await _v2_depth(reader, pair, meta)
            dex = "pancakeswap-v2"
        if price_q is None or quote_r is None or quote_r <= 0:
            continue

        stable_idx = meta["stable_idx"]
        if stable_idx is not None:
            # quote is a stablecoin: price and depth already in USD
            price_usd = price_q
            quote_usd = quote_r
        elif quote_price_usd is not None:
            # native quote anchored to USD by the caller
            price_usd = price_q * quote_price_usd
            quote_usd = quote_r * quote_price_usd
        else:
            # unanchored native quote: keep in quote-token units (router will skip
            # mixing with USD pools, but same-quote pools still route together)
            price_usd = price_q
            quote_usd = quote_r

        out.append({
            "pair_address": pair,
            "dex": dex,
            "fee_bps": meta["fee_bps"],
            "price_usd": price_usd,
            "reserve_usd": 2.0 * quote_usd,      # total TVL ~= 2x the quote leg
            "reserve_base_usd": quote_usd,       # single-sided value of the base leg
            "kind": meta["kind"],
        })
    return out


# --------------------------------------------------------------------------- #
# liquidity_graph: depth summary across pools                                 #
# --------------------------------------------------------------------------- #

def liquidity_graph(pools: list[dict]) -> dict:
    """Summarize the liquidity graph for a token: total depth across all pools, the
    per-DEX and per-fee-tier breakdown, and the dominant pool's share of total depth."""
    enriched = []
    total = 0.0
    for p in pools:
        r = _pool_reserve_usd(p)
        if r <= 0:
            continue
        total += r
        enriched.append({
            "pair": p.get("pair_address"),
            "dex": p.get("dex") or "unknown",
            "fee_bps": p.get("fee_bps") if p.get("fee_bps") is not None
                       else swap_fee_bps(p.get("dex")),
            "reserve_usd": r,
        })

    by_dex: dict[str, float] = {}
    by_fee: dict[str, float] = {}
    for e in enriched:
        by_dex[e["dex"]] = by_dex.get(e["dex"], 0.0) + e["reserve_usd"]
        key = f"{e['fee_bps']:.0f}bps"
        by_fee[key] = by_fee.get(key, 0.0) + e["reserve_usd"]

    dominant = max(enriched, key=lambda e: e["reserve_usd"]) if enriched else None
    return {
        "pool_count": len(enriched),
        "total_depth_usd": total,
        "by_dex": dict(sorted(by_dex.items(), key=lambda kv: kv[1], reverse=True)),
        "by_fee_tier": dict(sorted(by_fee.items(), key=lambda kv: kv[1], reverse=True)),
        "dominant_pool": (dominant["pair"] if dominant else None),
        "dominant_dex": (dominant["dex"] if dominant else None),
        "dominant_share": (dominant["reserve_usd"] / total) if (dominant and total > 0) else None,
        "pools": sorted(enriched, key=lambda e: e["reserve_usd"], reverse=True),
    }
