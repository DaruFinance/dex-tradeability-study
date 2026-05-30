"""EVM (BSC) HOLDER & WALLET analytics, reconstructed entirely on-chain from raw
ERC-20 `Transfer` logs via eth_getLogs. No third-party index, no key. This is the
data the paid platforms (Nansen/Birdeye) charge most for: holder distribution,
concentration, whale tracking, and per-wallet PnL, here we dig it from the chain.

The single primitive is the ERC-20 Transfer event:
    Transfer(address indexed from, address indexed to, uint256 value)
    topic0 = 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef
             == keccak256("Transfer(address,address,uint256)")  (verified)
    from  = topics[1]  (last 40 hex chars -> 20-byte address)
    to    = topics[2]  (last 40 hex chars -> 20-byte address)
    value = data       (word0, a single uint256)

A holder set is the net of every Transfer in the token's history: +value to the
recipient, -value from the sender. Mint = from 0x0; burn = to 0x0. Balances are
exact (integer arithmetic, scaled by decimals() only for the human-readable view).

THROUGHPUT, NOT DATA, IS THE LIMIT. Full reconstruction is feasible for new /
long-tail tokens (thousands of transfers) but heavy for mega-cap tokens (millions
of transfers), that is a throughput limit on free public RPCs, not a limit of the
method. Point BSC_ARCHIVE_RPC_URL at a local archive node to lift it. When a scan
is capped by MAX_CHUNKS this module emits a LOUD truncation warning and marks the
result `truncated=True`, there are no silent caps.

`holders_at(token, block)` re-runs the same reconstruction with to_block pinned to
a historical block, so distribution can be tracked OVER TIME (concentration at
launch vs now). `wallet_token_pnl` nets a single wallet's in/out flows of the token
and, given an optional price, computes realized/unrealized PnL, kept on-chain:
position = sum(in) - sum(out).
"""
from __future__ import annotations

import logging
import os

from .http import HttpError

log = logging.getLogger("chainscope.holders_evm")

# ERC-20 Transfer(address indexed from, address indexed to, uint256 value)
# topic0 == keccak256("Transfer(address,address,uint256)")  (verified on-chain/web)
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# function selectors
SEL_DECIMALS = "0x313ce567"      # decimals()
SEL_TOTAL_SUPPLY = "0x18160ddd"  # totalSupply()

# addresses that are not "holders": the zero address and the canonical dead/burn sink
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
DEAD_ADDRESS = "0x000000000000000000000000000000000000dead"
BURN_ADDRESSES = frozenset({ZERO_ADDRESS, DEAD_ADDRESS})

CHUNK = 5000          # blocks per getLogs call (safe on free archival RPCs)
MAX_CHUNKS = 400      # cap per reconstruction; beyond this, use a local archive node


def _topic_addr(topic: str) -> str:
    """Decode an indexed address topic (32-byte word, address in the low 20 bytes)."""
    return "0x" + topic[-40:].lower()


def _u(hexstr: str, start: int, end: int) -> int:
    return int(hexstr[start:end], 16)


class EvmHolders:
    """On-chain holder & wallet analytics for EVM tokens (BSC). Owns its own
    endpoint-rotating JSON-RPC client (it is NOT a Provider) so it can be used
    standalone. Mirrors the _rpc / getLogs-chunking / decoder style of
    providers.bsc_indexer.BscIndexerProvider.
    """

    def __init__(self, http, rpcs: list[str] | None = None):
        """`http` is a chainscope.http.HttpClient (provides post_json + HttpError)."""
        self.http = http
        # rotate across free archival endpoints that serve wide getLogs ranges;
        # set BSC_ARCHIVE_RPC_URL (or a comma list) to pin your own / a local node.
        env = os.environ.get("BSC_ARCHIVE_RPC_URL")
        if rpcs:
            self.rpcs = list(rpcs)
        elif env:
            self.rpcs = [u.strip() for u in env.split(",") if u.strip()]
        else:
            self.rpcs = [
                "https://bsc.publicnode.com",
                "https://bsc-rpc.publicnode.com",
                "https://binance.llamarpc.com",
            ]
        self._rpc_i = 0
        self._dec_cache: dict[str, int] = {}

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

    async def _eth_call(self, to: str, data: str) -> str | None:
        try:
            return await self._rpc("eth_call", [{"to": to, "data": data}, "latest"])
        except HttpError:
            return None

    async def latest_block(self) -> int:
        return int(await self._rpc("eth_blockNumber", []), 16)

    async def decimals(self, token: str) -> int:
        token = token.lower()
        if token in self._dec_cache:
            return self._dec_cache[token]
        d = await self._eth_call(token, SEL_DECIMALS)
        dec = int(d, 16) if d and d != "0x" else 18
        self._dec_cache[token] = dec
        return dec

    async def total_supply(self, token: str) -> int | None:
        """Raw (un-scaled) totalSupply, or None if the call fails."""
        s = await self._eth_call(token, SEL_TOTAL_SUPPLY)
        if not s or s == "0x":
            return None
        try:
            return int(s, 16)
        except ValueError:
            return None

    # ---- log fetching ----

    async def _transfer_logs(self, token: str, from_block: int, to_block: int,
                             extra_topics: list | None = None) -> tuple[list[dict], bool]:
        """Chunked eth_getLogs of Transfer events for `token`. Returns (logs, truncated).
        `extra_topics` lets callers add an indexed-address filter (topics[1]/[2]) so a
        single-wallet scan does not have to pull the whole token's transfer history."""
        topics: list = [TRANSFER_TOPIC]
        if extra_topics:
            topics.extend(extra_topics)
        out: list[dict] = []
        b = from_block
        chunks = 0
        while b <= to_block and chunks < MAX_CHUNKS:
            hi = min(b + CHUNK - 1, to_block)
            try:
                res = await self._rpc("eth_getLogs", [{
                    "address": token, "topics": topics,
                    "fromBlock": hex(b), "toBlock": hex(hi),
                }])
                out.extend(res or [])
            except HttpError as exc:
                log.debug("getLogs %s-%s failed: %s", b, hi, exc)
            b = hi + 1
            chunks += 1
        truncated = b <= to_block
        if truncated:
            log.warning(
                "EvmHolders: TRUNCATED reconstruction of %s, scanned %d blocks then hit "
                "MAX_CHUNKS=%d at block %d (target end %d). Holder set is INCOMPLETE. "
                "Point BSC_ARCHIVE_RPC_URL at a local archive node, or narrow the block "
                "range. This is a throughput cap, not a data limit.",
                token, b - from_block, MAX_CHUNKS, b, to_block,
            )
        return out, truncated

    # ---- core reconstruction ----

    async def holder_balances(self, token: str, from_block: int | None = None,
                              to_block: int | None = None) -> dict:
        """Reconstruct the full holder set for `token` by netting every Transfer.

        Returns a dict:
            {
              "token": <addr>, "decimals": int,
              "from_block": int, "to_block": int, "truncated": bool,
              "transfer_count": int,
              "balances": {address -> human-scaled float balance},   # holders only
              "balances_raw": {address -> int raw balance},          # holders only
              "minted": float, "burned": float,                       # human-scaled
            }
        Zero/burn addresses are EXCLUDED from `balances` (they are not holders), but
        their net flow is reported as `minted`/`burned` so the supply identity checks.

        NOTE: full reconstruction is feasible for new / long-tail tokens (thousands
        of transfers) but heavy for mega-cap tokens (millions of transfers), a
        throughput limit on free RPCs, not a data limit. A local archive node lifts
        it. If the scan is capped, `truncated` is True and a loud warning is logged.
        """
        token = token.lower()
        dec = await self.decimals(token)
        if to_block is None:
            to_block = await self.latest_block()
        if from_block is None:
            from_block = 0
        logs, truncated = await self._transfer_logs(token, from_block, to_block)

        raw: dict[str, int] = {}
        for lg in logs:
            tps = lg.get("topics") or []
            if len(tps) < 3:
                continue  # not a standard 3-topic Transfer (e.g. ERC-721 has 4; skip)
            data = lg.get("data") or "0x"
            d = data[2:]
            if len(d) < 64:
                continue
            value = _u(d, 0, 64)
            if value == 0:
                continue
            src = _topic_addr(tps[1])
            dst = _topic_addr(tps[2])
            raw[src] = raw.get(src, 0) - value
            raw[dst] = raw.get(dst, 0) + value

        scale = 10 ** dec
        minted_raw = -sum(raw.get(a, 0) for a in BURN_ADDRESSES if raw.get(a, 0) < 0)
        burned_raw = sum(raw.get(a, 0) for a in BURN_ADDRESSES if raw.get(a, 0) > 0)
        balances_raw: dict[str, int] = {}
        balances: dict[str, float] = {}
        for addr, bal in raw.items():
            if addr in BURN_ADDRESSES:
                continue
            if bal <= 0:
                continue  # net-zero / dust-negative (rounding of partial scans): not a holder
            balances_raw[addr] = bal
            balances[addr] = bal / scale

        return {
            "token": token,
            "decimals": dec,
            "from_block": from_block,
            "to_block": to_block,
            "truncated": truncated,
            "transfer_count": len(logs),
            "balances": balances,
            "balances_raw": balances_raw,
            "minted": minted_raw / scale,
            "burned": burned_raw / scale,
        }

    async def holders_at(self, token: str, block: int,
                         from_block: int | None = None) -> dict:
        """Holder balances reconstructed AS-OF a historical block (to_block=block).
        Lets distribution be tracked over time: e.g. concentration at launch vs now.
        Same return shape as holder_balances."""
        return await self.holder_balances(token, from_block=from_block, to_block=block)

    # ---- concentration / distribution stats ----

    @staticmethod
    def holder_stats(balances: dict, dev_address: str | None = None) -> dict:
        """Concentration & distribution metrics over a {address -> balance} map.

        Accepts either the full dict returned by holder_balances (uses its
        `balances` key) or a bare {address -> balance} mapping.

        Returns: holder_count, total_held, top10_pct, top50_pct, hhi (Herfindahl,
        sum of share^2, in [0,1]), gini ([0,1]), and dev/creator share if a
        dev_address with a balance is supplied. Percentages are fractions (0..1).
        """
        if isinstance(balances, dict) and "balances" in balances:
            balances = balances["balances"]
        vals = [float(v) for v in balances.values() if v and v > 0]
        n = len(vals)
        total = sum(vals)
        out: dict = {
            "holder_count": n,
            "total_held": total,
            "top10_pct": None,
            "top50_pct": None,
            "hhi": None,
            "gini": None,
            "dev_pct": None,
            "dev_address": dev_address.lower() if dev_address else None,
        }
        if n == 0 or total <= 0:
            return out
        ordered = sorted(vals, reverse=True)
        out["top10_pct"] = sum(ordered[:10]) / total
        out["top50_pct"] = sum(ordered[:50]) / total
        shares = [v / total for v in vals]
        out["hhi"] = sum(s * s for s in shares)
        out["gini"] = EvmHolders._gini(ordered, total)
        if dev_address:
            dev_bal = balances.get(dev_address.lower())
            if dev_bal:
                out["dev_pct"] = float(dev_bal) / total
        return out

    @staticmethod
    def _gini(ordered_desc: list[float], total: float) -> float:
        """Gini coefficient of the balance distribution (0 = perfectly equal,
        ->1 = one wallet holds everything). Computed on values sorted ascending."""
        n = len(ordered_desc)
        if n <= 1 or total <= 0:
            return 0.0
        asc = ordered_desc[:-1]  # ascending
        cum = 0.0
        weighted = 0.0
        for i, v in enumerate(asc, start=1):
            cum += v
            weighted += i * v
        # gini = (2*sum(i*x_i)/(n*sum(x))) - (n+1)/n
        return (2.0 * weighted) / (n * total) - (n + 1.0) / n

    # ---- per-wallet PnL ----

    async def wallet_token_pnl(self, wallet: str, token: str, price_usd: float | None = None,
                               from_block: int | None = None,
                               to_block: int | None = None) -> dict:
        """Per-wallet position & PnL for one token, reconstructed on-chain from the
        wallet's own Transfer in/out of the token.

        position = sum(received) - sum(sent)   (token units, exact)

        If `price_usd` (current token price) is supplied, computes unrealized PnL on
        the open position. Realized PnL requires per-trade USD prices we do not have
        on-chain alone, so it is reported as 0.0 with a note; pass a price feed
        upstream for full cost-basis accounting. With no price, quantities only.

        Returns: total_in, total_out, position (token units), transfer_count,
        position_usd, unrealized_pnl_usd (None if no price), realized_pnl_usd,
        truncated, notes.
        """
        wallet = wallet.lower()
        token = token.lower()
        dec = await self.decimals(token)
        if to_block is None:
            to_block = await self.latest_block()
        if from_block is None:
            from_block = 0

        wtopic = "0x" + "0" * 24 + wallet[2:]  # 32-byte left-padded indexed address
        # one scan for inflows (to == wallet), one for outflows (from == wallet);
        # the indexed-topic filter means we never pull the whole token's history.
        in_logs, tr_in = await self._transfer_logs(token, from_block, to_block,
                                                    extra_topics=[None, wtopic])
        out_logs, tr_out = await self._transfer_logs(token, from_block, to_block,
                                                     extra_topics=[wtopic])
        truncated = tr_in or tr_out

        def _sum(logs: list[dict]) -> int:
            s = 0
            for lg in logs:
                d = (lg.get("data") or "0x")[2:]
                if len(d) >= 64:
                    s += _u(d, 0, 64)
            return s

        scale = 10 ** dec
        total_in = _sum(in_logs) / scale
        total_out = _sum(out_logs) / scale
        position = total_in - total_out

        notes = []
        position_usd = None
        unrealized = None
        realized = 0.0
        if price_usd is not None:
            position_usd = position * price_usd
            unrealized = position * price_usd  # mark-to-market vs zero cost basis on-chain
            notes.append(
                "unrealized_pnl_usd is current mark-to-market of the net position; "
                "cost-basis-aware realized/unrealized PnL needs per-trade USD prices "
                "(supply a price feed upstream)."
            )
        else:
            notes.append("no price_usd supplied, quantities only (on-chain position).")
        if truncated:
            notes.append("TRUNCATED scan, flows incomplete; use a local archive node.")

        return {
            "wallet": wallet,
            "token": token,
            "decimals": dec,
            "from_block": from_block,
            "to_block": to_block,
            "transfer_count": len(in_logs) + len(out_logs),
            "total_in": total_in,
            "total_out": total_out,
            "position": position,
            "position_usd": position_usd,
            "unrealized_pnl_usd": unrealized,
            "realized_pnl_usd": realized,
            "truncated": truncated,
            "notes": notes,
        }
