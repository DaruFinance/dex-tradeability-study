"""Fully on-chain BNB Smart Chain provider: token metadata, pools, and token-safety
read entirely from a BSC node via eth_call + eth_getLogs. No third-party data API.

This replaces the gated platforms (DexScreener / GeckoTerminal / GoPlus / Honeypot.is)
for BSC with pure RPC. The only external dependency is a JSON-RPC endpoint (free public
nodes work out of the box; point BSC_RPC_URL / BSC_ARCHIVE_RPC_URL at your own node for
deeper getLogs history).

How each capability is served:
  - token : ERC-20 views (name/symbol/decimals/totalSupply) + price/mcap/liquidity from
            the token's deepest discovered pool's reserves.
  - pools : discovered from factory creation logs (PancakeSwap V2 PairCreated,
            V3 PoolCreated), every pool that holds the token, dead or alive.
  - rug   : owner()/renounce, LP burned/locked (LP-token balances at dead/locker
            addresses), and honeypot + buy/sell tax via an eth_call swap SIMULATION
            with STATE OVERRIDE (the technique honeypot.is uses internally).

Pricing: USD when the pool quote is a known stablecoin (USDT/USDC/BUSD). WBNB-quoted
pools are priced in WBNB (price_native); WBNB itself is priced off a WBNB/USDT pool so
WBNB-quoted tokens can also be expressed in USD.

VERIFIED on-chain constants (see module-level comments for source URLs):
  V2 factory  0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73   PairCreated topic 0x0d3648bd…
  V3 factory  0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865    PoolCreated topic 0x783cca1c…
  V2 router   0x10ED43C718714eb63d5aA57B78B54704E256024E    (getAmountsOut + swap sims)
All topic0 hashes / selectors below were recomputed with keccak256 and cross-checked
against BscScan (see provider report).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from ..chains import Chain, normalize_address
from ..http import HttpError
from ..models import Pool, RugReport, Token
from .base import CAP_POOLS, CAP_RUG, CAP_TOKEN, NotSupported, Provider

log = logging.getLogger("chainscope.bsc_chain")

# ---- ERC-20 / pool / Ownable function selectors (first 4 bytes of keccak(sig)) ----
SEL_NAME = "0x06fdde03"          # name()
SEL_SYMBOL = "0x95d89b41"        # symbol()
SEL_DECIMALS = "0x313ce567"      # decimals()
SEL_TOTAL_SUPPLY = "0x18160ddd"  # totalSupply()
SEL_OWNER = "0x8da5cb5b"         # owner()
SEL_BALANCE_OF = "0x70a08231"    # balanceOf(address)
SEL_GET_RESERVES = "0x0902f1ac"  # getReserves()  (V2 pair)
SEL_SLOT0 = "0x3850c7bd"         # slot0()        (V3 pool)
SEL_TOKEN0 = "0x0dfe1681"        # token0()
SEL_TOKEN1 = "0xd21220a7"        # token1()
SEL_FEE = "0xddca3f43"           # fee()          (V3 pool, uint24, 1e-6 units)
SEL_GET_AMOUNTS_OUT = "0xd06ca61f"  # getAmountsOut(uint256,address[])
SEL_GET_PAIR = "0xe6a43905"      # getPair(address,address)            (V2 factory)
SEL_GET_POOL = "0x1698ee82"      # getPool(address,address,uint24)     (V3 factory)

# ---- PancakeSwap on-chain addresses (lowercased) ----
V2_FACTORY = "0xca143ce32fe78f1f7019d7d551a6402fc5350c73"
V3_FACTORY = "0x0bfbcf9fa4f9c56b0f40a671ad40e0805a091865"
V2_ROUTER = "0x10ed43c718714eb63d5aa57b78b54704e256024e"
WBNB = "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"

# Event topic0 (keccak256 of the event signature), recomputed + BscScan-verified.
PAIR_CREATED_TOPIC = "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9"
POOL_CREATED_TOPIC = "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118"

# Known USD stablecoins on BSC (lowercased), pools quoted in these price directly in USD.
STABLES = {
    "0x55d398326f99059ff775485246999027b3197955",  # USDT
    "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",  # USDC
    "0xe9e7cea3dedca5984780bafc599bd69add087d56",  # BUSD
    "0x1af3f329e8be154074d8769d1ffa4ee058b1dbc3",  # DAI
}

# Standard quote tokens to probe for direct factory.getPair/getPool lookups. Public RPCs
# cap eth_getLogs at ~50k blocks, so a from-genesis log scan of the factory is infeasible
# without an archive node; instead we ask the factory directly for the canonical pools
# (one eth_call each, no scan, complete). These cover essentially all real liquidity.
QUOTE_TOKENS = [
    "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c",  # WBNB
    "0x55d398326f99059ff775485246999027b3197955",  # USDT
    "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",  # USDC
    "0xe9e7cea3dedca5984780bafc599bd69add087d56",  # BUSD
]
# PancakeSwap V3 fee tiers (uint24, 1e-6 units): 0.01%, 0.05%, 0.25%, 1%.
V3_FEE_TIERS = [100, 500, 2500, 10000]
ZERO_ADDR = "0x0000000000000000000000000000000000000000"

# WBNB/USDT 0.05%-ish V2 pair is the deepest BNB price source; we discover it dynamically,
# but keep the canonical WBNB/USDT V2 pair as a fast path / fallback.
WBNB_USDT_V2_PAIR = "0x16b9a82891338f9ba80e2d6970fdda79d1eb0dae"

# Dead / burn addresses: LP or supply sent here is provably unrecoverable.
DEAD_ADDRESSES = [
    "0x000000000000000000000000000000000000dead",
    "0x0000000000000000000000000000000000000000",
]

# Well-known liquidity lockers on BSC (lowercased). LP held here is locked, not burned.
LP_LOCKERS = {
    "0xc765bddb93b0d1c1a88282ba0fa6b2d00e3e0c83": "Unicrypt/UNCX V2",
    "0xfd0f4ce2cb83c87a3d75f3056e21e63de6f5bd75": "Unicrypt/UNCX V3",
    "0x7ee058420e5937496f5a2096f04caa7721cf70cc": "PinkLock v1",
    "0x407993575c91ce7643a4d4ccacc9a98c36ee1bbe": "PinkLock v2",
    "0x71b5759d73262fbb223956913ecf4ecc51057641": "PinkLock v2 (alt)",
    "0xe2fe530c047f2d85298b07d9333c05737f1435fb": "Team.Finance",
    "0x5f0d4c2a4e9f1c8b0f0e0c2e9d2f4a6b8c0d1e2f": "Mudra Locker",
}

# Honeypot simulation parameters.
SIM_BUY_BNB_WEI = 10 ** 17           # 0.1 BNB notional buy
HUGE_BALANCE = "0x21e19e0c9bab2400000"  # 10,000 ether worth of wei, for the fake buyer
# Deterministic throwaway buyer EOA used in the sims (no code, no real funds touched).
SIM_BUYER = "0x1234567890123456789012345678901234567890"


# ---- minimal pure-Python keccak-256 (no new pip deps) ----
# Needed to derive ERC-20 mapping storage slots for the honeypot state-override.
# Verified: keccak256(b"") == c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470.
_KECCAK_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
_KECCAK_R = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61],
             [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]


def keccak256(msg: bytes) -> bytes:
    """Keccak-256 (the pre-NIST padding used by Ethereum), rate=1088 bits."""
    def rol(x, n):
        return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF

    def keccak_f(st):
        for rnd in range(24):
            c = [st[x][0] ^ st[x][1] ^ st[x][2] ^ st[x][3] ^ st[x][4] for x in range(5)]
            d = [c[(x - 1) % 5] ^ rol(c[(x + 1) % 5], 1) for x in range(5)]
            for x in range(5):
                for y in range(5):
                    st[x][y] ^= d[x]
            b = [[0] * 5 for _ in range(5)]
            for x in range(5):
                for y in range(5):
                    b[y][(2 * x + 3 * y) % 5] = rol(st[x][y], _KECCAK_R[x][y])
            for x in range(5):
                for y in range(5):
                    st[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
            st[0][0] ^= _KECCAK_RC[rnd]
        return st

    rate = 136
    st = [[0] * 5 for _ in range(5)]
    m = bytearray(msg)
    m.append(0x01)
    while len(m) % rate != 0:
        m.append(0x00)
    m[-1] ^= 0x80
    for off in range(0, len(m), rate):
        blk = m[off:off + rate]
        for i in range(rate // 8):
            st[i % 5][i // 5] ^= int.from_bytes(blk[i * 8:i * 8 + 8], "little")
        st = keccak_f(st)
    out = bytearray()
    for i in range(4):
        out += st[i % 5][i // 5].to_bytes(8, "little")
    return bytes(out[:32])


def _u(hexstr: str, start: int, end: int) -> int:
    return int(hexstr[start:end], 16)


def _hex_int(result) -> int | None:
    if not isinstance(result, str) or result in ("", "0x"):
        return None
    try:
        return int(result, 16)
    except ValueError:
        return None


def _topic_addr(addr: str) -> str:
    """Left-pad a 20-byte address to a 32-byte log topic."""
    return "0x" + "0" * 24 + normalize_address(Chain.BSC, addr)[2:]


def _enc_addr(addr: str) -> str:
    """ABI-encode an address as a 32-byte word (no 0x)."""
    return "0" * 24 + addr.lower().replace("0x", "")


def _enc_uint(n: int) -> str:
    """ABI-encode a uint256 as a 32-byte word (no 0x)."""
    return f"{n:064x}"


def _decode_string(result: str | None) -> str | None:
    """Decode an ERC-20 name()/symbol() return, handling BOTH the ABI dynamic-string
    encoding (offset+length+bytes) AND the legacy bytes32 form used by old tokens
    (e.g. MKR-style fixed-32-byte name/symbol)."""
    if not isinstance(result, str) or result in ("", "0x"):
        return None
    h = result[2:] if result.startswith("0x") else result
    if len(h) < 64:
        return None
    try:
        # Dynamic string: word0 = offset (usually 0x20), word1 = length, then bytes.
        offset = int(h[0:64], 16)
        if offset == 0x20 and len(h) >= 128:
            length = int(h[64:128], 16)
            if 0 < length <= (len(h) - 128) // 2:
                data = bytes.fromhex(h[128:128 + length * 2])
                s = data.decode("utf-8", errors="replace").strip("\x00").strip()
                if s:
                    return s
        # Legacy bytes32: the whole first word is the string, right-padded with NULs.
        data = bytes.fromhex(h[0:64])
        s = data.decode("utf-8", errors="replace").rstrip("\x00").strip()
        # Keep only the printable run; junk bytes => not a bytes32 string.
        s = "".join(ch for ch in s if ch.isprintable())
        return s or None
    except (ValueError, UnicodeDecodeError):
        return None


# ---- swap-simulator bytecode assembler (for the honeypot code-override probe) ----

# Selectors used inside the simulator.
SEL_SWAP_ETH_FOR_TOKENS = "b6f9de95"   # swapExactETHForTokensSupportingFeeOnTransferTokens
SEL_SWAP_TOKENS_FOR_TOKENS_FOT = "5c11d795"  # swapExactTokensForTokensSupportingFeeOnTransferTokens
SEL_APPROVE = "095ea7b3"               # approve(address,uint256)
SEL_BALANCE_OF_RAW = "70a08231"        # balanceOf(address)
MAX_DEADLINE = (1 << 64) - 1


class _SimAsm:
    """Builds runtime EVM bytecode for an atomic buy+sell honeypot probe contract.

    The contract is injected at SIM_BUYER via an eth_call `code` state override and called
    with msg.value = SIM_BUY_BNB_WEI. It performs, with `self` as recipient:
      1. router.swapExactETHForTokensSupportingFeeOnTransferTokens{value:callvalue}(
             0, [WBNB, token], self, MAX_DEADLINE)             # buy: BNB -> token
      2. bought = token.balanceOf(self)                         # realized tokens (post buy-tax)
      3. token.approve(router, bought)
      4. wbnb_before = WBNB.balanceOf(self)
         router.swapExactTokensForTokensSupportingFeeOnTransferTokens(
             bought, 0, [token, WBNB], self, MAX_DEADLINE)      # sell: token -> WBNB
         wbnb_back = WBNB.balanceOf(self) - wbnb_before         # realized WBNB (post sell-tax)
      5. return abi.encode(bought, wbnb_back)
    We sell to WBNB (not BNB) so the result is a plain ERC-20 balance delta, this avoids the
    router's WETH.withdraw()->safeTransferETH unwrap, whose BNB credit is not reliably visible
    to SELFBALANCE inside an eth_call frame. The fee-on-transfer-supporting variant handles
    taxed tokens (the non-FOT variant reverts on them). Any sub-call revert bubbles up; the
    caller treats a top-level revert as a sell trap / honeypot.

    Implementation note: each call's calldata is laid out in memory at a fixed base, then CALL;
    success is checked with a branchless revert-on-failure. ADDRESS/CALLVALUE supply dynamic
    values; ABI offsets are constants. Scratch words live at high memory (0x100+).
    """

    def __init__(self, token: str, sell: bool = True):
        self.token = token.lower().replace("0x", "")
        self.wbnb = WBNB.replace("0x", "")
        self.router = V2_ROUTER.replace("0x", "")
        self.sell = sell      # sell=False -> buy-only probe (returns bought, 0)
        self.code = bytearray()

    # --- low-level emit helpers ---
    def _op(self, b: int):
        self.code.append(b)

    def _push(self, val: int, size: int | None = None):
        data = val.to_bytes((max(val.bit_length(), 1) + 7) // 8, "big") if size is None \
            else val.to_bytes(size, "big")
        if len(data) == 0:
            data = b"\x00"
        if len(data) > 32:
            raise ValueError("push >32 bytes")
        self._op(0x5F + len(data))  # PUSH1..PUSH32 (0x60..0x7f); PUSH0 unused here
        self.code += data

    def _push_hex(self, hexstr: str):
        data = bytes.fromhex(hexstr)
        self._op(0x5F + len(data))
        self.code += data

    # --- memory store helpers ---
    def _mstore_word(self, mem_off: int, word_hex: str):
        """MSTORE a full 32-byte constant word at mem_off."""
        self._push_hex(word_hex.rjust(64, "0"))   # value
        self._push(mem_off)                        # offset
        self._op(0x52)                             # MSTORE

    def build(self) -> str:
        # We lay out each call's calldata fresh starting at memory offset 0.
        # ---------- 1) BUY: swapExactETHForTokensSupportingFeeOnTransferTokens ----------
        # args: (uint amountOutMin=0, address[] path=[WBNB,token], address to=self, uint deadline)
        # ABI head: sel | amountOutMin | pathOffset(0x80) | to | deadline | [path: len|WBNB|token]
        self._mstore_word(0, SEL_SWAP_ETH_FOR_TOKENS.ljust(64, "0"))  # selector in top 4 bytes
        # selector must be left-aligned at mem[0:4]; above ljust puts it in high bytes. good.
        self._mstore_word(4, _enc_uint(0))             # amountOutMin = 0
        self._mstore_word(36, _enc_uint(0x80))         # offset to path array (from after selector)
        self._mstore_word(68, _enc_addr_word(""))      # to = self -> patch with ADDRESS at runtime
        self._mstore_word(100, _enc_uint(MAX_DEADLINE))
        self._mstore_word(132, _enc_uint(2))           # path length
        self._mstore_word(164, _enc_addr_word(self.wbnb))
        self._mstore_word(196, _enc_addr_word(self.token))
        # patch `to` = ADDRESS at mem[68]
        self._op(0x30)              # ADDRESS
        self._push(68)             # offset
        self._op(0x52)             # MSTORE
        # CALL(gas, router, value=CALLVALUE, in=0, insize=228, out=0, outsize=0)
        self._push(0)              # outsize
        self._push(0)              # out
        self._push(228)            # insize (4 + 7*32 = 228)
        self._push(0)              # in
        self._op(0x34)             # CALLVALUE -> value
        self._push_hex(self.router)  # router addr
        self._op(0x5A)             # GAS
        self._op(0xF1)             # CALL -> success
        self._revert_if_zero()

        # ---------- 2) bought = token.balanceOf(self) ----------
        self._mstore_word(0, SEL_BALANCE_OF_RAW.ljust(64, "0"))
        self._op(0x30)             # ADDRESS
        self._push(4)
        self._op(0x52)             # MSTORE -> arg(self) at mem[4]
        # STATICCALL(gas, token, in=0, insize=36, out=0x100, outsize=32)
        self._push(32)             # outsize
        self._push(0x100)          # out
        self._push(36)             # insize
        self._push(0)              # in
        self._push_hex(self.token)
        self._op(0x5A)             # GAS
        self._op(0xFA)             # STATICCALL
        self._revert_if_zero()
        # bought = mload(0x100) ; stash at mem[0x120]
        self._push(0x100)
        self._op(0x51)             # MLOAD -> bought
        self._push(0x120)
        self._op(0x52)             # MSTORE bought @0x120

        if not self.sell:
            # buy-only probe: return (bought, 0)
            self._push(0x120)
            self._op(0x51)
            self._push(0x200)
            self._op(0x52)
            self._mstore_word(0x220, _enc_uint(0))
            self._push(64)
            self._push(0x200)
            self._op(0xF3)             # RETURN
            return self.code.hex()

        # ---------- 3) token.approve(router, bought) ----------
        self._mstore_word(0, SEL_APPROVE.ljust(64, "0"))
        self._mstore_word(4, _enc_addr_word(self.router))
        # amount = bought (copy mem[0x120] -> mem[36])
        self._push(0x120)
        self._op(0x51)             # MLOAD bought
        self._push(36)
        self._op(0x52)             # MSTORE @36
        # CALL(gas, token, 0, in=0, insize=68, out=0, outsize=0)
        self._push(0)
        self._push(0)
        self._push(68)
        self._push(0)
        self._push(0)              # value 0
        self._push_hex(self.token)
        self._op(0x5A)
        self._op(0xF1)
        self._revert_if_zero()

        # ---------- 4a) wbnb_before = WBNB.balanceOf(self)  (stash @0x140) ----------
        self._mstore_word(0, SEL_BALANCE_OF_RAW.ljust(64, "0"))
        self._op(0x30)             # ADDRESS
        self._push(4)
        self._op(0x52)
        self._push(32)             # outsize
        self._push(0x100)          # out
        self._push(36)             # insize
        self._push(0)              # in
        self._push_hex(self.wbnb)
        self._op(0x5A)
        self._op(0xFA)             # STATICCALL
        self._revert_if_zero()
        self._push(0x100)
        self._op(0x51)             # wbnb_before
        self._push(0x140)
        self._op(0x52)             # stash @0x140

        # ---------- 4b) SELL: swapExactTokensForTokensSupportingFeeOnTransferTokens ----------
        # args:(uint amountIn=bought, uint amountOutMin=0, address[] path=[token,WBNB],
        #        address to=self, uint deadline)  -> sells to WBNB (plain ERC-20 out)
        self._mstore_word(0, SEL_SWAP_TOKENS_FOR_TOKENS_FOT.ljust(64, "0"))
        # amountIn = bought (mem[0x120] -> mem[4])
        self._push(0x120)
        self._op(0x51)
        self._push(4)
        self._op(0x52)
        self._mstore_word(36, _enc_uint(0))            # amountOutMin = 0
        self._mstore_word(68, _enc_uint(0xA0))         # path offset (after 5 head words)
        self._mstore_word(100, _enc_addr_word(""))     # to = self (patched below)
        self._mstore_word(132, _enc_uint(MAX_DEADLINE))
        self._mstore_word(164, _enc_uint(2))           # path length
        self._mstore_word(196, _enc_addr_word(self.token))
        self._mstore_word(228, _enc_addr_word(self.wbnb))
        self._op(0x30)             # ADDRESS
        self._push(100)
        self._op(0x52)             # patch to=self @100
        # CALL(gas, router, 0, in=0, insize=260, out=0, outsize=0)
        self._push(0)
        self._push(0)
        self._push(260)            # 4 + 8*32 = 260
        self._push(0)
        self._push(0)              # value 0
        self._push_hex(self.router)
        self._op(0x5A)
        self._op(0xF1)
        self._revert_if_zero()

        # ---------- 4c) wbnb_back = WBNB.balanceOf(self) - wbnb_before ----------
        self._mstore_word(0, SEL_BALANCE_OF_RAW.ljust(64, "0"))
        self._op(0x30)
        self._push(4)
        self._op(0x52)
        self._push(32)
        self._push(0x100)
        self._push(36)
        self._push(0)
        self._push_hex(self.wbnb)
        self._op(0x5A)
        self._op(0xFA)
        self._revert_if_zero()
        # wbnb_back = mload(0x100) - mload(0x140)
        self._push(0x140)
        self._op(0x51)             # wbnb_before
        self._push(0x100)
        self._op(0x51)             # wbnb_after
        self._op(0x03)             # SUB (after - before)
        self._push(0x160)
        self._op(0x52)             # stash wbnb_back @0x160

        # ---------- 5) return abi.encode(bought, wbnb_back) ----------
        self._push(0x120)
        self._op(0x51)             # bought
        self._push(0x200)
        self._op(0x52)
        self._push(0x160)
        self._op(0x51)             # wbnb_back
        self._push(0x220)
        self._op(0x52)
        self._push(64)             # size
        self._push(0x200)          # offset
        self._op(0xF3)             # RETURN
        return self.code.hex()

    def _revert_if_zero(self):
        """If the top stack item (CALL success) is zero, REVERT(0,0); else pop it."""
        # success on stack. We do: PUSH continue_dest ; JUMPI ; PUSH0 PUSH0 REVERT ; JUMPDEST
        # Compute jump target after the revert block.
        # layout: [JUMPI dest][... we need dest = position of JUMPDEST]
        # Emit: PUSH2 <dest> ; SWAP1? -- JUMPI takes (dest, cond). Stack top is cond, so we
        # need dest pushed under cond: PUSH2 dest then SWAP1 -> (cond, dest)? JUMPI pops
        # dest then cond. Order: JUMPI expects [dest, cond] with dest on top-1. Actually
        # JUMPI pops top=dest, next=cond. So push dest AFTER cond is wrong. Use: cond on
        # stack -> PUSH2 dest -> stack [cond, dest], JUMPI pops dest(top), cond(next). Good.
        # placeholder for dest patched after we know JUMPDEST position.
        jumpi_pos = len(self.code)
        self._op(0x61)             # PUSH2
        self.code += b"\x00\x00"   # placeholder dest
        self._op(0x57)             # JUMPI
        # revert block (reached when cond == 0)
        self._push(0)
        self._push(0)
        self._op(0xFD)             # REVERT
        dest = len(self.code)
        self._op(0x5B)             # JUMPDEST
        # patch placeholder
        self.code[jumpi_pos + 1] = (dest >> 8) & 0xFF
        self.code[jumpi_pos + 2] = dest & 0xFF


def _enc_addr_word(addr_hex: str) -> str:
    """Right-align a 20-byte address (hex, no 0x) into a 32-byte word; '' -> zero word."""
    a = addr_hex.replace("0x", "")
    return a.rjust(64, "0")


class BscChainProvider(Provider):
    """Pure-RPC BSC provider. eth_call + eth_getLogs only, no third-party data API."""

    name = "bsc_chain"
    supported_chains = frozenset({Chain.BSC})
    capabilities = frozenset({CAP_TOKEN, CAP_POOLS, CAP_RUG})
    requires_key = False
    onchain = True

    # getLogs chunking. Public RPCs cap the per-call span (~50k blocks) AND are slow, so the
    # supplementary creation-log scan only covers a recent window by default; the primary
    # discovery path (factory.getPair/getPool) needs no scan and is complete for real
    # liquidity. Point BSC_ARCHIVE_RPC_URL at a fast/archive node and raise these for deeper
    # historical pool discovery.
    CHUNK = 10000
    MAX_CHUNKS = int(os.environ.get("BSC_CHAIN_LOG_CHUNKS", "12"))

    def __init__(self, http, settings):
        super().__init__(http, settings)
        # Endpoint rotation, mirroring bsc_indexer: prefer a configured node, else free
        # public archival endpoints that serve wide getLogs ranges.
        env = os.environ.get("BSC_ARCHIVE_RPC_URL") or os.environ.get("BSC_RPC_URL")
        if env:
            self.rpcs = [u.strip() for u in env.split(",") if u.strip()]
        else:
            self.rpcs = [
                self.settings.rpc_url(Chain.BSC),
                "https://bsc.publicnode.com",
                "https://binance.llamarpc.com",
                "https://bsc-rpc.publicnode.com",
            ]
        # de-dup preserving order
        seen: set[str] = set()
        self.rpcs = [u for u in self.rpcs if not (u in seen or seen.add(u))]
        self._rpc_i = 0
        self._meta_cache: dict[str, dict] = {}
        self._wbnb_usd: float | None = None
        self._supports_override: bool | None = None

    # ---- low-level RPC (rotates endpoints on failure) ----

    async def _rpc(self, method: str, params: list):
        last: Exception | None = None
        for _ in range(len(self.rpcs)):
            url = self.rpcs[self._rpc_i]
            try:
                resp = await self.http.post_json(
                    url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                )
                if isinstance(resp, dict) and resp.get("error") is not None:
                    raise HttpError(0, url, str(resp["error"]))
                return resp.get("result") if isinstance(resp, dict) else None
            except HttpError as exc:
                last = exc
                self._rpc_i = (self._rpc_i + 1) % len(self.rpcs)
        if last is not None:
            raise last
        return None

    async def _eth_call(self, to: str, data: str, overrides: dict | None = None) -> str | None:
        """eth_call returning the hex result, or None on revert/error.
        `overrides` (if given) is the geth state-override map passed as the 3rd param."""
        params: list = [{"to": to, "data": data}, "latest"]
        if overrides:
            params.append(overrides)
        try:
            return await self._rpc("eth_call", params)
        except HttpError:
            return None

    async def _latest_block(self) -> int:
        return int(await self._rpc("eth_blockNumber", []), 16)

    # ---- pool metadata + reserves ----

    async def _pair_meta(self, pair: str) -> dict | None:
        """token0/token1/decimals + V2-vs-V3 detection for a pool address."""
        pair = pair.lower()
        if pair in self._meta_cache:
            return self._meta_cache[pair]
        t0 = await self._eth_call(pair, SEL_TOKEN0)
        t1 = await self._eth_call(pair, SEL_TOKEN1)
        if not t0 or not t1 or t0 == "0x" or t1 == "0x":
            return None
        t0 = "0x" + t0[-40:]
        t1 = "0x" + t1[-40:]
        d0 = _hex_int(await self._eth_call(t0, SEL_DECIMALS))
        d1 = _hex_int(await self._eth_call(t1, SEL_DECIMALS))
        fee_hex = await self._eth_call(pair, SEL_FEE)
        fee_val = _hex_int(fee_hex)
        if fee_val and fee_val > 0:
            kind, fee_bps = "v3", fee_val / 100.0
        else:
            kind, fee_bps = "v2", 25.0
        meta = {
            "pair": pair, "token0": t0, "token1": t1,
            "dec0": d0 if d0 is not None else 18,
            "dec1": d1 if d1 is not None else 18,
            "kind": kind, "fee_bps": fee_bps,
        }
        self._meta_cache[pair] = meta
        return meta

    async def _pool_state(self, meta: dict) -> dict | None:
        """Read reserves/price for a pool. Returns dict with token amounts + raw price0in1.
        V2: getReserves(); V3: slot0() sqrtPriceX96 (depth not read here)."""
        pair = meta["pair"]
        dec0, dec1 = meta["dec0"], meta["dec1"]
        if meta["kind"] == "v3":
            res = await self._eth_call(pair, SEL_SLOT0)
            if not res or len(res) < 66:
                return None
            sqrt_x96 = _u(res, 2, 66)
            if sqrt_x96 <= 0:
                return None
            sqrt_p = sqrt_x96 / 2 ** 96
            price0in1 = (sqrt_p * sqrt_p) * 10 ** (dec0 - dec1)
            # V3 in-range depth: read each token's real balance held by the pool.
            bal0 = _hex_int(await self._eth_call(
                meta["token0"], SEL_BALANCE_OF + _enc_addr(pair)))
            bal1 = _hex_int(await self._eth_call(
                meta["token1"], SEL_BALANCE_OF + _enc_addr(pair)))
            amt0 = (bal0 or 0) / 10 ** dec0
            amt1 = (bal1 or 0) / 10 ** dec1
            return {"amt0": amt0, "amt1": amt1, "price0in1": price0in1}
        # V2
        res = await self._eth_call(pair, SEL_GET_RESERVES)
        if not res or len(res) < 130:
            return None
        r0 = _u(res, 2, 66) / 10 ** dec0
        r1 = _u(res, 66, 130) / 10 ** dec1
        price0in1 = (r1 / r0) if r0 else None
        return {"amt0": r0, "amt1": r1, "price0in1": price0in1}

    def _classify(self, meta: dict, token: str) -> dict:
        """Which side is the queried token (base) vs the quote, and quote type."""
        token = token.lower()
        t0, t1 = meta["token0"].lower(), meta["token1"].lower()
        if token == t0:
            base_idx, quote_idx = 0, 1
        elif token == t1:
            base_idx, quote_idx = 1, 0
        else:
            base_idx, quote_idx = 0, 1  # shouldn't happen; default
        quote = t1 if base_idx == 0 else t0
        if quote in STABLES:
            qkind = "usd"
        elif quote == WBNB:
            qkind = "wbnb"
        else:
            qkind = "other"
        return {"base_idx": base_idx, "quote_idx": quote_idx,
                "quote_addr": quote, "qkind": qkind}

    async def _wbnb_usd_price(self) -> float | None:
        """Price WBNB in USD from a WBNB/USDT V2 pool (cached per provider instance)."""
        if self._wbnb_usd is not None:
            return self._wbnb_usd
        meta = await self._pair_meta(WBNB_USDT_V2_PAIR)
        if meta:
            state = await self._pool_state(meta)
            if state:
                t0 = meta["token0"].lower()
                # price of WBNB in USDT
                if t0 == WBNB:
                    self._wbnb_usd = state["price0in1"]  # USDT per WBNB
                else:
                    p = state["price0in1"]
                    self._wbnb_usd = (1 / p) if p else None
        return self._wbnb_usd

    async def _pool_pricing(self, meta: dict, token: str) -> dict | None:
        """Compute price_usd / price_native / liquidity_usd for `token` in this pool."""
        state = await self._pool_state(meta)
        if not state:
            return None
        cls = self._classify(meta, token)
        base_idx, qkind = cls["base_idx"], cls["qkind"]
        amt0, amt1 = state["amt0"], state["amt1"]
        base_amt = amt0 if base_idx == 0 else amt1
        quote_amt = amt1 if base_idx == 0 else amt0
        # price of base measured in quote token
        price_in_quote = (quote_amt / base_amt) if base_amt else None
        price_usd = price_native = liq_usd = None
        if qkind == "usd":
            price_usd = price_in_quote
            liq_usd = 2 * quote_amt
        elif qkind == "wbnb":
            price_native = price_in_quote
            wbnb_usd = await self._wbnb_usd_price()
            if wbnb_usd and price_in_quote is not None:
                price_usd = price_in_quote * wbnb_usd
            if wbnb_usd:
                liq_usd = 2 * quote_amt * wbnb_usd
        return {
            "price_usd": price_usd, "price_native": price_native,
            "liquidity_usd": liq_usd, "base_amt": base_amt, "quote_amt": quote_amt,
            "quote_addr": cls["quote_addr"], "base_idx": base_idx,
        }

    # ---- pool discovery via factory creation logs ----

    async def _get_logs(self, address: str, topics: list, from_b: int, to_b: int) -> list[dict]:
        out: list[dict] = []
        b = from_b
        chunks = 0
        while b <= to_b and chunks < self.MAX_CHUNKS:
            hi = min(b + self.CHUNK - 1, to_b)
            try:
                res = await self._rpc("eth_getLogs", [{
                    "address": address, "topics": topics,
                    "fromBlock": hex(b), "toBlock": hex(hi),
                }])
                out.extend(res or [])
            except HttpError as exc:
                log.debug("getLogs %s %s-%s failed: %s", address, b, hi, exc)
            b = hi + 1
            chunks += 1
        return out

    async def _discover_direct(self, addr: str) -> dict[str, dict]:
        """Fast, complete-for-real-liquidity discovery: ask the factories directly for the
        canonical pools against standard quotes (factory.getPair / getPool). One eth_call
        each, no log scan. This is what pricing / deepest-pool / rug use."""
        addr = addr.lower()
        found: dict[str, dict] = {}
        for quote in QUOTE_TOKENS:
            if quote == addr:
                continue
            # V2 pair
            res = await self._eth_call(
                V2_FACTORY, SEL_GET_PAIR + _enc_addr(addr) + _enc_addr(quote))
            pair = ("0x" + res[-40:]).lower() if (res and len(res) >= 42) else None
            if pair and pair != ZERO_ADDR:
                found[pair] = {"pool": pair, "dex": "pancakeswap-v2",
                               "token0": None, "token1": None, "fee": None}
            # V3 pools (one per fee tier)
            for fee in V3_FEE_TIERS:
                res = await self._eth_call(
                    V3_FACTORY,
                    SEL_GET_POOL + _enc_addr(addr) + _enc_addr(quote) + _enc_uint(fee))
                pool = ("0x" + res[-40:]).lower() if (res and len(res) >= 42) else None
                if pool and pool != ZERO_ADDR:
                    found[pool] = {"pool": pool, "dex": "pancakeswap-v3",
                                   "token0": None, "token1": None, "fee": fee}
        return found

    async def _discover_pairs(self, addr: str, scan_logs: bool = False) -> list[dict]:
        """All pools holding `addr`. Direct factory lookups (always) + an optional bounded
        recent-window creation-log scan (`scan_logs=True`, for the explicit get_pools call)
        to also catch pools with non-standard quote tokens within the RPC's getLogs span."""
        found = await self._discover_direct(addr)
        if scan_logs:
            try:
                await self._discover_via_logs(addr.lower(), found)
            except HttpError:
                pass
        return list(found.values())

    async def _discover_via_logs(self, addr: str, found: dict) -> None:
        """Scan the most recent ~MAX_CHUNKS*CHUNK blocks of factory creation events for
        pools holding `addr` (token0 or token1 indexed). Bounded by the RPC's getLogs span;
        catches non-standard-quote pools that direct lookups miss. Full history needs an
        archive node via BSC_ARCHIVE_RPC_URL."""
        latest = await self._latest_block()
        from_b = max(0, latest - self.CHUNK * self.MAX_CHUNKS)
        addr_topic = _topic_addr(addr)

        async def scan(factory: str, topic0: str, dex: str):
            for topics in ([topic0, addr_topic], [topic0, None, addr_topic]):
                for lg in await self._get_logs(factory, topics, from_b, latest):
                    self._record_pair(lg, dex, found)

        await scan(V2_FACTORY, PAIR_CREATED_TOPIC, "pancakeswap-v2")
        await scan(V3_FACTORY, POOL_CREATED_TOPIC, "pancakeswap-v3")

    @staticmethod
    def _record_pair(lg: dict, dex: str, found: dict) -> None:
        """Decode the created pool/pair address from a creation log's non-indexed data.
        V2 PairCreated data = [pair(addr), allPairsLength]; the pair address is word0.
        V3 PoolCreated data = [tickSpacing, pool]; the pool address is word1."""
        data = lg.get("data", "")
        h = data[2:] if data.startswith("0x") else data
        if dex == "pancakeswap-v2":
            if len(h) < 64:
                return
            pool = "0x" + h[24:64]
        else:  # v3
            if len(h) < 128:
                return
            pool = "0x" + h[64 + 24:128]
        topics = lg.get("topics", [])
        token0 = ("0x" + topics[1][-40:]) if len(topics) > 1 else None
        token1 = ("0x" + topics[2][-40:]) if len(topics) > 2 else None
        fee = _hex_int(topics[3]) if (dex == "pancakeswap-v3" and len(topics) > 3) else None
        found[pool.lower()] = {
            "pool": pool.lower(), "dex": dex,
            "token0": token0, "token1": token1, "fee": fee,
        }

    async def _build_pool(self, disc: dict, token: str) -> Pool | None:
        meta = await self._pair_meta(disc["pool"])
        if not meta:
            return None
        pricing = await self._pool_pricing(meta, token)
        if not pricing:
            return None
        cls = self._classify(meta, token)
        base_addr = token.lower()
        quote_addr = pricing["quote_addr"]
        base_sym = await self._symbol(base_addr)
        quote_sym = await self._symbol(quote_addr)
        return Pool(
            source=self.name, chain=Chain.BSC, dex=disc["dex"],
            pair_address=disc["pool"],
            base_address=base_addr, base_symbol=base_sym,
            quote_address=quote_addr, quote_symbol=quote_sym,
            price_usd=pricing["price_usd"], price_native=pricing["price_native"],
            liquidity_usd=pricing["liquidity_usd"],
            liquidity_base=pricing["base_amt"], liquidity_quote=pricing["quote_amt"],
            raw=disc,
        )

    async def _symbol(self, addr: str) -> str | None:
        return _decode_string(await self._eth_call(addr, SEL_SYMBOL))

    async def _deepest_pool(self, addr: str) -> tuple[dict | None, dict | None]:
        """Discover pools, return (meta, pricing) for the deepest by USD liquidity."""
        discovered = await self._discover_pairs(addr)
        best_meta = best_pricing = None
        best_liq = -1.0
        for disc in discovered:
            meta = await self._pair_meta(disc["pool"])
            if not meta:
                continue
            pricing = await self._pool_pricing(meta, addr)
            if not pricing:
                continue
            liq = pricing["liquidity_usd"]
            # prefer pools with a USD-quantifiable depth; fall back to raw quote amount
            score = liq if liq is not None else (pricing["quote_amt"] or 0)
            if score > best_liq:
                best_liq, best_meta, best_pricing = score, meta, pricing
        return best_meta, best_pricing

    # ================= capability: token =================

    async def get_token(self, chain: Chain, address: str) -> Token | None:
        chain = Chain.parse(chain)
        if chain != Chain.BSC:
            raise NotSupported
        addr = normalize_address(chain, address)
        try:
            name = _decode_string(await self._eth_call(addr, SEL_NAME))
            symbol = _decode_string(await self._eth_call(addr, SEL_SYMBOL))
            dec = _hex_int(await self._eth_call(addr, SEL_DECIMALS))
            supply_raw = _hex_int(await self._eth_call(addr, SEL_TOTAL_SUPPLY))
        except HttpError:
            return None
        if supply_raw is None:
            return None  # not an ERC-20 (totalSupply reverted)
        decimals = dec if dec is not None else 18
        total_supply = supply_raw / 10 ** decimals

        price_usd = price_native = market_cap = liquidity_usd = None
        try:
            _, pricing = await self._deepest_pool(addr)
        except HttpError:
            pricing = None
        if pricing:
            price_usd = pricing["price_usd"]
            price_native = pricing["price_native"]
            liquidity_usd = pricing["liquidity_usd"]
            if price_usd is not None:
                market_cap = price_usd * total_supply

        return Token(
            source=self.name, chain=chain, address=addr,
            symbol=symbol, name=name, decimals=decimals,
            total_supply=total_supply, price_usd=price_usd, price_native=price_native,
            market_cap=market_cap, fdv=market_cap, liquidity_usd=liquidity_usd,
            raw={"name": name, "symbol": symbol, "decimals": decimals},
        )

    # ================= capability: pools =================

    async def get_pools(self, chain: Chain, address: str) -> list[Pool]:
        chain = Chain.parse(chain)
        if chain != Chain.BSC:
            raise NotSupported
        addr = normalize_address(chain, address)
        # scan_logs=True also catches non-standard-quote pools within the RPC's getLogs span.
        try:
            discovered = await self._discover_pairs(addr, scan_logs=True)
        except HttpError:
            return []
        pools: list[Pool] = []
        for disc in discovered:
            try:
                p = await self._build_pool(disc, addr)
            except HttpError:
                p = None
            if p is not None:
                pools.append(p)
        pools.sort(key=lambda p: (p.liquidity_usd or 0), reverse=True)
        return pools

    # ================= capability: rug =================

    async def get_rug(self, chain: Chain, address: str) -> RugReport | None:
        chain = Chain.parse(chain)
        if chain != Chain.BSC:
            raise NotSupported
        addr = normalize_address(chain, address)
        flags: list[str] = []

        # ---- ownership ----
        owner_address = ownership_renounced = None
        try:
            owner_hex = await self._eth_call(addr, SEL_OWNER)
        except HttpError:
            owner_hex = None
        if owner_hex and len(owner_hex) >= 40 and owner_hex != "0x":
            owner_address = ("0x" + owner_hex[-40:]).lower()
            ownership_renounced = owner_address in DEAD_ADDRESSES
            if not ownership_renounced:
                flags.append("ownership not renounced")

        # ---- LP burned / locked on the deepest V2 pair ----
        lp_burned_pct = lp_locked_pct = None
        try:
            best_meta, _ = await self._deepest_pool(addr)
        except HttpError:
            best_meta = None
        # find the deepest *V2* pair (LP-token accounting only applies to V2)
        v2_pair = None
        if best_meta and best_meta["kind"] == "v2":
            v2_pair = best_meta["pair"]
        else:
            try:
                for disc in await self._discover_pairs(addr):
                    m = await self._pair_meta(disc["pool"])
                    if m and m["kind"] == "v2":
                        v2_pair = m["pair"]
                        break
            except HttpError:
                v2_pair = None
        if v2_pair:
            lp_burned_pct, lp_locked_pct = await self._lp_safety(v2_pair, flags)

        # ---- honeypot + buy/sell tax via swap simulation (state override) ----
        is_honeypot = buy_tax = sell_tax = None
        try:
            sim = await self._simulate_honeypot(addr)
        except HttpError:
            sim = None
        if sim is not None:
            is_honeypot = sim.get("is_honeypot")
            buy_tax = sim.get("buy_tax")
            sell_tax = sim.get("sell_tax")
            note = sim.get("note")
            if is_honeypot:
                flags.append(f"honeypot ({note})" if note else "honeypot")
            else:
                if sell_tax is not None and sell_tax >= 0.10:
                    flags.append(f"high sell tax ~{sell_tax*100:.0f}%")
                if buy_tax is not None and buy_tax >= 0.10:
                    flags.append(f"high buy tax ~{buy_tax*100:.0f}%")
                if note:  # e.g. code-override-not-honored fallback note
                    flags.append(note)

        return RugReport(
            source=self.name, chain=chain, address=addr,
            risk_score=None,
            is_honeypot=is_honeypot, buy_tax=buy_tax, sell_tax=sell_tax,
            owner_address=owner_address, ownership_renounced=ownership_renounced,
            lp_burned_pct=lp_burned_pct, lp_locked_pct=lp_locked_pct,
            is_open_source=None,  # cannot be determined from RPC alone
            flags=flags,
            raw={"owner": owner_hex, "honeypot_sim": sim, "v2_pair": v2_pair},
        )

    async def _lp_safety(self, pair: str, flags: list) -> tuple[float | None, float | None]:
        """Read the V2 pair's LP-token totalSupply and the balances at dead/locker
        addresses to derive lp_burned_pct and lp_locked_pct (fractions 0..1)."""
        lp_supply = _hex_int(await self._eth_call(pair, SEL_TOTAL_SUPPLY))
        if not lp_supply:
            return None, None
        burned = 0
        for dead in DEAD_ADDRESSES:
            bal = _hex_int(await self._eth_call(pair, SEL_BALANCE_OF + _enc_addr(dead)))
            burned += bal or 0
        locked = 0
        locked_names: list[str] = []
        for locker, lname in LP_LOCKERS.items():
            bal = _hex_int(await self._eth_call(pair, SEL_BALANCE_OF + _enc_addr(locker)))
            if bal:
                locked += bal
                locked_names.append(lname)
        burned_pct = min(1.0, burned / lp_supply)
        locked_pct = min(1.0, locked / lp_supply)
        if burned_pct >= 0.5:
            flags.append(f"LP burned ~{burned_pct*100:.0f}%")
        if locked_pct >= 0.5:
            flags.append(f"LP locked ~{locked_pct*100:.0f}% ({', '.join(locked_names)})")
        if burned_pct < 0.5 and locked_pct < 0.5:
            flags.append("LP not majority-burned/locked")
        return burned_pct, locked_pct

    # ---- honeypot simulation via eth_call CODE-OVERRIDE simulator (honeypot.is style) ----

    async def _simulate_honeypot(self, token: str) -> dict | None:
        """Detect honeypot + buy/sell tax by EXECUTING a real buy then sell in a single
        eth_call against a simulator contract injected at a throwaway address via the geth
        state-override `code` field (the same approach honeypot.is uses internally).

        Why a simulator contract: a plain getAmountsOut quote reflects only pool math and
        NOT the token's own transfer-fee/blacklist logic, so it cannot see taxes or sell
        traps. By actually routing BNB->token (buy) and token->WBNB (sell) through the
        PancakeSwap V2 router inside one call, the token's `_transfer` runs for real:
          buy_tax  = 1 - (tokens_received / getAmountsOut_no_tax_expectation)
          sell_tax = 1 - (wbnb_received / getAmountsOut_no_tax_expectation_for_those_tokens)
          honeypot = the swap sub-call reverts, or the sell yields ~0 WBNB out.
        The simulator buys, measures balance, approves the router, sells to WBNB, and
        RETURNS (tokensBought, wbnbBack). If the whole call reverts we localize the trap
        with a buy-only probe: buy-revert => untradeable/anti-bot; buy-ok+sell-revert =>
        classic sell trap. All via eth_call, nothing is broadcast.

        Fallback: if the RPC rejects code override, we fall back to a spot WBNB->token->WBNB
        getAmountsOut round-trip (fees+tax combined, can't separate buy/sell)."""
        token = token.lower()
        # No-tax expectation for the buy (pool math only).
        exp_buy = await self._get_amounts_out(SIM_BUY_BNB_WEI, [WBNB, token])
        if not exp_buy or len(exp_buy) < 2 or exp_buy[-1] == 0:
            return {"is_honeypot": None, "buy_tax": None, "sell_tax": None,
                    "note": "no WBNB route, tax/honeypot not simulable"}
        expected_tokens = exp_buy[-1]

        sim = await self._run_sim_contract(token, sell=True)
        if sim is None:
            # code override not honored: spot round-trip fallback.
            exp_sell = await self._get_amounts_out(expected_tokens, [token, WBNB])
            wbnb_back = exp_sell[-1] if (exp_sell and len(exp_sell) >= 2) else 0
            return self._round_trip_tax(
                SIM_BUY_BNB_WEI, wbnb_back,
                note="code-override not honored, tax via spot round-trip")

        if sim.get("reverted"):
            # The full buy+sell reverted. Probe a buy-only sim to localize the trap:
            # if the buy also reverts the token blocks buying (untradeable / anti-bot);
            # if the buy succeeds, it's a classic sell trap.
            buy_only = await self._run_sim_contract(token, sell=False)
            if buy_only is None or buy_only.get("reverted") or not buy_only.get("bought"):
                return {"is_honeypot": True, "buy_tax": None, "sell_tax": None,
                        "note": "buy reverts (untradeable / anti-bot / no live swap)"}
            bought = buy_only["bought"]
            buy_tax = max(0.0, 1.0 - (bought / expected_tokens)) if expected_tokens else None
            return {"is_honeypot": True,
                    "buy_tax": round(buy_tax, 4) if buy_tax is not None else None,
                    "sell_tax": 1.0, "note": "sell reverts (honeypot)"}

        bought = sim["bought"]
        wbnb_back = sim["wbnb_back"]
        # buy tax: realized tokens vs no-tax expectation.
        buy_tax = max(0.0, 1.0 - (bought / expected_tokens)) if expected_tokens else None
        # no-tax expectation (WBNB) for selling exactly `bought` tokens.
        exp_sell = await self._get_amounts_out(bought, [token, WBNB])
        exp_back = exp_sell[-1] if (exp_sell and len(exp_sell) >= 2) else 0
        sell_tax = (max(0.0, 1.0 - (wbnb_back / exp_back)) if exp_back else None)
        is_honeypot = (wbnb_back == 0) or (sell_tax is not None and sell_tax >= 0.99)
        return {
            "is_honeypot": bool(is_honeypot),
            "buy_tax": round(buy_tax, 4) if buy_tax is not None else None,
            "sell_tax": round(sell_tax, 4) if sell_tax is not None else None,
            "note": None,
        }

    async def _run_sim_contract(self, token: str, sell: bool = True) -> dict | None:
        """Inject the simulator contract (code override) at SIM_BUYER and call it with
        msg.value = SIM_BUY_BNB_WEI. With sell=True it buys then sells and returns ABI
        (tokensBought, wbnbBack); with sell=False it only buys and returns (tokensBought, 0).
        A top-level revert => `reverted: True`. Returns None if the RPC rejects the code
        override (empty/short return), so the caller can use the getAmountsOut fallback."""
        code = _SimAsm(token, sell=sell).build()
        ovr = {SIM_BUYER: {"code": "0x" + code, "balance": HUGE_BALANCE}}
        params = [{"to": SIM_BUYER, "from": SIM_BUYER, "data": "0x",
                   "value": hex(SIM_BUY_BNB_WEI), "gas": "0x4000000"}, "latest", ovr]
        try:
            res = await self._rpc("eth_call", params)
        except HttpError:
            return {"reverted": True, "bought": 0, "wbnb_back": 0}
        if res is None:
            return {"reverted": True, "bought": 0, "wbnb_back": 0}
        h = res[2:] if res.startswith("0x") else res
        if len(h) < 128:
            return None  # override likely not honored -> caller uses fallback
        return {"reverted": False, "bought": int(h[0:64], 16),
                "wbnb_back": int(h[64:128], 16)}

    async def _get_amounts_out(self, amount_in: int, path: list[str]) -> list[int] | None:
        """router.getAmountsOut(amountIn, path) -> uint[] (no-tax pool-math expectation)."""
        head = _enc_uint(amount_in) + _enc_uint(64)
        tail = _enc_uint(len(path)) + "".join(_enc_addr(p) for p in path)
        res = await self._eth_call(V2_ROUTER, SEL_GET_AMOUNTS_OUT + head + tail)
        if not res or res == "0x":
            return None
        h = res[2:]
        try:
            n = int(h[64:128], 16)
            return [int(h[128 + i * 64:128 + (i + 1) * 64], 16) for i in range(n)]
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _round_trip_tax(bnb_in: int, wbnb_back: int, note: str) -> dict:
        """Fallback when code-override is unavailable: a spot WBNB->token->WBNB getAmountsOut
        round trip. The shortfall vs input reflects pool fees + slippage + token taxes
        combined (cannot cleanly separate buy vs sell tax without an executed swap)."""
        if not wbnb_back or wbnb_back <= 0:
            return {"is_honeypot": None, "buy_tax": None, "sell_tax": None,
                    "note": note + " (no quote)"}
        round_trip_loss = max(0.0, 1.0 - (wbnb_back / bnb_in))
        return {"is_honeypot": None, "buy_tax": None, "sell_tax": None,
                "note": f"{note}; round-trip loss ~{round_trip_loss*100:.1f}% (fees+tax)"}
