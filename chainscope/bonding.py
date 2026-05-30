"""Pre-graduation bonding-curve trade decoder: the trades a DEX indexer can't see.

Meme launches trade on a *bonding-curve* contract/program BEFORE they "graduate"
to a real DEX pool (PancakeSwap on BSC, PumpSwap/Raydium on Solana). Until that
migration, every buy/sell happens against the launchpad's curve, so the PancakeSwap
/ Raydium log-scanning indexers (bsc_indexer / solana_indexer) see nothing. This
module reads those pre-graduation trades straight off the chain, no third-party
index, no key, stdlib only.

Two decoders (NOT Providers, the parent integrates these):

  FourMemeBonding : BSC four.meme curve, via eth_getLogs of the TokenManager2
                     TokenPurchase / TokenSale events. Priced in BNB; USD anchored
                     off the WBNB/USDT pool reserves (same approach as bsc_indexer).

  PumpFunBonding  : Solana pump.fun curve, via getSignaturesForAddress +
                     getTransaction, decoding the anchor `TradeEvent` emitted as a
                     `Program data:` log line (event-CPI: 8-byte self-CPI sentinel +
                     8-byte event discriminator + borsh body). Priced in SOL; USD
                     anchored off a Raydium SOL/USDC pool when cheaply available.

GRADUATION CUTOFF: these trades STOP once the token migrates (curve hits 100% ->
liquidity seeded on PancakeSwap / PumpSwap); from that point the DEX indexers take
over. Each decoder records bonding-curve progress / reserves in Trade.raw so the
caller can see how close to graduation a snapshot is.

Addresses, event signatures and the TradeEvent layout below were verified against
live on-chain logs + the official pump.fun IDL (2026-05), not guessed.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from datetime import datetime, timezone

from .chains import Chain
from .http import HttpError
from .models import Trade

log = logging.getLogger("chainscope.bonding")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# four.meme (BSC), bonding-curve buy/sell decoder
# ===========================================================================

# four.meme TokenManager2 (the bonding-curve manager; buys/sells fire here).
FOURMEME_MANAGER = "0x5c952063c7fc8610ffdb798152d69f0b9550762b"

# Event topic0 hashes. BOTH events share the same 8-word, all-non-indexed layout:
#   (address token, address account, uint256 price, uint256 amount,
#    uint256 cost, uint256 fee, uint256 offers, uint256 funds)
# - token   : the launched token
# - account : the trader (maker)
# - price   : contract-internal price scalar (we use cost/amount instead)
# - amount  : token amount (18-dec)
# - cost    : BNB moved (18-dec), quote leg
# - fee     : BNB fee (18-dec)
# - offers  : curve token offering remaining (18-dec), graduation progress input
# - funds   : BNB raised so far on the curve (18-dec), graduation progress input
# Verified live (2026-05): both topic0s observed on the manager, both decode to the
# 8-word layout with cost/amount == price (BNB per token).
TOKEN_PURCHASE_TOPIC = "0x7db52723a3b2cdd6164364b3b766e65e540d7be48ffa89582956d8eaebe62942"
TOKEN_SALE_TOPIC = "0x0a5575b3648bae2210cee56bf33254cc1ddfbc7bf637c0af2ac18b14fb1bae19"

# WBNB/USDT anchoring (mirrors bsc_indexer._bnb_usd).
WBNB_USDT_PAIR = "0x16b9a82891338f9ba80e2d6970fdda79d1eb0dae"  # PancakeSwap V2 USDT/WBNB
SEL_GET_RESERVES = "0x0902f1ac"

# four.meme tokens use 18 decimals (standard BEP-20); the curve quotes in BNB (18-dec).
FOURMEME_TOKEN_DEC = 18
BNB_DEC = 18

CHUNK = 5000          # blocks per getLogs call (safe on free archival RPCs)
MAX_CHUNKS = 60       # cap per request; beyond this, run a dedicated backfill / local node


def _u(hexstr: str, start: int, end: int) -> int:
    return int(hexstr[start:end], 16)


class FourMemeBonding:
    """Decode four.meme pre-graduation curve trades from TokenManager2 logs.

    Construct with the shared ``(http, settings)``. Trades are denominated in BNB
    on the curve; USD price/volume are anchored via the live WBNB/USDT pool. Once a
    token graduates to PancakeSwap, these events stop and bsc_indexer takes over.
    """

    def __init__(self, http, settings):
        self.http = http
        self.settings = settings
        # rotate across free archival endpoints that serve wide getLogs ranges
        # (mirrors bsc_indexer); pin your own/local node via BSC_ARCHIVE_RPC_URL.
        env = os.environ.get("BSC_ARCHIVE_RPC_URL")
        self.rpcs = [u.strip() for u in env.split(",")] if env else [
            settings.rpc_url(Chain.BSC),
            "https://bsc.publicnode.com",
            "https://binance.llamarpc.com",
            "https://bsc-rpc.publicnode.com",
        ]
        # de-dup while preserving order
        seen: set[str] = set()
        self.rpcs = [u for u in self.rpcs if not (u in seen or seen.add(u))]
        self._rpc_i = 0
        self._bnb: float | None = None
        self._clock: tuple[int, int, float] | None = None  # (block, ts, sec/block)

    # ---- low-level RPC (rotates endpoints on failure) ----

    async def _rpc(self, method: str, params: list):
        last: Exception | None = None
        for _ in range(len(self.rpcs)):
            url = self.rpcs[self._rpc_i]
            try:
                resp = await self.http.post_json(
                    url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                )
                if isinstance(resp, dict) and resp.get("error"):
                    raise HttpError(0, url, str(resp["error"]))
                return resp.get("result") if isinstance(resp, dict) else None
            except HttpError as exc:
                last = exc
                self._rpc_i = (self._rpc_i + 1) % len(self.rpcs)
        if last is not None:
            raise last
        return None

    async def _eth_call(self, to: str, data: str) -> str | None:
        try:
            return await self._rpc("eth_call", [{"to": to, "data": data}, "latest"])
        except HttpError:
            return None

    async def _latest_block(self) -> int:
        return int(await self._rpc("eth_blockNumber", []), 16)

    async def _bnb_usd(self) -> float | None:
        """BNB/USD from the WBNB/USDT V2 pool reserves (on-chain, cached)."""
        if self._bnb is not None:
            return self._bnb
        res = await self._eth_call(WBNB_USDT_PAIR, SEL_GET_RESERVES)
        if res and len(res) >= 130:
            r0 = int(res[2:66], 16) / 1e18   # USDT reserve
            r1 = int(res[66:130], 16) / 1e18  # WBNB reserve
            self._bnb = (r0 / r1) if r1 else None
        return self._bnb

    async def _block_ts(self, block: int) -> int:
        b = await self._rpc("eth_getBlockByNumber", [hex(block), False])
        return int(b["timestamp"], 16)

    async def _clock_calibrate(self):
        """Derive seconds/block by sampling two recent blocks (BSC block time varies)."""
        if self._clock is not None:
            return self._clock
        latest = await self._latest_block()
        back = max(1, latest - 20000)
        t_latest = await self._block_ts(latest)
        t_back = await self._block_ts(back)
        spb = (t_latest - t_back) / max(1, latest - back)
        self._clock = (latest, t_latest, spb if spb > 0 else 0.75)
        return self._clock

    def _ts_of_block(self, block: int) -> datetime:
        latest, t_latest, spb = self._clock  # type: ignore
        secs = t_latest - (latest - block) * spb
        return datetime.fromtimestamp(secs, tz=timezone.utc)

    async def _get_logs(self, topic: str, token: str, from_b: int, to_b: int) -> list[dict]:
        """Chunked getLogs on the manager, filtered to one event topic AND `token`.

        The token is the FIRST (non-indexed) data word, so it is NOT addressable via
        the topics filter; we fetch the manager's events for the topic across the
        range and post-filter by the decoded token word. Range is chunked to stay
        under free-RPC getLogs span caps."""
        token_word = token.lower().replace("0x", "").rjust(64, "0")
        out: list[dict] = []
        b = from_b
        chunks = 0
        while b <= to_b and chunks < MAX_CHUNKS:
            hi = min(b + CHUNK - 1, to_b)
            try:
                res = await self._rpc("eth_getLogs", [{
                    "address": FOURMEME_MANAGER, "topics": [topic],
                    "fromBlock": hex(b), "toBlock": hex(hi),
                }])
                for lg in res or []:
                    data = lg.get("data", "0x")[2:]
                    if data[0:64] == token_word:   # first word == token
                        out.append(lg)
            except HttpError as exc:
                log.debug("fourmeme getLogs %s-%s failed: %s", b, hi, exc)
            b = hi + 1
            chunks += 1
        if b <= to_b:
            log.warning("fourmeme: truncated at %d blocks (MAX_CHUNKS=%d); use a local "
                        "archive node for deeper history", b - from_b, MAX_CHUNKS)
        return out

    @staticmethod
    def _decode_trade_log(data: str) -> dict | None:
        """Decode the 8-word TokenPurchase/TokenSale data blob."""
        d = data[2:] if data.startswith("0x") else data
        if len(d) < 64 * 8:
            return None
        return {
            "token": "0x" + d[24:64],
            "account": "0x" + d[64 + 24:128],
            "price": _u(d, 128, 192),
            "amount": _u(d, 192, 256) / 10 ** FOURMEME_TOKEN_DEC,
            "cost": _u(d, 256, 320) / 10 ** BNB_DEC,
            "fee": _u(d, 320, 384) / 10 ** BNB_DEC,
            "offers": _u(d, 384, 448) / 10 ** FOURMEME_TOKEN_DEC,
            "funds": _u(d, 448, 512) / 10 ** BNB_DEC,
        }

    async def get_trades(self, token: str, from_block: int | None = None,
                         to_block: int | None = None, limit: int = 1000) -> list[Trade]:
        """Pre-graduation four.meme curve trades for `token`, newest-first.

        Scans TokenPurchase + TokenSale logs on TokenManager2 over [from_block,
        to_block] (defaults: a recent window ending at head), filters to `token`,
        decodes to Trade. amount_base = token amount, amount_quote = BNB cost,
        price/USD anchored via the live WBNB/USDT pool. Bonding-curve reserves
        (offers/funds) land in raw. HttpError is swallowed (returns what decoded)."""
        try:
            await self._clock_calibrate()
        except HttpError as exc:
            log.debug("fourmeme clock calibrate failed: %s", exc)
            return []
        latest, _t_latest, _ = self._clock  # type: ignore
        to_b = to_block if to_block is not None else latest
        from_b = from_block if from_block is not None else max(1, to_b - CHUNK * MAX_CHUNKS)

        bnb_usd = await self._bnb_usd()

        try:
            buys = await self._get_logs(TOKEN_PURCHASE_TOPIC, token, from_b, to_b)
            sells = await self._get_logs(TOKEN_SALE_TOPIC, token, from_b, to_b)
        except HttpError as exc:
            log.debug("fourmeme getLogs failed: %s", exc)
            return []

        out: list[Trade] = []
        for side, logs in (("buy", buys), ("sell", sells)):
            for lg in logs:
                dec = self._decode_trade_log(lg.get("data", "0x"))
                if dec is None or dec["amount"] <= 0:
                    continue
                blk = int(lg["blockNumber"], 16)
                amt_base = dec["amount"]
                amt_quote = dec["cost"]            # BNB moved
                price_native = (amt_quote / amt_base) if amt_base else None  # BNB per token
                price_usd = (price_native * bnb_usd) if (price_native is not None and bnb_usd) else None
                amt_usd = (amt_quote * bnb_usd) if bnb_usd else None
                # curve progress: BNB raised so far vs. the curve's offering depth.
                reserve_usd = (dec["funds"] * bnb_usd) if bnb_usd else None
                out.append(Trade(
                    source="fourmeme_bonding", chain=Chain.BSC,
                    pair_address=token,                # the curve is keyed by token
                    block_time=self._ts_of_block(blk), side=side, dex="fourmeme",
                    price_usd=price_usd,
                    price_native=price_native if bnb_usd is None else None,
                    amount_base=amt_base, amount_quote=amt_quote, amount_usd=amt_usd,
                    reserve_usd=reserve_usd, maker=dec["account"],
                    tx_hash=lg.get("transactionHash"), block_number=blk,
                    log_index=int(lg["logIndex"], 16) if lg.get("logIndex") else None,
                    fee_bps=None,
                    raw={
                        "bonding_curve": True,
                        "curve_funds_bnb": dec["funds"],     # BNB raised on the curve so far
                        "curve_offers_tokens": dec["offers"],  # tokens still offered by the curve
                        "fee_bnb": dec["fee"],
                        "price_scalar": dec["price"],
                        "bnb_usd": bnb_usd,
                        "log": lg,
                    },
                ))
        out.sort(key=lambda t: (t.block_number or 0, t.log_index or 0), reverse=True)
        return out[:limit]


# ===========================================================================
# pump.fun (Solana), bonding-curve TradeEvent decoder
# ===========================================================================

PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# Anchor event-CPI framing of a `Program data:` log line:
#   bytes[0:8]  = self-CPI event sentinel  = sha256("anchor:event")[:8]
#   bytes[8:16] = event discriminator       = sha256("event:<Name>")[:8]
#   bytes[16:]  = borsh-encoded event body
# Older programs emit just (event_disc | body) without the sentinel; we accept both.
_ANCHOR_EVENT_SENTINEL = hashlib.sha256(b"anchor:event").digest()[:8]   # 1d9acb512ea545e4
_TRADE_EVENT_DISC = hashlib.sha256(b"event:TradeEvent").digest()[:8]    # bddb7fd34ee661ee

# pump.fun token decimals (always 6) and SOL decimals (9).
PUMP_TOKEN_DEC = 6
SOL_DEC = 9

# On-chain SOL/USD anchor: Raydium AMM v4 SOL/USDC pool vaults. We read the two
# vault SPL-token balances directly (getTokenAccountBalance) and divide. This is
# the canonical, high-liquidity SOL/USDC v4 pool.
RAYDIUM_SOL_USDC_QUOTE_VAULT = "HLmqeL62xR1QoZ1HKKbXRrdN1p3phKpxRMb2VVopvBBz"  # USDC vault
RAYDIUM_SOL_USDC_BASE_VAULT = "DQyrAcCrDXQ7NeoqGgDCZwBvWDcYmFCjSb9JtteuvPpz"   # wSOL vault

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


def _b58encode(raw: bytes) -> str:
    """Pure-Python base58 (Bitcoin/Solana alphabet). Preserves leading-zero bytes as '1's."""
    n = int.from_bytes(raw, "big")
    out = ""
    while n > 0:
        n, rem = divmod(n, 58)
        out = _B58_ALPHABET[rem] + out
    pad = 0
    for b in raw:
        if b == 0:
            pad += 1
        else:
            break
    return "1" * pad + out


def _b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        n = n * 58 + _B58_INDEX[ch]
    full = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = 0
    for ch in s:
        if ch == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + full


class _BorshReader:
    """Minimal little-endian borsh reader: only the primitives TradeEvent needs."""

    def __init__(self, data: bytes):
        self.d = data
        self.i = 0

    def _take(self, n: int) -> bytes:
        if self.i + n > len(self.d):
            raise ValueError("borsh: out of bounds")
        b = self.d[self.i:self.i + n]
        self.i += n
        return b

    def u64(self) -> int:
        return int.from_bytes(self._take(8), "little")

    def i64(self) -> int:
        return int.from_bytes(self._take(8), "little", signed=True)

    def bool(self) -> bool:
        return self._take(1)[0] != 0

    def pubkey(self) -> str:
        return _b58encode(self._take(32))


class PumpFunBonding:
    """Decode pump.fun pre-graduation curve trades (anchor TradeEvent log lines).

    Construct with the shared ``(http, settings)``. Pages getSignaturesForAddress on
    the token mint, pulls each tx (getTransaction, jsonParsed, v0), and scans
    meta.logMessages for the `Program data:` line carrying the TradeEvent. Priced in
    SOL; USD anchored off a Raydium SOL/USDC pool when cheaply readable. Once a token
    graduates to PumpSwap, these events stop and solana_indexer takes over.

    Public Solana RPC throttles getTransaction hard, keep `limit` small (<=20).
    """

    def __init__(self, http, settings):
        self.http = http
        self.settings = settings
        self._sol_usd: float | None = None
        self._sol_usd_tried = False

    # ---- low-level RPC ----

    async def _rpc(self, method: str, params: list):
        url = self.settings.rpc_url(Chain.SOLANA)
        resp = await self.http.post_json(
            url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        )
        if isinstance(resp, dict) and resp.get("error"):
            raise HttpError(0, url, str(resp["error"]))
        return resp.get("result") if isinstance(resp, dict) else None

    async def _sol_usd_price(self) -> float | None:
        """SOL/USD from a Raydium SOL/USDC pool: USDC vault balance / wSOL vault balance.

        Cheap (two getTokenAccountBalance calls), cached. Returns None on any failure
        so the caller can fall back to SOL-denominated pricing."""
        if self._sol_usd_tried:
            return self._sol_usd
        self._sol_usd_tried = True
        try:
            usdc = await self._rpc("getTokenAccountBalance", [RAYDIUM_SOL_USDC_QUOTE_VAULT])
            sol = await self._rpc("getTokenAccountBalance", [RAYDIUM_SOL_USDC_BASE_VAULT])
            usdc_ui = ((usdc or {}).get("value") or {}).get("uiAmount")
            sol_ui = ((sol or {}).get("value") or {}).get("uiAmount")
            if usdc_ui and sol_ui:
                self._sol_usd = float(usdc_ui) / float(sol_ui)
        except (HttpError, TypeError, ValueError, ZeroDivisionError) as exc:
            log.debug("pumpfun SOL/USD anchor failed: %s", exc)
        return self._sol_usd

    def _bonding_curve_pda(self, mint: str) -> str | None:
        """Derive the bonding-curve PDA: findProgramAddress([b"bonding-curve", mint], program).

        Pure-Python: brute-force the bump (255..0) over sha256(seeds | bump | program |
        "ProgramDerivedAddress"), skipping points on the ed25519 curve."""
        try:
            mint_bytes = _b58decode(mint)
            prog_bytes = _b58decode(PUMPFUN_PROGRAM)
        except (KeyError, ValueError):
            return None
        if len(mint_bytes) != 32 or len(prog_bytes) != 32:
            return None
        seed = b"bonding-curve"
        for bump in range(255, -1, -1):
            h = hashlib.sha256(
                seed + mint_bytes + bytes([bump]) + prog_bytes + b"ProgramDerivedAddress"
            ).digest()
            if not _is_on_curve(h):
                return _b58encode(h)
        return None

    async def _signatures(self, address: str, limit: int, before: str | None) -> list[dict]:
        opts: dict = {"limit": max(1, min(limit, 1000))}
        if before:
            opts["before"] = before
        try:
            res = await self._rpc("getSignaturesForAddress", [address, opts])
        except HttpError as exc:
            log.debug("getSignaturesForAddress %s failed: %s", address, exc)
            return []
        return res or []

    async def _get_transaction(self, sig: str) -> dict | None:
        try:
            return await self._rpc("getTransaction", [sig, {
                "maxSupportedTransactionVersion": 0, "encoding": "jsonParsed",
            }])
        except HttpError as exc:
            log.debug("getTransaction %s failed: %s", sig, exc)
            return None

    # ---- TradeEvent decode ----

    @staticmethod
    def _decode_trade_event(blob: bytes) -> dict | None:
        """Match the TradeEvent discriminator and borsh-decode the prefix we need.

        Frame may be [sentinel(8) | disc(8) | body] (anchor event-CPI) or [disc(8) |
        body]. Only the leading borsh fields are read, the official IDL appends many
        trailing fields (fees, a variable-length ix_name string, a Shareholder vec),
        so the body is NOT fixed-length; we decode through real_token_reserves and stop.
        """
        body = None
        if len(blob) >= 16 and blob[0:8] == _ANCHOR_EVENT_SENTINEL and blob[8:16] == _TRADE_EVENT_DISC:
            body = blob[16:]
        elif len(blob) >= 8 and blob[0:8] == _TRADE_EVENT_DISC:
            body = blob[8:]
        if body is None:
            return None
        r = _BorshReader(body)
        try:
            mint = r.pubkey()
            sol_amount = r.u64()
            token_amount = r.u64()
            is_buy = r.bool()
            user = r.pubkey()
            timestamp = r.i64()
            v_sol = r.u64()
            v_tok = r.u64()
            r_sol = r.u64()
            r_tok = r.u64()
        except (ValueError, IndexError):
            return None
        return {
            "mint": mint, "sol_amount": sol_amount, "token_amount": token_amount,
            "is_buy": is_buy, "user": user, "timestamp": timestamp,
            "virtual_sol_reserves": v_sol, "virtual_token_reserves": v_tok,
            "real_sol_reserves": r_sol, "real_token_reserves": r_tok,
        }

    @staticmethod
    def _extract_program_data(log_messages: list[str]) -> list[bytes]:
        """Pull base64 payloads from every `Program data:` log line."""
        out: list[bytes] = []
        for line in log_messages or []:
            if "Program data:" in line:
                b64 = line.split("Program data:", 1)[1].strip()
                try:
                    out.append(base64.b64decode(b64))
                except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
                    continue
        return out

    def _trade_from_event(self, mint: str, sig: str, slot: int | None,
                          block_time: int | None, ev: dict, sol_usd: float | None) -> Trade:
        amount_base = ev["token_amount"] / 10 ** PUMP_TOKEN_DEC
        amount_quote = ev["sol_amount"] / 10 ** SOL_DEC          # SOL
        price_native = (amount_quote / amount_base) if amount_base else None  # SOL per token
        price_usd = (price_native * sol_usd) if (price_native is not None and sol_usd) else None
        amount_usd = (amount_quote * sol_usd) if sol_usd else None
        v_sol = ev["virtual_sol_reserves"] / 10 ** SOL_DEC
        v_tok = ev["virtual_token_reserves"] / 10 ** PUMP_TOKEN_DEC
        reserve_usd = (2.0 * v_sol * sol_usd) if sol_usd else None
        bt = datetime.fromtimestamp(block_time, tz=timezone.utc) if block_time else _now()
        return Trade(
            source="pumpfun_bonding", chain=Chain.SOLANA, pair_address=mint,
            block_time=bt, tx_hash=sig, side="buy" if ev["is_buy"] else "sell",
            dex="pumpfun-bc",
            price_usd=price_usd,
            price_native=price_native if sol_usd is None else None,
            amount_base=amount_base, amount_quote=amount_quote, amount_usd=amount_usd,
            reserve_usd=reserve_usd, maker=ev["user"], block_number=slot,
            raw={
                "bonding_curve": True,
                "virtual_sol_reserves": v_sol,
                "virtual_token_reserves": v_tok,
                "real_sol_reserves": ev["real_sol_reserves"] / 10 ** SOL_DEC,
                "real_token_reserves": ev["real_token_reserves"] / 10 ** PUMP_TOKEN_DEC,
                "sol_usd": sol_usd,
                "event_timestamp": ev["timestamp"],
            },
        )

    async def get_trades(self, mint: str, limit: int = 200,
                         before: str | None = None) -> list[Trade]:
        """Pre-graduation pump.fun curve trades for `mint`, newest-first.

        Pages signatures on the mint (falls back from the bonding-curve PDA), then
        getTransaction + scans logMessages for the TradeEvent. Caps getTransaction
        calls to `limit` (public RPC throttles hard, keep it small). HttpError is
        swallowed (returns what decoded). USD fields are populated only if the
        Raydium SOL/USDC anchor was cheaply readable, else price_native is in SOL."""
        # Prefer the mint (every curve trade touches it); the PDA is derived as a
        # secondary address in case the caller wants curve-only signatures.
        addresses = [mint]
        pda = self._bonding_curve_pda(mint)
        if pda and pda != mint:
            addresses.append(pda)

        sol_usd = await self._sol_usd_price()

        sigs: list[dict] = []
        for addr in addresses:
            page = await self._signatures(addr, limit, before)
            for s in page:
                if s.get("err") is not None:
                    continue
                if s.get("signature"):
                    sigs.append(s)
            if sigs:
                break  # the mint already yielded signatures; no need for the PDA

        # de-dup + cap candidate signatures
        seen: set[str] = set()
        ordered: list[dict] = []
        for s in sigs:
            sg = s["signature"]
            if sg not in seen:
                seen.add(sg)
                ordered.append(s)
        ordered = ordered[:limit]

        out: list[Trade] = []
        for s in ordered:
            if len(out) >= limit:
                break
            sig = s["signature"]
            tx = await self._get_transaction(sig)
            if not tx:
                continue
            meta = (tx or {}).get("meta") or {}
            if meta.get("err") is not None:
                continue
            blobs = self._extract_program_data(meta.get("logMessages") or [])
            for blob in blobs:
                ev = self._decode_trade_event(blob)
                if ev is None:
                    continue
                if ev["mint"] != mint:
                    continue   # a TradeEvent for a different mint in the same tx
                out.append(self._trade_from_event(
                    mint, sig, tx.get("slot"), s.get("blockTime"), ev, sol_usd
                ))
                break   # one TradeEvent per tx for this mint
        out.sort(key=lambda t: t.block_time, reverse=True)
        return out[:limit]


# ---------------------------------------------------------------------------
# ed25519 on-curve test (for the bonding-curve PDA derivation), pure Python.
# A point hash is a valid PDA only if it is NOT a point on the ed25519 curve.
# ---------------------------------------------------------------------------
_ED_P = 2 ** 255 - 19
_ED_D = (-121665 * pow(121666, _ED_P - 2, _ED_P)) % _ED_P


def _is_on_curve(b: bytes) -> bool:
    """True if the 32-byte value decodes to a valid ed25519 point (compressed Y)."""
    if len(b) != 32:
        return False
    y = int.from_bytes(b, "little")
    sign = (y >> 255) & 1
    y &= (1 << 255) - 1
    if y >= _ED_P:
        return False
    # recover x^2 = (y^2 - 1) / (d*y^2 + 1)
    y2 = (y * y) % _ED_P
    num = (y2 - 1) % _ED_P
    den = (_ED_D * y2 + 1) % _ED_P
    try:
        x2 = (num * pow(den, _ED_P - 2, _ED_P)) % _ED_P
    except ValueError:
        return False
    if x2 == 0:
        return sign == 0
    x = pow(x2, (_ED_P + 3) // 8, _ED_P)
    if (x * x - x2) % _ED_P != 0:
        x = (x * pow(2, (_ED_P - 1) // 4, _ED_P)) % _ED_P
    if (x * x - x2) % _ED_P != 0:
        return False   # no square root -> not on curve
    return True
