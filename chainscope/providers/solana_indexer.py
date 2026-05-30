"""Solana chain indexer: reconstructs the trade tape, OHLCV, and per-trade pool
RESERVES directly from raw Solana JSON-RPC. No third-party index, no key.

This is the data the paid platforms charge to serve; here we dig it from the chain.
The core trick is a BALANCE-DIFF on the pool's two vault token accounts, which works
uniformly across DEX programs without per-program instruction decoding:
  - per-trade reserves  <- the POST token balance of the quote vault (the slippage input)
  - trades              <- the pre/post delta on each vault across a transaction
  - OHLCV               <- bucketed from the above

Pool vaults + mints are resolved once per pool from the on-chain account, branching
on the owning program id (Raydium AMM v4, PumpSwap, Orca Whirlpool). Layout byte
offsets are taken from each program's SDK/IDL (see the *_LAYOUT constants below);
they are not guessed.

Pricing is in USD when the pool is quoted in a known stablecoin (USDC/USDT); for
SOL-quoted pools price is in SOL (price_native) and USD fields are left None.

Deep history pages by transaction signature (getSignaturesForAddress + `before`),
unlike the EVM block-range backfiller: fetch_trades_page() exposes that unit for a
future Solana backfiller. (follow-up, see module note in get_trades.)
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone

from ..chains import Chain
from ..http import HttpError
from ..models import OHLCV, Trade
from .base import CAP_OHLCV, CAP_TRADES, NotSupported, Provider

log = logging.getLogger("chainscope.solana_indexer")

# ---- DEX program ids (account owners) ----
RAYDIUM_AMM_V4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
PUMPSWAP_AMM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
ORCA_WHIRLPOOL = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"

# ---- on-chain account layouts (byte offset of each 32-byte pubkey) ----
# Raydium LIQUIDITY_STATE_LAYOUT_V4 (raydium-io/raydium-sdk-v1 src/liquidity/layout.ts):
# 32 u64 (256B) + swapBaseInAmount u128 + swapQuoteOutAmount u128 + swapBase2QuoteFee u64
# + swapQuoteInAmount u128 + swapBaseOutAmount u128 + swapQuote2BaseFee u64 -> baseVault @336.
RAYDIUM_V4_LAYOUT = {
    "base_vault": 336, "quote_vault": 368, "base_mint": 400, "quote_mint": 432,
}
# PumpSwap Pool (pump-fun/pump-public-docs idl/pump_amm.json), after 8-byte anchor disc:
# pool_bump u8 (1) + index u16 (2) + creator pk (32) -> base_mint @43, quote_mint @75,
# lp_mint @107, pool_base_token_account @139, pool_quote_token_account @171.
PUMPSWAP_LAYOUT = {
    "base_mint": 43, "quote_mint": 75,
    "base_vault": 139, "quote_vault": 171,
}
# Orca Whirlpool (docs.rs whirlpools:state:Whirlpool), after 8-byte anchor disc:
# config pk(32) + bump(1) + tick_spacing u16(2) + tick_spacing_seed(2) + fee_rate u16(2)
# + protocol_fee_rate u16(2) + liquidity u128(16) + sqrt_price u128(16)
# + tick_current_index i32(4) + protocol_fee_owed_a/b u64(8+8) -> token_mint_a @101,
# token_vault_a @133, fee_growth_global_a u128(16) -> token_mint_b @181, token_vault_b @213.
WHIRLPOOL_LAYOUT = {
    "base_mint": 101, "base_vault": 133, "quote_mint": 181, "quote_vault": 213,
}

DEX_NAMES = {
    RAYDIUM_AMM_V4: "raydium-v4",
    PUMPSWAP_AMM: "pumpswap",
    ORCA_WHIRLPOOL: "orca-whirlpool",
}

# quote-side mints. USD stables price 1:1; wSOL is priced in SOL (USD left None).
USD_STABLES = {
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",   # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",   # USDT
}
WSOL = "So11111111111111111111111111111111111111112"
# preferred quote ordering when both sides are quote-able (shouldn't happen, but be safe)
QUOTE_MINTS = USD_STABLES | {WSOL}

_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600,
               "4h": 14400, "12h": 43200, "1d": 86400}

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


def b58encode(raw: bytes) -> str:
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


def b58decode(s: str) -> bytes:
    """Inverse of b58encode."""
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


class SolanaIndexerProvider(Provider):
    name = "solana_indexer"
    supported_chains = frozenset({Chain.SOLANA})
    capabilities = frozenset({CAP_OHLCV, CAP_TRADES})
    requires_key = False

    def __init__(self, http, settings):
        super().__init__(http, settings)
        self._meta_cache: dict[str, dict] = {}
        self._dec_cache: dict[str, int] = {}   # mint -> decimals

    # ---- low-level RPC ----

    async def _rpc(self, method: str, params: list):
        url = self.settings.rpc_url(Chain.SOLANA)
        resp = await self.http.post_json(
            url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        )
        if isinstance(resp, dict) and resp.get("error"):
            raise HttpError(0, url, str(resp["error"]))
        return resp.get("result") if isinstance(resp, dict) else None

    async def _account_info(self, pubkey: str, encoding: str = "base64"):
        return await self._rpc("getAccountInfo", [pubkey, {"encoding": encoding}])

    # ---- pool resolution (cached) ----

    @staticmethod
    def _pk_at(data: bytes, offset: int) -> str:
        return b58encode(data[offset:offset + 32])

    async def _mint_decimals(self, mint: str) -> int:
        if mint in self._dec_cache:
            return self._dec_cache[mint]
        try:
            res = await self._account_info(mint, encoding="jsonParsed")
            info = (((res or {}).get("value") or {}).get("data") or {}).get("parsed", {})
            dec = info.get("info", {}).get("decimals")
            if dec is not None:
                self._dec_cache[mint] = int(dec)
                return int(dec)
        except HttpError as exc:
            log.debug("decimals lookup failed for %s: %s", mint, exc)
        # SPL Mint layout: decimals is a single byte at offset 44. Fall back to that.
        try:
            res = await self._account_info(mint, encoding="base64")
            blob = ((res or {}).get("value") or {}).get("data")
            if blob and isinstance(blob, list):
                raw = base64.b64decode(blob[0])
                if len(raw) > 44:
                    dec = raw[44]
                    self._dec_cache[mint] = int(dec)
                    return int(dec)
        except (HttpError, ValueError) as exc:
            log.debug("decimals fallback failed for %s: %s", mint, exc)
        return 9  # SOL default; safest single guess for an unknown SPL token

    async def _pool_meta(self, pool: str) -> dict:
        if pool in self._meta_cache:
            return self._meta_cache[pool]
        res = await self._account_info(pool, encoding="base64")
        value = (res or {}).get("value")
        if not value:
            raise NotSupported(f"{pool}: account not found")
        owner = value.get("owner")
        blob = value.get("data")
        if not blob or not isinstance(blob, list):
            raise NotSupported(f"{pool}: no base64 account data")
        data = base64.b64decode(blob[0])

        if owner == RAYDIUM_AMM_V4:
            layout = RAYDIUM_V4_LAYOUT
        elif owner == PUMPSWAP_AMM:
            layout = PUMPSWAP_LAYOUT
        elif owner == ORCA_WHIRLPOOL:
            layout = WHIRLPOOL_LAYOUT
        else:
            raise NotSupported(f"{pool}: unknown owner program {owner}")

        base_mint = self._pk_at(data, layout["base_mint"])
        quote_mint = self._pk_at(data, layout["quote_mint"])
        base_vault = self._pk_at(data, layout["base_vault"])
        quote_vault = self._pk_at(data, layout["quote_vault"])

        # Decide which side is the quote (stable/SOL). If the layout's "base" side is
        # actually the quote-able mint, swap so amount_quote is always the quote leg.
        if base_mint in QUOTE_MINTS and quote_mint not in QUOTE_MINTS:
            base_mint, quote_mint = quote_mint, base_mint
            base_vault, quote_vault = quote_vault, base_vault

        if quote_mint in USD_STABLES:
            quote_kind = "usd"
        elif quote_mint == WSOL:
            quote_kind = "sol"
        else:
            quote_kind = None   # exotic quote -> price in quote units, no USD/native

        base_dec = await self._mint_decimals(base_mint)
        quote_dec = await self._mint_decimals(quote_mint)

        meta = {
            "owner": owner, "dex": DEX_NAMES.get(owner, owner),
            "base_mint": base_mint, "quote_mint": quote_mint,
            "base_vault": base_vault, "quote_vault": quote_vault,
            "base_dec": base_dec, "quote_dec": quote_dec,
            "quote_kind": quote_kind,
        }
        self._meta_cache[pool] = meta
        return meta

    # ---- balance-diff helpers ----

    @staticmethod
    def _vault_balance(token_balances: list, vault: str, account_keys: list) -> float | None:
        """Resolve a vault's ui balance from a pre/postTokenBalances list.

        Each entry carries an `accountIndex` into the tx's account keys; match the
        vault by resolving that index to a pubkey, then read uiTokenAmount.
        """
        for tb in token_balances or []:
            idx = tb.get("accountIndex")
            owner_pk = None
            if idx is not None and 0 <= idx < len(account_keys):
                owner_pk = account_keys[idx]
            if owner_pk != vault:
                continue
            uta = tb.get("uiTokenAmount") or {}
            ui = uta.get("uiAmount")
            if ui is not None:
                return float(ui)
            amt = uta.get("amount")
            dec = uta.get("decimals")
            if amt is not None and dec is not None:
                try:
                    return int(amt) / (10 ** int(dec))
                except (ValueError, TypeError):
                    return None
        return None

    @staticmethod
    def _account_keys(tx_meta_msg: dict, meta: dict) -> list[str]:
        """Flatten static + loaded (address-lookup-table) account keys to pubkey strings.

        jsonParsed keys are {"pubkey": ...} dicts; loaded addresses are bare strings."""
        keys: list[str] = []
        for k in (tx_meta_msg.get("accountKeys") or []):
            keys.append(k["pubkey"] if isinstance(k, dict) else k)
        loaded = meta.get("loadedAddresses") or {}
        keys.extend(loaded.get("writable") or [])
        keys.extend(loaded.get("readonly") or [])
        return keys

    def _trade_from_tx(self, pool: str, meta: dict, sig: str,
                       block_time: int | None, tx: dict) -> Trade | None:
        tx_meta = (tx or {}).get("meta") or {}
        if tx_meta.get("err") is not None:
            return None
        message = (tx.get("transaction") or {}).get("message") or {}
        account_keys = self._account_keys(message, tx_meta)
        if not account_keys:
            return None

        bv, qv = meta["base_vault"], meta["quote_vault"]
        pre = tx_meta.get("preTokenBalances")
        post = tx_meta.get("postTokenBalances")
        pre_base = self._vault_balance(pre, bv, account_keys)
        post_base = self._vault_balance(post, bv, account_keys)
        pre_quote = self._vault_balance(pre, qv, account_keys)
        post_quote = self._vault_balance(post, qv, account_keys)
        if None in (pre_base, post_base, pre_quote, post_quote):
            return None  # this tx didn't touch both vaults -> not a swap of this pool

        delta_base = post_base - pre_base
        delta_quote = post_quote - pre_quote
        amount_base = abs(delta_base)
        amount_quote = abs(delta_quote)
        # ignore dust / non-swaps (e.g. fee-only or add/remove-liquidity matched both ways)
        if amount_base <= 0 or amount_quote <= 0:
            return None

        # side from base perspective: base flows INTO pool => trader SOLD base.
        side = "sell" if delta_base > 0 else "buy"
        price = amount_quote / amount_base  # in quote units per base

        qk = meta["quote_kind"]
        if qk == "usd":
            price_usd, price_native = price, None
            amount_usd = amount_quote
            reserve_usd = 2.0 * post_quote
        elif qk == "sol":
            price_usd, price_native = None, price
            amount_usd, reserve_usd = None, None
        else:
            price_usd = price_native = amount_usd = reserve_usd = None

        bt = datetime.fromtimestamp(block_time, tz=timezone.utc) if block_time else _now()
        maker = account_keys[0] if account_keys else None
        gas_native = (tx_meta.get("fee") or 0) / 1e9  # lamports -> SOL (exact, incl. priority fee)
        return Trade(
            source="solana_indexer", chain=Chain.SOLANA, pair_address=pool,
            block_time=bt, tx_hash=sig, side=side, dex=meta["dex"],
            price_usd=price_usd, price_native=price_native,
            amount_base=amount_base, amount_quote=amount_quote, amount_usd=amount_usd,
            reserve_usd=reserve_usd, maker=maker,
            block_number=tx.get("slot"), gas_native=gas_native or None,
            raw={"signature": sig, "blockTime": block_time},
        )

    # ---- signature paging ----

    async def _signatures(self, pool: str, limit: int,
                          before: str | None = None, until: str | None = None) -> list[dict]:
        opts: dict = {"limit": max(1, min(limit, 1000))}
        if before:
            opts["before"] = before
        if until:
            opts["until"] = until
        try:
            res = await self._rpc("getSignaturesForAddress", [pool, opts])
        except HttpError as exc:
            log.debug("getSignaturesForAddress %s failed: %s", pool, exc)
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

    # ---- capabilities ----

    async def get_trades(self, chain: Chain, pair_address: str, since: int | None = None,
                         until: int | None = None, limit: int = 1000) -> list[Trade]:
        """Trade tape for a Solana pool, newest-first.

        Pages getSignaturesForAddress (1000/call, `before`=oldest sig collected) until
        `limit` candidate signatures are gathered or `since` is passed, then balance-diffs
        each transaction. Caps getTransaction calls to `limit`. HttpError is swallowed
        (returns whatever was decoded so far).

        NOTE: the existing block-range Backfiller is EVM-shaped; Solana deep backfill pages
        by signature via fetch_trades_page(), wiring that into backfill.py is a follow-up.
        """
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported
        meta = await self._pool_meta(pair_address)

        # collect candidate signatures (newest-first), honoring time bounds.
        sigs: list[dict] = []
        before: str | None = None
        while len(sigs) < limit:
            page = await self._signatures(pair_address, min(1000, limit - len(sigs)), before=before)
            if not page:
                break
            stop = False
            for s in page:
                if s.get("err") is not None:
                    continue
                bt = s.get("blockTime")
                if until is not None and bt is not None and bt > until:
                    continue  # too new; skip until we reach the window
                if since is not None and bt is not None and bt < since:
                    stop = True
                    break     # older than window; signatures are newest-first -> done
                sigs.append(s)
                if len(sigs) >= limit:
                    break
            if stop or len(page) < min(1000, limit):
                break
            before = page[-1].get("signature")  # paginate older

        out: list[Trade] = []
        for s in sigs:
            if len(out) >= limit:
                break
            sig = s.get("signature")
            if not sig:
                continue
            tx = await self._get_transaction(sig)
            if not tx:
                continue
            tr = self._trade_from_tx(pair_address, meta, sig, s.get("blockTime"), tx)
            if tr is not None:
                out.append(tr)
        out.sort(key=lambda t: t.block_time, reverse=True)
        return out[:limit]

    async def fetch_trades_page(self, pool: str, before_sig: str | None = None,
                                limit: int = 1000) -> tuple[list[Trade], str | None]:
        """Decode one signature page of trades for `pool`, oldest-first paging unit.

        Returns (trades_newest_first, oldest_signature_in_page). Pass the returned
        oldest signature back as `before_sig` to walk deeper history, the unit a
        future Solana backfiller drives (signature-based, not block-range)."""
        meta = await self._pool_meta(pool)
        page = await self._signatures(pool, limit, before=before_sig)
        if not page:
            return [], None
        oldest_sig = page[-1].get("signature")
        out: list[Trade] = []
        for s in page:
            if s.get("err") is not None:
                continue
            sig = s.get("signature")
            if not sig:
                continue
            tx = await self._get_transaction(sig)
            if not tx:
                continue
            tr = self._trade_from_tx(pool, meta, sig, s.get("blockTime"), tx)
            if tr is not None:
                out.append(tr)
        out.sort(key=lambda t: t.block_time, reverse=True)
        return out, oldest_sig

    async def get_ohlcv(self, chain: Chain, pair_address: str, timeframe: str = "1h",
                        limit: int = 1000, before: int | None = None) -> list[OHLCV]:
        """Bucket the trade tape into OHLCV bars (oldest-first).

        volume = sum of trade amount_usd in the bucket; reserve_usd = the last trade's
        reserve in the bucket (post-trade pool depth); trade_count = trades in the bucket.
        """
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported
        tf_sec = _TF_SECONDS.get(timeframe, 3600)
        # pull enough trades to cover the window; cap candidate sigs so we stay polite.
        since = (before - tf_sec * limit) if before else None
        trades = await self.get_trades(chain, pair_address, since=since, until=before,
                                       limit=max(limit * 10, 200))
        if not trades:
            return []

        bars: dict[int, dict] = {}
        # iterate oldest-first so open/close are chronologically correct
        for tr in sorted(trades, key=lambda t: t.block_time):
            price = tr.price_usd if tr.price_usd is not None else tr.price_native
            if price is None:
                continue
            ts = int(tr.block_time.timestamp())
            bucket = ts - (ts % tf_sec)
            b = bars.get(bucket)
            if b is None:
                b = bars[bucket] = {"o": price, "h": price, "l": price, "c": price,
                                    "v": 0.0, "res": tr.reserve_usd, "n": 0}
            b["h"] = max(b["h"], price)
            b["l"] = min(b["l"], price)
            b["c"] = price
            if tr.reserve_usd is not None:
                b["res"] = tr.reserve_usd
            if tr.amount_usd is not None:
                b["v"] += tr.amount_usd
            b["n"] += 1

        out: list[OHLCV] = []
        for bucket in sorted(bars):
            b = bars[bucket]
            out.append(OHLCV(
                source="solana_indexer", chain=Chain.SOLANA, pair_address=pair_address,
                timeframe=timeframe, timestamp=datetime.fromtimestamp(bucket, tz=timezone.utc),
                open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                volume=b["v"] or None, reserve_usd=b["res"], trade_count=b["n"] or None,
            ))
        return out[-limit:]


def _now() -> datetime:
    return datetime.now(timezone.utc)
