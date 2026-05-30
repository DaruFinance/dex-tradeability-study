"""Fully ON-CHAIN Solana core provider — token / pools / rug from pure JSON-RPC.

This is the no-key, no-third-party-API replacement for the gated DEX/security
platforms (DexScreener, RugCheck, ...) on Solana. Everything here is read straight
from a mainnet RPC node:

  token  <- getAccountInfo(mint) for decimals/supply/authorities; the Metaplex
            metadata PDA for name/symbol; the deepest pool's quote vault for price.
  pools  <- getProgramAccounts on each DEX program, memcmp-filtered on the base AND
            quote mint offset, then vault balances for reserves/liquidity.
  rug    <- mint/freeze authorities (null == revoked), getTokenLargestAccounts for
            top-10 concentration, and the deepest pool's LP-mint burn (when present).

Metaplex name/symbol (verified offsets — see METADATA_PROGRAM / _metadata_pda):
  metadata account = key(1) + update_authority(32) + mint(32) -> name borsh string
  at offset 65 (u32 LE length + utf8), then symbol borsh string, then uri.
  The metadata PDA = findProgramAddress(["metadata", program, mint], program); the
  off-curve bump scan is implemented in pure Python (no solana-py dependency) using
  the ed25519 on-curve test from the Bernstein reference (ed25519.cr.yp.to/python).

getProgramAccounts is HEAVY and aggressively throttled on the public RPC. We use a
dataSlice of length 0 (we only need the matched pubkeys, not the account bodies) and
cap the number of pools we hydrate. On a throttled endpoint pools may come back empty;
token/rug degrade gracefully and never raise.

Docs: https://solana.com/docs/rpc/http/getprogramaccounts ,
      https://developers.metaplex.com/token-metadata .
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging

from ..chains import Chain, normalize_address
from ..http import HttpError
from ..models import Pool, RugReport, Token
from .base import CAP_POOLS, CAP_RUG, CAP_TOKEN, NotSupported, Provider
from .solana_indexer import (
    DEX_NAMES,
    ORCA_WHIRLPOOL,
    PUMPSWAP_AMM,
    PUMPSWAP_LAYOUT,
    RAYDIUM_AMM_V4,
    RAYDIUM_V4_LAYOUT,
    USD_STABLES,
    WHIRLPOOL_LAYOUT,
    WSOL,
    b58decode,
    b58encode,
)

log = logging.getLogger("chainscope.solana_chain")

# Metaplex Token Metadata program (fixed mainnet address).
METADATA_PROGRAM = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

# Per-DEX layout used for getProgramAccounts memcmp discovery: which byte offset holds
# the base mint and which holds the quote mint. Same constants the indexer verified.
DEX_LAYOUTS = {
    RAYDIUM_AMM_V4: RAYDIUM_V4_LAYOUT,
    PUMPSWAP_AMM: PUMPSWAP_LAYOUT,
    ORCA_WHIRLPOOL: WHIRLPOOL_LAYOUT,
}
QUOTE_MINTS = USD_STABLES | {WSOL}

# Cap on getProgramAccounts hydration so a fat token doesn't melt the public RPC.
_MAX_POOLS_PER_PROGRAM = 12

# ---- ed25519 on-curve test (Bernstein reference, ed25519.cr.yp.to/python) ----
_ED_Q = 2 ** 255 - 19


def _ed_inv(x: int) -> int:
    return pow(x, _ED_Q - 2, _ED_Q)


_ED_D = (-121665 * _ed_inv(121666)) % _ED_Q
_ED_I = pow(2, (_ED_Q - 1) // 4, _ED_Q)


def _ed_xrecover(y: int) -> int:
    xx = (y * y - 1) * _ed_inv(_ED_D * y * y + 1)
    x = pow(xx, (_ED_Q + 3) // 8, _ED_Q)
    if (x * x - xx) % _ED_Q != 0:
        x = (x * _ED_I) % _ED_Q
    if x % 2 != 0:
        x = _ED_Q - x
    return x


def _is_on_curve(pubkey: bytes) -> bool:
    """True if the 32-byte value decodes to a valid ed25519 point (has a private key).

    A PDA must be OFF the curve. Mirrors the reference decodepoint(): take y as the
    low 255 bits (little-endian), recover x, flip on the sign bit, verify the twisted
    Edwards curve equation. Any arithmetic failure -> treat as off-curve.
    """
    if len(pubkey) != 32:
        return False
    try:
        n = int.from_bytes(pubkey, "little")
        y = n & ((1 << 255) - 1)
        sign = (n >> 255) & 1
        x = _ed_xrecover(y)
        if x & 1 != sign:
            x = _ED_Q - x
        return (-x * x + y * y - 1 - _ED_D * x * x * y * y) % _ED_Q == 0
    except (ValueError, ZeroDivisionError):
        return False


def _find_program_address(seeds: list[bytes], program_id: bytes) -> str:
    """find_program_address(seeds, program) -> base58 PDA (canonical, off-curve bump).

    Hashes seeds || bump || program || "ProgramDerivedAddress" with SHA-256, scanning
    the bump from 255 downward, returning the first 32-byte digest that is OFF the
    ed25519 curve. Pure stdlib (hashlib); no solana-py.
    """
    for bump in range(255, -1, -1):
        h = hashlib.sha256()
        for s in seeds:
            h.update(s)
        h.update(bytes([bump]))
        h.update(program_id)
        h.update(b"ProgramDerivedAddress")
        cand = h.digest()
        if not _is_on_curve(cand):
            return b58encode(cand)
    raise ValueError("unable to find a program-derived address (off-curve bump)")


def _f(x) -> float | None:
    try:
        return float(x) if x is not None and x != "" else None
    except (TypeError, ValueError):
        return None


def _i(x) -> int | None:
    try:
        return int(x) if x is not None and x != "" else None
    except (TypeError, ValueError):
        return None


class SolanaChainProvider(Provider):
    name = "solana_chain"
    supported_chains = frozenset({Chain.SOLANA})
    capabilities = frozenset({CAP_TOKEN, CAP_POOLS, CAP_RUG})
    requires_key = False
    onchain = True

    def __init__(self, http, settings):
        super().__init__(http, settings)
        self._dec_cache: dict[str, int] = {}   # mint -> decimals
        # None = unknown; False once the endpoint reports getProgramAccounts is
        # disabled for these programs (public RPC error -32010 "excluded from
        # account secondary indexes"), so we stop wasting retry-backoff on it.
        self._gpa_available: bool | None = None

    # ---- low-level RPC ----

    async def _rpc(self, method: str, params: list):
        """Call a JSON-RPC method, returning `result` (or raising HttpError on error).

        The shared HttpClient retries transport/5xx/HTTP-429, but the public Solana RPC
        returns its per-method rate limit as an HTTP-200 body with JSON-RPC error code
        429; we retry that small number of times with backoff before giving up."""
        url = self.settings.rpc_url(Chain.SOLANA)
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        for attempt in range(3):
            resp = await self.http.post_json(url, json=body)
            err = resp.get("error") if isinstance(resp, dict) else None
            if err:
                code = err.get("code") if isinstance(err, dict) else None
                if code == 429 and attempt < 2:
                    await asyncio.sleep(0.6 * (attempt + 1))
                    continue
                raise HttpError(0, url, str(err))
            return resp.get("result") if isinstance(resp, dict) else None
        return None

    async def _account_info(self, pubkey: str, encoding: str = "base64"):
        return await self._rpc("getAccountInfo", [pubkey, {"encoding": encoding}])

    @staticmethod
    def _mint_info(result) -> dict | None:
        """Extract the SPL mint `parsed.info` block from a jsonParsed getAccountInfo."""
        if not isinstance(result, dict):
            return None
        value = result.get("value")
        if value is None:
            return None
        info = (((value.get("data") or {}).get("parsed") or {}).get("info")) or {}
        return info if isinstance(info, dict) else None

    @staticmethod
    def _pk_at(data: bytes, offset: int) -> str:
        return b58encode(data[offset:offset + 32])

    async def _mint_decimals(self, mint: str) -> int | None:
        if mint in self._dec_cache:
            return self._dec_cache[mint]
        try:
            info = self._mint_info(await self._account_info(mint, encoding="jsonParsed"))
        except HttpError:
            info = None
        dec = _i((info or {}).get("decimals"))
        if dec is not None:
            self._dec_cache[mint] = dec
        return dec

    # ---- Metaplex metadata ----

    def _metadata_pda(self, mint: str) -> str:
        seeds = [b"metadata", b58decode(METADATA_PROGRAM), b58decode(mint)]
        return _find_program_address(seeds, b58decode(METADATA_PROGRAM))

    @staticmethod
    def _read_borsh_string(data: bytes, offset: int) -> tuple[str | None, int]:
        """Read a borsh string (u32 LE length + utf8) at `offset`. Returns (text, next)."""
        if offset + 4 > len(data):
            return None, offset
        length = int.from_bytes(data[offset:offset + 4], "little")
        start = offset + 4
        end = start + length
        if length < 0 or end > len(data) or length > 4096:
            return None, end
        try:
            text = data[start:end].decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - defensive on arbitrary on-chain bytes
            return None, end
        # On-chain strings are fixed-width and null-padded; trim trailing NULs/space.
        return text.replace("\x00", "").strip() or None, end

    async def _metaplex_name_symbol(self, mint: str) -> tuple[str | None, str | None]:
        """Resolve (name, symbol) from the Metaplex metadata PDA. None on absence."""
        try:
            pda = self._metadata_pda(mint)
        except ValueError:
            return None, None
        try:
            res = await self._account_info(pda, encoding="base64")
        except HttpError:
            return None, None
        value = (res or {}).get("value")
        blob = (value or {}).get("data")
        if not blob or not isinstance(blob, list):
            return None, None
        try:
            data = base64.b64decode(blob[0])
        except (ValueError, TypeError):
            return None, None
        # key(1) + update_authority(32) + mint(32) -> name borsh string at offset 65.
        name, nxt = self._read_borsh_string(data, 65)
        symbol, _ = self._read_borsh_string(data, nxt)
        return name, symbol

    # ---- pool discovery ----

    async def _program_pools_for_mint(self, program: str, mint: str) -> list[str]:
        """getProgramAccounts(program) matching `mint` at either base OR quote offset.

        Returns the matched pool account pubkeys. dataSlice length 0 keeps the response
        tiny (we re-read each pool's body only for the ones we hydrate). If the endpoint
        has disabled getProgramAccounts for these programs (public RPC error -32010), we
        flip a flag and short-circuit so token/rug aren't dragged down by dead calls."""
        if self._gpa_available is False:
            return []
        layout = DEX_LAYOUTS[program]
        bytes58 = mint  # memcmp `bytes` is a base58 string; the mint already is one.
        out: list[str] = []
        seen: set[str] = set()
        for offset in (layout["base_mint"], layout["quote_mint"]):
            params = [
                program,
                {
                    "encoding": "base64",
                    "dataSlice": {"offset": 0, "length": 0},
                    "filters": [{"memcmp": {"offset": offset, "bytes": bytes58}}],
                },
            ]
            try:
                res = await self._rpc("getProgramAccounts", params)
            except HttpError as exc:
                # -32010 == endpoint has the program out of its secondary index;
                # getProgramAccounts is unavailable here, so stop trying.
                if "secondary index" in str(exc) or "-32010" in str(exc):
                    self._gpa_available = False
                    return []
                log.debug("getProgramAccounts %s @%d failed: %s", program, offset, exc)
                continue
            self._gpa_available = True
            for acc in res or []:
                pk = acc.get("pubkey") if isinstance(acc, dict) else None
                if pk and pk not in seen:
                    seen.add(pk)
                    out.append(pk)
        return out

    async def _hydrate_pool(self, pool: str, mint: str) -> Pool | None:
        """Read a discovered pool account -> Pool with reserves/price for `mint`."""
        try:
            res = await self._account_info(pool, encoding="base64")
        except HttpError:
            return None
        value = (res or {}).get("value")
        owner = (value or {}).get("owner")
        blob = (value or {}).get("data")
        if owner not in DEX_LAYOUTS or not blob or not isinstance(blob, list):
            return None
        try:
            data = base64.b64decode(blob[0])
        except (ValueError, TypeError):
            return None
        layout = DEX_LAYOUTS[owner]
        if len(data) < layout["quote_vault"] + 32:
            return None

        base_mint = self._pk_at(data, layout["base_mint"])
        quote_mint = self._pk_at(data, layout["quote_mint"])
        base_vault = self._pk_at(data, layout["base_vault"])
        quote_vault = self._pk_at(data, layout["quote_vault"])

        # Orient so `mint` is the base leg and the quote-able side is the quote leg.
        if quote_mint == mint and base_mint != mint:
            base_mint, quote_mint = quote_mint, base_mint
            base_vault, quote_vault = quote_vault, base_vault
        elif base_mint in QUOTE_MINTS and quote_mint not in QUOTE_MINTS:
            base_mint, quote_mint = quote_mint, base_mint
            base_vault, quote_vault = quote_vault, base_vault

        base_bal = await self._token_account_ui(base_vault)
        quote_bal = await self._token_account_ui(quote_vault)

        price_usd = price_native = liquidity_usd = None
        if base_bal and quote_bal and base_bal > 0:
            price = quote_bal / base_bal       # quote units per base
            if quote_mint in USD_STABLES:
                price_usd = price
                liquidity_usd = 2.0 * quote_bal
            elif quote_mint == WSOL:
                price_native = price

        return Pool(
            source=self.name,
            chain=Chain.SOLANA,
            dex=DEX_NAMES.get(owner, owner),
            pair_address=pool,
            base_address=base_mint,
            quote_address=quote_mint,
            quote_symbol=self._known_quote_symbol(quote_mint),
            price_usd=price_usd,
            price_native=price_native,
            liquidity_usd=liquidity_usd,
            liquidity_base=base_bal,
            liquidity_quote=quote_bal,
            raw={"owner": owner, "base_vault": base_vault, "quote_vault": quote_vault},
        )

    async def _token_account_ui(self, token_account: str) -> float | None:
        try:
            res = await self._rpc("getTokenAccountBalance", [token_account])
        except HttpError:
            return None
        val = (res or {}).get("value") or {}
        ui = _f(val.get("uiAmount"))
        if ui is not None:
            return ui
        amt, dec = _i(val.get("amount")), _i(val.get("decimals"))
        if amt is not None and dec is not None:
            return amt / 10 ** dec
        return None

    @staticmethod
    def _known_quote_symbol(quote_mint: str) -> str | None:
        if quote_mint in USD_STABLES:
            return "USDC" if quote_mint.startswith("EPjF") else "USDT"
        if quote_mint == WSOL:
            return "SOL"
        return None

    # ---- capability: pools ----

    async def get_pools(self, chain: Chain, address: str) -> list[Pool]:
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported
        mint = normalize_address(chain, address)
        out: list[Pool] = []
        for program in DEX_LAYOUTS:
            pools = await self._program_pools_for_mint(program, mint)
            for pk in pools[:_MAX_POOLS_PER_PROGRAM]:
                pool = await self._hydrate_pool(pk, mint)
                if pool is not None:
                    out.append(pool)
        # deepest (USD-liquid) first
        out.sort(key=lambda p: (p.liquidity_usd or -1.0), reverse=True)
        return out

    # ---- capability: token ----

    async def get_token(self, chain: Chain, address: str) -> Token | None:
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported
        mint = normalize_address(chain, address)
        try:
            info = self._mint_info(await self._account_info(mint, encoding="jsonParsed"))
        except HttpError:
            return None
        if info is None:
            return None

        decimals = _i(info.get("decimals"))
        supply_raw = _i(info.get("supply"))
        total_supply = None
        if supply_raw is not None and decimals is not None:
            total_supply = supply_raw / 10 ** decimals
            self._dec_cache[mint] = decimals

        name, symbol = await self._metaplex_name_symbol(mint)

        # Price/mcap from the deepest discovered pool. getProgramAccounts may be empty
        # under throttling; the rest of the token still resolves.
        price_usd = price_native = liquidity_usd = market_cap = None
        try:
            pools = await self.get_pools(chain, mint)
        except (HttpError, NotSupported):
            pools = []
        if pools:
            deepest = pools[0]
            price_usd = deepest.price_usd
            price_native = deepest.price_native
            liquidity_usd = deepest.liquidity_usd
            if price_usd is not None and total_supply is not None:
                market_cap = price_usd * total_supply

        return Token(
            source=self.name,
            chain=Chain.SOLANA,
            address=mint,
            symbol=symbol,
            name=name,
            decimals=decimals,
            total_supply=total_supply,
            price_usd=price_usd,
            price_native=price_native,
            market_cap=market_cap,
            fdv=market_cap,   # SPL has no separate circulating measure on-chain
            liquidity_usd=liquidity_usd,
            pair_count=len(pools) or None,
            raw=info,
        )

    # ---- capability: rug ----

    async def get_rug(self, chain: Chain, address: str) -> RugReport | None:
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported
        mint = normalize_address(chain, address)
        try:
            info = self._mint_info(await self._account_info(mint, encoding="jsonParsed"))
        except HttpError:
            return None
        if info is None:
            return None

        decimals = _i(info.get("decimals"))
        supply_raw = _i(info.get("supply"))
        mint_authority_active = info.get("mintAuthority") is not None
        freeze_authority_active = info.get("freezeAuthority") is not None

        # top-10 concentration from the largest token accounts (max 20 returned).
        top10_holder_pct = None
        largest_raw = None
        try:
            largest = await self._rpc("getTokenLargestAccounts", [mint])
        except HttpError:
            largest = None
        if isinstance(largest, dict):
            accounts = largest.get("value") or []
            largest_raw = accounts
            if accounts and supply_raw is not None and decimals is not None and supply_raw > 0:
                top10_ui = sum(
                    (_f(a.get("uiAmount")) or 0.0)
                    for a in accounts[:10] if isinstance(a, dict)
                )
                total_ui = supply_raw / 10 ** decimals
                if total_ui > 0:
                    top10_holder_pct = top10_ui / total_ui

        # LP burn for the deepest pool's LP mint, where the program exposes one.
        lp_burned_pct = None
        lp_raw = None
        try:
            lp_burned_pct, lp_raw = await self._lp_burn(chain, mint)
        except (HttpError, NotSupported):
            pass

        flags: list[str] = []
        if mint_authority_active:
            flags.append("mint authority active")
        if freeze_authority_active:
            flags.append("freeze authority active")
        if top10_holder_pct is not None and top10_holder_pct > 0.5:
            flags.append("top10 hold >50%")
        if lp_burned_pct is not None and lp_burned_pct >= 0.99:
            flags.append("LP burned")

        return RugReport(
            source=self.name,
            chain=Chain.SOLANA,
            address=mint,
            risk_score=None,
            mint_authority_active=mint_authority_active,
            freeze_authority_active=freeze_authority_active,
            top10_holder_pct=top10_holder_pct,
            lp_burned_pct=lp_burned_pct,
            holder_count=None,   # not enumerable from these RPC calls
            flags=flags,
            raw={"mint": info, "largestAccounts": largest_raw, "lp": lp_raw},
        )

    async def _lp_burn(self, chain: Chain, mint: str) -> tuple[float | None, dict | None]:
        """For the deepest Raydium pool, estimate LP-mint burn from supply.

        Only Raydium v4 exposes an SPL LP mint at a fixed offset; PumpSwap has an LP
        mint too but Orca's CLMM has no fungible LP token. We read the deepest pool's
        owning program, grab its LP mint, and compare LP supply against burn addresses
        is non-trivial on-chain — a fully-burned LP mint typically has supply 0 or its
        authority revoked. We report supply==0 as fully burned, else None.
        """
        pools = await self.get_pools(chain, mint)
        for pool in pools:
            owner = (pool.raw or {}).get("owner")
            lp_off = None
            if owner == RAYDIUM_AMM_V4:
                lp_off = 464   # RAYDIUM v4: lpMint @464 (after quoteMint@432)
            elif owner == PUMPSWAP_AMM:
                lp_off = 107   # PumpSwap lp_mint @107
            if lp_off is None:
                continue
            try:
                res = await self._account_info(pool.pair_address, encoding="base64")
            except HttpError:
                continue
            blob = ((res or {}).get("value") or {}).get("data")
            if not blob or not isinstance(blob, list):
                continue
            try:
                data = base64.b64decode(blob[0])
            except (ValueError, TypeError):
                continue
            if len(data) < lp_off + 32:
                continue
            lp_mint = self._pk_at(data, lp_off)
            lp_info = self._mint_info(await self._account_info(lp_mint, encoding="jsonParsed"))
            if lp_info is None:
                continue
            lp_supply = _i(lp_info.get("supply"))
            if lp_supply is not None:
                burned = 1.0 if lp_supply == 0 else None
                return burned, {"lp_mint": lp_mint, "lp_supply": lp_supply}
        return None, None
