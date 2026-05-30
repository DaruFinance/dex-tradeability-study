"""Fully ON-CHAIN Solana HOLDER & WALLET analytics, pure JSON-RPC, no paid API.

Reconstructs the things the paid holder/whale platforms charge for, straight from
a mainnet RPC node:

  top_holders      <- getTokenLargestAccounts(mint): the 20 largest token ACCOUNTS
                      (uiAmount), each optionally resolved to its OWNER wallet via
                      getAccountInfo(jsonParsed).info.owner. Works on the public RPC.
  all_holders      <- getProgramAccounts on the SPL Token program(s), memcmp-filtered
                      on the mint at offset 0 over dataSize-165 accounts -> EVERY token
                      account holding the mint, aggregated by owner. This is the only
                      way to get a true holder_count, but the public RPC (mainnet-beta)
                      HARD-disables getProgramAccounts on the token programs (JSON-RPC
                      error -32010 "excluded from account secondary indexes"). We detect
                      that and return an empty map with a clear flag; a dedicated RPC
                      (Helius / Triton / a local node with the index enabled) lights it up.
  holder_stats     <- top10/top20 concentration + HHI from either the full owner map or
                      the top-20 accounts. holder_count is only reported when the full
                      set is available.
  wallet_position  <- getTokenAccountsByOwner(wallet, {mint}) (jsonParsed) for the live
                      balance, plus a best-effort net flow by paging the token account's
                      signatures and balance-diffing each transaction.

SPL Token-account layout (verified, solana-program-library Account state; also the
165-byte canonical size that Token-2022 preserves for base accounts):
  mint   @ 0  (32 bytes, pubkey)
  owner  @ 32 (32 bytes, pubkey)
  amount @ 64 (u64 little-endian)
  ... delegate / state / is_native / delegated_amount / close_authority -> 165 bytes total.

Token program ids (fixed mainnet addresses):
  SPL Token   = TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
  Token-2022  = TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb

Docs: https://solana.com/docs/rpc/http/gettokenlargestaccounts,
      https://solana.com/docs/rpc/http/getprogramaccounts,
      https://solana.com/docs/rpc/http/gettokenaccountsbyowner,
      https://github.com/solana-labs/solana-program-library (token Account state) .
"""
from __future__ import annotations

import base64
import logging

from .chains import Chain
from .http import HttpError

log = logging.getLogger("chainscope.holders_sol")

# ---- SPL token programs (account owners of token accounts) ----
SPL_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
TOKEN_PROGRAMS = (SPL_TOKEN_PROGRAM, TOKEN_2022_PROGRAM)

# ---- SPL token-account layout (verified byte offsets) ----
TOKEN_ACCOUNT_SIZE = 165          # canonical base token-account dataSize
TA_MINT_OFFSET = 0                # pubkey (32)
TA_OWNER_OFFSET = 32              # pubkey (32)
TA_AMOUNT_OFFSET = 64             # u64 little-endian


# Public-RPC rejections that mean "getProgramAccounts can't enumerate token accounts
# here": detected so all_holders() degrades to a clearly-flagged empty result rather
# than faking a zero-holder set. Two distinct limits show up on api.mainnet-beta:
#   -32010  "...excluded from account secondary indexes..."   (program de-indexed)
#   -32012  "scan aborted: The accumulated scan results exceeded the limit"  (too many)
_GPA_DISABLED_MARKERS = (
    "-32010", "secondary index", "secondary indexes",
    "-32012", "scan aborted", "exceeded the limit",
)
_GPA_UNAVAILABLE_REASON = (
    "getProgramAccounts cannot enumerate token accounts on this RPC "
    "(public mainnet-beta returns -32010 'excluded from account secondary indexes' "
    "or -32012 'scan aborted: results exceeded the limit'); use a dedicated RPC "
    "(Helius/Triton) or a local node with the token secondary index enabled"
)


def _is_gpa_disabled(err_text: str) -> bool:
    low = err_text.lower()
    return any(m in low for m in _GPA_DISABLED_MARKERS)


def _amount_from_account_data(raw: bytes) -> tuple[str | None, str | None, int | None]:
    """Decode (mint, owner, raw_amount) from a 165-byte SPL token-account body."""
    if len(raw) < TA_AMOUNT_OFFSET + 8:
        return None, None, None
    mint = _b58encode(raw[TA_MINT_OFFSET:TA_MINT_OFFSET + 32])
    owner = _b58encode(raw[TA_OWNER_OFFSET:TA_OWNER_OFFSET + 32])
    amount = int.from_bytes(raw[TA_AMOUNT_OFFSET:TA_AMOUNT_OFFSET + 8], "little")
    return mint, owner, amount


# ---- pure-Python base58 (same alphabet/pattern as solana_indexer) ----
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(raw: bytes) -> str:
    """Base58-encode bytes (Bitcoin/Solana alphabet), preserving leading-zero bytes as '1'."""
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


class SolHolders:
    """On-chain holder/wallet analytics for a Solana SPL token.

    Owns its own JSON-RPC channel via ``self.settings.rpc_url(Chain.SOLANA)``, it is
    NOT a Provider (no registry capability), just a focused analytics helper that mirrors
    the raw-RPC style of solana_chain / solana_indexer.
    """

    def __init__(self, http, settings):
        self.http = http
        self.settings = settings
        self._dec_cache: dict[str, int] = {}      # mint -> decimals
        self._supply_cache: dict[str, float] = {}  # mint -> ui supply
        # None = unknown; False once the endpoint reports getProgramAccounts is disabled
        # for the token programs (public RPC error -32010 "secondary index"), so we stop
        # hammering a call that will never succeed here.
        self._gpa_available: bool | None = None

    # ---- low-level RPC (own channel; same shape/retry note as solana_chain._rpc) ----

    async def _rpc(self, method: str, params: list):
        """Call a JSON-RPC method, returning ``result`` or raising HttpError on error.

        The public Solana RPC returns its per-method rate limit as an HTTP-200 body with
        JSON-RPC error code 429; we surface every JSON-RPC error as HttpError so callers
        can branch on the message (e.g. the -32010 secondary-index limit)."""
        url = self.settings.rpc_url(Chain.SOLANA)
        resp = await self.http.post_json(
            url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        )
        if isinstance(resp, dict) and resp.get("error"):
            raise HttpError(0, url, str(resp["error"]))
        return resp.get("result") if isinstance(resp, dict) else None

    async def _account_info(self, pubkey: str, encoding: str = "jsonParsed"):
        return await self._rpc("getAccountInfo", [pubkey, {"encoding": encoding}])

    # ---- mint supply / decimals (cached) ----

    async def _mint_supply_decimals(self, mint: str) -> tuple[float | None, int | None]:
        """(ui_supply, decimals) for a mint from getAccountInfo(jsonParsed)."""
        try:
            res = await self._account_info(mint, encoding="jsonParsed")
        except HttpError as exc:
            log.debug("mint info failed for %s: %s", mint, exc)
            return self._supply_cache.get(mint), self._dec_cache.get(mint)
        value = (res or {}).get("value") or {}
        info = (((value.get("data") or {}).get("parsed") or {}).get("info")) or {}
        decimals = _i(info.get("decimals"))
        supply_raw = _i(info.get("supply"))
        ui_supply = None
        if supply_raw is not None and decimals is not None:
            ui_supply = supply_raw / 10 ** decimals
            self._supply_cache[mint] = ui_supply
            self._dec_cache[mint] = decimals
        return ui_supply, decimals

    # ---- top holders (works on public RPC) ----

    async def top_holders(self, mint: str, n: int = 20) -> list[dict]:
        """The largest token ACCOUNTS for ``mint`` (max 20 from getTokenLargestAccounts).

        For each account we optionally resolve the OWNER wallet via getAccountInfo
        (jsonParsed -> info.owner). Returns up to ``n`` dicts:
          {account, owner, amount, pct_of_supply}
        sorted by balance descending. ``pct_of_supply`` is a fraction (0..1) of the mint
        supply, or None if supply is unavailable. Owner resolution is best-effort: a
        throttled / failing getAccountInfo leaves ``owner`` None but keeps the account.
        """
        try:
            largest = await self._rpc("getTokenLargestAccounts", [mint])
        except HttpError as exc:
            log.debug("getTokenLargestAccounts failed for %s: %s", mint, exc)
            return []
        accounts = (largest or {}).get("value") if isinstance(largest, dict) else None
        if not accounts:
            return []

        ui_supply, _dec = await self._mint_supply_decimals(mint)

        out: list[dict] = []
        for entry in accounts[: max(1, n)]:
            if not isinstance(entry, dict):
                continue
            account = entry.get("address")
            amount = _f(entry.get("uiAmount"))
            if amount is None:
                # uiAmount is deprecated/None on some endpoints -> derive from raw/decimals
                amt, dec = _i(entry.get("amount")), _i(entry.get("decimals"))
                if amt is not None and dec is not None:
                    amount = amt / 10 ** dec
            owner = await self._resolve_owner(account) if account else None
            pct = (amount / ui_supply) if (amount is not None and ui_supply) else None
            out.append({
                "account": account,
                "owner": owner,
                "amount": amount,
                "pct_of_supply": pct,
            })
        out.sort(key=lambda h: (h["amount"] or -1.0), reverse=True)
        return out

    async def _resolve_owner(self, token_account: str) -> str | None:
        """OWNER wallet behind a token account, via getAccountInfo(jsonParsed).info.owner."""
        try:
            res = await self._account_info(token_account, encoding="jsonParsed")
        except HttpError:
            return None
        value = (res or {}).get("value") or {}
        info = (((value.get("data") or {}).get("parsed") or {}).get("info")) or {}
        owner = info.get("owner")
        return owner if isinstance(owner, str) and owner else None

    # ---- full holder set (needs a getProgramAccounts-capable RPC) ----

    async def all_holders(self, mint: str) -> dict:
        """EVERY owner wallet -> aggregated ui balance for ``mint``.

        Uses getProgramAccounts on each SPL token program with filters
          [{dataSize: 165}, {memcmp: {offset: 0, bytes: <mint base58>}}]
        decoding the base64 account body (owner @32, amount @64 u64 LE) and summing by
        owner. The public RPC (api.mainnet-beta) HARD-disables getProgramAccounts on the
        token programs (JSON-RPC error -32010 "excluded from account secondary indexes");
        we DETECT that and return a clearly-flagged empty result rather than faking it.

        Returns a dict:
          {
            "balances": {owner -> ui_balance},   # empty if unavailable
            "available": bool,                   # True only if gPA actually ran
            "holder_count": int | None,          # len(balances) when available
            "supply": float | None,
            "decimals": int | None,
            "reason": str | None,                # why it's empty, when unavailable
          }
        A capable RPC (Helius / Triton / local node with the token secondary index, or
        any node where getProgramAccounts is permitted for the token programs) lights it up.
        """
        ui_supply, decimals = await self._mint_supply_decimals(mint)
        result: dict = {
            "balances": {},
            "available": False,
            "holder_count": None,
            "supply": ui_supply,
            "decimals": decimals,
            "reason": None,
        }
        if decimals is None:
            result["reason"] = "mint not found or supply/decimals unavailable"
            return result
        if self._gpa_available is False:
            result["reason"] = _GPA_UNAVAILABLE_REASON
            return result

        balances: dict[str, int] = {}
        scale = 10 ** decimals
        ran = False  # did getProgramAccounts actually succeed on any program?
        for program in TOKEN_PROGRAMS:
            params = [
                program,
                {
                    "encoding": "base64",
                    "filters": [
                        {"dataSize": TOKEN_ACCOUNT_SIZE},
                        {"memcmp": {"offset": TA_MINT_OFFSET, "bytes": mint}},
                    ],
                },
            ]
            try:
                accounts = await self._rpc("getProgramAccounts", params)
            except HttpError as exc:
                if _is_gpa_disabled(str(exc)):
                    # the public-RPC limit: getProgramAccounts can't enumerate token
                    # accounts here (-32010 de-indexed, or -32012 scan too large).
                    self._gpa_available = False
                    result["reason"] = _GPA_UNAVAILABLE_REASON
                    return result
                log.debug("getProgramAccounts %s failed for %s: %s", program, mint, exc)
                continue
            self._gpa_available = True
            ran = True
            for acc in accounts or []:
                blob = ((acc or {}).get("account") or {}).get("data")
                if not blob or not isinstance(blob, list):
                    continue
                try:
                    raw = base64.b64decode(blob[0])
                except (ValueError, TypeError):
                    continue
                _m, owner, amount_raw = _amount_from_account_data(raw)
                if owner is None or amount_raw is None:
                    continue
                if amount_raw == 0:
                    continue  # zero-balance accounts aren't holders
                balances[owner] = balances.get(owner, 0) + amount_raw

        if not ran:
            # neither token program returned a scan (and none tripped a known limit),
            # don't pretend this is a clean zero-holder result.
            result["reason"] = "getProgramAccounts did not complete on any token program"
            return result

        ui_balances = {owner: amt / scale for owner, amt in balances.items()}
        result["balances"] = ui_balances
        result["available"] = True
        result["holder_count"] = len(ui_balances)
        return result

    # ---- distribution / concentration stats ----

    @staticmethod
    def holder_stats(holders, supply: float | None = None) -> dict:
        """Concentration stats from either the full owner map or the top-20 accounts.

        ``holders`` accepts:
          - dict[owner -> balance]  (full set; from all_holders()["balances"]), OR
          - list[dict]              (top_holders() output, each with "amount"/"pct_of_supply").

        Returns:
          {
            "holder_count": int | None,   # only when the FULL owner set is provided
            "top10_pct": float | None,    # fraction 0..1 of supply held by the top 10
            "top20_pct": float | None,    # fraction 0..1 of supply held by the top 20
            "hhi": float | None,          # Herfindahl-Hirschman index = sum(share^2)
            "full_set": bool,             # True if computed from the full owner map
            "supply": float | None,
          }
        With only the top-20 accounts (public RPC), holder_count is None and the figures
        are floors over those 20: flagged via full_set=False. HHI is computed over share
        of supply (0..1); over a full set it is the true HHI, over the top-20 it is a lower
        bound. ``hhi`` is None when no supply is known (top-20 input without pct_of_supply).
        """
        stats: dict = {
            "holder_count": None,
            "top10_pct": None,
            "top20_pct": None,
            "hhi": None,
            "full_set": False,
            "supply": supply,
        }

        if isinstance(holders, dict):
            # full owner map -> true holder_count + concentration over the whole supply.
            amounts = sorted((v for v in holders.values() if v and v > 0), reverse=True)
            total = supply if (supply and supply > 0) else sum(amounts)
            stats["full_set"] = True
            stats["holder_count"] = len(amounts)
            stats["supply"] = total or None
            if total and total > 0:
                stats["top10_pct"] = sum(amounts[:10]) / total
                stats["top20_pct"] = sum(amounts[:20]) / total
                stats["hhi"] = sum((a / total) ** 2 for a in amounts)
            return stats

        # list-of-dicts (top_holders output): a floor over the top 20.
        items = [h for h in (holders or []) if isinstance(h, dict)]
        amounts = sorted((_f(h.get("amount")) or 0.0 for h in items), reverse=True)
        # prefer pct_of_supply when present; else fall back to amounts / supply.
        pcts = [h.get("pct_of_supply") for h in items if h.get("pct_of_supply") is not None]
        if pcts:
            pcts.sort(reverse=True)
            stats["top10_pct"] = sum(pcts[:10])
            stats["top20_pct"] = sum(pcts[:20])
            stats["hhi"] = sum(p * p for p in pcts)
        elif supply and supply > 0:
            stats["top10_pct"] = sum(amounts[:10]) / supply
            stats["top20_pct"] = sum(amounts[:20]) / supply
            stats["hhi"] = sum((a / supply) ** 2 for a in amounts)
        return stats

    # ---- per-wallet position ----

    async def wallet_position(self, wallet: str, mint: str) -> dict:
        """A wallet's current position in ``mint`` (on-chain only).

        Live balance via getTokenAccountsByOwner(wallet, {mint}) (jsonParsed -> uiAmount),
        summed across the wallet's token accounts for the mint (there can be more than one,
        e.g. an ATA plus an auxiliary account). Best-effort NET FLOW is reconstructed by
        paging each token account's signatures (getSignaturesForAddress) and balance-diffing
        the wallet's pre/post token balance in each transaction.

        Returns:
          {
            "wallet": str,
            "mint": str,
            "balance": float | None,        # current ui balance (None if no account)
            "token_accounts": [str, ...],   # the wallet's token accounts for this mint
            "net_flow": float | None,       # signed ui delta over the sampled window
            "inflow": float | None,         # sum of positive deltas (received)
            "outflow": float | None,        # sum of negative deltas (sent), as a positive number
            "tx_sampled": int,              # transactions actually decoded for flow
            "flow_complete": bool,          # False => paging was capped (flow is partial)
          }
        """
        out: dict = {
            "wallet": wallet,
            "mint": mint,
            "balance": None,
            "token_accounts": [],
            "net_flow": None,
            "inflow": None,
            "outflow": None,
            "tx_sampled": 0,
            "flow_complete": True,
        }

        try:
            res = await self._rpc(
                "getTokenAccountsByOwner",
                [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
            )
        except HttpError as exc:
            log.debug("getTokenAccountsByOwner failed for %s/%s: %s", wallet, mint, exc)
            return out

        accounts = (res or {}).get("value") if isinstance(res, dict) else None
        if not accounts:
            return out

        balance = 0.0
        token_accounts: list[str] = []
        for acc in accounts:
            if not isinstance(acc, dict):
                continue
            pubkey = acc.get("pubkey")
            info = (((acc.get("account") or {}).get("data") or {}).get("parsed") or {}).get("info") or {}
            ui = _f((info.get("tokenAmount") or {}).get("uiAmount"))
            if ui is None:
                ta = info.get("tokenAmount") or {}
                amt, dec = _i(ta.get("amount")), _i(ta.get("decimals"))
                if amt is not None and dec is not None:
                    ui = amt / 10 ** dec
            if pubkey:
                token_accounts.append(pubkey)
            if ui is not None:
                balance += ui
        out["balance"] = balance
        out["token_accounts"] = token_accounts

        # best-effort net flow from the token accounts' signature history.
        flow = await self._wallet_flow(token_accounts, wallet)
        out.update(flow)
        return out

    async def _wallet_flow(self, token_accounts: list[str], wallet: str,
                           max_sigs: int = 200) -> dict:
        """Net token flow across ``token_accounts`` by balance-diffing recent transactions.

        Pages getSignaturesForAddress per token account (capped at ``max_sigs`` each to stay
        polite on the public RPC), fetches each transaction, and reads the wallet's pre/post
        token-balance delta for the mint's accounts. Returns net/in/out/sampled/complete."""
        result = {
            "net_flow": None, "inflow": None, "outflow": None,
            "tx_sampled": 0, "flow_complete": True,
        }
        if not token_accounts:
            return result

        ta_set = set(token_accounts)
        seen_sigs: set[str] = set()
        net = 0.0
        inflow = 0.0
        outflow = 0.0
        sampled = 0
        complete = True

        for ta in token_accounts:
            try:
                sigs = await self._rpc(
                    "getSignaturesForAddress", [ta, {"limit": max_sigs}]
                )
            except HttpError as exc:
                log.debug("getSignaturesForAddress %s failed: %s", ta, exc)
                complete = False
                continue
            sigs = sigs or []
            if len(sigs) >= max_sigs:
                complete = False  # there is older history we didn't page
            for s in sigs:
                if not isinstance(s, dict):
                    continue
                if s.get("err") is not None:
                    continue
                sig = s.get("signature")
                if not sig or sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                delta = await self._tx_token_delta(sig, ta_set, wallet)
                if delta is None:
                    continue
                sampled += 1
                net += delta
                if delta > 0:
                    inflow += delta
                elif delta < 0:
                    outflow += -delta

        result["tx_sampled"] = sampled
        result["flow_complete"] = complete
        if sampled:
            result["net_flow"] = net
            result["inflow"] = inflow
            result["outflow"] = outflow
        return result

    async def _tx_token_delta(self, sig: str, ta_set: set[str], wallet: str) -> float | None:
        """Signed ui-balance delta for the wallet's token accounts within one transaction.

        Matches pre/postTokenBalances entries that belong to the wallet's token accounts
        (by accountIndex -> pubkey, falling back to the entry's own ``owner``), and returns
        post - pre summed over them. None if the tx didn't touch the wallet's accounts."""
        try:
            tx = await self._rpc(
                "getTransaction",
                [sig, {"maxSupportedTransactionVersion": 0, "encoding": "jsonParsed"}],
            )
        except HttpError:
            return None
        if not isinstance(tx, dict):
            return None
        meta = tx.get("meta") or {}
        if meta.get("err") is not None:
            return None

        message = (tx.get("transaction") or {}).get("message") or {}
        account_keys: list[str] = []
        for k in (message.get("accountKeys") or []):
            account_keys.append(k["pubkey"] if isinstance(k, dict) else k)
        loaded = meta.get("loadedAddresses") or {}
        account_keys.extend(loaded.get("writable") or [])
        account_keys.extend(loaded.get("readonly") or [])

        def _matches(tb: dict) -> bool:
            idx = tb.get("accountIndex")
            pk = account_keys[idx] if (idx is not None and 0 <= idx < len(account_keys)) else None
            if pk is not None and pk in ta_set:
                return True
            # fall back to the balance entry's own owner field (jsonParsed provides it).
            return tb.get("owner") == wallet

        def _ui(tb: dict) -> float:
            uta = tb.get("uiTokenAmount") or {}
            ui = _f(uta.get("uiAmount"))
            if ui is not None:
                return ui
            amt, dec = _i(uta.get("amount")), _i(uta.get("decimals"))
            if amt is not None and dec is not None:
                return amt / 10 ** dec
            return 0.0

        pre = sum(_ui(tb) for tb in (meta.get("preTokenBalances") or []) if _matches(tb))
        post = sum(_ui(tb) for tb in (meta.get("postTokenBalances") or []) if _matches(tb))
        # if neither pre nor post touched our accounts, this tx is irrelevant.
        touched_pre = any(_matches(tb) for tb in (meta.get("preTokenBalances") or []))
        touched_post = any(_matches(tb) for tb in (meta.get("postTokenBalances") or []))
        if not touched_pre and not touched_post:
            return None
        return post - pre
