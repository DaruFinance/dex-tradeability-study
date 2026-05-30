"""GoPlus Security — free, no API key. Token-security ("rug") reports for both
Solana and BSC.

Capabilities: rug. Covers Solana and BSC via two different endpoints.
BSC uses the EVM endpoint /token_security/{chain_id} (booleans come back as the
strings "1"/"0", taxes as string fractions). Solana uses /solana/token_security
(authorities expressed as {status, authority}, several flags as 0/1 ints).
Docs: https://docs.gopluslabs.io/reference/api-overview
"""
from __future__ import annotations

from ..chains import Chain, normalize_address, spec
from ..http import HttpError
from ..models import RugReport
from .base import CAP_RUG, NotSupported, Provider

BASE = "https://api.gopluslabs.io/api/v1"

# Addresses that mean "no owner" / "burned" for EVM.
_BURN_ADDRS = {
    "0x000000000000000000000000000000000000dead",
    "0x0000000000000000000000000000000000000000",
}
_ZERO_OWNER = {""} | _BURN_ADDRS


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


def _b(x) -> bool | None:
    """Coerce GoPlus boolean encodings: '1'/'0' strings, 1/0 ints, real bools."""
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None


class GoPlusProvider(Provider):
    name = "goplus"
    supported_chains = frozenset({Chain.SOLANA, Chain.BSC})
    capabilities = frozenset({CAP_RUG})
    requires_key = False

    async def get_rug(self, chain: Chain, address: str) -> RugReport | None:
        chain = Chain.parse(chain)
        if chain == Chain.BSC:
            return await self._get_evm(chain, address)
        if chain == Chain.SOLANA:
            return await self._get_solana(chain, address)
        raise NotSupported

    # ---- EVM (BSC) -------------------------------------------------------

    async def _get_evm(self, chain: Chain, address: str) -> RugReport | None:
        chain_id = spec(chain).goplus_chain_id
        addr = normalize_address(chain, address)
        try:
            data = await self.http.get_json(
                f"{BASE}/token_security/{chain_id}",
                params={"contract_addresses": addr},
                cache_ttl=60,
            )
        except HttpError:
            return None
        result = (data or {}).get("result") or {}
        # GoPlus keys the result by the lowercased address.
        item = result.get(addr) or result.get(address) or {}
        if not item and result:
            # fall back to the single returned entry if keying differs
            item = next(iter(result.values()), {}) or {}
        if not item:
            return None

        owner_address = item.get("owner_address")
        owner_norm = (owner_address or "").strip().lower()

        lp_locked_pct, lp_burned_pct = self._evm_lp(item.get("lp_holders") or [])
        top10 = self._evm_top10(item.get("holders") or [])

        flags = self._evm_flags(item)

        return RugReport(
            source=self.name,
            chain=chain,
            address=addr,
            risk_score=None,  # GoPlus exposes no single native risk score
            is_honeypot=_b(item.get("is_honeypot")),
            buy_tax=_f(item.get("buy_tax")),
            sell_tax=_f(item.get("sell_tax")),
            transfer_tax=_f(item.get("transfer_tax")),
            mintable=_b(item.get("is_mintable")),
            owner_address=owner_address or None,
            ownership_renounced=(owner_norm in _ZERO_OWNER) if owner_address is not None else None,
            can_take_back_ownership=_b(item.get("can_take_back_ownership")),
            hidden_owner=_b(item.get("hidden_owner")),
            lp_locked_pct=lp_locked_pct,
            lp_burned_pct=lp_burned_pct,
            top10_holder_pct=top10,
            creator_pct=_f(item.get("creator_percent")),
            holder_count=_i(item.get("holder_count")),
            is_open_source=_b(item.get("is_open_source")),
            is_proxy=_b(item.get("is_proxy")),
            transfer_pausable=_b(item.get("transfer_pausable")),
            has_blacklist=_b(item.get("is_blacklisted")),
            has_whitelist=_b(item.get("is_whitelisted")),
            anti_whale=_b(item.get("is_anti_whale")),
            flags=flags,
            raw=item,
        )

    @staticmethod
    def _evm_lp(lp_holders: list) -> tuple[float | None, float | None]:
        if not lp_holders:
            return None, None
        locked = 0.0
        burned = 0.0
        for h in lp_holders:
            if not isinstance(h, dict):
                continue
            pct = _f(h.get("percent")) or 0.0
            if _b(h.get("is_locked")):
                locked += pct
            addr = (h.get("address") or "").strip().lower()
            tag = (h.get("tag") or "").lower()
            if addr in _BURN_ADDRS or "burn" in tag:
                burned += pct
        return locked, burned

    @staticmethod
    def _evm_top10(holders: list) -> float | None:
        if not holders:
            return None
        # GoPlus returns holders sorted descending by balance.
        return sum((_f(h.get("percent")) or 0.0) for h in holders[:10] if isinstance(h, dict))

    @staticmethod
    def _evm_flags(item: dict) -> list[str]:
        # (raw key, human-readable flag) for any boolean risk that is set.
        checks = [
            ("is_honeypot", "honeypot"),
            ("cannot_buy", "cannot buy"),
            ("cannot_sell_all", "cannot sell all"),
            ("is_mintable", "mintable"),
            ("can_take_back_ownership", "can take back ownership"),
            ("owner_change_balance", "owner can change balance"),
            ("hidden_owner", "hidden owner"),
            ("selfdestruct", "self destruct"),
            ("external_call", "external call"),
            ("transfer_pausable", "transfer pausable"),
            ("is_blacklisted", "has blacklist"),
            ("is_whitelisted", "has whitelist"),
            ("is_anti_whale", "anti whale"),
            ("anti_whale_modifiable", "anti-whale modifiable"),
            ("slippage_modifiable", "modifiable tax"),
            ("personal_slippage_modifiable", "personal slippage modifiable"),
            ("trading_cooldown", "trading cooldown"),
            ("is_proxy", "proxy contract"),
            ("honeypot_with_same_creator", "honeypot with same creator"),
        ]
        flags = [label for key, label in checks if _b(item.get(key)) is True]
        if _b(item.get("is_open_source")) is False:
            flags.append("not open source")
        return flags

    # ---- Solana ----------------------------------------------------------

    async def _get_solana(self, chain: Chain, address: str) -> RugReport | None:
        mint = normalize_address(chain, address)
        try:
            data = await self.http.get_json(
                f"{BASE}/solana/token_security",
                params={"contract_addresses": mint},
                cache_ttl=60,
            )
        except HttpError:
            return None
        result = (data or {}).get("result") or {}
        item = result.get(mint) or result.get(address) or {}
        if not item and result:
            item = next(iter(result.values()), {}) or {}
        if not item:
            return None

        mintable = item.get("mintable") or {}
        freezable = item.get("freezable") or {}
        transfer_fee = item.get("transfer_fee") or {}

        return RugReport(
            source=self.name,
            chain=chain,
            address=mint,
            risk_score=None,
            mint_authority_active=self._sol_status(mintable),
            freeze_authority_active=self._sol_status(freezable),
            transfer_tax=self._sol_transfer_fee(transfer_fee),
            top10_holder_pct=self._sol_top10(item.get("holders") or []),
            holder_count=_i(item.get("holder_count")),
            lp_burned_pct=self._sol_lp_burned(item.get("dex") or []),
            flags=self._sol_flags(item),
            raw=item,
        )

    @staticmethod
    def _sol_status(node) -> bool | None:
        """{status:'1'/'0', authority:[...]} -> True if authority active."""
        if not isinstance(node, dict) or not node:
            return None
        st = node.get("status")
        b = _b(st)
        return b

    @staticmethod
    def _sol_transfer_fee(fee) -> float | None:
        if not isinstance(fee, dict) or not fee:
            return None
        # GoPlus may express this as basis points or as a percent string.
        for key in ("fee_rate", "current_fee_rate", "newer_transfer_fee", "fee", "pct"):
            v = _f(fee.get(key))
            if v is not None:
                # If it looks like basis points (>1), convert assuming bps.
                return v / 10000.0 if v > 1 else v
        return None

    @staticmethod
    def _sol_top10(holders: list) -> float | None:
        if not holders:
            return None
        return sum((_f(h.get("percent")) or 0.0) for h in holders[:10] if isinstance(h, dict))

    @staticmethod
    def _sol_lp_burned(dex: list) -> float | None:
        if not dex:
            return None
        best = None
        for d in dex:
            if not isinstance(d, dict):
                continue
            bp = _f(d.get("burn_percent"))
            if bp is None:
                continue
            # burn_percent appears as a 0..100 percentage; normalize to fraction.
            frac = bp / 100.0 if bp > 1 else bp
            best = frac if best is None else max(best, frac)
        return best

    @staticmethod
    def _sol_flags(item: dict) -> list[str]:
        flags: list[str] = []

        def status_true(node) -> bool:
            return isinstance(node, dict) and _b(node.get("status")) is True

        if status_true(item.get("mintable")):
            flags.append("mint authority active")
        if status_true(item.get("freezable")):
            flags.append("freeze authority active")
        if status_true(item.get("closable")):
            flags.append("account closable")
        if status_true(item.get("balance_mutable_authority")):
            flags.append("balance mutable")
        if status_true(item.get("metadata_mutable")):
            flags.append("mutable metadata")
        if _b(item.get("non_transferable")) is True:
            flags.append("non-transferable")
        # transfer_hook is a list; non-empty means a hook is set.
        if item.get("transfer_hook"):
            flags.append("transfer hook")
        # default_account_state == 1 means accounts start frozen.
        if _i(item.get("default_account_state")):
            flags.append("default account state frozen")
        fee = item.get("transfer_fee") or {}
        if isinstance(fee, dict) and fee:
            flags.append("transfer fee")
        # any creator marked malicious
        for c in item.get("creators") or []:
            if isinstance(c, dict) and _b(c.get("malicious_address")) is True:
                flags.append("malicious creator")
                break
        return flags
