"""Solana JSON-RPC: free, no API key. Ground-truth token-safety signals read
directly from a mainnet RPC node, complementing the security-API providers.

Capabilities: token, rug. Solana only.
Authorities are null on-chain when revoked, so mint/freeze_authority_active are
False when the corresponding authority is absent. top10_holder_pct is derived
from getTokenLargestAccounts (max 20 entries) over the UI total supply.
holder_count cannot be enumerated from these calls and is left None.
Docs: https://solana.com/docs/rpc
"""
from __future__ import annotations

from ..chains import Chain, normalize_address
from ..http import HttpError
from ..models import RugReport, Token
from .base import CAP_RUG, CAP_TOKEN, NotSupported, Provider


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


class SolanaRpcProvider(Provider):
    name = "solana_rpc"
    supported_chains = frozenset({Chain.SOLANA})
    capabilities = frozenset({CAP_RUG, CAP_TOKEN})
    requires_key = False

    async def _rpc(self, method: str, params: list) -> dict | None:
        """Call a JSON-RPC method; return the `result` object or None on error."""
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        data = await self.http.post_json(self.settings.rpc_url(Chain.SOLANA), json=body)
        if not isinstance(data, dict) or data.get("error") is not None:
            return None
        return data.get("result")

    @staticmethod
    def _mint_info(result) -> dict | None:
        """Extract the SPL mint `parsed.info` block from a getAccountInfo result."""
        if not isinstance(result, dict):
            return None
        value = result.get("value")
        if value is None:
            return None
        info = (((value.get("data") or {}).get("parsed") or {}).get("info")) or {}
        return info if isinstance(info, dict) else {}

    async def get_token(self, chain: Chain, address: str) -> Token | None:
        chain = Chain.parse(chain)
        if chain != Chain.SOLANA:
            raise NotSupported
        mint = normalize_address(chain, address)
        try:
            result = await self._rpc(
                "getAccountInfo", [mint, {"encoding": "jsonParsed"}]
            )
        except HttpError:
            return None
        info = self._mint_info(result)
        if info is None:
            return None

        decimals = _i(info.get("decimals"))
        supply_raw = _i(info.get("supply"))
        total_supply = None
        if supply_raw is not None and decimals is not None:
            total_supply = supply_raw / 10 ** decimals

        return Token(
            source=self.name,
            chain=chain,
            address=mint,
            decimals=decimals,
            total_supply=total_supply,
            raw=info or None,
        )

    async def get_rug(self, chain: Chain, address: str) -> RugReport | None:
        chain = Chain.parse(chain)
        if chain != Chain.SOLANA:
            raise NotSupported
        mint = normalize_address(chain, address)

        try:
            acct = await self._rpc(
                "getAccountInfo", [mint, {"encoding": "jsonParsed"}]
            )
        except HttpError:
            return None
        info = self._mint_info(acct)
        if info is None:
            return None

        decimals = _i(info.get("decimals"))
        supply_raw = _i(info.get("supply"))
        mint_auth = info.get("mintAuthority")
        freeze_auth = info.get("freezeAuthority")
        mint_authority_active = mint_auth is not None
        freeze_authority_active = freeze_auth is not None

        # Largest accounts -> approximate top-10 concentration.
        top10_holder_pct = None
        largest_raw = None
        try:
            largest = await self._rpc("getTokenLargestAccounts", [mint])
        except HttpError:
            largest = None
        if isinstance(largest, dict):
            accounts = largest.get("value") or []
            largest_raw = accounts
            if (
                accounts
                and supply_raw is not None
                and decimals is not None
                and supply_raw > 0
            ):
                top10_ui = sum(
                    (_f(a.get("uiAmount")) or 0.0)
                    for a in accounts[:10]
                    if isinstance(a, dict)
                )
                total_ui = supply_raw / 10 ** decimals
                if total_ui > 0:
                    top10_holder_pct = top10_ui / total_ui

        flags: list[str] = []
        if mint_authority_active:
            flags.append("mint authority active")
        if freeze_authority_active:
            flags.append("freeze authority active")
        if top10_holder_pct is not None and top10_holder_pct > 0.5:
            flags.append("top10 hold >50%")

        return RugReport(
            source=self.name,
            chain=chain,
            address=mint,
            risk_score=None,
            mint_authority_active=mint_authority_active,
            freeze_authority_active=freeze_authority_active,
            top10_holder_pct=top10_holder_pct,
            holder_count=None,
            flags=flags,
            raw={"mint": info, "largestAccounts": largest_raw},
        )
