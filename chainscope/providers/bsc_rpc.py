"""BNB Smart Chain JSON-RPC: free, no API key. Ground-truth token-safety
signals read directly from a BSC node via eth_call, complementing the
security-API providers.

Capabilities: token, rug. BSC only.
Reads standard ERC-20 views (decimals/totalSupply) and the Ownable owner()
view. A reverting eth_call returns a JSON-RPC error (or empty "0x"); we treat
those as "unavailable" rather than failures. Contracts without Ownable simply
report owner_address=None / ownership_renounced=None.
Docs: https://docs.bnbchain.org/docs/rpc
"""
from __future__ import annotations

from ..chains import Chain, normalize_address
from ..http import HttpError
from ..models import RugReport, Token
from .base import CAP_RUG, CAP_TOKEN, NotSupported, Provider

# ERC-20 / Ownable function selectors (first 4 bytes of keccak(signature)).
SEL_DECIMALS = "0x313ce567"      # decimals()
SEL_TOTAL_SUPPLY = "0x18160ddd"  # totalSupply()
SEL_OWNER = "0x8da5cb5b"         # owner()

_DEAD_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}


def _hex_int(result) -> int | None:
    """Decode a hex-string RPC result as an unsigned integer."""
    if not isinstance(result, str):
        return None
    if result in ("", "0x"):
        return None
    try:
        return int(result, 16)
    except ValueError:
        return None


def _hex_address(result) -> str | None:
    """Decode the last 20 bytes of a hex-string RPC result as a lowercased address."""
    if not isinstance(result, str) or len(result) < 40:
        return None
    if result in ("", "0x"):
        return None
    return ("0x" + result[-40:]).lower()


class BscRpcProvider(Provider):
    name = "bsc_rpc"
    supported_chains = frozenset({Chain.BSC})
    capabilities = frozenset({CAP_RUG, CAP_TOKEN})
    requires_key = False

    async def _eth_call(self, to: str, data: str) -> str | None:
        """Single eth_call; return the hex result string, or None on revert/error."""
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [{"to": to, "data": data}, "latest"],
        }
        resp = await self.http.post_json(self.settings.rpc_url(Chain.BSC), json=body)
        if not isinstance(resp, dict) or resp.get("error") is not None:
            return None
        result = resp.get("result")
        return result if isinstance(result, str) else None

    async def get_token(self, chain: Chain, address: str) -> Token | None:
        chain = Chain.parse(chain)
        if chain != Chain.BSC:
            raise NotSupported
        addr = normalize_address(chain, address)

        try:
            dec_hex = await self._eth_call(addr, SEL_DECIMALS)
            supply_hex = await self._eth_call(addr, SEL_TOTAL_SUPPLY)
        except HttpError:
            return None

        decimals = _hex_int(dec_hex)
        if decimals is None:  # decimals() reverted -> assume standard 18
            decimals = 18
        raw_supply = _hex_int(supply_hex)
        if raw_supply is None:  # not a token / totalSupply() reverted
            return None

        return Token(
            source=self.name,
            chain=chain,
            address=addr,
            decimals=decimals,
            total_supply=raw_supply / 10 ** decimals,
            raw={"decimals": dec_hex, "totalSupply": supply_hex},
        )

    async def get_rug(self, chain: Chain, address: str) -> RugReport | None:
        chain = Chain.parse(chain)
        if chain != Chain.BSC:
            raise NotSupported
        addr = normalize_address(chain, address)

        try:
            owner_hex = await self._eth_call(addr, SEL_OWNER)
        except HttpError:
            return None

        owner_address = _hex_address(owner_hex)
        ownership_renounced = None
        flags: list[str] = []
        if owner_address is not None:
            ownership_renounced = owner_address in _DEAD_ADDRESSES
            if not ownership_renounced:
                flags.append("ownership not renounced")

        return RugReport(
            source=self.name,
            chain=chain,
            address=addr,
            risk_score=None,
            owner_address=owner_address,
            ownership_renounced=ownership_renounced,
            flags=flags,
            raw={"owner": owner_hex},
        )
