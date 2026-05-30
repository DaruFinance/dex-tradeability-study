"""Honeypot.is, free, no API key. BSC-only honeypot / tax simulator.

Capabilities: rug. BSC only (the v2 endpoint also covers a few other EVM chains,
but chainscope only models BSC here).
Taxes come back as PERCENT numbers (5.0 == 5%); we store fractions.
Docs: https://docs.honeypot.is/
"""
from __future__ import annotations

from ..chains import Chain, normalize_address
from ..http import HttpError
from ..models import RugReport
from .base import CAP_RUG, NotSupported, Provider

BASE = "https://api.honeypot.is/v2"


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


def _pct_to_frac(x) -> float | None:
    """Honeypot.is reports tax as a percent number; convert to fraction."""
    v = _f(x)
    return None if v is None else v / 100.0


class HoneypotProvider(Provider):
    name = "honeypot"
    supported_chains = frozenset({Chain.BSC})
    capabilities = frozenset({CAP_RUG})
    requires_key = False

    async def get_rug(self, chain: Chain, address: str) -> RugReport | None:
        chain = Chain.parse(chain)
        if chain != Chain.BSC:
            raise NotSupported
        addr = normalize_address(chain, address)
        try:
            data = await self.http.get_json(
                f"{BASE}/IsHoneypot",
                params={"address": addr, "chainID": "56"},
                cache_ttl=60,
            )
        except HttpError:
            return None
        if not isinstance(data, dict) or not data:
            return None

        summary = data.get("summary") or {}
        hp = data.get("honeypotResult") or {}
        sim = data.get("simulationResult") or {}
        code = data.get("contractCode") or {}
        token = data.get("token") or {}

        flags = list(summary.get("flags") or [])
        reason = hp.get("honeypotReason")
        if reason and reason not in flags:
            flags.append(reason)

        return RugReport(
            source=self.name,
            chain=chain,
            address=addr,
            risk_score=_f(summary.get("riskLevel")),
            is_honeypot=hp.get("isHoneypot"),
            buy_tax=_pct_to_frac(sim.get("buyTax")),
            sell_tax=_pct_to_frac(sim.get("sellTax")),
            transfer_tax=_pct_to_frac(sim.get("transferTax")),
            is_open_source=code.get("openSource"),
            is_proxy=code.get("isProxy"),
            holder_count=_i(token.get("totalHolders")),
            flags=flags,
            raw=data,
        )
