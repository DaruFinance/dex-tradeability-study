"""RugCheck.xyz, free, no API key. Solana-only token-safety report.

Capabilities: rug. Solana only.
score_normalised is 0-100 (higher = riskier). Authorities are null when revoked.
topHolders pct / lpLockedPct / transferFee.pct are PERCENT numbers; we store
top10/lp as fractions and transfer_tax as a fraction.
Docs: https://api.rugcheck.xyz/swagger/index.html
"""
from __future__ import annotations

from ..chains import Chain, normalize_address
from ..http import HttpError
from ..models import RugReport
from .base import CAP_RUG, NotSupported, Provider

BASE = "https://api.rugcheck.xyz/v1"


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


class RugCheckProvider(Provider):
    name = "rugcheck"
    supported_chains = frozenset({Chain.SOLANA})
    capabilities = frozenset({CAP_RUG})
    requires_key = False

    async def get_rug(self, chain: Chain, address: str) -> RugReport | None:
        chain = Chain.parse(chain)
        if chain != Chain.SOLANA:
            raise NotSupported
        mint = normalize_address(chain, address)
        try:
            data = await self.http.get_json(
                f"{BASE}/tokens/{mint}/report",
                cache_ttl=60,
            )
        except HttpError:
            # 404/400 for unindexed tokens.
            return None
        if not isinstance(data, dict) or not data:
            return None

        mint_auth = data.get("mintAuthority")
        freeze_auth = data.get("freezeAuthority")
        transfer_fee = data.get("transferFee") or {}

        return RugReport(
            source=self.name,
            chain=chain,
            address=mint,
            risk_score=_f(data.get("score_normalised")),
            rugged=data.get("rugged"),
            mint_authority_active=bool(mint_auth) if mint_auth is not None else None,
            freeze_authority_active=bool(freeze_auth) if freeze_auth is not None else None,
            top10_holder_pct=self._top10(data.get("topHolders") or []),
            creator_pct=None,
            holder_count=_i(data.get("totalHolders")),
            lp_locked_pct=self._lp_locked(data.get("markets") or []),
            transfer_tax=self._transfer_tax(transfer_fee),
            flags=[r.get("name") for r in (data.get("risks") or []) if isinstance(r, dict) and r.get("name")],
            raw=data,
        )

    @staticmethod
    def _top10(top_holders: list) -> float | None:
        if not top_holders:
            return None
        total = sum((_f(h.get("pct")) or 0.0) for h in top_holders[:10] if isinstance(h, dict))
        return total / 100.0

    @staticmethod
    def _lp_locked(markets: list) -> float | None:
        best = None
        for m in markets:
            if not isinstance(m, dict):
                continue
            lp = m.get("lp") or {}
            pct = _f(lp.get("lpLockedPct"))
            if pct is None:
                continue
            best = pct if best is None else max(best, pct)
        return None if best is None else best / 100.0

    @staticmethod
    def _transfer_tax(fee) -> float | None:
        if not isinstance(fee, dict) or not fee:
            return None
        pct = _f(fee.get("pct"))
        return None if pct is None else pct / 100.0
