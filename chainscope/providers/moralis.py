"""Moralis: Solana pump.fun launch & graduation event feed (requires MORALIS_API_KEY).

Capabilities: launch (single-token bonding status). Solana only.
Also exposes three direct event-feed helpers the aggregator calls outside the
base capability set: recent_launches / recent_bonding / recent_graduated, the
new/bonding/graduated pump.fun streams that are the actual tradeable signal.

Base: https://solana-gateway.moralis.io  (header X-API-Key)
Docs:
  pump.fun new/bonding/graduated . https://docs.moralis.com/get-started/tutorials/data-api/tokens-and-markets/get-pump-fun-new-bonding-and-graduated-tokens
  single-token bonding status .... https://docs.moralis.com/web3-data-api/solana/reference/get-bonding-status-by-token-address
  graduated by exchange .......... https://docs.moralis.com/web3-data-api/solana/reference/get-graduated-tokens-by-exchange
  bonding by exchange ............ https://docs.moralis.com/web3-data-api/solana/reference/get-bonding-tokens-by-exchange
  token metadata ................. https://docs.moralis.com/web3-data-api/solana/reference/get-token-metadata
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..chains import Chain
from ..http import HttpError
from ..models import Launch
from .base import CAP_LAUNCH, NotSupported, Provider

BASE = "https://solana-gateway.moralis.io"


def _f(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _i(x) -> int | None:
    try:
        return int(float(x)) if x is not None else None
    except (TypeError, ValueError):
        return None


def _parse_dt(s) -> datetime | None:
    if not s:
        return None
    txt = str(s).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _progress_to_fraction(x) -> float | None:
    """Moralis returns bonding progress as a percent (0..100); normalize to 0..1.
    Guard against an already-fractional value (<=1) just in case."""
    v = _f(x)
    if v is None:
        return None
    frac = v / 100.0 if v > 1.0 else v
    return max(0.0, min(1.0, frac))


class MoralisProvider(Provider):
    name = "moralis"
    supported_chains = frozenset({Chain.SOLANA})
    capabilities = frozenset({CAP_LAUNCH})
    requires_key = True
    key_env = "MORALIS_API_KEY"

    def _headers(self) -> dict:
        return {"X-API-Key": self.settings.get_key(self.key_env) or "",
                "accept": "application/json"}

    def _token_to_launch(self, t: dict, *, complete: bool = False,
                         progress_field: str = "bondingCurveProgress") -> Launch:
        """Map one entry from a new/bonding/graduated list payload -> Launch."""
        return Launch(
            source="moralis",
            chain=Chain.SOLANA,
            address=t.get("tokenAddress") or t.get("mint") or "",
            launchpad="pumpfun",
            name=t.get("name"),
            symbol=t.get("symbol"),
            market_cap_usd=_f(t.get("fullyDilutedValuation")) or _f(t.get("marketCap")),
            bonding_curve_progress=(1.0 if complete
                                    else _progress_to_fraction(
                                        t.get(progress_field, t.get("bondingProgress")))),
            complete=complete or None,
            graduated_at=_parse_dt(t.get("graduatedAt")) if complete else None,
            created_at=_parse_dt(t.get("createdAt")),
            target_dex="pumpswap",
            raw=t,
        )

    # ---- launch (base capability) ----

    async def get_launch(self, chain: Chain, address: str) -> Launch | None:
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported
        # bonding-status -> {"mint", "bondingProgress" (0..100)}; metadata -> name/symbol.
        status: dict = {}
        try:
            status = await self.http.get_json(
                f"{BASE}/token/mainnet/{address}/bonding-status",
                headers=self._headers(), cache_ttl=30,
            ) or {}
        except HttpError as exc:
            if exc.status == 404:
                return None
            raise

        meta: dict = {}
        try:
            meta = await self.http.get_json(
                f"{BASE}/token/mainnet/{address}/metadata",
                headers=self._headers(), cache_ttl=30,
            ) or {}
        except HttpError as exc:
            if exc.status != 404:
                raise

        progress = _progress_to_fraction(
            status.get("bondingCurveProgress", status.get("bondingProgress"))
        )
        graduated_flag = status.get("graduated")
        if graduated_flag is None:
            graduated_flag = status.get("isGraduated")
        complete = bool(graduated_flag) or (progress is not None and progress >= 1.0)

        return Launch(
            source="moralis",
            chain=Chain.SOLANA,
            address=address,
            launchpad="pumpfun",
            name=meta.get("name"),
            symbol=meta.get("symbol"),
            market_cap_usd=_f(meta.get("marketCap")) or _f(meta.get("fullyDilutedValue")),
            bonding_curve_progress=progress,
            complete=complete or None,
            target_dex="pumpswap",
            raw={"bonding_status": status, "metadata": meta},
        )

    # ---- direct event-feed helpers (NOT routed through base capabilities) ----

    async def _exchange_feed(self, path: str, limit: int) -> list[dict]:
        try:
            data = await self.http.get_json(
                f"{BASE}/token/mainnet/exchange/pumpfun/{path}",
                params={"limit": max(1, min(100, int(limit)))},
                headers=self._headers(), cache_ttl=30,
            )
        except HttpError as exc:
            if exc.status == 404:
                return []
            raise
        if isinstance(data, list):
            return data
        return (data or {}).get("result") or []

    async def recent_launches(self, chain: Chain, limit: int = 100) -> list[Launch]:
        """GET /token/mainnet/exchange/pumpfun/new: newly created pump.fun tokens."""
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported
        rows = await self._exchange_feed("new", limit)
        return [self._token_to_launch(t) for t in rows]

    async def recent_bonding(self, chain: Chain, limit: int = 100) -> list[Launch]:
        """GET /token/mainnet/exchange/pumpfun/bonding, tokens mid bonding curve."""
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported
        rows = await self._exchange_feed("bonding", limit)
        return [self._token_to_launch(t) for t in rows]

    async def recent_graduated(self, chain: Chain, limit: int = 100) -> list[Launch]:
        """GET /token/mainnet/exchange/pumpfun/graduated, graduated tokens (complete=True)."""
        if Chain.parse(chain) != Chain.SOLANA:
            raise NotSupported
        rows = await self._exchange_feed("graduated", limit)
        return [self._token_to_launch(t, complete=True) for t in rows]
