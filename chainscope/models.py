"""Canonical, chain-agnostic data models. Every provider normalizes its raw
payload into one of these so downstream research code sees a single schema
regardless of source chain or API.

Every record carries `source` (which provider produced it) and `observed_at`
(UTC wall-clock when we fetched it) so stored data is point-in-time and free of
lookahead: a row never contains information that did not exist at observed_at.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from .chains import Chain


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Record(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    source: str = "unknown"
    observed_at: datetime = Field(default_factory=_utcnow)
    raw: dict | None = Field(default=None, repr=False)

    def to_row(self) -> dict:
        """Flat dict for parquet/dataframe storage (excludes the raw payload)."""
        return self.model_dump(exclude={"raw"})


class Token(Record):
    chain: Chain
    address: str
    symbol: str | None = None
    name: str | None = None
    decimals: int | None = None
    total_supply: float | None = None
    price_usd: float | None = None
    price_native: float | None = None
    market_cap: float | None = None
    fdv: float | None = None
    liquidity_usd: float | None = None
    volume_5m: float | None = None
    volume_1h: float | None = None
    volume_24h: float | None = None
    price_change_5m: float | None = None
    price_change_1h: float | None = None
    price_change_24h: float | None = None
    txns_24h_buys: int | None = None
    txns_24h_sells: int | None = None
    holder_count: int | None = None
    pair_count: int | None = None
    created_at: datetime | None = None
    image_url: str | None = None
    websites: list[str] = Field(default_factory=list)
    socials: list[str] = Field(default_factory=list)


class Pool(Record):
    chain: Chain
    dex: str | None = None
    pair_address: str
    base_address: str | None = None
    base_symbol: str | None = None
    quote_address: str | None = None
    quote_symbol: str | None = None
    price_usd: float | None = None
    price_native: float | None = None
    liquidity_usd: float | None = None
    liquidity_base: float | None = None
    liquidity_quote: float | None = None
    fdv: float | None = None
    market_cap: float | None = None
    volume_5m: float | None = None
    volume_1h: float | None = None
    volume_24h: float | None = None
    price_change_24h: float | None = None
    txns_24h_buys: int | None = None
    txns_24h_sells: int | None = None
    created_at: datetime | None = None
    url: str | None = None


class OHLCV(Record):
    """A price bar. For backtesting, prefer the finest granularity available and
    use high/low (not close) for TP/SL fill logic. `reserve_usd` (when the source
    provides it) is the pool depth at bar time, needed to model slippage."""
    chain: Chain
    pair_address: str
    timeframe: str            # e.g. "1m", "5m", "1h", "1d"
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    reserve_usd: float | None = None     # pool liquidity at bar time (cost model)
    trade_count: int | None = None


class Trade(Record):
    """An individual on-chain swap: the ground truth for fill/slippage modeling.
    `side` is from the base token's perspective ("buy" = base bought with quote)."""
    chain: Chain
    pair_address: str
    block_time: datetime
    tx_hash: str | None = None
    side: str | None = None            # "buy" | "sell"
    dex: str | None = None
    price_usd: float | None = None
    price_native: float | None = None
    amount_base: float | None = None
    amount_quote: float | None = None
    amount_usd: float | None = None
    reserve_usd: float | None = None   # pool depth at/after the trade, if available
    maker: str | None = None
    block_number: int | None = None    # EVM block / Solana slot
    log_index: int | None = None       # intra-block ordering (EVM logIndex / position in slot)
    fee_bps: float | None = None        # pool fee tier actually charged
    gas_native: float | None = None     # gas paid in native token (BNB/SOL), exact when available
    gas_usd: float | None = None
    sandwiched: bool | None = None      # flagged by the MEV detector


class PoolCreation(Record):
    """A pool/pair creation event: the unit of the historical token universe.
    Scanning these across all of history (incl. pools now dead/rugged) is what
    makes a cross-sectional backtest survivorship-free."""
    chain: Chain
    dex: str | None = None
    pair_address: str
    token0: str | None = None
    token1: str | None = None
    fee_bps: float | None = None
    created_block: int | None = None
    created_at: datetime | None = None
    creator: str | None = None


class HolderSnapshot(Record):
    """Point-in-time holder distribution. `address` is the screener's pair_address (so it
    joins the cross-sectional view); `token` is the underlying coin for provenance."""
    chain: Chain
    address: str
    token: str | None = None
    holder_count: int | None = None
    top10_pct: float | None = None
    hhi: float | None = None


class RugReport(Record):
    """Normalized token-safety profile. risk_score is 0-100, higher = riskier."""
    chain: Chain
    address: str
    risk_score: float | None = None
    rugged: bool | None = None
    is_honeypot: bool | None = None
    buy_tax: float | None = None          # fraction, 0.05 == 5%
    sell_tax: float | None = None
    transfer_tax: float | None = None
    # authority / ownership
    mint_authority_active: bool | None = None     # solana: mint authority not revoked
    freeze_authority_active: bool | None = None    # solana: freeze authority not revoked
    mintable: bool | None = None                   # evm: supply can be inflated
    owner_address: str | None = None
    ownership_renounced: bool | None = None
    can_take_back_ownership: bool | None = None
    hidden_owner: bool | None = None
    # liquidity safety
    lp_locked_pct: float | None = None             # fraction 0..1
    lp_burned_pct: float | None = None             # fraction 0..1
    # distribution
    top10_holder_pct: float | None = None          # fraction 0..1
    creator_pct: float | None = None
    holder_count: int | None = None
    # contract
    is_open_source: bool | None = None
    is_proxy: bool | None = None
    transfer_pausable: bool | None = None
    has_blacklist: bool | None = None
    has_whitelist: bool | None = None
    anti_whale: bool | None = None
    flags: list[str] = Field(default_factory=list)  # human-readable risk flags


class Launch(Record):
    """A launchpad / bonding-curve token and its graduation status."""
    chain: Chain
    address: str
    launchpad: str | None = None        # "pumpfun", "fourmeme", "letsbonk", ...
    name: str | None = None
    symbol: str | None = None
    creator: str | None = None
    created_at: datetime | None = None
    market_cap_usd: float | None = None
    bonding_curve_progress: float | None = None   # fraction 0..1 (1 == graduated)
    virtual_sol_reserves: float | None = None
    virtual_token_reserves: float | None = None
    real_sol_reserves: float | None = None
    real_token_reserves: float | None = None
    complete: bool | None = None                   # graduated?
    graduated_at: datetime | None = None
    migrated_pool: str | None = None
    target_dex: str | None = None                  # "pumpswap", "raydium", "pancakeswap", ...


class Ticker(Record):
    """One trading venue/pair for a token (CEX or DEX)."""
    exchange: str
    exchange_type: str | None = None    # "cex" or "dex"
    base: str | None = None
    target: str | None = None
    price_usd: float | None = None
    volume_usd: float | None = None
    trust_score: str | None = None
    trade_url: str | None = None


class ExchangePresence(Record):
    """Where a token currently trades, across CEX and DEX venues."""
    chain: Chain | None = None
    address: str | None = None
    coingecko_id: str | None = None
    symbol: str | None = None
    cex_exchanges: list[str] = Field(default_factory=list)
    dex_exchanges: list[str] = Field(default_factory=list)
    tickers: list[Ticker] = Field(default_factory=list, repr=False)

    def to_row(self) -> dict:
        d = self.model_dump(exclude={"raw", "tickers"})
        d["cex_count"] = len(self.cex_exchanges)
        d["dex_count"] = len(self.dex_exchanges)
        return d
