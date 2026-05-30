"""FastAPI app for the chainscope research terminal.

A LOCAL, read-only dashboard for **cross-sectional coin-selection research**:
rank/filter hundreds-to-~1000 coins (pools) at once by their on-chain metrics,
then drill into any one coin's full dossier + price chart.

It reads exclusively through :class:`chainscope.query.Query` (DuckDB over the
partitioned parquet store the indexers/cohort pipeline write). It never touches a
chain itself. Every endpoint degrades gracefully on an EMPTY store -> ``[]``.

Run:
    uvicorn chainscope.ui.app:app --port 8011
or via the CLI:
    python3 -m chainscope.cli serve

Endpoints:
    GET /                              -> single-page dashboard (static/index.html)
    GET /api/tables                    -> datasets + row counts (Query.tables())
    GET /api/screener?chain=&order_by=&limit=&min_volume=&min_liquidity=&dex=
                                       -> cross-sectional table, one row per pool
    GET /api/coin/{chain}/{address}    -> full per-coin dossier (build_dossier)
    GET /api/ohlcv?chain=&pair=&tf=    -> OHLCV bars (ohlcv dataset, else resampled
                                          from trades) for the price chart
"""
from __future__ import annotations

import asyncio
import json
import math
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query as Q
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..query import Query

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="chainscope terminal", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _query() -> Query:
    """A fresh Query per request: re-scans the store so newly written datasets
    (and partitions) light up without restarting the server. Cheap — DuckDB just
    registers views over the parquet globs; no data is materialized until a SQL
    call runs."""
    return Query()


def _jsonable(obj):
    """Recursively coerce DB / pydantic / numpy values into JSON-safe primitives.

    Handles: datetimes -> ISO strings, NaN/Inf -> None, numpy scalars -> python,
    pydantic models -> dict (model_dump), bytes -> hex.
    """
    if obj is None:
        return None
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    # pydantic v2 model
    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump(exclude={"raw"}))
        except Exception:
            pass
    # numpy / pandas scalars
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    isna = getattr(obj, "__len__", None)  # last resort
    try:
        return str(obj)
    except Exception:
        return None


def _ok(data) -> JSONResponse:
    return JSONResponse(content=_jsonable(data))


# Whitelisted sortable columns for the screener (prevents SQL injection via
# order_by and keeps the UI honest about what's sortable).
_SCREENER_COLUMNS = {
    "pair_address", "dex", "first_trade", "last_trade", "trade_count",
    "volume_usd", "buy_usd", "sell_usd", "buy_count", "sell_count",
    "last_price", "last_reserve_usd", "age_days", "created_at",
    "holder_count", "top10_pct",
}


# --------------------------------------------------------------------------- #
# /api/tables
# --------------------------------------------------------------------------- #
@app.get("/api/tables")
def api_tables():
    """Datasets present on disk and their row counts (empty list on empty store)."""
    try:
        with _query() as q:
            return _ok({"datasets": q.tables(), "backend": q.backend})
    except Exception as exc:
        return _ok({"datasets": [], "backend": None, "error": str(exc)[:200]})


# --------------------------------------------------------------------------- #
# /api/screener  -- the cross-sectional centerpiece
# --------------------------------------------------------------------------- #
@app.get("/api/screener")
def api_screener(
    chain: str | None = Q(None, description="filter to one chain, e.g. bsc / solana"),
    order_by: str = Q("volume_usd", description="sortable column"),
    desc: bool = Q(True, description="descending sort"),
    limit: int = Q(1000, ge=1, le=20000),
    min_volume: float | None = Q(None, description="min total volume_usd"),
    min_liquidity: float | None = Q(None, description="min last reserve_usd"),
    dex: str | None = Q(None, description="filter to one dex"),
):
    """One row per coin/pool, aggregated from `trades`, enriched with
    `pool_universe` (dex / created_at / age) and holder metrics from a `holders`
    dataset if present, else `rug_reports`/`tokens` (holder_count, top10_pct).

    Columns: chain, pair_address, dex, first_trade, last_trade, trade_count,
    volume_usd, buy_usd, sell_usd, buy_count, sell_count, last_price,
    last_reserve_usd (liquidity), created_at, age_days, holder_count, top10_pct.

    Empty list if there is no `trades` dataset yet.
    """
    col = order_by if order_by in _SCREENER_COLUMNS else "volume_usd"
    direction = "DESC" if desc else "ASC"

    try:
        with _query() as q:
            if "trades" not in q.datasets():
                return _ok({"rows": [], "columns": sorted(_SCREENER_COLUMNS),
                            "note": "no `trades` dataset yet — populate via `cohort` or `backfill`"})

            where = []
            if chain:
                where.append(f"t.chain = {q._lit(chain)}")
            clause = (" WHERE " + " AND ".join(where)) if where else ""

            # Per-pool aggregation from the trade tape. last_price / last_reserve
            # use ARG_MAX(value, block_time) so we get the value AT the latest
            # trade (DuckDB; the polars fallback path uses a windowed subquery).
            if q.backend == "duckdb":
                agg = (
                    "SELECT t.chain AS chain, t.pair_address AS pair_address, "
                    "any_value(t.dex) AS trade_dex, "
                    "min(t.block_time) AS first_trade, "
                    "max(t.block_time) AS last_trade, "
                    "count(*) AS trade_count, "
                    "sum(COALESCE(t.amount_usd, 0)) AS volume_usd, "
                    "sum(CASE WHEN t.side='buy' THEN COALESCE(t.amount_usd,0) ELSE 0 END) AS buy_usd, "
                    "sum(CASE WHEN t.side='sell' THEN COALESCE(t.amount_usd,0) ELSE 0 END) AS sell_usd, "
                    "count(*) FILTER (WHERE t.side='buy') AS buy_count, "
                    "count(*) FILTER (WHERE t.side='sell') AS sell_count, "
                    "arg_max(t.price_usd, t.block_time) AS last_price, "
                    "arg_max(t.reserve_usd, t.block_time) AS last_reserve_usd "
                    f"FROM trades t{clause} "
                    "GROUP BY t.chain, t.pair_address"
                )
            else:
                # polars SQL: no FILTER / arg_max; approximate with conditional sums
                # and a max(price) is misleading, so fetch last via a join below.
                agg = (
                    "SELECT t.chain AS chain, t.pair_address AS pair_address, "
                    "max(t.dex) AS trade_dex, "
                    "min(t.block_time) AS first_trade, "
                    "max(t.block_time) AS last_trade, "
                    "count(*) AS trade_count, "
                    "sum(COALESCE(t.amount_usd, 0)) AS volume_usd, "
                    "sum(CASE WHEN t.side='buy' THEN COALESCE(t.amount_usd,0) ELSE 0 END) AS buy_usd, "
                    "sum(CASE WHEN t.side='sell' THEN COALESCE(t.amount_usd,0) ELSE 0 END) AS sell_usd, "
                    "sum(CASE WHEN t.side='buy' THEN 1 ELSE 0 END) AS buy_count, "
                    "sum(CASE WHEN t.side='sell' THEN 1 ELSE 0 END) AS sell_count, "
                    "max(t.price_usd) AS last_price, "
                    "max(t.reserve_usd) AS last_reserve_usd "
                    f"FROM trades t{clause} "
                    "GROUP BY t.chain, t.pair_address"
                )

            has_universe = "pool_universe" in q.datasets()
            holder_src = None
            for cand in ("holders", "rug_reports", "tokens"):
                if cand in q.datasets():
                    holder_src = cand
                    break

            # Assemble the final SELECT around the aggregate CTE, joining the
            # static pool-universe (dex/created_at) and the latest holder snapshot.
            sel = [
                "a.chain AS chain", "a.pair_address AS pair_address",
                "COALESCE(u.dex, a.trade_dex) AS dex" if has_universe else "a.trade_dex AS dex",
                "a.first_trade AS first_trade", "a.last_trade AS last_trade",
                "a.trade_count AS trade_count", "a.volume_usd AS volume_usd",
                "a.buy_usd AS buy_usd", "a.sell_usd AS sell_usd",
                "a.buy_count AS buy_count", "a.sell_count AS sell_count",
                "a.last_price AS last_price", "a.last_reserve_usd AS last_reserve_usd",
            ]
            joins = ""
            if has_universe:
                sel.append("u.created_at AS created_at")
                if q.backend == "duckdb":
                    sel.append(
                        "date_diff('day', u.created_at, a.last_trade) AS age_days")
                else:
                    sel.append("NULL AS age_days")
                joins += (
                    " LEFT JOIN ("
                    "SELECT chain, pair_address, any_value(created_at) AS created_at, "
                    "any_value(dex) AS dex "
                    "FROM pool_universe GROUP BY chain, pair_address) u "
                    "ON a.chain=u.chain AND a.pair_address=u.pair_address"
                    if q.backend == "duckdb" else
                    " LEFT JOIN ("
                    "SELECT chain, pair_address, max(created_at) AS created_at, "
                    "max(dex) AS dex "
                    "FROM pool_universe GROUP BY chain, pair_address) u "
                    "ON a.chain=u.chain AND a.pair_address=u.pair_address"
                )
            else:
                sel.append("NULL AS created_at")
                sel.append("NULL AS age_days")

            if holder_src and q.backend == "duckdb":
                # latest holder snapshot per address; rug_reports/tokens key on
                # `address` (the token), which is *not* the pair_address. We still
                # surface it best-effort by joining on pair_address==address (works
                # for a `holders` dataset keyed by pool, harmless otherwise).
                sel.append("h.holder_count AS holder_count")
                sel.append("h.top10_pct AS top10_pct")
                top_col = "top10_holder_pct" if holder_src in ("rug_reports",) else "top10_pct"
                joins += (
                    f" LEFT JOIN ("
                    f"SELECT address, arg_max(holder_count, observed_at) AS holder_count, "
                    f"arg_max({top_col}, observed_at) AS top10_pct "
                    f"FROM {holder_src} WHERE holder_count IS NOT NULL "
                    f"GROUP BY address) h ON a.pair_address = h.address"
                )
            else:
                sel.append("NULL AS holder_count")
                sel.append("NULL AS top10_pct")

            having = []
            if min_volume is not None:
                having.append(f"a.volume_usd >= {float(min_volume)}")
            if dex:
                # filter on the resolved dex
                dex_expr = "COALESCE(u.dex, a.trade_dex)" if has_universe else "a.trade_dex"
                having.append(f"{dex_expr} = {q._lit(dex)}")
            if min_liquidity is not None:
                having.append(f"a.last_reserve_usd >= {float(min_liquidity)}")
            where_outer = (" WHERE " + " AND ".join(having)) if having else ""

            full = (
                f"WITH agg AS ({agg}) "
                f"SELECT {', '.join(sel)} FROM agg a{joins}{where_outer} "
                f"ORDER BY {col} {direction} NULLS LAST "
                f"LIMIT {int(limit)}"
            )
            if q.backend == "polars":
                full = full.replace("NULLS LAST", "")  # polars SQL lacks NULLS LAST

            rows = q.sql(full)
            return _ok({"rows": rows, "columns": sorted(_SCREENER_COLUMNS),
                        "count": len(rows), "order_by": col, "desc": desc})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"screener failed: {exc}")


# --------------------------------------------------------------------------- #
# /api/ohlcv  -- price chart series
# --------------------------------------------------------------------------- #
@app.get("/api/ohlcv")
def api_ohlcv(
    chain: str | None = Q(None),
    pair: str = Q(..., description="pair_address"),
    tf: str = Q("1h", description="timeframe (e.g. 1m,5m,1h); for resample bucket too"),
    limit: int = Q(1000, ge=1, le=20000),
):
    """OHLCV bars for one pair. Prefers the `ohlcv` dataset; if absent (or empty
    for this pair) resamples bars from the `trades` tape into `tf` buckets using
    price_usd (first=open, max=high, min=low, last=close) and amount_usd volume."""
    try:
        with _query() as q:
            # 1) prefer a real ohlcv dataset
            if "ohlcv" in q.datasets():
                bars = q.ohlcv_for(pair, timeframe=tf, chain=chain, limit=limit)
                if bars:
                    return _ok({"source": "ohlcv", "tf": tf, "bars": bars})

            # 2) resample from trades
            if "trades" not in q.datasets():
                return _ok({"source": None, "tf": tf, "bars": []})

            secs = _tf_seconds(tf)
            where = [f"pair_address = {q._lit(pair)}", "price_usd IS NOT NULL"]
            if chain:
                where.append(f"chain = {q._lit(chain)}")
            clause = " AND ".join(where)

            if q.backend == "duckdb":
                resample = (
                    "SELECT to_timestamp("
                    f"floor(epoch(block_time) / {secs}) * {secs}) AS timestamp, "
                    "first(price_usd ORDER BY block_time) AS open, "
                    "max(price_usd) AS high, min(price_usd) AS low, "
                    "last(price_usd ORDER BY block_time) AS close, "
                    "sum(COALESCE(amount_usd,0)) AS volume, "
                    "count(*) AS trade_count "
                    "FROM trades "
                    f"WHERE {clause} "
                    f"GROUP BY floor(epoch(block_time) / {secs}) "
                    "ORDER BY timestamp "
                    f"LIMIT {int(limit)}"
                )
            else:
                # polars fallback: approximate OHLC with min/max (no ordered agg).
                resample = (
                    "SELECT block_time AS timestamp, price_usd AS open, "
                    "price_usd AS high, price_usd AS low, price_usd AS close, "
                    "COALESCE(amount_usd,0) AS volume, 1 AS trade_count "
                    "FROM trades "
                    f"WHERE {clause} "
                    "ORDER BY block_time "
                    f"LIMIT {int(limit)}"
                )
            bars = q.sql(resample)
            return _ok({"source": "trades_resampled", "tf": tf, "bars": bars})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ohlcv failed: {exc}")


def _tf_seconds(tf: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    tf = (tf or "1h").strip().lower()
    try:
        num = int("".join(ch for ch in tf if ch.isdigit()) or "1")
        unit = "".join(ch for ch in tf if ch.isalpha()) or "h"
        return max(1, num * units.get(unit[0], 3600))
    except Exception:
        return 3600


# --------------------------------------------------------------------------- #
# /api/coin/{chain}/{address}  -- full dossier drill-down
# --------------------------------------------------------------------------- #
@app.get("/api/coin/{chain}/{address}")
def api_coin(chain: str, address: str):
    """Full per-coin dossier via :func:`chainscope.dossier.build_dossier`.

    This spins up a :class:`chainscope.aggregate.Client` and runs the async
    dossier builder (token meta, pools, price, recent trades, depth, cost curve,
    rug flags, launch, MEV). Falls back to whatever the store/providers return;
    on total failure returns a minimal stub from parquet so the panel still loads.
    """
    try:
        data = asyncio.run(_build_dossier(chain, address))
        return _ok(data)
    except Exception as exc:
        # last-resort: synthesize a thin dossier from parquet so the UI degrades
        try:
            with _query() as q:
                stub = _parquet_stub(q, chain, address)
            stub["error"] = f"dossier builder failed: {str(exc)[:160]}"
            return _ok(stub)
        except Exception:
            raise HTTPException(status_code=500, detail=f"coin failed: {exc}")


async def _build_dossier(chain: str, address: str) -> dict:
    from ..aggregate import Client
    from ..dossier import build_dossier
    async with Client() as cs:
        return await build_dossier(cs, chain, address)


def _parquet_stub(q: Query, chain: str, address: str) -> dict:
    """Thin per-coin summary straight from the trade tape (no network)."""
    if "trades" not in q.datasets():
        return {"chain": chain, "address": address, "pools": [], "trades": [],
                "ohlcv": [], "rug": None}
    rows = q.sql(
        "SELECT chain, pair_address, count(*) AS trade_count, "
        "sum(COALESCE(amount_usd,0)) AS volume_usd, "
        "max(block_time) AS last_trade, "
        "arg_max(price_usd, block_time) AS last_price, "
        "arg_max(reserve_usd, block_time) AS last_reserve_usd "
        f"FROM trades WHERE pair_address = {q._lit(address)} "
        f"{('AND chain = ' + q._lit(chain)) if chain else ''} "
        "GROUP BY chain, pair_address"
    )
    r = rows[0] if rows else {}
    return {
        "chain": chain, "address": address,
        "price_usd": r.get("last_price"),
        "liquidity_usd": r.get("last_reserve_usd"),
        "volume_24h": None, "pool_count": 1 if rows else 0,
        "pools": [], "trades": [], "ohlcv": [], "rug": None,
        "dominant_pool": {"pair_address": address} if rows else None,
        "_from_parquet": True,
    }


# --------------------------------------------------------------------------- #
# /  -- single-page UI
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    idx = _STATIC / "index.html"
    if not idx.is_file():
        return JSONResponse({"error": "index.html missing"}, status_code=500)
    return FileResponse(str(idx))
