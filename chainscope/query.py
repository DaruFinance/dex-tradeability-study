"""SQL analytics layer over the partitioned parquet datasets the indexers and
backfiller write.

The `ParquetStore` lays every dataset out as
``{data_dir}/{dataset}/chain=.../dt=.../*.parquet`` — a hive-partitioned tree
where ``chain`` and ``dt`` (the event date) are encoded in the directory names.
This module points DuckDB at each of those trees with ``hive_partitioning=true``
so the partition columns come back as real, queryable columns, then exposes the
whole store as a set of SQL views you can run arbitrary queries against::

    from chainscope.query import Query
    q = Query()                       # uses get_settings().data_dir
    q.tables()                        # [{"name": "trades", "rows": 1234}, ...]
    q.sql("SELECT side, count(*) FROM trades GROUP BY side")
    q.df("SELECT * FROM ohlcv LIMIT 10")
    q.top_tokens_by_volume("bsc", since_dt="2026-05-01")

DuckDB is the right tool here (it reads partitioned parquet natively and pushes
predicates into the partition pruning). If DuckDB cannot be imported the class
transparently falls back to a polars ``SQLContext`` (``Query.backend == "polars"``);
the polars path lacks DuckDB-specific functions but covers the convenience
methods and straightforward SQL.

The known datasets mirror what `ParquetStore.write` is called with across the
codebase (backfill / universe / cli): tokens, pools, ohlcv, trades, rug_reports,
launches, pool_universe — plus a couple of forward-looking names (listings,
holders) that are registered only if they happen to exist on disk.
"""
from __future__ import annotations

import glob
from pathlib import Path

from .config import get_settings

# Datasets the store is known to write, in a sensible display order. Any name
# here that has no data on disk is simply skipped (no crash on an empty store);
# any *other* directory found under data_dir that contains parquet is registered
# too, so new indexer datasets light up automatically.
KNOWN_DATASETS = [
    "tokens",
    "pools",
    "pool_universe",
    "ohlcv",
    "trades",
    "rug_reports",
    "launches",
    "listings",
    "holders",
]

try:  # DuckDB is the preferred backend.
    import duckdb  # type: ignore

    _HAVE_DUCKDB = True
except Exception:  # pragma: no cover - exercised only when duckdb is absent
    duckdb = None  # type: ignore
    _HAVE_DUCKDB = False


class Query:
    """Read-only SQL surface over the parquet store.

    Parameters
    ----------
    data_dir:
        Root of the parquet store. Defaults to ``get_settings().data_dir``.
    """

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else get_settings().data_dir
        # name -> on-disk dataset directory, only for datasets that have parquet
        self._datasets: dict[str, Path] = {}
        self.backend = "duckdb" if _HAVE_DUCKDB else "polars"
        self._con = duckdb.connect(database=":memory:") if _HAVE_DUCKDB else None
        self.refresh()

    # ------------------------------------------------------------------ #
    # discovery / registration
    # ------------------------------------------------------------------ #
    def _glob_for(self, name: str) -> str:
        """Recursive parquet glob for one dataset directory."""
        return str(self.data_dir / name / "**" / "*.parquet")

    def _has_parquet(self, name: str) -> bool:
        return bool(glob.glob(self._glob_for(name), recursive=True))

    def _discover(self) -> list[str]:
        """All dataset names with parquet on disk: the known ones first (for a
        stable order), then any other directory under data_dir that has parquet."""
        names: list[str] = [n for n in KNOWN_DATASETS if self._has_parquet(n)]
        seen = set(names)
        if self.data_dir.is_dir():
            for child in sorted(self.data_dir.iterdir()):
                if child.is_dir() and child.name not in seen and self._has_parquet(child.name):
                    names.append(child.name)
                    seen.add(child.name)
        return names

    def refresh(self) -> dict[str, Path]:
        """(Re)scan the store and (re)register a SQL view per existing dataset.

        Each view reads ``{data_dir}/{name}/**/*.parquet`` with
        ``hive_partitioning=true`` so the ``chain`` and ``dt`` partition columns
        are available. Datasets that don't exist yet are skipped. Returns the
        registered name -> directory map.
        """
        self._datasets = {}
        for name in self._discover():
            self._datasets[name] = self.data_dir / name
        if self.backend == "duckdb":
            self._register_duckdb_views()
        return dict(self._datasets)

    def _register_duckdb_views(self) -> None:
        assert self._con is not None
        for name in list(self._datasets):
            pattern = self._glob_for(name).replace("'", "''")
            # union_by_name tolerates schema drift across appended parquet files
            # (older writes may be missing columns added later).
            self._con.execute(
                f'CREATE OR REPLACE VIEW "{name}" AS '
                f"SELECT * FROM read_parquet('{pattern}', "
                f"hive_partitioning=true, union_by_name=true)"
            )

    # ------------------------------------------------------------------ #
    # core query API
    # ------------------------------------------------------------------ #
    def datasets(self) -> list[str]:
        """Names of the registered (existing) dataset views."""
        return list(self._datasets)

    def sql(self, query: str) -> list[dict]:
        """Run a SQL query and return rows as a list of dicts."""
        df = self.df(query)
        return df.to_dict(orient="records")

    def df(self, query: str):
        """Run a SQL query and return a pandas DataFrame."""
        if self.backend == "duckdb":
            assert self._con is not None
            return self._con.execute(query).fetch_df()
        # polars fallback
        return self._polars_df(query)

    def _polars_ctx(self):
        import polars as pl  # local import; only needed on the fallback path

        ctx = pl.SQLContext()
        for name, root in self._datasets.items():
            pattern = self._glob_for(name)
            lf = pl.scan_parquet(pattern, hive_partitioning=True)
            ctx.register(name, lf)
        return ctx

    def _polars_df(self, query: str):
        ctx = self._polars_ctx()
        return ctx.execute(query, eager=True).to_pandas()

    def tables(self) -> list[dict]:
        """Available dataset views with their row counts.

        Returns a list of ``{"name": ..., "rows": ...}`` dicts. A dataset whose
        files can't be read (corrupt / mid-write) reports ``rows=None`` rather
        than raising, so this stays usable on a live, growing store.
        """
        out: list[dict] = []
        for name in self._datasets:
            try:
                rows = self.sql(f'SELECT count(*) AS n FROM "{name}"')[0]["n"]
                rows = int(rows)
            except Exception:
                rows = None
            out.append({"name": name, "rows": rows})
        return out

    # ------------------------------------------------------------------ #
    # convenience analytics — each is resilient to a missing dataset
    # ------------------------------------------------------------------ #
    def _missing(self, *names: str) -> bool:
        return any(n not in self._datasets for n in names)

    @staticmethod
    def _lit(value) -> str:
        """SQL string literal with single-quote escaping."""
        return "'" + str(value).replace("'", "''") + "'"

    def top_tokens_by_volume(
        self, chain: str | None = None, since_dt: str | None = None, limit: int = 25
    ) -> list[dict]:
        """Most-traded pairs by total USD volume from the `trades` dataset.

        Sums ``amount_usd`` per ``pair_address`` (the on-chain unit a trade is
        attributed to), optionally scoped to a chain and to trades on/after
        ``since_dt`` (a ``YYYY-MM-DD`` partition value). Empty list if there is
        no `trades` dataset yet.
        """
        if self._missing("trades"):
            return []
        where = []
        if chain:
            where.append(f"chain = {self._lit(chain)}")
        if since_dt:
            where.append(f"dt >= {self._lit(since_dt)}")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        q = (
            "SELECT chain, pair_address, "
            "count(*) AS trade_count, "
            "sum(COALESCE(amount_usd, 0)) AS volume_usd, "
            "sum(CASE WHEN side = 'buy' THEN COALESCE(amount_usd, 0) ELSE 0 END) AS buy_usd, "
            "sum(CASE WHEN side = 'sell' THEN COALESCE(amount_usd, 0) ELSE 0 END) AS sell_usd "
            f"FROM trades{clause} "
            "GROUP BY chain, pair_address "
            "ORDER BY volume_usd DESC "
            f"LIMIT {int(limit)}"
        )
        return self.sql(q)

    def pool_liquidity_timeseries(self, pair: str, chain: str | None = None) -> list[dict]:
        """Liquidity-over-time for one pool from the `pools` snapshots.

        Returns one row per ``observed_at`` snapshot of the pair, with the USD
        liquidity / price / 24h volume we recorded at that time. Falls back to
        an empty list if the `pools` dataset doesn't exist.
        """
        if self._missing("pools"):
            return []
        where = [f"pair_address = {self._lit(pair)}"]
        if chain:
            where.append(f"chain = {self._lit(chain)}")
        clause = " AND ".join(where)
        q = (
            "SELECT chain, pair_address, observed_at, dt, "
            "liquidity_usd, price_usd, volume_24h "
            "FROM pools "
            f"WHERE {clause} "
            "ORDER BY observed_at"
        )
        return self.sql(q)

    def ohlcv_for(self, pair: str, timeframe: str | None = None, chain: str | None = None,
                  limit: int = 1000) -> list[dict]:
        """OHLCV bars for one pair, optionally filtered to a timeframe."""
        if self._missing("ohlcv"):
            return []
        where = [f"pair_address = {self._lit(pair)}"]
        if timeframe:
            where.append(f"timeframe = {self._lit(timeframe)}")
        if chain:
            where.append(f"chain = {self._lit(chain)}")
        clause = " AND ".join(where)
        q = (
            "SELECT chain, pair_address, timeframe, timestamp, "
            "open, high, low, close, volume "
            "FROM ohlcv "
            f"WHERE {clause} "
            "ORDER BY timestamp "
            f"LIMIT {int(limit)}"
        )
        return self.sql(q)

    def new_launches(self, chain: str | None = None, since_dt: str | None = None,
                     limit: int = 100) -> list[dict]:
        """Newly created pools/pairs from the `pool_universe` dataset.

        This is the survivorship-free historical token universe: every pool
        creation event, including pools now dead. Ordered newest-first by
        creation date. Empty list if `pool_universe` hasn't been built yet.
        """
        if self._missing("pool_universe"):
            return []
        where = []
        if chain:
            where.append(f"chain = {self._lit(chain)}")
        if since_dt:
            where.append(f"dt >= {self._lit(since_dt)}")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        q = (
            "SELECT chain, dt, dex, pair_address, token0, token1, "
            "created_block, created_at, creator "
            f"FROM pool_universe{clause} "
            "ORDER BY COALESCE(created_at, dt::TIMESTAMP) DESC "
            f"LIMIT {int(limit)}"
        )
        if self.backend == "polars":
            # polars SQL has no ::TIMESTAMP cast on the dt string; order by created_at only.
            q = (
                "SELECT chain, dt, dex, pair_address, token0, token1, "
                "created_block, created_at, creator "
                f"FROM pool_universe{clause} "
                "ORDER BY created_at DESC "
                f"LIMIT {int(limit)}"
            )
        return self.sql(q)

    def holder_count_over_time(self, address: str, chain: str | None = None) -> list[dict]:
        """Holder-count time series for a token.

        Prefers a dedicated `holders` dataset if one exists; otherwise derives
        the series from the ``holder_count`` column on `tokens` snapshots (and,
        failing that, `rug_reports`). Empty list if no source carries it.
        """
        # Dedicated holders dataset (placeholder — register & use if present).
        if "holders" in self._datasets:
            where = [f"address = {self._lit(address)}"]
            if chain:
                where.append(f"chain = {self._lit(chain)}")
            clause = " AND ".join(where)
            q = (
                "SELECT chain, address, observed_at, dt, holder_count "
                "FROM holders "
                f"WHERE {clause} "
                "ORDER BY observed_at"
            )
            return self.sql(q)
        # Derive from snapshot datasets that carry holder_count.
        for src in ("tokens", "rug_reports"):
            if src in self._datasets:
                where = [f"address = {self._lit(address)}", "holder_count IS NOT NULL"]
                if chain:
                    where.append(f"chain = {self._lit(chain)}")
                clause = " AND ".join(where)
                q = (
                    f"SELECT chain, address, observed_at, dt, holder_count "
                    f"FROM {src} "
                    f"WHERE {clause} "
                    "ORDER BY observed_at"
                )
                rows = self.sql(q)
                if rows:
                    return rows
        return []

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "Query":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
