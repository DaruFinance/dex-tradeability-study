"""Parquet storage, append-only and partitioned for point-in-time research.

Every record is written natively to parquet (never CSV-then-convert), partitioned
by chain and observation date. Rows carry `observed_at`, so the store is a
point-in-time log with no lookahead: re-reading a date gives only what was known then.

Datasets: tokens, pools, ohlcv, trades, rug_reports, launches, listings.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config import get_settings
from .models import Record


def _dt_to_date(dt) -> str:
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return str(dt)[:10]


class ParquetStore:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else get_settings().data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def write(self, dataset: str, records: list[Record], time_field: str = "observed_at") -> int:
        """Append records to {data_dir}/{dataset}/chain=.../dt=.../*.parquet. Returns row count.

        Partitions by `time_field`'s date. For market data pass the EVENT time
        (e.g. time_field="block_time" for trades, "timestamp" for OHLCV) so the
        store is partitioned by when things happened, not when we fetched them;
        `observed_at` is still kept as a column for lookahead audits."""
        rows = []
        for r in records:
            row = r.to_row()
            # partition keys
            row.setdefault("chain", getattr(r, "chain", None))
            row["chain"] = str(row.get("chain") or "unknown")
            part_ts = getattr(r, time_field, None) or getattr(r, "observed_at", None)
            row["dt"] = _dt_to_date(part_ts or datetime.now(timezone.utc))
            rows.append(row)
        if not rows:
            return 0
        table = pa.Table.from_pylist(rows)
        root = self.data_dir / dataset
        pq.write_to_dataset(
            table,
            root_path=str(root),
            partition_cols=["chain", "dt"],
        )
        return len(rows)

    def path(self, dataset: str) -> Path:
        return self.data_dir / dataset

    def read(self, dataset: str, columns: list[str] | None = None):
        """Load a dataset (all partitions) into a pandas DataFrame. Requires the
        dataset to exist; partition columns chain/dt are restored automatically."""
        import pandas as pd  # local import; pandas is heavy

        root = self.data_dir / dataset
        if not root.exists():
            return pd.DataFrame()
        # DuckDB unions files whose schemas differ (a column that's all-None in one
        # write infers a null type that pyarrow can't cast across files); fall back
        # to pyarrow when duckdb isn't installed.
        try:
            import duckdb
            cols = ", ".join(columns) if columns else "*"
            return duckdb.sql(
                f"SELECT {cols} FROM read_parquet('{root}/**/*.parquet', "
                f"union_by_name=true, hive_partitioning=true)"
            ).df()
        except ImportError:
            return pq.read_table(str(root), columns=columns).to_pandas()
