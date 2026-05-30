"""Backfill engine: turns a chain indexer into downloaded parquet datasets.

Generic over any `fetch(from_block, to_block) -> list[Record]` callable, so it
drives the BSC indexer today and a Solana indexer later. Properties:

  - resumable: a per-target cursor (last fully-flushed block) is persisted to
    {data_dir}/_cursors.json; re-running continues from there, never restarts.
  - bounded memory: records are flushed to parquet every flush_rows / flush_blocks,
    so a multi-million-block scan never holds the whole tape in RAM.
  - crash-safe: the cursor only advances past a block range once its rows are on
    disk, so a mid-run crash re-scans (at most) the last unflushed window, no gaps.
  - endpoint resilience: handled in the indexer's _rpc (rotates free archival RPCs).

Hot-loop note: the decode/scan is pure CPU once logs are fetched, a Rust port of
fetch_*_range is the next speedup for full-chain scans. (follow-up)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Awaitable, Callable

from .models import Record
from .storage import ParquetStore

log = logging.getLogger("chainscope.backfill")

FetchFn = Callable[[int, int], Awaitable[list[Record]]]


class Backfiller:
    def __init__(self, store: ParquetStore | None = None, cursor_file: str | Path | None = None):
        self.store = store or ParquetStore()
        self.cursor_file = Path(cursor_file) if cursor_file else self.store.data_dir / "_cursors.json"
        self._cursors: dict[str, int] = self._load()

    def _load(self) -> dict[str, int]:
        try:
            return json.loads(self.cursor_file.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        self.cursor_file.parent.mkdir(parents=True, exist_ok=True)
        self.cursor_file.write_text(json.dumps(self._cursors, indent=0))

    def cursor(self, key: str) -> int | None:
        return self._cursors.get(key)

    async def run(self, *, dataset: str, key: str, fetch: FetchFn,
                  start: int | None = None, end: int, batch_blocks: int = 5000,
                  flush_rows: int = 20000, flush_blocks: int = 100000,
                  time_field: str = "block_time", progress=None) -> dict:
        """Backfill [start..end] for one target. Resumes from the stored cursor if present."""
        cur = self.cursor(key)
        from_b = (cur + 1) if cur is not None else start
        if from_b is None:
            raise ValueError("provide start= (no existing cursor for this key)")
        if from_b > end:
            return {"dataset": dataset, "key": key, "rows": 0, "from_block": from_b,
                    "to_block": end, "note": "already up to date"}

        buf: list[Record] = []
        total = 0
        last_flush_block = from_b - 1
        b = from_b
        while b <= end:
            hi = min(b + batch_blocks - 1, end)
            recs = await fetch(b, hi)
            buf.extend(recs)
            advance = (len(buf) >= flush_rows) or (hi - last_flush_block >= flush_blocks) or (hi == end)
            if advance:
                if buf:
                    self.store.write(dataset, buf, time_field=time_field)
                    total += len(buf)
                    buf = []
                self._cursors[key] = hi      # only advance cursor past flushed rows
                self._save()
                last_flush_block = hi
                if progress:
                    progress(hi, end, total)
            b = hi + 1
        return {"dataset": dataset, "key": key, "rows": total,
                "from_block": from_b, "to_block": end}


async def backfill_bsc_trades(client, pair_address: str, *, days: float | None = None,
                              from_block: int | None = None, batch_blocks: int = 5000,
                              store: ParquetStore | None = None, progress=None) -> dict:
    """Convenience: backfill the V2 trade tape (with per-trade reserves) for a BSC pair."""
    from .chains import Chain

    idx = client.registry.get("bsc_indexer")
    end = await idx.finalized_head()   # reorg-safe: only backfill finalized blocks
    bf = Backfiller(store=store or ParquetStore())
    key = f"bsc:trades:{pair_address.lower()}"
    start = from_block
    if start is None and bf.cursor(key) is None:
        if days is None:
            days = 1.0
        _, t_latest, spb = await idx._clock_calibrate()
        start = max(1, end - int(days * 86400 / spb))
    return await bf.run(
        dataset="trades", key=key,
        fetch=lambda a, z: idx.fetch_trades_range(pair_address, a, z),
        start=start, end=end, batch_blocks=batch_blocks,
        time_field="block_time", progress=progress,
    )
