"""Path B ingest: ESTABLISHED (month+ old) DEX coins with their RECENT tape, archive-backed.

Discovers PairCreated launches in an OLDER window (coins are month+ old), then for each:
  1. cheap LIVENESS PROBE — pull only the last ~1 day; if no swaps, the coin is dead/illiquid
     now -> record it (survivorship!) and SKIP the expensive full pull.
  2. if alive -> pull the recent N-day tape (Sync+Swap, per-bar reserves) + holders -> store.

Survivorship-honest: dead coins are recorded in _deep_universe.jsonl (they existed and were
un-tradeable in the eval window); we just don't waste getLogs paging their empty history.

NodeReal archive caps getLogs at 10k blocks; we raise bsc_indexer MAX_CHUNKS to fit the window.

Usage:
  python3 _research/deep_ingest.py <from_block> <to_block> <days_back> <n> <concurrency>
"""
from __future__ import annotations
import asyncio, json, logging, math, sys, time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
sys.path.insert(0, ".")

import chainscope.providers.bsc_indexer as bidx
import chainscope.universe as _uni

from chainscope.aggregate import Client
from chainscope.universe import UniverseBuilder
from chainscope.cohort import _coin_side
from chainscope.holders_evm import EvmHolders
from chainscope.storage import ParquetStore

BLOCKS_PER_DAY = int(2.22 * 86400)  # BSC ~2.22 blk/s (measured 2026-05-28)
PROBE_DAYS = 1.0                    # liveness probe depth

FROM_B, TO_B, DAYS_BACK, N, CONC = (int(sys.argv[1]), int(sys.argv[2]),
                                    int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

# getLogs chunking sized for the NodeReal 10k cap + the requested window.
bidx.CHUNK = 10_000
_uni.BSC_CHUNK = 10_000
bidx.MAX_CHUNKS = math.ceil(DAYS_BACK * BLOCKS_PER_DAY / bidx.CHUNK) + 30
_uni.BSC_MAX_CHUNKS = max(_uni.BSC_MAX_CHUNKS, math.ceil((TO_B - FROM_B) / _uni.BSC_CHUNK) + 30)


async def main():
    t0 = time.time()
    store = ParquetStore()
    done_file = store.data_dir / "_deep_done.json"
    univ_file = store.data_dir / "_deep_universe.jsonl"
    done = set(json.loads(done_file.read_text())) if done_file.exists() else set()

    async with Client() as cs:
        idx = cs.registry.get("bsc_indexer")
        head = await idx.head_block()
        recent_start = max(1, head - int(DAYS_BACK * BLOCKS_PER_DAY))
        probe_start = max(1, head - int(PROBE_DAYS * BLOCKS_PER_DAY))
        print(f"[discover] window [{FROM_B},{TO_B}] (~{(head-FROM_B)/BLOCKS_PER_DAY:.0f}.."
              f"{(head-TO_B)/BLOCKS_PER_DAY:.0f}d old); tape from {recent_start} (~{DAYS_BACK}d); "
              f"probe last {PROBE_DAYS}d; head {head}", flush=True)
        creations = await ub_scan(cs, FROM_B, TO_B)
        pool = [c for c in creations if not (c.dex and "four" in (c.dex or "").lower())]
        seen, launches = set(), []
        for c in sorted(pool, key=lambda x: x.created_block or 0, reverse=True):
            coin = _coin_side(c)
            if coin and coin not in seen:
                seen.add(coin); launches.append(c)
            if len(launches) >= N:
                break
        print(f"[discover] {len(creations)} creations -> {len(pool)} dex pools -> {len(launches)} unique coins", flush=True)
        if launches:
            store.write("pool_universe", launches, time_field="created_at")

        sem = asyncio.Semaphore(CONC)
        st = {"live": 0, "dead": 0, "err": 0, "skip": 0}
        ufh = open(univ_file, "a")

        async def one(c):
            coin = _coin_side(c); pair = c.pair_address
            if coin in done:
                st["skip"] += 1; return
            status = "err"; recent_n = 0
            async with sem:
                try:
                    probe = await idx.fetch_trades_range(pair, probe_start, head)  # cheap liveness
                    if not probe:
                        status = "dead"; st["dead"] += 1
                    else:
                        trades = await idx.fetch_trades_range(pair, recent_start, head)
                        recent_n = len(trades)
                        if trades:
                            store.write("trades", trades, time_field="block_time")
                            try:
                                res = await EvmHolders(cs.http).holder_balances(coin, from_block=recent_start)
                                hs = EvmHolders.holder_stats(res)
                                if hs:
                                    store.write("holders", [{
                                        "source": "bsc_indexer",
                                        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "address": pair, "token": coin, "holder_count": hs.get("holder_count"),
                                        "top10_pct": hs.get("top10_pct"), "hhi": hs.get("hhi"), "chain": "bsc",
                                    }], time_field="observed_at")
                            except Exception:
                                pass
                            status = "live"; st["live"] += 1
                        else:
                            status = "dead"; st["dead"] += 1
                except Exception as e:
                    st["err"] += 1
                    logging.warning("coin %s: %s", (coin or "?")[:12], str(e)[:80])
            ufh.write(json.dumps({"coin": coin, "pair": pair, "created_block": c.created_block,
                                  "status": status, "recent_trades": recent_n}) + "\n"); ufh.flush()
            done.add(coin)
            try:
                done_file.write_text(json.dumps(sorted(done)))
            except OSError:
                pass
            n = sum(st.values())
            if n % 10 == 0:
                print(f"  [{n}/{len(launches)}] live={st['live']} dead={st['dead']} err={st['err']} "
                      f"({time.time()-t0:.0f}s, {(time.time()-t0)/max(1,n):.1f}s/coin)", flush=True)

        await asyncio.gather(*[one(c) for c in launches])
        ufh.close()
        print(f"[done] {st} in {time.time()-t0:.0f}s", flush=True)


async def ub_scan(cs, fr, to):
    ub = UniverseBuilder(cs.http, cs.settings)
    return await ub.scan_bsc(from_block=fr, to_block=to)


asyncio.run(main())
