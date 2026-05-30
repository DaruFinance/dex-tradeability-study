"""Pull raw trade tapes (with BUY/SELL sides + reserves) for the top BSC DEX-only coins via NodeReal.

This gets the on-chain FLOW data GeckoTerminal cannot (taker buy/sell imbalance per trade) — the
input to the actual thesis (flow archetype) that simple OHLCV TA can't express. Slow (block-bound
getLogs) but targeted to a handful of coins. Writes to the 'flow_trades' parquet dataset.

Usage: python3 _research/raw_flow_pull.py <n_coins> <days_back> <concurrency>
"""
from __future__ import annotations
import asyncio, json, math, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import chainscope.providers.bsc_indexer as bidx
from chainscope.aggregate import Client
from chainscope.storage import ParquetStore

DATA = Path("./data")
BPD = int(2.22 * 86400)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 4
bidx.CHUNK = 10_000
bidx.MAX_CHUNKS = math.ceil(DAYS * BPD / bidx.CHUNK) + 20


def top_bsc_dexonly(n):
    univ = {}
    for l in open(DATA / "_gt_universe.jsonl"):
        r = json.loads(l); univ[r["pair"]] = r
    dex = []
    for l in open(DATA / "_cex_verdicts_universe.jsonl"):
        r = json.loads(l)
        if r["status"] == "dex_only" and r.get("chain", "bsc") == "bsc":
            u = univ.get(r["pair"], {})
            dex.append((r["pair"], u.get("reserve_usd") or 0))
    dex.sort(key=lambda x: -x[1])
    return [p for p, _ in dex[:n]]


async def main():
    pairs = top_bsc_dexonly(N)
    store = ParquetStore()
    done_file = DATA / "_flow_done.json"
    done = set(json.loads(done_file.read_text())) if done_file.exists() else set()
    print(f"{len(pairs)} top BSC dex_only coins, {DAYS}d tape via NodeReal (MAX_CHUNKS={bidx.MAX_CHUNKS})", flush=True)
    t0 = time.time()
    async with Client() as cs:
        idx = cs.registry.get("bsc_indexer")
        head = await idx.head_block()
        start = max(1, head - DAYS * BPD)
        sem = asyncio.Semaphore(CONC)
        st = {"ok": 0, "empty": 0, "err": 0}

        async def one(pair):
            if pair in done:
                return
            async with sem:
                try:
                    trades = await idx.fetch_trades_range(pair, start, head)
                    if trades:
                        store.write("flow_trades", trades, time_field="block_time")
                        st["ok"] += 1
                    else:
                        st["empty"] += 1
                except Exception as e:
                    st["err"] += 1
                    print(f"  err {pair[:12]}: {str(e)[:60]}", flush=True)
            done.add(pair); done_file.write_text(json.dumps(sorted(done)))
            n = sum(st.values())
            print(f"  [{n}/{len(pairs)}] ok={st['ok']} empty={st['empty']} err={st['err']} "
                  f"({time.time()-t0:.0f}s, {(time.time()-t0)/max(1,n):.0f}s/coin)", flush=True)

        await asyncio.gather(*[one(p) for p in pairs])
    print(f"[done] {st} in {time.time()-t0:.0f}s -> flow_trades dataset", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
