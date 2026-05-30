"""Cohort pipeline: the end-to-end "give me N coins that launched in a window and
pull all their data until today" flow, entirely from our own on-chain digging.

  discover_launches  -> scan factory creation events for the window (survivorship-free)
  coin_data          -> per coin: movements (trade tape) + volume + per-block liquidity
                        + holders/concentration + (optional) exact gas, from its creation
                        block to now, written to parquet
  run_cohort         -> orchestrates the two over a cohort and returns a summary table

BSC is fully supported on free RPC (throughput-bound for deep windows; a local archive
node makes 100-coins-x-30-days routine). Solana historical launch discovery needs the
Old Faithful archive or a getProgramAccounts-capable RPC (the public endpoint can't
enumerate past launches), flagged, not faked.

Note: pre-graduation four.meme / pump.fun bonding-curve trades are NOT DEX swaps, so the
DEX indexer doesn't see them until the token graduates to a real pool. This pipeline
covers DEX-pool (PancakeSwap V2/V3) trading; bonding-curve trade decoding is a separate
launchpad decoder (gap, noted).
"""
from __future__ import annotations

import asyncio
import json

from .chains import Chain
from .holders_evm import EvmHolders
from .storage import ParquetStore
from .universe import UniverseBuilder

WBNB = "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"
STABLES = {
    "0x55d398326f99059ff775485246999027b3197955",
    "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",
    "0xe9e7cea3dedca5984780bafc599bd69add087d56",
}
QUOTE = {WBNB, *STABLES}


def _coin_from_tokens(t0: str | None, t1: str | None) -> str | None:
    t0 = (t0 or "").lower()
    t1 = (t1 or "").lower()
    if t0 in QUOTE and t1 and t1 not in QUOTE:
        return t1
    if t1 in QUOTE and t0 and t0 not in QUOTE:
        return t0
    return t0 or None


def _coin_side(c) -> str | None:
    """The launched coin is the non-quote side of the pair."""
    return _coin_from_tokens(c.token0, c.token1)


async def discover_launches(client, chain: str | Chain = "bsc", recent_blocks: int = 50_000,
                            n: int = 100, dex_pools_only: bool = True):
    """Find up to n coins launched in the recent block window (survivorship-free)."""
    if Chain.parse(chain) != Chain.BSC:
        return [], {"note": "Solana historical launch discovery needs Old Faithful / a "
                            "getProgramAccounts-capable RPC; the public endpoint can't enumerate past launches"}
    idx = client.registry.get("bsc_indexer")
    head = await idx.head_block()
    ub = UniverseBuilder(client.http, client.settings)
    creations = await ub.scan_bsc(from_block=max(1, head - recent_blocks), to_block=head)
    fourmeme = sum(1 for c in creations if c.dex and "four" in c.dex.lower())
    pool = [c for c in creations if not (c.dex and "four" in c.dex.lower())] if dex_pools_only else creations
    seen: set[str] = set()
    out = []
    for c in sorted(pool, key=lambda x: x.created_block or 0, reverse=True):
        coin = _coin_side(c)
        if not coin or coin in seen:
            continue
        seen.add(coin)
        out.append(c)
        if len(out) >= n:
            break
    disc = {"scanned_creations": len(creations), "fourmeme_launches": fourmeme,
            "dex_pool_launches": len(pool), "head_block": head}
    return out, disc


async def coin_data(client, creation, store: ParquetStore | None = None,
                    with_holders: bool = True, gas_actual: bool = False) -> dict:
    """Reconstruct one coin's full data from its creation block to now."""
    idx = client.registry.get("bsc_indexer")
    pair = creation.pair_address
    coin = _coin_side(creation)
    head = await idx.head_block()
    start = creation.created_block or max(1, head - 200_000)

    trades = await idx.fetch_trades_range(pair, start, head)
    if gas_actual and trades:
        await idx.enrich_gas(trades, limit=200)
    volume_usd = sum((t.amount_usd or 0) for t in trades)
    last = trades[0] if trades else None

    hstats = None
    if with_holders and coin:
        try:
            res = await EvmHolders(client.http).holder_balances(coin, from_block=start)
            hstats = EvmHolders.holder_stats(res)
        except Exception:
            hstats = None

    if store and trades:
        store.write("trades", trades, time_field="block_time")

    return {
        "coin": coin, "pair": pair, "dex": creation.dex,
        "created_at": creation.created_at, "created_block": creation.created_block,
        "trades": len(trades), "volume_usd": volume_usd,
        "last_price_usd": last.price_usd if last else None,
        "liquidity_usd": last.reserve_usd if last else None,
        "holder_count": (hstats or {}).get("holder_count"),
        "top10_pct": (hstats or {}).get("top10_pct"),
        "hhi": (hstats or {}).get("hhi"),
        "gas_native_last": (trades[0].gas_native if trades else None),
    }


async def run_cohort(client, chain: str | Chain = "bsc", recent_blocks: int = 50_000,
                     n: int = 10, with_holders: bool = True, gas_actual: bool = False,
                     concurrency: int = 4, resume: bool = True) -> dict:
    """Discover N launches and pull each coin's full data CONCURRENTLY to parquet.

    Scales to large cohorts (e.g. n=1000): `concurrency` coins are processed at once
    (the HttpClient still rate-limits per host, so it won't hammer a free RPC), and
    `resume` skips coins already completed (tracked in _cohort_done.json) so a long
    run can be interrupted and continued. For 1000 coins, raise `recent_blocks` enough
    to contain that many launches and point at a local archive node for speed."""
    launches, disc = await discover_launches(client, chain, recent_blocks=recent_blocks, n=n)
    store = ParquetStore()
    if launches:
        store.write("pool_universe", launches, time_field="created_at")  # so screener age/created_at fill
    done_file = store.data_dir / "_cohort_done.json"
    done: set[str] = set()
    if resume and done_file.exists():
        try:
            done = set(json.loads(done_file.read_text()))
        except (OSError, ValueError):
            done = set()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(c) -> dict:
        coin = _coin_side(c)
        if resume and coin in done:
            return {"coin": coin, "skipped": True}
        async with sem:
            try:
                row = await coin_data(client, c, store=store,
                                      with_holders=with_holders, gas_actual=gas_actual)
            except Exception as exc:
                return {"coin": coin, "error": str(exc)[:90]}
        if coin:
            done.add(coin)
            try:
                done_file.write_text(json.dumps(sorted(done)))  # checkpoint for resume
            except OSError:
                pass
        return row

    rows = await asyncio.gather(*[_one(c) for c in launches])
    return {"discovery": disc, "discovered": len(launches),
            "processed": sum(1 for r in rows if not r.get("skipped")), "coins": rows}


async def run_solana_cohort(client, n: int = 15, collect_seconds: int = 45,
                            trade_limit: int = 60) -> dict:
    """Solana ingest: collect recent pump.fun launches from the live stream, then pull
    each one's bonding-curve trades (PumpFunBonding) and write chain=solana Trade records
    to the store so they appear in the screener. (Deep/historical Solana launch discovery
    still needs Old Faithful; this captures the live recent-launch flow, which is the
    Solana parallel to the BSC factory scan.)"""
    from .bonding import PumpFunBonding

    store = ParquetStore()
    mints: list[tuple[str, str | None]] = []
    gen = client.stream_launches("solana")

    async def _collect():
        async for lc in gen:
            if lc.address and not lc.complete:      # a new token (not a migration)
                mints.append((lc.address, lc.symbol))
                if len(mints) >= n:
                    break
    try:
        await asyncio.wait_for(_collect(), timeout=collect_seconds)
    except asyncio.TimeoutError:
        pass
    finally:
        await gen.aclose()

    pb = PumpFunBonding(client.http, client.settings)
    rows = []
    for mint, sym in mints:
        try:
            trs = await pb.get_trades(mint, limit=trade_limit)
            if trs:
                store.write("trades", trs, time_field="block_time")
            rows.append({"coin": mint, "symbol": sym, "trades": len(trs),
                         "volume_usd": sum((t.amount_usd or 0) for t in trs)})
        except Exception as exc:
            rows.append({"coin": mint, "error": str(exc)[:90]})
    return {"discovered": len(mints), "coins": rows}


async def enrich_holders(client, chain: str | Chain | None = None,
                         limit: int | None = None, concurrency: int = 4) -> dict:
    """Compute holder distribution for the pools ALREADY in the store and write a
    `holders` dataset (keyed by pair_address) so the screener's holder_count + top10
    columns fill in. A separate pass over existing pools, does NOT re-pull trades, so
    no double-counting. BSC via Transfer-event reconstruction; Solana via Helius."""
    from .holders_evm import EvmHolders
    from .holders_sol import SolHolders
    from .models import Chain as _C, HolderSnapshot

    store = ParquetStore()
    want = chain.lower() if isinstance(chain, str) else (chain.value if chain else None)
    targets: list[tuple[str, str, str, int | None]] = []  # (chain, pair, token, from_block)

    if want in (None, "bsc"):
        pu = store.read("pool_universe")
        if pu is not None and len(pu):
            for _, r in pu.iterrows():
                if str(r.get("chain")) != "bsc":
                    continue
                coin = _coin_from_tokens(r.get("token0"), r.get("token1"))
                cb = r.get("created_block")
                fromblk = int(cb) if cb is not None and cb == cb else None  # cb==cb filters NaN
                if coin:
                    targets.append(("bsc", r["pair_address"], coin, fromblk))
    if want in (None, "solana"):
        tr = store.read("trades")
        if tr is not None and len(tr):
            for m in tr[tr["chain"] == "solana"]["pair_address"].dropna().unique().tolist():
                targets.append(("solana", m, m, None))

    seen: set[str] = set()
    uniq = [t for t in targets if not (t[1] in seen or seen.add(t[1]))]
    if limit:
        uniq = uniq[:limit]

    eh = EvmHolders(client.http)
    sh = SolHolders(client.http, client.settings)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(t):
        ch, pair, token, fromblk = t
        async with sem:
            try:
                if ch == "bsc":
                    res = await eh.holder_balances(token, from_block=fromblk)
                    st = EvmHolders.holder_stats(res)
                    return HolderSnapshot(source="holders_evm", chain=_C.BSC, address=pair,
                                          token=token, holder_count=st.get("holder_count"),
                                          top10_pct=st.get("top10_pct"), hhi=st.get("hhi"))
                th = await sh.top_holders(token, n=20)
                top10 = sum((h.get("pct_of_supply") or 0) for h in th[:10]) if th else None
                hc = None
                try:
                    ah = await sh.all_holders(token)
                    if ah.get("available"):
                        hc = ah.get("holder_count") or len(ah.get("balances") or {})
                except Exception:
                    pass
                if not hc and th:
                    hc = len(th)   # fallback: count of largest accounts (gPA empty for new mints)
                return HolderSnapshot(source="holders_sol", chain=_C.SOLANA, address=pair,
                                      token=token, holder_count=hc, top10_pct=top10)
            except Exception:
                return None

    results = await asyncio.gather(*[_one(t) for t in uniq])
    # the screener join requires holder_count IS NOT NULL
    snaps = [r for r in results if r and r.holder_count is not None]
    if snaps:
        store.write("holders", snaps, time_field="observed_at")
    return {"targets": len(uniq), "enriched": len(snaps),
            "bsc": sum(1 for r in snaps if r.chain == "bsc"),
            "solana": sum(1 for r in snaps if r.chain == "solana")}
