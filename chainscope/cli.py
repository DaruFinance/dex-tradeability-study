"""chainscope CLI — search tokens and pull rich on-chain data for Solana + BSC."""
from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

from .aggregate import Client
from .backtest import backtest_frame
from .chains import Chain
from .storage import ParquetStore

app = typer.Typer(add_completion=False, help="Multi-chain on-chain token research data source.")
console = Console()


def _run(coro):
    return asyncio.run(coro)


def _fmt(v, money=False):
    if v is None:
        return "-"
    if money and isinstance(v, (int, float)):
        return f"${v:,.2f}" if abs(v) < 1000 else f"${v:,.0f}"
    if isinstance(v, float):
        return f"{v:,.6g}"
    return str(v)


@app.command()
def providers():
    """Show every provider, whether it's enabled, and what it supplies."""
    async def go():
        async with Client() as cs:
            return cs.registry.status()
    rows = _run(go())
    t = Table(title="chainscope providers", show_lines=False)
    for col in ("provider", "enabled", "key env", "chains", "capabilities"):
        t.add_column(col)
    for r in rows:
        t.add_row(
            r["name"],
            "[green]yes[/green]" if r["enabled"] else "[dim]no (set key)[/dim]",
            r["key_env"] or "-",
            ",".join(r["chains"]),
            ",".join(r["capabilities"]),
        )
    console.print(t)


@app.command()
def search(query: str, chain: str = typer.Option(None, help="solana|bsc (optional)"),
           limit: int = 20):
    """Search tokens by name/symbol/address across DEXes."""
    async def go():
        async with Client() as cs:
            return await cs.search(query, chain, limit)
    toks = _run(go())
    t = Table(title=f"search: {query}")
    for col in ("chain", "symbol", "name", "price", "mcap", "liq", "vol24h", "pairs", "address"):
        t.add_column(col)
    for tok in toks:
        t.add_row(str(tok.chain), tok.symbol or "-", (tok.name or "-")[:24],
                  _fmt(tok.price_usd, True), _fmt(tok.market_cap, True),
                  _fmt(tok.liquidity_usd, True), _fmt(tok.volume_24h, True),
                  str(tok.pair_count or "-"), tok.address)
    console.print(t)


@app.command()
def token(chain: str, address: str):
    """Merged token snapshot (price, mcap, liquidity, volume) from all sources."""
    async def go():
        async with Client() as cs:
            return await cs.token(chain, address)
    tok = _run(go())
    if not tok:
        console.print("[red]not found[/red]")
        raise typer.Exit(1)
    for k, v in tok.to_row().items():
        if v not in (None, [], ""):
            console.print(f"[cyan]{k:18}[/cyan] {_fmt(v)}")


@app.command()
def pools(chain: str, address: str):
    """List liquidity pools for a token."""
    async def go():
        async with Client() as cs:
            return await cs.pools(chain, address)
    ps = _run(go())
    ps.sort(key=lambda p: p.liquidity_usd or 0, reverse=True)
    t = Table(title="pools")
    for col in ("dex", "pair", "base/quote", "price", "liq", "vol24h", "created", "pair_address"):
        t.add_column(col)
    for p in ps:
        t.add_row(p.dex or "-", "", f"{p.base_symbol or '?'}/{p.quote_symbol or '?'}",
                  _fmt(p.price_usd, True), _fmt(p.liquidity_usd, True),
                  _fmt(p.volume_24h, True), str(p.created_at.date() if p.created_at else "-"),
                  p.pair_address)
    console.print(t)


@app.command()
def rug(chain: str, address: str):
    """Merged token-safety / rug profile from every security source."""
    async def go():
        async with Client() as cs:
            return await cs.rug(chain, address)
    r = _run(go())
    if not r:
        console.print("[yellow]no rug data[/yellow]")
        raise typer.Exit(1)
    for k, v in r.to_row().items():
        if v not in (None, [], ""):
            console.print(f"[cyan]{k:24}[/cyan] {_fmt(v)}")
    if r.raw and r.raw.get("sources"):
        console.print(f"[dim]sources: {', '.join(r.raw['sources'])}[/dim]")


@app.command()
def listings(chain: str, address: str):
    """Where the token trades (CEX + DEX venues), via CoinGecko."""
    async def go():
        async with Client() as cs:
            return await cs.listings(chain, address)
    e = _run(go())
    if not e:
        console.print("[yellow]not tracked by CoinGecko[/yellow]")
        raise typer.Exit(1)
    console.print(f"[cyan]coingecko_id[/cyan] {e.coingecko_id}")
    console.print(f"[cyan]CEX ({len(e.cex_exchanges)})[/cyan] {', '.join(e.cex_exchanges[:40])}")
    console.print(f"[cyan]DEX ({len(e.dex_exchanges)})[/cyan] {', '.join(e.dex_exchanges[:40])}")


@app.command()
def launch(chain: str, address: str):
    """Launchpad / bonding-curve / graduation status for a token."""
    async def go():
        async with Client() as cs:
            return await cs.launch(chain, address)
    lc = _run(go())
    if not lc:
        console.print("[yellow]no launchpad data[/yellow]")
        raise typer.Exit(1)
    for k, v in lc.to_row().items():
        if v not in (None, [], ""):
            console.print(f"[cyan]{k:22}[/cyan] {_fmt(v)}")


@app.command()
def ohlcv(chain: str, pair_address: str, tf: str = "1h", limit: int = 1000,
          save: bool = typer.Option(False, help="append to parquet store")):
    """Fetch OHLCV price bars for a pool (backtest price path)."""
    async def go():
        async with Client() as cs:
            return await cs.ohlcv(chain, pair_address, tf, limit)
    bars = _run(go())
    console.print(f"{len(bars)} bars ({tf})")
    for b in bars[:5] + bars[-3:]:
        console.print(f"  {b.timestamp}  O {b.open:.6g} H {b.high:.6g} "
                      f"L {b.low:.6g} C {b.close:.6g} V {_fmt(b.volume)}")
    if save and bars:
        n = ParquetStore().write("ohlcv", bars)
        console.print(f"[green]wrote {n} rows to parquet[/green]")


@app.command()
def trades(chain: str, pair_address: str, limit: int = 300,
           save: bool = typer.Option(False)):
    """Fetch the trade tape (individual swaps) for a pool."""
    async def go():
        async with Client() as cs:
            return await cs.trades(chain, pair_address, limit=limit)
    ts = _run(go())
    console.print(f"{len(ts)} trades")
    for tr in ts[:10]:
        console.print(f"  {tr.block_time}  {tr.side or '?':4}  "
                      f"px {_fmt(tr.price_usd, True)}  usd {_fmt(tr.amount_usd, True)}")
    if save and ts:
        n = ParquetStore().write("trades", ts)
        console.print(f"[green]wrote {n} rows to parquet[/green]")


@app.command()
def profile(chain: str, address: str, save: bool = typer.Option(False)):
    """Full rich record: token + top pool + merged rug + listings + launch."""
    async def go():
        async with Client() as cs:
            return await cs.profile(chain, address)
    p = _run(go())
    tok, pool, r, e, lc = p["token"], p["top_pool"], p["rug"], p["listings"], p["launch"]
    console.rule(f"{chain} {address}")
    if tok:
        console.print(f"[bold]{tok.symbol or '?'}[/bold] {tok.name or ''}  "
                      f"price {_fmt(tok.price_usd, True)}  mcap {_fmt(tok.market_cap, True)}  "
                      f"liq {_fmt(tok.liquidity_usd, True)}  vol24h {_fmt(tok.volume_24h, True)}")
    if pool:
        console.print(f"top pool: {pool.dex} {pool.base_symbol}/{pool.quote_symbol} "
                      f"liq {_fmt(pool.liquidity_usd, True)} ({p['pool_count']} pools)")
    if r:
        console.print(f"rug: risk={_fmt(r.risk_score)} honeypot={r.is_honeypot} "
                      f"top10={_fmt(r.top10_holder_pct)} flags={r.flags} "
                      f"[dim]({'+'.join((r.raw or {}).get('sources', []))})[/dim]")
    if e:
        console.print(f"listed: {len(e.cex_exchanges)} CEX / {len(e.dex_exchanges)} DEX")
    if lc:
        console.print(f"launch: {lc.launchpad} progress={_fmt(lc.bonding_curve_progress)} "
                      f"graduated={lc.complete}")
    if save:
        store = ParquetStore()
        if tok:
            store.write("tokens", [tok])
        if r:
            store.write("rug_reports", [r])
        console.print("[green]saved token + rug to parquet[/green]")


@app.command()
def graduated(chain: str = typer.Argument("solana"), limit: int = 50):
    """Recently GRADUATED launchpad tokens (needs MORALIS_API_KEY)."""
    async def go():
        async with Client() as cs:
            return await cs.recent_graduated(chain, limit)
    ls = _run(go())
    if not ls:
        console.print("[yellow]no data (set MORALIS_API_KEY?)[/yellow]")
        return
    t = Table(title="recently graduated")
    for col in ("symbol", "name", "mcap", "graduated_at", "address"):
        t.add_column(col)
    for lc in ls:
        t.add_row(lc.symbol or "-", (lc.name or "-")[:24], _fmt(lc.market_cap_usd, True),
                  str(lc.graduated_at or "-"), lc.address)
    console.print(t)


@app.command()
def watch(chain: str = typer.Argument("solana"),
          limit: int = typer.Option(0, help="stop after N events; 0 = forever")):
    """Stream live token launches + graduations (pump.fun via PumpPortal, no key)."""
    async def go():
        n = 0
        async with Client() as cs:
            gen = cs.stream_launches(chain)
            try:
                async for lc in gen:
                    tag = "[magenta]GRADUATED[/magenta]" if lc.complete else "[green]NEW[/green]"
                    console.print(f"{tag} {lc.symbol or '?':10} {lc.address}  "
                                  f"{('-> ' + (lc.migrated_pool or '')) if lc.complete else ''}")
                    n += 1
                    if limit and n >= limit:
                        break
            finally:
                await gen.aclose()  # complete the ws close handshake in-loop
    try:
        _run(go())
    except KeyboardInterrupt:
        console.print("[dim]stopped[/dim]")


@app.command()
def backtest(chain: str, pair_address: str, tf: str = "1h", limit: int = 1000,
             size_usd: float = 1000.0, dex: str = typer.Option(None),
             reserve_usd: float = typer.Option(None, help="pool liquidity for cost model"),
             save: bool = typer.Option(False)):
    """Build a backtest-ready OHLCV frame with per-bar round-trip DEX cost."""
    async def go():
        async with Client() as cs:
            return await backtest_frame(cs, chain, pair_address, tf, limit,
                                        size_usd=size_usd, dex=dex, reserve_usd=reserve_usd)
    df = _run(go())
    if df.empty:
        console.print("[red]no OHLCV returned[/red]")
        raise typer.Exit(1)
    console.print(df.tail(8).to_string())
    console.print(f"[cyan]bars[/cyan] {len(df)}  "
                  f"[cyan]median rt_cost[/cyan] {df['rt_cost_frac'].median():.4%}  "
                  f"(min move to break even on a ${size_usd:,.0f} trade)")
    if save:
        out = ParquetStore().data_dir / "backtest" / f"{chain}_{pair_address}_{tf}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        console.print(f"[green]wrote {out}[/green]")


@app.command()
def backfill(chain: str, pair_address: str, days: float = typer.Option(1.0, help="history depth if no saved cursor"),
             from_block: int = typer.Option(None, help="explicit start block (overrides days)"),
             batch: int = 5000):
    """Dig the on-chain trade tape (with per-trade reserves) into parquet. Resumable. BSC V2 today."""
    if chain.lower() not in ("bsc", "bnb"):
        console.print("[yellow]backfill currently supports BSC; the Solana indexer is the next build[/yellow]")
        raise typer.Exit(1)
    from .backfill import backfill_bsc_trades

    def prog(hi, end, rows):
        console.print(f"  ...block {hi}/{end}  rows {rows}", end="\r")

    async def go():
        async with Client() as cs:
            return await backfill_bsc_trades(cs, pair_address, days=days,
                                             from_block=from_block, batch_blocks=batch, progress=prog)
    stats = _run(go())
    note = f"  ({stats['note']})" if stats.get("note") else ""
    console.print(f"\n[green]backfilled[/green] {stats['rows']} trades, "
                  f"blocks {stats['from_block']}->{stats['to_block']}{note}")
    console.print(f"parquet -> {ParquetStore().path('trades')}")


@app.command()
def cohort(chain: str = typer.Argument("bsc"), recent_blocks: int = 50000, n: int = 10,
           holders: bool = True, concurrency: int = 4,
           gas_actual: bool = typer.Option(False, help="exact per-tx gas (slow)")):
    """Discover recent launches and pull each coin's movements/volume/holders to parquet.
    Scales to large cohorts (e.g. --n 1000); concurrent + resumable."""
    from .cohort import run_cohort, run_solana_cohort
    _sol = chain.lower() in ("solana", "sol")

    async def go():
        async with Client() as cs:
            if _sol:
                return await run_solana_cohort(cs, n=n)
            return await run_cohort(cs, chain, recent_blocks=recent_blocks, n=n,
                                    with_holders=holders, gas_actual=gas_actual,
                                    concurrency=concurrency)
    res = _run(go())
    if res.get("discovery"):
        console.print(f"discovery: {res['discovery']}")
    t = Table(title=f"cohort: {res['discovered']} coins")
    for col in ("coin", "dex", "created", "trades", "vol_usd", "liq_usd", "holders", "top10"):
        t.add_column(col)
    for r in res["coins"]:
        if r.get("error"):
            t.add_row((r.get("coin") or "?")[:14], "ERR", "", "", "", "", "", r["error"][:24])
            continue
        t.add_row((r["coin"] or "?")[:14], (r.get("dex") or "")[:14],
                  str(r.get("created_at") or "")[:10], str(r.get("trades")),
                  _fmt(r.get("volume_usd"), True), _fmt(r.get("liquidity_usd"), True),
                  str(r.get("holder_count") or "-"), _fmt(r.get("top10_pct")))
    console.print(t)


@app.command(name="enrich-holders")
def enrich_holders_cmd(chain: str = typer.Argument(None, help="bsc|solana (default: both)"),
                       limit: int = typer.Option(None), concurrency: int = 4):
    """Compute holder distribution for pools already in the store -> `holders` dataset,
    filling the screener's holder_count + top10 columns. Does not re-pull trades."""
    from .cohort import enrich_holders

    async def go():
        async with Client() as cs:
            return await enrich_holders(cs, chain=chain, limit=limit, concurrency=concurrency)
    res = _run(go())
    console.print(f"[green]enriched holders[/green] for {res['enriched']}/{res['targets']} pools "
                  f"(bsc={res['bsc']}, solana={res['solana']})")


@app.command()
def dossier(chain: str, address: str, save: bool = typer.Option(False)):
    """Full on-chain dossier: metadata, pools, price, trades, depth, cost curve, rug, launch, MEV."""
    from .dossier import build_dossier

    async def go():
        async with Client() as cs:
            return await build_dossier(cs, chain, address)
    d = _run(go())
    console.rule(f"{d['chain']} {d.get('symbol') or ''} {address}")
    console.print(f"[bold]{d.get('symbol') or '?'}[/bold] {d.get('name') or ''}  dec {d.get('decimals')}")
    console.print(f"price {_fmt(d['price_usd'], True)}  mcap {_fmt(d['market_cap'], True)}  "
                  f"fdv {_fmt(d['fdv'], True)}  liq {_fmt(d['liquidity_usd'], True)}  "
                  f"vol24h {_fmt(d['volume_24h'], True)}")
    console.print(f"pools {d['pool_count']}  created {d.get('created_at')}  "
                  f"bars {len(d['ohlcv'])}  trades {len(d['trades'])}")
    dom = d.get("dominant_pool")
    if dom:
        console.print(f"dominant: {dom.dex} {dom.base_symbol}/{dom.quote_symbol} "
                      f"liq {_fmt(dom.liquidity_usd, True)}  {dom.pair_address}")
    for r in d.get("cost_curve") or []:
        rt, ri = r["single_pool_round_trip_frac"], r.get("routed_impact_frac")
        line = f"  ${r['size_usd']:>10,.0f}  round-trip {rt:.3%}"
        if ri is not None:
            line += f"  routed-impact {ri:.3%}"
        console.print(line)
    rug = d.get("rug")
    if rug:
        console.print(f"rug: risk={_fmt(rug.risk_score)} honeypot={rug.is_honeypot} "
                      f"top10={_fmt(rug.top10_holder_pct)} flags={rug.flags} "
                      f"[dim]({'+'.join((rug.raw or {}).get('sources', []))})[/dim]")
    if d.get("launch"):
        lc = d["launch"]
        console.print(f"launch: {lc.launchpad} progress={_fmt(lc.bonding_curve_progress)} "
                      f"graduated={lc.complete} creator={d.get('creator')}")
    if d.get("mev"):
        console.print(f"mev: {d['mev']}")
    if save:
        store = ParquetStore()
        if d.get("token_obj"):
            store.write("tokens", [d["token_obj"]])
        if rug:
            store.write("rug_reports", [rug])
        if d["ohlcv"]:
            store.write("ohlcv", d["ohlcv"], time_field="timestamp")
        if d["trades"]:
            store.write("trades", d["trades"], time_field="block_time")
        console.print(f"[green]saved dossier datasets to {store.data_dir}[/green]")


@app.command()
def serve(host: str = typer.Option("0.0.0.0", help="bind address (0.0.0.0 so WSL is reachable from Windows)"),
          port: int = typer.Option(8011, help="port"),
          reload: bool = typer.Option(False, help="auto-reload (dev)")):
    """Launch the LOCAL web dashboard (cross-sectional coin screener + per-coin
    drill-down) over the parquet store. Needs `pip install fastapi uvicorn`.

    Binds 0.0.0.0 by default — required for WSL: a 127.0.0.1-only bind inside WSL
    is not reliably reachable from the Windows browser. Open http://localhost:PORT
    on Windows (or http://<wsl-ip>:PORT if localhost forwarding is off)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]missing deps[/red] — run: pip install fastapi uvicorn")
        raise typer.Exit(1)
    console.print(f"[green]chainscope terminal[/green] -> open http://localhost:{port} on Windows")
    uvicorn.run("chainscope.ui.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
