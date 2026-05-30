"""CEX-listing filter for the T_DEX universe.

Decides, per coin token, whether it is listed on any MAJOR CEX (-> EXCLUDE from the
DEX-only universe) using a SINGLE CoinGecko contract call per token (its inline `tickers`
array already enumerates venues; we do NOT page /tickers — that rate-limits the keyless
tier into uselessness). 404 = token not tracked by CoinGecko = obscure/DEX-only = INCLUDE.

Conservative: exclude any coin EVER listed on a major CEX (CoinGecko gives current
listings, not per-exchange listing dates). Verdicts are cached with observed_at.

Usage:
    python3 _research/cex_filter.py            # run over coins in pool_universe, cache verdicts
    python3 _research/cex_filter.py --selftest # CAKE must be EXCLUDED; a junk addr -> 404/include
"""
from __future__ import annotations
import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

DATA = Path("./data")
CACHE = Path("./_research/cex_verdicts.json")
CG = "https://api.coingecko.com/api/v3"
PLATFORM = {"bsc": "binance-smart-chain", "solana": "solana", "base": "base",
            "eth": "ethereum", "ethereum": "ethereum", "arbitrum": "arbitrum-one",
            "polygon_pos": "polygon-pos"}

# Demo key (x-cg-demo-api-key header) lifts the keyless throttle to ~30/min, 10k/mo.
def _cg_key():
    k = os.environ.get("COINGECKO_API_KEY")
    if k:
        return k
    env = Path("./.env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("COINGECKO_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None
CG_KEY = _cg_key()
RATE_S = 2.2 if CG_KEY else 8.0  # ~27/min with demo key, slow & polite without

# CoinGecko market.identifier values for MAJOR CEXs (the ones with real algo-trader
# competition). Avoiding these venues is the whole point of the universe.
MAJOR_CEX = {
    "binance", "binanceus", "binance_us", "coinbase_exchange", "gdax", "coinbase",
    "okex", "okx", "bybit_spot", "bybit", "kraken", "upbit", "kucoin", "gate",
    "bitget", "htx", "huobi", "mxc", "mexc", "bitfinex", "crypto_com", "cryptocom",
    "bitstamp", "gemini", "bithumb",
}
# Pool base/quote assets that are NEVER the "coin" (so we pick the other side as the token).
BASE_ASSETS = {
    # BSC
    "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c",  # WBNB
    "0x55d398326f99059ff775485246999027b3197955",  # USDT
    "0xe9e7cea3dedca5984780bafc599bd69add087d56",  # BUSD
    "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",  # USDC
    "0x2170ed0880ac9a755fd29b2688956bd959f933f8",  # ETH
    "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c",  # BTCB
}


def _get_json(url, timeout=15):
    headers = {"User-Agent": "chainscope-research"}
    if CG_KEY:
        headers["x-cg-demo-api-key"] = CG_KEY
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError):
                raise  # real HTTP status (404/429/etc) handled by caller
            time.sleep(3 * (attempt + 1))  # transient DNS/network blip -> retry
    raise urllib.error.URLError("network retries exhausted")


def coin_token_of_pair(token0, token1):
    """Pick the side of a pool that is the traded coin (not WBNB/stable/major base)."""
    t0 = (token0 or "").lower(); t1 = (token1 or "").lower()
    if t0 in BASE_ASSETS and t1 not in BASE_ASSETS:
        return t1
    if t1 in BASE_ASSETS and t0 not in BASE_ASSETS:
        return t0
    # both or neither base -> default to token0 (rare; flag later)
    return t0 or t1


def verdict_for(chain, token):
    """Return dict: is_major_cex (bool|None), cex_venues, n_tickers, tracked, source."""
    plat = PLATFORM.get(chain)
    if not plat or not token:
        return {"is_major_cex": None, "cex_venues": [], "n_tickers": 0, "tracked": False,
                "note": "no platform/token"}
    url = f"{CG}/coins/{plat}/contract/{token}"
    try:
        d = _get_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"is_major_cex": False, "cex_venues": [], "n_tickers": 0, "tracked": False,
                    "note": "404 not tracked -> DEX-only, include"}
        if e.code == 429:
            return {"is_major_cex": None, "rate_limited": True, "note": "429"}
        return {"is_major_cex": None, "note": f"http {e.code}"}
    except Exception as e:
        return {"is_major_cex": None, "note": f"err {type(e).__name__}"}
    tickers = d.get("tickers") or []
    venues = sorted({(t.get("market") or {}).get("identifier") for t in tickers
                     if (t.get("market") or {}).get("identifier")})
    cex_hits = sorted(v for v in venues if v in MAJOR_CEX)
    return {"is_major_cex": bool(cex_hits), "cex_venues": cex_hits, "n_tickers": len(tickers),
            "tracked": True, "symbol": d.get("symbol")}


def load_universe_tokens():
    import duckdb
    con = duckdb.connect()
    rows = con.sql(f"""SELECT chain, token0, token1 FROM
        read_parquet('{DATA}/pool_universe/**/*.parquet', union_by_name=true)""").df()
    out = {}  # (chain, token) -> None
    for _, r in rows.iterrows():
        tok = coin_token_of_pair(r["token0"], r["token1"])
        if tok:
            out[(r["chain"], tok.lower())] = None
    return list(out.keys())


def selftest():
    print(f"CG_KEY={'set' if CG_KEY else 'NONE'}  RATE_S={RATE_S}")
    cake = verdict_for("bsc", "0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82")
    print("CAKE:", cake)
    assert cake["is_major_cex"] is True, "CAKE must be flagged major-CEX"
    time.sleep(RATE_S)
    junk = verdict_for("bsc", "0x000000000000000000000000000000000000dead")
    print("junk:", junk)
    assert junk["is_major_cex"] is False, "untracked must be include"
    print("SELFTEST PASS")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    toks = load_universe_tokens()
    print(f"{len(toks)} unique coin tokens in pool_universe; {len(cache)} cached; rate={RATE_S}s")
    n_new = 0
    for chain, tok in toks:
        k = f"{chain}:{tok}"
        if k in cache and cache[k].get("is_major_cex") is not None:
            continue
        v = verdict_for(chain, tok)
        v["observed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cache[k] = v
        n_new += 1
        if n_new % 10 == 0:
            CACHE.write_text(json.dumps(cache, indent=0))
            print(f"  {n_new} new... last {k[:24]} -> cex={v.get('is_major_cex')} {v.get('cex_venues')}")
        time.sleep(RATE_S)
    CACHE.write_text(json.dumps(cache, indent=0))
    inc = sum(1 for v in cache.values() if v.get("is_major_cex") is False)
    exc = sum(1 for v in cache.values() if v.get("is_major_cex") is True)
    print(f"DONE: {len(cache)} verdicts | include(DEX-only)={inc} exclude(major-CEX)={exc}")


if __name__ == "__main__":
    main()
