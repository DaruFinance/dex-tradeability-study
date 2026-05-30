"""Fast OHLCV ingest via GeckoTerminal (real chain-derived data, ~1 call = 42d hourly / 180d daily).

Discovery stays survivorship-free (chain factory scan -> pool_universe). This fills price/volume
history fast. Per-bar reserve isn't in GT OHLCV; we capture CURRENT pool reserve (multi endpoint)
for the liquidity gate + cost-model depth proxy (validated vs raw indexer on a sample separately).

Usage: python3 _research/gt_ingest.py <timeframe day|hour|minute> <limit_coins>
Reads pool_universe (bsc), writes ohlcv parquet + a gt_pool_meta.jsonl (reserve/vol/dex).
"""
from __future__ import annotations
import json, sys, time, urllib.request, urllib.error
from pathlib import Path
import duckdb

DATA = Path("./data")
OUT = DATA / "ohlcv_gt"; OUT.mkdir(exist_ok=True)
META = DATA / "_gt_pool_meta.jsonl"
GT = "https://api.geckoterminal.com/api/v2"
TF = sys.argv[1] if len(sys.argv) > 1 else "hour"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
RATE_S = 2.4  # ~25/min, under GT's 30/min cap


_last = [0.0]
def gt(path, t=25, _retries=6):
    """Rate-limited GT fetch with 429 backoff. Enforces >=RATE_S between calls + honors Retry-After."""
    for attempt in range(_retries):
        wait = RATE_S - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(GT + path, headers={"User-Agent": "cs", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=t) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                back = max(float(ra) if ra and ra.isdigit() else 0, 5 * (attempt + 1))
                time.sleep(back)
                continue
            raise
    raise urllib.error.HTTPError(GT + path, 429, "rate-limited after retries", None, None)


def pairs_from_universe():
    con = duckdb.connect()
    df = con.sql(f"""SELECT DISTINCT pair_address, dex FROM
        read_parquet('{DATA}/pool_universe/**/*.parquet', union_by_name=true)
        WHERE chain='bsc'""").df()
    return list(df.itertuples(index=False, name=None))


def multi_pool_meta(addrs):
    """Batch current reserve/volume for up to 30 pools/call."""
    out = {}
    for i in range(0, len(addrs), 30):
        chunk = addrs[i:i+30]
        try:
            d = gt(f"/networks/bsc/pools/multi/{','.join(chunk)}")
            for p in d.get("data", []):
                a = p.get("attributes", {})
                out[a.get("address", "").lower()] = {
                    "reserve_usd": a.get("reserve_in_usd"),
                    "vol24": (a.get("volume_usd") or {}).get("h24"),
                    "name": a.get("name"), "created": a.get("pool_created_at"),
                    "base_token": (p.get("relationships", {}).get("base_token", {})
                                   .get("data", {}) or {}).get("id"),
                }
        except Exception as e:
            print(f"  multi err {i}: {type(e).__name__}", flush=True)
    return out


def main():
    pairs = pairs_from_universe()[:LIMIT]
    addrs = [p[0] for p in pairs]
    print(f"{len(pairs)} bsc pairs from pool_universe; tf={TF}", flush=True)

    t0 = time.time()
    meta = multi_pool_meta(addrs)
    print(f"pool meta for {len(meta)} pools in {time.time()-t0:.0f}s", flush=True)
    with open(META, "w") as fh:
        for a, m in meta.items():
            fh.write(json.dumps({"pair": a, **m}) + "\n")

    rows = []
    stats = {"ok": 0, "nodata": 0, "err": 0}
    for n, (pair, dex) in enumerate(pairs):
        try:
            d = gt(f"/networks/bsc/pools/{pair}/ohlcv/{TF}?aggregate=1&limit=1000&currency=usd")
            lst = d.get("data", {}).get("attributes", {}).get("ohlcv_list", []) or []
            if lst:
                for ts, o, h, l, c, v in lst:
                    rows.append({"chain": "bsc", "pair_address": pair, "timeframe": TF,
                                 "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
                stats["ok"] += 1
            else:
                stats["nodata"] += 1
        except urllib.error.HTTPError as e:
            if e.code == 404: stats["nodata"] += 1
            else: stats["err"] += 1
        except Exception:
            stats["err"] += 1
        if (n + 1) % 25 == 0:
            print(f"  [{n+1}/{len(pairs)}] ok={stats['ok']} nodata={stats['nodata']} err={stats['err']} "
                  f"rows={len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    import pandas as pd
    if rows:
        pd.DataFrame(rows).to_parquet(OUT / f"bsc_{TF}.parquet", index=False)
    print(f"[done] {stats} | {len(rows)} bars from {stats['ok']} pools in {time.time()-t0:.0f}s "
          f"-> {OUT}/bsc_{TF}.parquet", flush=True)


if __name__ == "__main__":
    main()
