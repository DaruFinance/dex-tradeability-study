"""Generalized single-chain GT universe+OHLCV puller (run one chain at a time; GT limit is per-IP).

Usage: python3 _research/gt_chain.py <network> <max_pages> <min_age_days> <min_reserve_usd> <tf>
  network: GT slug, eth | base | bsc | arbitrum | polygon_pos | ...
Writes _gt_universe_<network>.jsonl and ohlcv_gt/<network>_<tf>.parquet.
"""
from __future__ import annotations
import json, sys, time, urllib.request, urllib.error, datetime as dt
from pathlib import Path

DATA = Path("./data"); OUT = DATA / "ohlcv_gt"; OUT.mkdir(exist_ok=True)
GT = "https://api.geckoterminal.com/api/v2"
RATE_S = 2.4
NET = sys.argv[1]
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MIN_AGE_D = int(sys.argv[3]) if len(sys.argv) > 3 else 30
MIN_RES = float(sys.argv[4]) if len(sys.argv) > 4 else 20_000
TF = sys.argv[5] if len(sys.argv) > 5 else "day"
UNIV = DATA / f"_gt_universe_{NET}.jsonl"
NOW = dt.datetime.now(dt.timezone.utc)

_last = [0.0]
def gt(path, t=25, _retries=6):
    for attempt in range(_retries):
        w = RATE_S - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(GT + path, headers={"User-Agent": "cs", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=t) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                time.sleep(max(float(ra) if ra and ra.isdigit() else 0, 5 * (attempt + 1)))
                continue
            if e.code in (404, 401):
                return None
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(5 * (attempt + 1)); continue   # transient DNS/network blip -> retry
    return None

def _f(x):
    try: return float(x)
    except: return None

def age_days(c):
    if not c: return None
    try: return (NOW - dt.datetime.fromisoformat(c.replace("Z", "+00:00"))).days
    except: return None

def main():
    t0 = time.time(); seen, kept = set(), []
    print(f"[{NET}] discover <= {MAX_PAGES} pages, age>={MIN_AGE_D}d, reserve>=${MIN_RES:,.0f}", flush=True)
    for page in range(1, MAX_PAGES + 1):
        d = gt(f"/networks/{NET}/pools?page={page}")
        data = (d or {}).get("data", [])
        if not data:
            print(f"  [{NET}] page {page}: empty/cap", flush=True); break
        for p in data:
            a = p.get("attributes", {}); addr = (a.get("address") or "").lower()
            if not addr or addr in seen: continue
            seen.add(addr)
            res = _f(a.get("reserve_in_usd")) or 0.0; ad = age_days(a.get("pool_created_at"))
            if res >= MIN_RES and (ad is None or ad >= MIN_AGE_D):
                kept.append({"pair": addr, "name": a.get("name"), "reserve_usd": res,
                             "vol24": _f((a.get("volume_usd") or {}).get("h24")), "age_days": ad,
                             "base_token": (p.get("relationships", {}).get("base_token", {}).get("data", {}) or {}).get("id"),
                             "dex": (p.get("relationships", {}).get("dex", {}).get("data", {}) or {}).get("id"), "chain": NET})
        print(f"  [{NET}] page {page}: seen {len(seen)} kept {len(kept)}", flush=True)
    with open(UNIV, "w") as fh:
        for r in kept: fh.write(json.dumps(r) + "\n")
    print(f"[{NET}] {len(kept)} established pools in {time.time()-t0:.0f}s", flush=True)

    rows, ok = [], 0
    for n, r in enumerate(kept):
        d = gt(f"/networks/{NET}/pools/{r['pair']}/ohlcv/{TF}?aggregate=1&limit=1000&currency=usd")
        lst = ((d or {}).get("data", {}).get("attributes", {}) or {}).get("ohlcv_list", []) or []
        for ts, o, h, l, c, v in lst:
            rows.append({"chain": NET, "pair_address": r["pair"], "timeframe": TF,
                         "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
        ok += bool(lst)
        if (n + 1) % 25 == 0:
            print(f"  [{NET}] ohlcv [{n+1}/{len(kept)}] ok={ok} rows={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    import pandas as pd
    if rows:
        pd.DataFrame(rows).to_parquet(OUT / f"{NET}_{TF}.parquet", index=False)
    print(f"[{NET}] DONE established={len(kept)} ohlcv_ok={ok} bars={len(rows)} -> {OUT}/{NET}_{TF}.parquet", flush=True)

if __name__ == "__main__":
    main()
