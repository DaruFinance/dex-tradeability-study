"""ONE sequential GT pipeline (single shared 30/min IP limit): discover -> filter -> OHLCV.

Discovery via GeckoTerminal's live BSC pool listings (paginated) = currently-liquid coins, which
we filter to ESTABLISHED (age >= MIN_AGE_D) + reserve >= MIN_RES. Chain launch-discovery is the
wrong tool (yields mostly dead dust). Survivorship caveat: this seed is current-survivors; coins
that died mid-window are under-represented -> Phase-1 baseline is optimistic, to be corrected for
production by adding chain-discovered in-window deaths. Flagged, not hidden.

CEX filter (CoinGecko, separate host) is applied later by cex_filter.py.

Usage: python3 _research/gt_universe.py <max_pages> <min_age_days> <min_reserve_usd> <ohlcv_tf>
"""
from __future__ import annotations
import json, sys, time, urllib.request, urllib.error, datetime as dt
from pathlib import Path

DATA = Path("./data")
OUT = DATA / "ohlcv_gt"; OUT.mkdir(exist_ok=True)
UNIV = DATA / "_gt_universe.jsonl"
GT = "https://api.geckoterminal.com/api/v2"
RATE_S = 2.4
MAX_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MIN_AGE_D = int(sys.argv[2]) if len(sys.argv) > 2 else 30
MIN_RES = float(sys.argv[3]) if len(sys.argv) > 3 else 30_000
TF = sys.argv[4] if len(sys.argv) > 4 else "day"
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
                return None  # 404 = no data; 401 = GT free pagination cap (~page 10) -> stop gracefully
            raise
    return None

def _f(x):
    try: return float(x)
    except: return None

def age_days(created):
    if not created: return None
    try: return (NOW - dt.datetime.fromisoformat(created.replace("Z", "+00:00"))).days
    except: return None

def discover():
    """Paginate live BSC pools; keep established + liquid."""
    seen, kept = set(), []
    for page in range(1, MAX_PAGES + 1):
        d = gt(f"/networks/bsc/pools?page={page}")
        data = (d or {}).get("data", [])
        if not data:
            print(f"  page {page}: empty/cap reached", flush=True); break
        for p in data:
            a = p.get("attributes", {}); addr = (a.get("address") or "").lower()
            if not addr or addr in seen: continue
            seen.add(addr)
            res = _f(a.get("reserve_in_usd")) or 0.0
            ad = age_days(a.get("pool_created_at"))
            base = (p.get("relationships", {}).get("base_token", {}).get("data", {}) or {}).get("id", "")
            rec = {"pair": addr, "name": a.get("name"), "reserve_usd": res,
                   "vol24": _f((a.get("volume_usd") or {}).get("h24")), "age_days": ad,
                   "base_token": base, "dex": (p.get("relationships", {}).get("dex", {}).get("data", {}) or {}).get("id")}
            if res >= MIN_RES and (ad is None or ad >= MIN_AGE_D):
                kept.append(rec)
        print(f"  page {page}: total seen {len(seen)}, kept {len(kept)}", flush=True)
    return kept

def main():
    t0 = time.time()
    print(f"discover: GT BSC pools, <= {MAX_PAGES} pages, age>={MIN_AGE_D}d, reserve>=${MIN_RES:,.0f}", flush=True)
    universe = discover()
    with open(UNIV, "w") as fh:
        for r in universe: fh.write(json.dumps(r) + "\n")
    print(f"discovered {len(universe)} established+liquid pools in {time.time()-t0:.0f}s", flush=True)

    rows, st = [], {"ok": 0, "nodata": 0}
    for n, r in enumerate(universe):
        d = gt(f"/networks/bsc/pools/{r['pair']}/ohlcv/{TF}?aggregate=1&limit=1000&currency=usd")
        lst = ((d or {}).get("data", {}).get("attributes", {}) or {}).get("ohlcv_list", []) or []
        if lst:
            for ts, o, h, l, c, v in lst:
                rows.append({"chain": "bsc", "pair_address": r["pair"], "timeframe": TF,
                             "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
            st["ok"] += 1
        else:
            st["nodata"] += 1
        if (n + 1) % 20 == 0:
            print(f"  ohlcv [{n+1}/{len(universe)}] ok={st['ok']} rows={len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    import pandas as pd
    if rows:
        pd.DataFrame(rows).to_parquet(OUT / f"bsc_univ_{TF}.parquet", index=False)
    print(f"[done] discovered={len(universe)} ohlcv_ok={st['ok']} bars={len(rows)} in {time.time()-t0:.0f}s "
          f"-> {OUT}/bsc_univ_{TF}.parquet", flush=True)

if __name__ == "__main__":
    main()
