"""Build a SURVIVORSHIP-FREE daily panel: chain-discover launches from an OLD window (includes
coins that have since died), then pull GT daily OHLCV (GT retains dead-pool history -> captures the
pump AND the rug). This is the panel needed to honestly test cross-sectional momentum: if the edge
is just 'survivors kept rising', it will COLLAPSE here because momentum will buy coins right before
they die.

Usage: python3 _research/surv_free_panel.py <from_block> <to_block> <n_coins>
"""
from __future__ import annotations
import asyncio, json, sys, time, urllib.request, urllib.error
from pathlib import Path
sys.path.insert(0, ".")
import chainscope.universe as _uni
from chainscope.aggregate import Client
from chainscope.universe import UniverseBuilder
from chainscope.cohort import _coin_side

DATA = Path("./data"); OUT = DATA / "ohlcv_gt"
GT = "https://api.geckoterminal.com/api/v2"; RATE_S = 2.4
_uni.BSC_CHUNK = 10_000
_uni.BSC_MAX_CHUNKS = 600
FROM_B, TO_B, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

_last = [0.0]
def gt(path, t=25, _r=6):
    for a in range(_r):
        w = RATE_S - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(GT + path, headers={"User-Agent": "cs", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=t) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("Retry-After"); time.sleep(max(float(ra) if ra and ra.isdigit() else 0, 5 * (a + 1))); continue
            if e.code in (404, 401): return None
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(5 * (a + 1)); continue
    return None


async def discover():
    async with Client() as cs:
        ub = UniverseBuilder(cs.http, cs.settings)
        creations = await ub.scan_bsc(from_block=FROM_B, to_block=TO_B)
    pool = [c for c in creations if not (c.dex and "four" in (c.dex or "").lower())]
    seen, out = set(), []
    for c in sorted(pool, key=lambda x: x.created_block or 0, reverse=True):
        coin = _coin_side(c)
        if c.pair_address and c.pair_address.lower() not in seen:
            seen.add(c.pair_address.lower()); out.append(c.pair_address.lower())
        if len(out) >= N: break
    return out


def main():
    pairs = asyncio.run(discover())
    print(f"chain-discovered {len(pairs)} pools (survivorship-free incl. since-dead) in window", flush=True)
    rows, ok, nodata = [], 0, 0
    t0 = time.time()
    for n, pair in enumerate(pairs):
        d = gt(f"/networks/bsc/pools/{pair}/ohlcv/day?aggregate=1&limit=1000&currency=usd")
        lst = ((d or {}).get("data", {}).get("attributes", {}) or {}).get("ohlcv_list", []) or []
        if lst:
            for ts, o, h, l, c, v in lst:
                rows.append({"chain": "bsc_survfree", "pair_address": pair, "timeframe": "day",
                             "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
            ok += 1
        else:
            nodata += 1
        if (n + 1) % 25 == 0:
            print(f"  ohlcv [{n+1}/{len(pairs)}] ok={ok} nodata={nodata} rows={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    import pandas as pd
    if rows:
        pd.DataFrame(rows).to_parquet(OUT / "survfree_bsc_day.parquet", index=False)
    # bar-count distribution: how many coins have a multi-week life (usable for daily panel)
    import collections
    bc = collections.Counter()
    for r in rows: bc[r["pair_address"]] += 1
    usable = sum(1 for v in bc.values() if v >= 20)
    print(f"[done] {ok} pools with GT history ({nodata} none); {usable} have >=20 daily bars (usable). "
          f"-> {OUT}/survfree_bsc_day.parquet", flush=True)


if __name__ == "__main__":
    main()
