"""MASS data collection via GeckoTerminal — thousands of tradeable coin histories across many chains.

Beats GT's ~200/list cap by hitting MANY endpoints per chain (default pools, volume/tx sorts,
new_pools, top-DEX pools) and dedup. Two resumable stages:
  A) DISCOVER  -> _mega_universe.jsonl  (pair, chain, reserve, vol24, age, dex, base_token)
  B) OHLCV     -> ohlcv_gt/mega_<chain>_day.parquet  (resumable via _mega_ohlcv_done.json)

Real chain-derived OHLCV. 'Tradeable' gate = reserve >= MIN_RES (recorded so we can tighten later).
Single sequential process (GT limit is per-IP). Runs for hours; safe to kill/resume.

Usage: python3 _research/gt_megacollect.py <stage discover|ohlcv|both> [min_reserve]
"""
from __future__ import annotations
import json, sys, time, urllib.request, urllib.error, datetime as dt
from pathlib import Path

DATA = Path("./data"); OUT = DATA / "ohlcv_gt"; OUT.mkdir(exist_ok=True)
# Route through CoinGecko's on-chain API (same data as GeckoTerminal) authenticated with our demo
# key -> separate, far-higher throughput than the throttled public GT (verified ~hundreds/min).
def _cg_key():
    for l in open(Path("./.env")):
        if l.startswith("COINGECKO_API_KEY="): return l.split("=", 1)[1].strip()
    return None
CG_KEY = _cg_key()
GT = "https://api.coingecko.com/api/v3/onchain"
RATE_S = 0.5  # ~120/min via authenticated CoinGecko-onchain (429 backoff handles any cap)
UNIV = DATA / "_mega_universe.jsonl"
STAGE = sys.argv[1] if len(sys.argv) > 1 else "both"
MIN_RES = float(sys.argv[2]) if len(sys.argv) > 2 else 5_000
TF = sys.argv[3] if len(sys.argv) > 3 else "day"          # day | hour | minute
DONE = DATA / f"_mega_ohlcv_done_{TF}.json"
NOW = dt.datetime.now(dt.timezone.utc)

# Priority chains first, then a broad long tail (28 high-activity GT networks). Skips any slug that 404s.
CHAINS = ["eth", "solana", "base", "bsc", "arbitrum", "avax", "sui-network", "tron",
          "polygon_pos", "hyperliquid", "hyperevm", "optimism", "blast", "scroll",
          "mantle", "linea", "sei-network", "ronin", "zksync", "mode", "sonic",
          "abstract", "berachain", "unichain", "ton", "pulsechain", "aptos", "ftm"]

_last = [0.0]
_PUB = "https://api.geckoterminal.com/api/v2"   # public GT (no key) = separate rate bucket
def gt(path, t=25, _r=8):
    for a in range(_r):
        w = RATE_S - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        _last[0] = time.time()
        for base, h in ((GT, {"User-Agent": "cs", "Accept": "application/json", **({"x-cg-demo-api-key": CG_KEY} if CG_KEY else {})}),
                        (_PUB, {"User-Agent": "cs", "Accept": "application/json"})):
            try:
                with urllib.request.urlopen(urllib.request.Request(base + path, headers=h), timeout=t) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (404, 401): return None
                if e.code in (429, 402, 503): continue   # throttled -> try the other source
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                continue
        time.sleep(4 * (a + 1))
    return None

def _f(x):
    try: return float(x)
    except: return None

def age_days(c):
    if not c: return None
    try: return (NOW - dt.datetime.fromisoformat(c.replace("Z", "+00:00"))).days
    except: return None

def parse_pools(chain, data, seen, out):
    for p in data or []:
        a = p.get("attributes", {}); addr = (a.get("address") or "").lower()
        if not addr or (chain, addr) in seen: continue
        res = _f(a.get("reserve_in_usd")) or 0.0
        seen.add((chain, addr))
        out.append({"chain": chain, "pair": addr, "name": a.get("name"), "reserve_usd": res,
                    "vol24": _f((a.get("volume_usd") or {}).get("h24")), "age_days": age_days(a.get("pool_created_at")),
                    "dex": (p.get("relationships", {}).get("dex", {}).get("data", {}) or {}).get("id"),
                    "base_token": (p.get("relationships", {}).get("base_token", {}).get("data", {}) or {}).get("id")})

def discover():
    seen = set(); out = []
    if UNIV.exists():
        for l in open(UNIV):
            r = json.loads(l); seen.add((r["chain"], r["pair"])); out.append(r)
        print(f"resume: {len(out)} pools already in universe", flush=True)
    base_n = len(out)
    for chain in CHAINS:
        c0 = len(out)
        # endpoint families that each return distinct ~200 sets
        eps = [f"/networks/{chain}/pools?page={{p}}",
               f"/networks/{chain}/pools?sort=h24_volume_usd_desc&page={{p}}",
               f"/networks/{chain}/pools?sort=h24_tx_count_desc&page={{p}}",
               f"/networks/{chain}/new_pools?page={{p}}"]
        chain_ok = False
        for ep in eps:
            for pg in range(1, 11):
                d = gt(ep.format(p=pg))
                data = (d or {}).get("data", [])
                if d is None and pg == 1 and ep == eps[0]:
                    break  # chain slug invalid
                if not data: break
                chain_ok = True; parse_pools(chain, data, seen, out)
        # top DEXs on this chain -> their pools (big breadth multiplier)
        if chain_ok:
            dx = gt(f"/networks/{chain}/dexes?page=1")
            dexes = [x.get("id") for x in (dx or {}).get("data", [])][:8]
            for dex in dexes:
                for pg in range(1, 6):
                    d = gt(f"/networks/{chain}/dexes/{dex}/pools?page={pg}")
                    data = (d or {}).get("data", [])
                    if not data: break
                    parse_pools(chain, data, seen, out)
        # checkpoint per chain
        with open(UNIV, "w") as fh:
            for r in out: fh.write(json.dumps(r) + "\n")
        trad = sum(1 for r in out if (r["reserve_usd"] or 0) >= MIN_RES)
        print(f"[discover] {chain}: +{len(out)-c0} (total {len(out)}, tradeable>={MIN_RES:.0f}: {trad})", flush=True)
    print(f"[discover DONE] {len(out)} pools ({len(out)-base_n} new this run)", flush=True)
    return out

def ohlcv():
    univ = [json.loads(l) for l in open(UNIV)] if UNIV.exists() else []
    targets = [r for r in univ if (r["reserve_usd"] or 0) >= MIN_RES]
    targets.sort(key=lambda r: -(r["reserve_usd"] or 0))   # highest-reserve (most tradeable) first
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    print(f"[ohlcv] {len(targets)} tradeable pools (reserve>={MIN_RES:.0f}); {len(done)} already pulled", flush=True)
    buf = {}; t0 = time.time(); pulled = 0
    def flush():
        import pandas as pd
        for chain, rows in buf.items():
            path = OUT / f"mega_{chain}_{TF}.parquet"
            new = pd.DataFrame(rows)
            if path.exists():
                old = pd.read_parquet(path); new = pd.concat([old, new], ignore_index=True)
            new.to_parquet(path, index=False)
        buf.clear()
    for n, r in enumerate(targets):
        key = f"{r['chain']}:{r['pair']}"
        if key in done: continue
        d = gt(f"/networks/{r['chain']}/pools/{r['pair']}/ohlcv/{TF}?aggregate=1&limit=1000&currency=usd")
        lst = ((d or {}).get("data", {}).get("attributes", {}) or {}).get("ohlcv_list", []) or []
        for ts, o, h, l, c, v in lst:
            buf.setdefault(r["chain"], []).append({"chain": r["chain"], "pair_address": r["pair"],
                "timeframe": TF, "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
        done.add(key); pulled += 1
        if pulled % 25 == 0:
            flush(); DONE.write_text(json.dumps(sorted(done)))
            print(f"[ohlcv] pulled {pulled} (done {len(done)}/{len(targets)}, {time.time()-t0:.0f}s, "
                  f"{(time.time()-t0)/pulled:.1f}s/coin)", flush=True)
    flush(); DONE.write_text(json.dumps(sorted(done)))
    print(f"[ohlcv DONE] pulled {pulled} this run; total done {len(done)}", flush=True)

if __name__ == "__main__":
    if STAGE in ("discover", "both"): discover()
    if STAGE in ("ohlcv", "both"): ohlcv()
