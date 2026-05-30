"""Targeted OHLCV pull (a given timeframe) for ONLY the DEX-only non-CEX coins.

Reads _cex_verdicts_universe.jsonl (status==dex_only) joined to the universe files for pair+chain,
pulls OHLCV at <tf> for just those, appends to ohlcv_gt/<chain>_<tf>.parquet (dedup by pair).
Efficient: skips the ~80% of discovered pools that are major-CEX / major-pair / dust.

Usage: python3 _research/gt_ohlcv_targeted.py <tf hour|day> [chain_filter]
"""
from __future__ import annotations
import json, glob, sys, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd

DATA = Path("./data"); OUT = DATA / "ohlcv_gt"; OUT.mkdir(exist_ok=True)
GT = "https://api.geckoterminal.com/api/v2"; RATE_S = 2.4
TF = sys.argv[1] if len(sys.argv) > 1 else "hour"
CHAIN_FILTER = sys.argv[2] if len(sys.argv) > 2 else None

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
                ra = e.headers.get("Retry-After")
                time.sleep(max(float(ra) if ra and ra.isdigit() else 0, 5 * (a + 1))); continue
            if e.code in (404, 401): return None
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            time.sleep(5 * (a + 1)); continue   # transient DNS/network blip -> backoff + retry
    return None

def main():
    verds = {}
    f = DATA / "_cex_verdicts_universe.jsonl"
    for l in open(f):
        r = json.loads(l)
        if r["status"] == "dex_only":
            verds[r["pair"]] = r.get("chain", "bsc")
    # chain per pair from universe files (authoritative chain slug)
    chain_of = {}
    for uf in glob.glob(f"{DATA}/_gt_universe*.jsonl"):
        for l in open(uf):
            u = json.loads(l); chain_of[u["pair"]] = u.get("chain", "bsc")
    targets = [(p, chain_of.get(p, c)) for p, c in verds.items()]
    if CHAIN_FILTER:
        targets = [(p, c) for (p, c) in targets if c == CHAIN_FILTER]
    print(f"{len(targets)} dex_only targets for tf={TF}" + (f" (chain={CHAIN_FILTER})" if CHAIN_FILTER else ""), flush=True)

    by_chain = {}
    t0 = time.time(); ok = 0
    for n, (pair, chain) in enumerate(targets):
        d = gt(f"/networks/{chain}/pools/{pair}/ohlcv/{TF}?aggregate=1&limit=1000&currency=usd")
        lst = ((d or {}).get("data", {}).get("attributes", {}) or {}).get("ohlcv_list", []) or []
        for ts, o, h, l, c, v in lst:
            by_chain.setdefault(chain, []).append({"chain": chain, "pair_address": pair, "timeframe": TF,
                                                   "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
        ok += bool(lst)
        if (n + 1) % 20 == 0:
            print(f"  [{n+1}/{len(targets)}] ok={ok} ({time.time()-t0:.0f}s)", flush=True)
    for chain, rows in by_chain.items():
        path = OUT / f"{chain}_{TF}.parquet"
        new = pd.DataFrame(rows)
        if path.exists():
            old = pd.read_parquet(path)
            new = pd.concat([old[~old.pair_address.isin(new.pair_address.unique())], new], ignore_index=True)
        new.to_parquet(path, index=False)
        print(f"  wrote {path} ({new.pair_address.nunique()} pools, {len(new)} bars)", flush=True)
    print(f"[done] ok={ok}/{len(targets)} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
