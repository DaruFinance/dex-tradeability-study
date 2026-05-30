"""Recover non-EVM chains (Solana, Tron, Aptos, TON) whose base58/non-hex pool addresses are
CASE-SENSITIVE: the mega-collector lowercased them and broke OHLCV lookups (dropped ~1,360 Solana
pools, the biggest memecoin market). This re-discovers them PRESERVING CASE and pulls daily+hourly
OHLCV via CoinGecko-onchain. Writes mega_<chain>_<tf>.parquet (joins the corpus). Resumable.

Usage: python3 _research/gt_noncase.py <tf day|hour> [min_reserve]
"""
from __future__ import annotations
import json, sys, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd

DATA = Path("./data"); OUT = DATA / "ohlcv_gt"
CHAINS = ["solana", "tron", "aptos", "ton"]
TF = sys.argv[1] if len(sys.argv) > 1 else "day"
MIN_RES = float(sys.argv[2]) if len(sys.argv) > 2 else 5_000
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
CG = "https://api.coingecko.com/api/v3/onchain"
CG_KEY = open("./.env").read().split("COINGECKO_API_KEY=")[1].split()[0]
DONE = DATA / f"_noncase_done_{TF}.json"; UNIV = DATA / "_noncase_universe.jsonl"
_last = [0.0]

PUB = "https://api.geckoterminal.com/api/v2"   # public GT (no key) = separate rate bucket
def _one(base, path, hdr):
    req = urllib.request.Request(base + path, headers=hdr)
    with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)
def gt(path, _r=8):
    for a in range(_r):
        w = 0.6 - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        _last[0] = time.time()
        # try CG-onchain (key) first; on throttle fall back to public GT (no key, separate bucket)
        for base, hdr in ((CG, {"User-Agent": UA, "x-cg-demo-api-key": CG_KEY}),
                          (PUB, {"User-Agent": UA})):
            try:
                return _one(base, path, hdr)
            except urllib.error.HTTPError as e:
                if e.code in (404, 401): return None
                if e.code in (429, 402, 503): continue   # try the other source
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                continue
        time.sleep(4 * (a + 1))   # both throttled -> backoff and retry
    return None

def discover():
    seen, out = set(), []
    if UNIV.exists():
        for l in open(UNIV):
            r = json.loads(l); seen.add((r["chain"], r["pair"])); out.append(r)
    for chain in CHAINS:
        c0 = len(out)
        eps = [f"/networks/{chain}/pools?page={{p}}",
               f"/networks/{chain}/pools?sort=h24_volume_usd_desc&page={{p}}",
               f"/networks/{chain}/new_pools?page={{p}}"]
        ok = False
        for ep in eps:
            for pg in range(1, 11):
                d = gt(ep.format(p=pg)); data = (d or {}).get("data", [])
                if d is None and pg == 1 and ep == eps[0]: break
                if not data: break
                ok = True
                for p in data:
                    a = p.get("attributes", {}); addr = a.get("address")    # PRESERVE CASE (no .lower())
                    if not addr or (chain, addr) in seen: continue
                    seen.add((chain, addr))
                    out.append({"chain": chain, "pair": addr, "name": a.get("name"),
                                "reserve_usd": float(a.get("reserve_in_usd") or 0) if a.get("reserve_in_usd") else 0.0,
                                "base_token": (p.get("relationships", {}).get("base_token", {}).get("data", {}) or {}).get("id")})
        if ok:
            dx = gt(f"/networks/{chain}/dexes?page=1")
            for dex in [x.get("id") for x in (dx or {}).get("data", [])][:6]:
                for pg in range(1, 6):
                    d = gt(f"/networks/{chain}/dexes/{dex}/pools?page={pg}"); data = (d or {}).get("data", [])
                    if not data: break
                    for p in data:
                        a = p.get("attributes", {}); addr = a.get("address")
                        if not addr or (chain, addr) in seen: continue
                        seen.add((chain, addr))
                        out.append({"chain": chain, "pair": addr, "name": a.get("name"),
                                    "reserve_usd": float(a.get("reserve_in_usd") or 0) if a.get("reserve_in_usd") else 0.0,
                                    "base_token": (p.get("relationships", {}).get("base_token", {}).get("data", {}) or {}).get("id")})
        with open(UNIV, "w") as fh:
            for r in out: fh.write(json.dumps(r) + "\n")
        print(f"[discover] {chain}: +{len(out)-c0} (total {len(out)})", flush=True)
    return out

def main():
    # skip redundant re-discovery if universe already populated (resume straight to OHLCV)
    if UNIV.exists() and sum(1 for _ in open(UNIV)) > 500:
        univ = [json.loads(l) for l in open(UNIV)]
        print(f"[resume] {len(univ)} non-EVM pools already discovered; skipping re-discovery", flush=True)
    else:
        univ = discover()
    targets = sorted([r for r in univ if (r["reserve_usd"] or 0) >= MIN_RES], key=lambda r: -(r["reserve_usd"] or 0))
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    print(f"[ohlcv {TF}] {len(targets)} tradeable non-EVM coins; {len(done)} done", flush=True)
    buf = {}; t0 = time.time(); pulled = 0
    def flush():
        for chain, rows in buf.items():
            path = OUT / f"mega_{chain}_{TF}.parquet"
            new = pd.DataFrame(rows)
            if path.exists():
                old = pd.read_parquet(path); new = pd.concat([old[~old.pair_address.isin(new.pair_address.unique())], new], ignore_index=True)
            new.to_parquet(path, index=False)
        buf.clear()
    for r in targets:
        key = f"{r['chain']}:{r['pair']}"
        if key in done: continue
        d = gt(f"/networks/{r['chain']}/pools/{r['pair']}/ohlcv/{TF}?aggregate=1&limit=1000&currency=usd")
        lst = ((d or {}).get("data", {}).get("attributes", {}) or {}).get("ohlcv_list", []) or []
        for ts, o, h, l, c, v in lst:
            buf.setdefault(r["chain"], []).append({"chain": r["chain"], "pair_address": r["pair"], "timeframe": TF,
                "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
        done.add(key); pulled += 1
        if pulled % 25 == 0:
            flush(); DONE.write_text(json.dumps(sorted(done)))
            print(f"  [{TF}] pulled {pulled}/{len(targets)} ({time.time()-t0:.0f}s)", flush=True)
    flush(); DONE.write_text(json.dumps(sorted(done)))
    print(f"[done {TF}] pulled {pulled}", flush=True)

if __name__ == "__main__":
    main()
