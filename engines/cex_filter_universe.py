"""Apply the CEX-exclusion filter to the GT-discovered universe (_gt_universe.jsonl).

Per pool: coin = base_token address. If it's a known major/stable (WBNB/USDT/...) the pool is a
major-pair we wouldn't trade -> 'major_pair'. Else check CoinGecko listings -> 'major_cex'
(exclude) or 'dex_only' (keep). Writes _cex_verdicts_universe.jsonl. CoinGecko host (separate
rate limit from GeckoTerminal) so this can run concurrently with the GT OHLCV pull.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from _research.cex_filter import verdict_for, BASE_ASSETS, RATE_S
import glob

DATA = Path("./data")
OUT = DATA / "_cex_verdicts_universe.jsonl"


def main():
    rows = []
    for f in glob.glob(f"{DATA}/_gt_universe*.jsonl"):
        rows += [json.loads(l) for l in open(f)]
    # chain from dex slug suffix or default bsc; base_token "chain_0x.."
    done = {}
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l); done[r["pair"]] = r
    out = open(OUT, "a")
    n_dex = n_cex = n_major = n_new = 0
    for r in rows:
        if r["pair"] in done:
            continue
        bt = r.get("base_token") or ""
        parts = bt.split("_", 1)
        chain = parts[0] if len(parts) == 2 else "bsc"
        addr = parts[-1].lower()
        if not addr.startswith("0x"):
            continue
        if addr in BASE_ASSETS:
            status, venues = "major_pair", []
        else:
            v = verdict_for(chain, addr)
            if v.get("is_major_cex") is True:
                status, venues = "major_cex", v.get("cex_venues", [])
            elif v.get("is_major_cex") is False:
                status, venues = "dex_only", []
            else:
                continue  # rate-limited/err -> retry next run
        rec = {"pair": r["pair"], "chain": chain, "token": addr, "status": status,
               "cex_venues": venues, "name": r.get("name")}
        out.write(json.dumps(rec) + "\n"); out.flush()
        n_new += 1
        n_dex += status == "dex_only"; n_cex += status == "major_cex"; n_major += status == "major_pair"
        if n_new % 20 == 0:
            print(f"  {n_new} new | dex_only={n_dex} major_cex={n_cex} major_pair={n_major}", flush=True)
        time.sleep(RATE_S)
    print(f"[done] new={n_new} dex_only={n_dex} major_cex={n_cex} major_pair={n_major}", flush=True)


if __name__ == "__main__":
    main()
