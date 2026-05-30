"""Pull historical per-coin FLOW from The Graph Token API /v1/evm/swaps (the differentiated signal).

Pages a pool's swaps (with USD value + trader wallet), classifies buy/sell of the coin, aggregates
to DAILY buy_usd / sell_usd / unique buyers / trade count. The Token API is rate-limited (no points
cap), so this is paced SLOW with long backoff and runs for hours — resumable. Writes
_tokenapi_flow_daily.parquet. This is what price data can't give: net-buy pressure + smart-money.

Usage: python3 _research/tokenapi_flow.py <max_coins> <pace_seconds>
"""
from __future__ import annotations
import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd

DATA = Path("./data")
KEY = os.environ["THEGRAPH_TOKEN_API_KEY"]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
B = "https://token-api.thegraph.com"
OUT = DATA / "_tokenapi_flow_daily.parquet"; DONE = DATA / "_tapi_flow_done.json"
MAXC = int(sys.argv[1]) if len(sys.argv) > 1 else 800
PACE = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
START = "2025-11-01T00:00:00Z"
NET = {"bsc": "bsc", "base": "base", "eth": "mainnet", "arbitrum": "arbitrum-one",
       "polygon_pos": "polygon", "avax": "avalanche", "optimism": "optimism"}
_last = [0.0]

def get(path, _r=7):
    for a in range(_r):
        w = PACE - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(B + path, headers={"User-Agent": UA, "Accept": "application/json", "X-Api-Key": KEY})
            with urllib.request.urlopen(req, timeout=45) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 503, 502):
                time.sleep(min(20 * (a + 1), 180)); continue   # long backoff on throttle
            return {"_http": e.code}
        except Exception:
            time.sleep(10 * (a + 1)); continue
    return {"_err": "retries"}

def main():
    univ = [json.loads(l) for l in open(DATA / "_mega_universe.jsonl")]
    cand = []
    for r in univ:
        ch = r["chain"]; bt = r.get("base_token") or ""
        if ch not in NET or "_" not in bt: continue
        tok = bt.split("_", 1)[1].lower()
        if tok.startswith("0x"): cand.append((ch, r["pair"], tok, r.get("reserve_usd") or 0))
    cand = sorted({(c, p, t): rv for c, p, t, rv in cand}.items(), key=lambda x: -x[1])
    cand = [k for k, _ in cand][:MAXC]
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    rows = pd.read_parquet(OUT).to_dict("records") if OUT.exists() else []
    print(f"{len(cand)} coins; {len(done)} done; pace {PACE}s", flush=True)
    t0 = time.time(); pulled = 0
    for ch, pair, tok in cand:
        key = f"{ch}:{pair}"
        if key in done: continue
        net = NET[ch]; day = {}; page = 1
        while page <= 25:    # cap pages/coin to bound cost
            d = get(f"/v1/evm/swaps?network={net}&pool={pair}&start_time={int(time.mktime(time.strptime(START,'%Y-%m-%dT%H:%M:%SZ')))}&limit=1000&page={page}")
            sw = d.get("data") if isinstance(d, dict) else None
            if not sw: break
            for s in sw:
                dd = (s.get("datetime") or "")[:10]
                if not dd: continue
                out = s.get("output_token") or {}; inp = s.get("input_token") or {}
                out_addr = (out.get("address", "") if isinstance(out, dict) else str(out)).lower()
                in_addr = (inp.get("address", "") if isinstance(inp, dict) else str(inp)).lower()
                usd = float(s.get("output_value") or 0)
                buyer = (s.get("transaction_from") or "").lower()
                e = day.setdefault(dd, {"buy_usd": 0.0, "sell_usd": 0.0, "buyers": set(), "n": 0})
                if out_addr == tok:   e["buy_usd"] += usd; e["buyers"].add(buyer)   # coin received = BUY
                elif in_addr == tok:  e["sell_usd"] += usd                          # coin spent = SELL
                e["n"] += 1
            if len(sw) < 1000: break
            page += 1
        for dd, e in day.items():
            rows.append({"chain": ch, "token": tok, "pair": pair, "date": dd,
                         "buy_usd": e["buy_usd"], "sell_usd": e["sell_usd"], "buyers": len(e["buyers"]), "n": e["n"]})
        done.add(key); pulled += 1
        if pulled % 10 == 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False); DONE.write_text(json.dumps(sorted(done)))
            print(f"  pulled {pulled} (done {len(done)}/{len(cand)}, {time.time()-t0:.0f}s, {(time.time()-t0)/pulled:.0f}s/coin, rows {len(rows)})", flush=True)
    pd.DataFrame(rows).to_parquet(OUT, index=False); DONE.write_text(json.dumps(sorted(done)))
    print(f"[done] {pulled} coins, {len(rows)} coin-day flow rows", flush=True)

if __name__ == "__main__":
    main()
