"""Pull per-coin DAILY FLOW (buy/sell USD, unique buyers, trade count) from Bitquery V2 DEXTrades.

Server-side aggregation: ~2 queries/coin (buy-side + sell-side) → full daily flow time series.
This is the SELECTION signal price data can't give: net-buy pressure + buyer growth + smart-money.
Resumable. Writes _bitquery_flow_daily.parquet. Conservative rate (free tier).

Usage: python3 _research/bitquery_flow.py <max_coins> [chains_csv]
"""
from __future__ import annotations
import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd

DATA = Path("./data")
TOK = os.environ["BITQUERY_TOKEN"]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
OUT = DATA / "_bitquery_flow_daily.parquet"; DONE = DATA / "_bq_flow_done.json"
MAXC = int(sys.argv[1]) if len(sys.argv) > 1 else 400
CHAINS = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["bsc", "base", "eth", "arbitrum", "polygon_pos", "avax"])
BQ_NET = {"bsc": "bsc", "base": "base", "eth": "eth", "arbitrum": "arbitrum", "polygon_pos": "matic", "avax": "avalanche"}
START = "2025-11-01T00:00:00Z"   # ~6mo back (covers our daily OHLCV window)

def gql(query, _r=5):
    for a in range(_r):
        try:
            req = urllib.request.Request("https://streaming.bitquery.io/graphql",
                data=json.dumps({"query": query}).encode(),
                headers={"Content-Type": "application/json", "User-Agent": UA, "Authorization": f"Bearer {TOK}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
                if "errors" in d and not d.get("data"): return {"_err": str(d["errors"])[:120]}
                return d
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 502): time.sleep(5 * (a + 1)); continue
            return {"_http": e.code}
        except Exception: time.sleep(3 * (a + 1)); continue
    return {"_err": "retries"}

def side_query(net, token, side):
    # FIX: DEXTradeByTokens + Side.Type + high limit -> FULL daily history (DEXTrades default-limited to
    # recent trades -> only ~2 days/coin, the bug that wrecked the first pull). Validate when points avail.
    return f'''{{ EVM(network: {net}, dataset: combined) {{ DEXTradeByTokens(
      where: {{Trade:{{Currency:{{SmartContract:{{is:"{token}"}}}}, Side:{{Type:{{is: {side}}}}}}}, Block:{{Time:{{after:"{START}"}}}}}}
      limit: {{count: 5000}} orderBy: {{ascendingByField: "d"}}) {{
      d: Block {{ Time(interval:{{in:days count:1}}) }}
      usd: sum(of: Trade_Side_AmountInUSD)
      traders: count(distinct: Transaction_From)
      n: count }} }} }}'''

def main():
    univ = [json.loads(l) for l in open(DATA / "_mega_universe.jsonl")]
    # token address from base_token "chain_0x..", restrict to mappable EVM chains + reserve-sorted
    cand = []
    for r in univ:
        ch = r["chain"]; bt = r.get("base_token") or ""
        if ch not in CHAINS or ch not in BQ_NET or "_" not in bt: continue
        tok = bt.split("_", 1)[1]
        if tok.startswith("0x"): cand.append((ch, tok, r.get("reserve_usd") or 0))
    cand = sorted({(c, t): rv for c, t, rv in cand}.items(), key=lambda x: -x[1])
    cand = [(c, t) for (c, t), _ in cand][:MAXC]
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    print(f"{len(cand)} EVM coins to pull flow for; {len(done)} done", flush=True)
    rows = []
    if OUT.exists(): rows = pd.read_parquet(OUT).to_dict("records")
    t0 = time.time(); pulled = 0
    for ch, tok in cand:
        key = f"{ch}:{tok}"
        if key in done: continue
        net = BQ_NET[ch]; day = {}
        for side in ("buy", "sell"):
            d = gql(side_query(net, tok, side))
            tr = (((d or {}).get("data") or {}).get("EVM") or {}).get("DEXTradeByTokens") if isinstance(d, dict) else None
            for row in (tr or []):
                dd = row["Block"]["d"][:10]
                day.setdefault(dd, {}).update({f"{side}_usd": float(row["usd"] or 0),
                                               f"{side}_traders": int(row["traders"] or 0), f"{side}_n": int(row["n"] or 0)})
            time.sleep(0.6)
        for dd, vals in day.items():
            rows.append({"chain": ch, "token": tok, "date": dd, **vals})
        done.add(key); pulled += 1
        if pulled % 25 == 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False); DONE.write_text(json.dumps(sorted(done)))
            print(f"  pulled {pulled} (done {len(done)}/{len(cand)}, {time.time()-t0:.0f}s, {(time.time()-t0)/pulled:.1f}s/coin)", flush=True)
    pd.DataFrame(rows).to_parquet(OUT, index=False); DONE.write_text(json.dumps(sorted(done)))
    print(f"[done] pulled {pulled}; {OUT} has {len(rows)} coin-day flow rows", flush=True)

if __name__ == "__main__":
    main()
