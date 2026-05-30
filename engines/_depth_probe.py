import os, urllib.request, json, time
KEY=os.environ["THEGRAPH_TOKEN_API_KEY"]
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
B="https://token-api.thegraph.com"; pool="0x0ed7e52944161450477ee417de9cd3a859b14fd0"
def get(url, tries=5):
    for a in range(tries):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":UA,"Accept":"application/json","X-Api-Key":KEY})
            with urllib.request.urlopen(req,timeout=40) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403,429): time.sleep(20*(a+1)); continue
            return {"_http":e.code}
        except Exception as ex: time.sleep(15); continue
    return {"_err":"exhausted"}
time.sleep(30)  # let any burst-block clear
now=int(time.time())
for days in (90,365):
    s=now-days*86400; e=s+3*86400
    d=get(f"{B}/v1/evm/swaps?network=bsc&pool={pool}&start_time={s}&end_time={e}&limit=20")
    rows=d.get("data") if isinstance(d,dict) else None
    if rows: print(f"~{days}d ago: {len(rows)} swaps, oldest {rows[-1]['datetime']}", flush=True)
    else: print(f"~{days}d ago: {d}", flush=True)
    time.sleep(25)
