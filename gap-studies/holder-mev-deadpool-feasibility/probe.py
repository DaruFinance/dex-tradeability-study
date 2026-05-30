#!/usr/bin/env python3
"""Bounded feasibility probe: holder data, MEV/flow metric, dead-pool universe.

GAP under test: are the three uncollected datasets (holders, MEV, survivorship-free
dead-pool universe) FEASIBLE to compute from on-disk data + bounded public RPC, and
which gap-test does each unblock? This is a PROBE + PLAN, not a backfill. Serial RPC
only, with sleeps, to respect rate limits (CoinGecko 429 / public Solana 429 / dataseed
getLogs limits are all real and documented in the results).

Reproducible: deterministic (no randomness); RPC results may vary with chain head.
Run: python3 probe.py
"""
import glob
import json
import time
import urllib.request

import numpy as np
import pandas as pd

OUT = "./_gaps/holder-mev-deadpool-feasibility"
DATA = "./data"
TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
PAIRCREATED = "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9"
PANCAKE_V2_FACTORY = "0xca143ce32fe78f1f7019d7d551a6402fc5350c73"


def rpc(url, method, params, tmo=20):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    t = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=tmo)
        d = json.load(r)
        return d.get("result"), round(time.time() - t, 2), d.get("error")
    except Exception as e:  # noqa
        return None, round(time.time() - t, 2), str(e)[:140]


# ----------------------------------------------------------------------------
# PART A: MEV sandwich metric on the real BSC trade tape (fully on-disk)
# Re-implements chainscope.mev.detect_sandwiches logic per (pair, block).
# ----------------------------------------------------------------------------
def detect_block(g):
    g = g.reset_index(drop=True)
    n = len(g)
    if n < 3:
        return 0, 0, []
    pre = None
    for p in g.price_usd.values:
        if pd.notna(p) and p > 0:
            pre = p
            break
    side = g.side.str.lower().str[0].values
    mk = g.maker.values
    px = g.price_usd.values
    used, attacks, victims, fr = set(), 0, 0, []
    for i in range(n):
        if side[i] != "b" or pd.isna(mk[i]):
            continue
        a = mk[i]
        j = None
        for k in range(i + 1, n):
            if k in used:
                continue
            if mk[k] == a and side[k] == "s":
                j = k
                break
        if j is None:
            continue
        brack = [m for m in range(i + 1, j) if mk[m] != a and side[m] == "b"]
        if not brack:
            continue
        used.add(j)
        attacks += 1
        for m in brack:
            victims += 1
            if pre and pd.notna(px[m]) and px[m] > 0:
                fr.append(max(0.0, (px[m] - pre) / pre))
    return attacks, victims, fr


def part_a_mev():
    files = sorted(glob.glob(f"{DATA}/trades/chain=bsc/dt=*/*.parquet"))
    frames = []
    for f in files:
        d = pd.read_parquet(f, columns=["pair_address", "block_number", "log_index",
                                        "side", "price_usd", "maker"])
        if len(d):
            frames.append(d)
    df = pd.concat(frames, ignore_index=True).sort_values(["pair_address", "block_number", "log_index"])

    # intra-block density (the binding limitation)
    g = df.groupby(["pair_address", "block_number"]).size()
    dens = g.value_counts().sort_index()
    ge3 = float((g >= 3).mean())

    rows = []
    for pair, sub in df.groupby("pair_address"):
        if len(sub) < 10:
            continue
        ta = tv = 0
        allfr = []
        nblk = sub.block_number.nunique()
        for _, grp in sub.groupby("block_number"):
            a, v, fr = detect_block(grp)
            ta += a
            tv += v
            allfr += fr
        rows.append(dict(pair=pair, n_trades=len(sub), n_blocks=nblk, attacks=ta, victims=tv,
                         victim_rate=tv / len(sub),
                         mean_extra_slip_bp=round(np.mean(allfr) * 1e4, 1) if allfr else 0.0))
    res = pd.DataFrame(rows).sort_values("attacks", ascending=False)
    res.to_csv(f"{OUT}/mev_sandwich_per_pair.csv", index=False)
    return df, dens, ge3, res


# ----------------------------------------------------------------------------
# PART B: Holder feasibility (bounded, serial RPC)
# ----------------------------------------------------------------------------
def part_b_holders():
    log = []
    # EVM: confirm eth_getLogs Transfer works on a busy contract (WBNB), measure throughput
    WBNB = "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"
    EP = "https://bsc.publicnode.com"
    lat, _, _ = rpc(EP, "eth_blockNumber", [])
    if isinstance(lat, str):
        lat = int(lat, 16)
        res, dt, err = rpc(EP, "eth_getLogs",
                           [{"fromBlock": hex(lat - 30), "toBlock": hex(lat),
                             "address": WBNB, "topics": [TRANSFER]}])
        log.append(("evm_getLogs_wbnb_30blk", None if err else len(res), dt, str(err)))
    time.sleep(1)
    # Solana: getTokenSupply works on public RPC; getTokenLargestAccounts is 429-limited
    SOL = "https://api.mainnet-beta.solana.com"
    mint = "G74wEtdgLbzdkUwaeAjW98dktkA6E6hfJ4vZ4hy7pump"
    res, dt, err = rpc(SOL, "getTokenSupply", [mint])
    log.append(("sol_getTokenSupply", None if err else res.get("value", {}).get("uiAmount"), dt, str(err)))
    time.sleep(3)
    res, dt, err = rpc(SOL, "getTokenLargestAccounts", [mint])
    log.append(("sol_getTokenLargestAccounts", "OK" if not err else "RATE_LIMITED", dt, str(err)))
    return pd.DataFrame(log, columns=["probe", "result", "seconds", "error_or_note"])


# ----------------------------------------------------------------------------
# PART C: Dead-pool / survivorship
# ----------------------------------------------------------------------------
def part_c_deadpool():
    # (1) near-dead proxy from on-disk OHLCV decay within the LIVE universe
    res = []
    for ch in ["bsc", "base", "eth"]:
        d = pd.read_parquet(f"{DATA}/ohlcv_gt/mega_{ch}_day.parquet")
        for pair, sub in d.groupby("pair_address"):
            sub = sub.sort_values("ts")
            if len(sub) < 10:
                continue
            vol = sub.volume.values
            last5 = np.nanmean(vol[-5:])
            peak = np.nanmax(vol)
            res.append(dict(chain=ch, pair=pair, n_bars=len(sub), peak_vol=peak,
                            last5_vol=last5, decay=last5 / peak if peak > 0 else np.nan,
                            dead_proxy=bool(peak > 0 and last5 < 0.01 * peak)))
    r = pd.DataFrame(res)
    r.to_csv(f"{OUT}/deadpool_proxy.csv", index=False)

    # (2) PairCreated rate (proves the survivorship-free universe is enumerable on-chain)
    EP = "https://bsc.publicnode.com"
    lat, _, _ = rpc(EP, "eth_blockNumber", [])
    pc = None
    if isinstance(lat, str):
        lat = int(lat, 16)
        rr, dt, err = rpc(EP, "eth_getLogs",
                          [{"fromBlock": hex(lat - 200), "toBlock": hex(lat),
                            "address": PANCAKE_V2_FACTORY, "topics": [PAIRCREATED]}])
        if not err:
            pc = dict(window_blocks=200, pairs_created=len(rr), seconds=dt,
                      pairs_per_block=len(rr) / 200, est_pairs_per_day=int(len(rr) / 200 * 28800))
    return r, pc


if __name__ == "__main__":
    print("PART A: MEV sandwich on BSC tape")
    df, dens, ge3, mev = part_a_mev()
    print(dens.to_string())
    print("ge3 frac", ge3, "attacks", mev.attacks.sum(), "victims", mev.victims.sum())
    print("\nPART B: holder feasibility")
    hb = part_b_holders()
    print(hb.to_string())
    hb.to_csv(f"{OUT}/holder_feasibility_probe.csv", index=False)
    print("\nPART C: dead-pool")
    dp, pc = part_c_deadpool()
    print("near-dead frac", round(dp.dead_proxy.mean(), 3), "PairCreated:", pc)
