"""On-chain MEV / sandwich + failed-transaction (revert) analysis.

Why this exists: a DEX backtest that fills at the observed trade price is dishonest.
In a public mempool your swap can be SANDWICHED (an attacker front-runs your buy to
push price up, lets you fill at the worse price, then back-runs to dump) and, more
mundanely, swaps REVERT, you pay gas and get no fill. Both are pure realized-cost
phenomena that the tape already records; this module quantifies them from the same
`Trade` records the indexers produce, then exposes a slippage pad the cost model can
absorb. No third-party MEV index, no API key, on-chain data only.

Module-level functions (deliberately NOT a Provider, this is post-processing over a
tape an indexer already returned, plus one direct revert query against an indexer):

  detect_sandwiches(trades)        -> attack/victim summary; mutates Trade.sandwiched
  revert_stats(indexer, ...)       -> gas-lost-no-fill revert rate (async)
  sandwich_padding_frac(summary)   -> a slippage pad (notional fraction) for costs.py

ORDERING / ASSUMPTIONS
----------------------
Sandwich detection is an INTRA-BLOCK pattern: front-run, victim, back-run all land in
the SAME block, and the attacker controls execution order via priority fee / builder.
We therefore order trades by (block_number, log_index, block_time) and only ever look
for the pattern *within one block_number*. `block_number`/`log_index` are populated by
the indexers from the raw log (EVM blockNumber/logIndex; Solana slot/in-slot position).
If they are missing (e.g. the indexer hasn't been wired to set them yet, see the
parent's in-flight change), we fall back to block_time ordering and, because there is
then no reliable intra-block resolution, we report zero attacks with `note` set rather
than guessing across block boundaries (which would manufacture false positives).

Attacker identity is `Trade.maker` (the tx signer / `from`). A classic sandwich is one
maker A appearing on BOTH sides (a BUY then a SELL) bracketing at least one OTHER
maker's trade B in the same block, where the bracketed victim trades in the same
direction the front-run pushed price (a BUY victim, after an attacker BUY raised price).

LIMITATIONS (documented, not faked)
------------------------------------
- BSC revert detection: a swap that reverts emits NO Swap/Sync event, so eth_getLogs
  (the bsc_indexer's only window onto the chain) literally cannot see it. Honest revert
  counting on BSC requires scanning full blocks (eth_getBlockByNumber with
  full-tx=True) or trace_block, filtering to the router, and checking each tx's receipt
  status: none of which the bsc_indexer exposes today. `revert_stats` therefore returns
  a `supported=False` stub for BSC with the approach written out, rather than a number.
- Sandwich detection needs `maker` set on trades. The bsc_indexer's V2/V3 trade builders
  do not currently populate `maker` (the swap `to`/recipient is in the log, but the real
  attacker is the tx signer, which needs the tx, not just the log). Until `maker` is
  populated, BSC sandwich detection will find no attacks and say so via `note`.
"""
from __future__ import annotations

import logging

from .chains import Chain
from .models import Trade

log = logging.getLogger("chainscope.mev")


# ---------------------------------------------------------------------------
# sandwich detection
# ---------------------------------------------------------------------------

def _order_key(t: Trade):
    """Deterministic intra-block ordering key. Missing block_number/log_index sort to 0
    so they don't crash; the caller separately decides whether intra-block resolution
    actually exists before trusting the pattern."""
    return (
        t.block_number if t.block_number is not None else 0,
        t.log_index if t.log_index is not None else 0,
        t.block_time,
    )


def _side(t: Trade) -> str | None:
    if t.side is None:
        return None
    return "buy" if str(t.side).lower().startswith("b") else "sell"


def detect_sandwiches(trades: list[Trade]) -> dict:
    """Detect classic single-attacker sandwiches and mark victims in place.

    Pattern (within one block): attacker A does a BUY (front-run), one or more OTHER
    makers' trades B execute, then the SAME attacker A does a SELL (back-run). Every
    bracketed victim that trades in the front-run direction (BUY, they pay the price A
    just inflated) is flagged `sandwiched=True`.

    Victim extra cost is estimated as the price gap between the victim's fill price and
    the block's PRE-sandwich price (the price right before A's front-run), as a fraction:
        (victim_price - pre_price) / pre_price          for a buy victim.
    This is a lower bound: it ignores A's second leg and multi-attacker stacks.

    Returns:
        {
          "sandwiched": [victim Trade, ...],
          "attacks": int,            # number of distinct A-buy/.../A-sell brackets
          "victim_count": int,       # victims flagged
          "est_extra_slippage_frac": float,   # mean over victims (0.0 if none)
          "intra_block_resolved": bool,       # did we have real ordering to work with?
          "note": str | None,                 # set when detection was degraded/skipped
        }
    """
    result = {
        "sandwiched": [],
        "attacks": 0,
        "victim_count": 0,
        "est_extra_slippage_frac": 0.0,
        "intra_block_resolved": False,
        "note": None,
    }
    if not trades:
        result["note"] = "empty trade tape"
        return result

    have_block = any(t.block_number is not None for t in trades)
    have_logidx = any(t.log_index is not None for t in trades)
    have_maker = any(t.maker is not None for t in trades)

    ordered = sorted(trades, key=_order_key)

    # We can only resolve the intra-block ordering a sandwich requires if trades carry
    # block_number (to group a block) AND some intra-block tiebreak (log_index). Without
    # both, fall back to time order but DON'T scan for the pattern across block
    # boundaries: that would invent attacks. Report zero with a note instead.
    if not (have_block and have_logidx):
        missing = []
        if not have_block:
            missing.append("block_number")
        if not have_logidx:
            missing.append("log_index")
        result["note"] = (
            "no intra-block ordering ({} unset on the tape), indexers must populate "
            "these for sandwich detection; returning zero attacks".format(", ".join(missing))
        )
        return result
    result["intra_block_resolved"] = True

    if not have_maker:
        result["note"] = (
            "trades have no `maker` set: cannot identify the attacker across both legs; "
            "returning zero attacks (indexer must populate Trade.maker)"
        )
        return result

    # group by block_number
    blocks: dict[int, list[Trade]] = {}
    for t in ordered:
        blocks.setdefault(t.block_number, []).append(t)

    victims: list[Trade] = []
    extra_fracs: list[float] = []
    attacks = 0

    for blk, group in blocks.items():
        if len(group) < 3:
            continue  # need front-run + victim + back-run
        # pre-sandwich price = first trade in the block that has a usable price.
        pre_price = None
        for t in group:
            p = t.price_usd if t.price_usd is not None else t.price_native
            if p is not None and p > 0:
                pre_price = p
                break

        n = len(group)
        # For each potential front-run BUY by A, find the nearest later SELL by the SAME
        # A. Any DISTINCT-maker trade strictly between them (same direction = buy) is a
        # victim. Consume the matched back-run so one A-leg pair counts as one attack.
        used_backrun: set[int] = set()
        for i in range(n):
            a_front = group[i]
            if _side(a_front) != "buy" or a_front.maker is None:
                continue
            attacker = a_front.maker
            # nearest later SELL by the same attacker not already consumed
            j = None
            for k in range(i + 1, n):
                if k in used_backrun:
                    continue
                if group[k].maker == attacker and _side(group[k]) == "sell":
                    j = k
                    break
            if j is None:
                continue
            bracketed = [
                group[m] for m in range(i + 1, j)
                if group[m].maker != attacker and _side(group[m]) == "buy"
            ]
            if not bracketed:
                continue  # an A-buy/A-sell pair with nobody trapped is not a sandwich
            used_backrun.add(j)
            attacks += 1
            for v in bracketed:
                if v.sandwiched:
                    continue  # already flagged by an outer/earlier bracket
                v.sandwiched = True
                victims.append(v)
                vp = v.price_usd if v.price_usd is not None else v.price_native
                if pre_price and vp is not None and vp > 0:
                    extra_fracs.append(max(0.0, (vp - pre_price) / pre_price))

    result["attacks"] = attacks
    result["victim_count"] = len(victims)
    result["sandwiched"] = victims
    result["est_extra_slippage_frac"] = (
        sum(extra_fracs) / len(extra_fracs) if extra_fracs else 0.0
    )
    if attacks == 0 and result["note"] is None:
        result["note"] = "no sandwiches found in this window"
    return result


# ---------------------------------------------------------------------------
# revert (gas-lost-no-fill) statistics
# ---------------------------------------------------------------------------

async def revert_stats(indexer, chain, pool: str, limit: int = 500) -> dict:
    """Quantify swaps that paid gas and got NO fill (reverts).

    SOLANA: getSignaturesForAddress on the pool already returns each signature's `err`
    field (null == ok, non-null == the tx reverted). We count ok vs err over the most
    recent `limit` signatures touching the pool. Each err entry is, to first order, a
    swap attempt that paid the fee and reverted. We fetch the fee for a sample of the
    failed txs (getTransaction -> meta.fee, lamports) and sum/extrapolate gas lost. Note
    a pool address sees more than just swaps (LP ops, etc.), so this is the revert rate
    of *interactions with the pool*, an upper-bound proxy for swap reverts specifically.

    BSC: NOT directly supported, reverted swaps emit no Swap/Sync event, so the
    bsc_indexer's eth_getLogs view is blind to them. Returns a stub documenting the
    real approach (full-block / trace scan filtered to the router) rather than a number.

    Returns (Solana):
        {"chain","pool","supported":True,"revert_rate","revert_count","ok_count",
         "total","gas_lost_native","gas_lost_usd":None,"sampled_failed","note"}
    Returns (BSC): {"chain","pool","supported":False,"note": <approach>}
    """
    c = Chain.parse(chain)

    if c == Chain.BSC:
        return {
            "chain": c.value,
            "pool": pool,
            "supported": False,
            "revert_rate": None,
            "revert_count": None,
            "ok_count": None,
            "total": 0,
            "gas_lost_native": None,
            "gas_lost_usd": None,
            "note": (
                "BSC revert detection not implemented: a reverted swap emits no "
                "Swap/Sync log, so eth_getLogs (the bsc_indexer's only data window) "
                "cannot see it. Honest counting requires scanning full blocks "
                "(eth_getBlockByNumber with full-tx=True) or trace_block over the "
                "window, filtering txs whose `to` is the PancakeSwap router, then "
                "reading each tx's receipt `status` (0x0 == reverted) and `gasUsed * "
                "effectiveGasPrice` for gas lost. The bsc_indexer exposes none of "
                "these today; wiring a router-tx scanner is a follow-up. No numbers "
                "are fabricated here."
            ),
        }

    if c != Chain.SOLANA:
        return {
            "chain": c.value, "pool": pool, "supported": False,
            "note": f"revert_stats not implemented for chain {c.value}",
            "total": 0,
        }

    # --- Solana ---
    # Use the indexer's own signature lister (paging by `before`) to gather up to `limit`
    # raw signature entries, which carry `err`. We must NOT use get_trades here: it drops
    # reverted txs by design, so it would erase exactly what we're trying to count.
    if not hasattr(indexer, "_signatures"):
        return {
            "chain": c.value, "pool": pool, "supported": False, "total": 0,
            "note": "indexer lacks a _signatures() lister; cannot enumerate reverts",
        }

    entries: list[dict] = []
    before: str | None = None
    while len(entries) < limit:
        page = await indexer._signatures(pool, min(1000, limit - len(entries)), before=before)
        if not page:
            break
        entries.extend(page)
        if len(page) < min(1000, limit):
            break
        before = page[-1].get("signature")
    entries = entries[:limit]

    total = len(entries)
    if total == 0:
        return {
            "chain": c.value, "pool": pool, "supported": True, "total": 0,
            "revert_rate": None, "revert_count": 0, "ok_count": 0,
            "gas_lost_native": None, "gas_lost_usd": None, "sampled_failed": 0,
            "note": "no signatures returned for this pool in the window",
        }

    failed = [e for e in entries if e.get("err") is not None]
    revert_count = len(failed)
    ok_count = total - revert_count

    # Best-effort gas-lost: fetch fee for a bounded sample of failed txs and extrapolate.
    # getTransaction returns meta.fee in lamports even for failed txs (fees still charged).
    gas_lost_native = None
    sampled = 0
    if failed and hasattr(indexer, "_get_transaction"):
        sample_fees_lamports: list[int] = []
        for e in failed[: min(20, len(failed))]:
            sig = e.get("signature")
            if not sig:
                continue
            tx = await indexer._get_transaction(sig)
            fee = (((tx or {}).get("meta") or {}).get("fee"))
            if fee is not None:
                try:
                    sample_fees_lamports.append(int(fee))
                    sampled += 1
                except (TypeError, ValueError):
                    pass
        if sample_fees_lamports:
            avg_fee_lamports = sum(sample_fees_lamports) / len(sample_fees_lamports)
            # extrapolate the sample mean across all failed txs; lamports -> SOL (1e9)
            gas_lost_native = (avg_fee_lamports * revert_count) / 1e9

    return {
        "chain": c.value,
        "pool": pool,
        "supported": True,
        "revert_rate": revert_count / total,
        "revert_count": revert_count,
        "ok_count": ok_count,
        "total": total,
        "gas_lost_native": gas_lost_native,     # SOL, extrapolated from a sample
        "gas_lost_usd": None,                   # left None: no price feed in this module
        "sampled_failed": sampled,
        "note": (
            "revert rate over ALL interactions with the pool address (swaps + LP/other "
            "ops); an upper-bound proxy for swap reverts. gas_lost_native extrapolates a "
            "sampled mean fee across all failed txs." if revert_count
            else "no reverts in this window"
        ),
    }


# ---------------------------------------------------------------------------
# cost-model feed
# ---------------------------------------------------------------------------

def sandwich_padding_frac(sandwich_summary: dict, base_pad_frac: float = 0.0) -> float:
    """A slippage pad (fraction of notional) to ADD to a backtest's cost model,
    derived from observed sandwich frequency × severity.

    Idea: a backtester can't know in advance which fills get sandwiched, so the honest
    move is to load EVERY modeled fill with the expected sandwich cost:

        pad = sandwich_probability * mean_victim_extra_slippage

    where sandwich_probability ~= victim_count / total_trades_seen (taken from the
    summary's victim_count and the length of `sandwiched` is insufficient, so callers
    should pass the tape length via the summary; if absent we use a conservative proxy).

    The returned pad slots straight into costs.apply_costs(..., slip_floor_bps=...) as an
    additive floor, or can be added to round_trip_cost_frac's result. It is intentionally
    a single notional fraction so it composes with the existing fee/impact/gas model
    without touching costs.py.

    Args:
        sandwich_summary: the dict from detect_sandwiches(). If it additionally carries a
            "total_trades" key (number of trades scanned), probability uses it; otherwise
            probability falls back to victim_count over (victim_count + a smoothing 1)
            which is conservative (over-states the pad on tiny samples, the safe side).
        base_pad_frac: an optional floor added unconditionally (e.g. a minimum MEV tax you
            always want to assume on a public-mempool chain).

    Returns:
        float >= base_pad_frac, the per-fill slippage pad.
    """
    if not sandwich_summary:
        return base_pad_frac
    victims = int(sandwich_summary.get("victim_count") or 0)
    severity = float(sandwich_summary.get("est_extra_slippage_frac") or 0.0)
    total = sandwich_summary.get("total_trades")
    if total and total > 0:
        prob = victims / float(total)
    else:
        prob = victims / float(victims + 1)  # conservative smoothing on unknown denom
    pad = prob * max(0.0, severity)
    return max(base_pad_frac, pad)
