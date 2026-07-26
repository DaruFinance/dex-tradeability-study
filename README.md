# No Edge Without Information

### An empirical study of tradeability in decentralized-exchange-only cryptocurrencies

**Daniel Gatto**

Working paper, under review · [SSRN 6858778](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6858778) · [Interactive write-up](https://daru.finance/research/dex-only)

## Overview

This repository contains the analysis engines for an empirical study of
*tradeability* in DEX-only cryptocurrencies: tokens that trade on decentralized
exchanges and have never been listed on a centralized venue. The study assembles
a survivorship-aware corpus of 4,990 trading pairs across 27 chains and asks a
narrow, falsifiable question: net of realistic execution costs, is there any
exploitable edge available to a price-based timing strategy in these assets?

The methodology is deliberately conservative. Every backtest is **long-only**
(spot tokens cannot be shorted without a borrow facility), carries a **per-fill
cost model** (swap fee plus constant-product slippage plus gas, applied on each
entry and exit), and is evaluated under **walk-forward** out-of-sample testing
with in-sample parameter selection and intrabar take-profit/stop on the bar high
and low. Crucially, every reported result is paired with a **mandatory
bar-shuffled null control**: the identical pipeline is re-run on surrogate series
that destroy temporal order while keeping each coin's return distribution, so any
apparent edge can be measured against what pure noise produces under the same
selection procedure. Results are reported as **full distributions** rather than
cherry-picked top performers, and the corpus is **survivorship-aware** — sourced
from currently-live pools, so every cross-sectional and holding metric is read as
an upper bound.

The central finding is that price-based timing carries **no edge** in this
universe once costs and the null control are accounted for: the live
distributions are statistically indistinguishable from, or worse than, the
shuffled-bar nulls. A large battery of extended studies (order-flow features,
cross-sectional selection, machine-learning ranking, liquidity-provision and
market-making proxies, launch-cohort and dispersion effects) is reported in the
same disciplined frame. The one qualified positive is a **leakage-free
cross-sectional machine-learning ranker** (point-in-time features only): its
out-of-sample rank-IC is ~0.06–0.08 at the 7- and 14-day horizons
(label-permutation p ~ 0.005), and insignificant at 30 days. It does not
generate alpha; its skill is **crash-avoidance** (every predicted decile's
median forward return is negative, so it flags which coins fall, not which rise),
and a long-only AMM trader cannot reach the left tail. We therefore report it as
a lead, not an edge. See `ml_ranker_pit.py` for the leakage-free re-run (the
earlier `ml_ranker.py` used end-of-sample snapshot features and is superseded).

## Repository layout

- `engines/` holds the core analysis and data-collection scripts: phased pipeline
  (`phase1`…`phase10`), GeckoTerminal universe and OHLCV ingestion (`gt_*`),
  the centralized-listing exclusion filter (`cex_filter*`), and the on-chain
  flow collectors (`bitquery_flow.py`, `tokenapi_flow.py`). These cover universe
  construction, baseline backtests, walk-forward optimization, flow-signal
  selection, and the null controls.
- `chainscope/` holds the supporting Python package: the per-fill cost model
  (`costs.py`), data storage and caching, chain definitions, provider clients,
  and shared utilities.
- `gap-studies/` holds fifteen self-contained extended-study engines, one per
  subdirectory (beta-hedged baskets, BSC order flow, BTC-regime timing,
  cost/depth sensitivity, intrabar TP/SL bracketing, in-sample noise fitting,
  launch cohorts, LP market-making, cross-sectional and meta selection,
  ML cross-sectional ranking, multipool dispersion, and a portfolio of marginal
  streams).
- `web-demos/` holds the React/TypeScript sources for the interactive figures
  in the companion write-up.

## Reproducing

The underlying market data is **not** included in this repository (it is large
and licensed from third-party providers). Scripts expect a local `./data`
directory and read all provider credentials from **environment variables**
(e.g. `COINGECKO_API_KEY`, `BITQUERY_TOKEN`, `THEGRAPH_TOKEN_API_KEY`,
`BIRDEYE_API_KEY`, `MORALIS_API_KEY`, `HELIUS_API_KEY`). Set these in your
environment before running any collector. With data and keys in place, the
phased engines in `engines/` reproduce the panel, the backtests, the
walk-forward results, and the null controls in sequence.

Companion write-up with interactive figures:
[daru.finance/research/dex-only](https://daru.finance/research/dex-only)

## Data

This study was made possible by on-chain and market data from
[Bitquery](https://bitquery.io) and [CoinGecko](https://www.coingecko.com). A
substantially expanded version 2, with broader coverage and deeper analysis
thanks to their support, is in progress.

## Cite

> Gatto, D. V. (2026). *No Edge Without Information: An Empirical Study of
> Tradeability in Decentralized-Exchange-Only Cryptocurrencies.* SSRN Working
> Paper 6858778. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6858778

```bibtex
@techreport{gatto2026dex,
  author = {Gatto, Daniel V.},
  title  = {No Edge Without Information: An Empirical Study of
            Tradeability in Decentralized-Exchange-Only Cryptocurrencies},
  year   = {2026},
  type   = {SSRN Working Paper},
  number = {6858778},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6858778}
}
```

## Status

Working paper, under review. Posted on SSRN:
[papers.ssrn.com/abstract=6858778](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6858778)
