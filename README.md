# DEX-Only Tradeability: An Empirical Study

**Daniel Gatto**

## Overview

This repository contains the analysis engines for an empirical
study of *tradeability* in DEX-only cryptocurrencies: tokens that trade on
decentralized exchanges and have never been listed on a centralized venue. The
study assembles a survivorship-free panel spanning 27 chains and asks a narrow,
falsifiable question: net of realistic execution costs, is there any exploitable
edge available to a price-based timing strategy in these assets?

The methodology is deliberately conservative. Every backtest is **long-only**
(spot tokens cannot be shorted without a borrow facility), carries a
**per-fill cost model** (fees plus slippage applied on each entry and exit), and
is evaluated under **walk-forward** out-of-sample testing with in-sample
parameter selection. Crucially, every reported result is paired with a
**mandatory bar-shuffled null control**: the identical pipeline is run on
phase-randomized series so that any apparent edge can be measured against what
pure noise produces under the same selection procedure. Results are reported as
**full distributions** rather than cherry-picked top performers, so the reader
sees the entire opportunity set, not its right tail.

The central finding is that price-based timing carries **no edge** in this
universe once costs and the null control are accounted for: the live
distributions are statistically indistinguishable from, or worse than, the
shuffled-bar nulls. A large battery of extended studies (order-flow features,
cross-sectional selection, machine-learning ranking, liquidity-provision and
market-making proxies, launch-cohort and dispersion effects) is reported in the
same disciplined frame. The one qualified positive is a **crash-avoidance
signal** built on a market-regime filter: it does not generate alpha, but it
reduces drawdown participation in adverse regimes, and that effect survives the
null control under stated assumptions.

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
  in the companion research preview.

## Reproducing

The underlying market data is **not** included in this repository (it is large
and licensed from third-party providers). Scripts expect a local `./data`
directory and read all provider credentials from **environment variables**
(e.g. `COINGECKO_API_KEY`, `BITQUERY_TOKEN`, `THEGRAPH_TOKEN_API_KEY`,
`BIRDEYE_API_KEY`, `MORALIS_API_KEY`, `HELIUS_API_KEY`). Set these in your
environment before running any collector. With data and keys in place, the
phased engines in `engines/` reproduce the panel, the backtests, the
walk-forward results, and the null controls in sequence.

Companion write-up: daru.finance (research preview)

## Status

Working paper; SSRN posting forthcoming.
