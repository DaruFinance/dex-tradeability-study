"""Bitquery: premium deep-history backtest source (requires BITQUERY_TOKEN).

Capabilities: ohlcv, trades, launch. Covers both Solana and BSC.
Full trade tape + time-bucketed OHLCV across years (uses the `archive` dataset
for deep history), plus launchpad create / bonding-curve / graduation timelines
for pump.fun (Solana) and four.meme (BSC).

GraphQL V2 endpoint: https://streaming.bitquery.io/graphql
Docs:
  Solana DEX trades ........ https://docs.bitquery.io/docs/blockchain/Solana/solana-dextrades/
  pump.fun OHLCV ........... https://docs.bitquery.io/docs/blockchain/Solana/Pumpfun/Pump-Fun-API/
  pump.fun marketcap/curve . https://docs.bitquery.io/docs/blockchain/Solana/Pumpfun/Pump-Fun-Marketcap-Bonding-Curve-API/
  pump.fun -> PumpSwap ..... https://docs.bitquery.io/docs/blockchain/Solana/Pumpfun/pump-fun-to-pump-swap/
  BSC four.meme ............ https://docs.bitquery.io/docs/blockchain/BSC/four-meme-api/
  BSC PancakeSwap .......... https://docs.bitquery.io/docs/blockchain/BSC/pancake-swap-api/
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..chains import Chain, normalize_address
from ..http import HttpError
from ..models import OHLCV, Launch, Trade
from .base import CAP_LAUNCH, CAP_OHLCV, CAP_TRADES, NotSupported, Provider

ENDPOINT = "https://streaming.bitquery.io/graphql"

# Launchpad program / factory identifiers.
PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMPSWAP_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
FOURMEME_FACTORY = "0x5c952063c7fc8610ffdb798152d69f0b9550762b"

# pump.fun bonding-curve constants (Solana). progress = 100 - (left*100/initial).
PF_INITIAL_REAL_TOKEN_RESERVES = 793_100_000.0
PF_RESERVED_TOKENS = 206_900_000.0
# four.meme bonding-curve constants (BSC). 200M reserved of a 1B supply.
FM_RESERVED_TOKENS = 200_000_000.0
FM_INITIAL_REAL_TOKEN_RESERVES = 800_000_000.0
FM_TOTAL_SUPPLY = 1_000_000_000.0

# our timeframe -> (count, Bitquery interval unit)
_TIMEFRAMES: dict[str, tuple[int, str]] = {
    "1m": (1, "minutes"),
    "5m": (5, "minutes"),
    "15m": (15, "minutes"),
    "1h": (1, "hours"),
    "4h": (4, "hours"),
    "1d": (1, "days"),
}


def _f(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _i(x) -> int | None:
    try:
        return int(float(x)) if x is not None else None
    except (TypeError, ValueError):
        return None


def _ts_to_iso(ts: int | None) -> str | None:
    """Unix seconds -> ISO-8601 UTC string (Bitquery time filter format)."""
    v = _i(ts)
    if v is None:
        return None
    return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_dt(s) -> datetime | None:
    if not s:
        return None
    txt = str(s).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    # Bitquery may return "2024-01-01 12:00:00" (space) or full ISO.
    for candidate in (txt, txt.replace(" ", "T")):
        try:
            dt = datetime.fromisoformat(candidate)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class BitqueryProvider(Provider):
    name = "bitquery"
    supported_chains = frozenset({Chain.SOLANA, Chain.BSC})
    capabilities = frozenset({CAP_OHLCV, CAP_TRADES, CAP_LAUNCH})
    requires_key = True
    key_env = "BITQUERY_TOKEN"

    # ---- GraphQL plumbing ----

    async def _gql(self, query: str, variables: dict, *, cache_ttl: float | None = None) -> dict:
        """POST a GraphQL query, return resp["data"] (or {} on errors)."""
        token = self.settings.get_key(self.key_env)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {"query": query, "variables": variables}
        # post_json has no cache; only get_json does. Trades intentionally uncached;
        # ohlcv/launch callers pre-check enabled state, cache handled by caller intent.
        resp = await self.http.post_json(ENDPOINT, json=payload, headers=headers)
        if not isinstance(resp, dict):
            return {}
        errors = resp.get("errors")
        if errors:
            msg = "; ".join(
                str(e.get("message", e)) if isinstance(e, dict) else str(e) for e in errors
            )
            raise HttpError(400, ENDPOINT, f"GraphQL errors: {msg}")
        return resp.get("data") or {}

    # ---- trades ----

    async def get_trades(self, chain: Chain, pair_address: str,
                         since: int | None = None, until: int | None = None,
                         limit: int = 1000) -> list[Trade]:
        chain = Chain.parse(chain)
        if chain == Chain.SOLANA:
            return await self._trades_solana(pair_address, since, until, limit)
        if chain == Chain.BSC:
            return await self._trades_bsc(pair_address, since, until, limit)
        raise NotSupported

    def _time_filter(self, since: int | None, until: int | None) -> tuple[str, dict]:
        """Build a `Time: {since:, till:}` GraphQL fragment + variables dict."""
        since_iso, till_iso = _ts_to_iso(since), _ts_to_iso(until)
        vars_: dict[str, str] = {}
        clauses = []
        if since_iso:
            clauses.append("since: $since")
            vars_["since"] = since_iso
        if till_iso:
            clauses.append("till: $till")
            vars_["till"] = till_iso
        frag = f"Block: {{Time: {{{', '.join(clauses)}}}}}" if clauses else ""
        return frag, vars_

    async def _trades_solana(self, pair_address, since, until, limit) -> list[Trade]:
        time_frag, time_vars = self._time_filter(since, until)
        # Filter to the market/pool address. DEXTradeByTokens gives per-token rows
        # with Side (the counter currency) -> clean buy/sell + USD amounts.
        query = """
        query ($market: String!, $limit: Int!, $since: DateTime, $till: DateTime) {
          Solana(dataset: archive) {
            DEXTradeByTokens(
              limit: {count: $limit}
              orderBy: {descending: Block_Time}
              where: {
                Trade: {Market: {MarketAddress: {is: $market}}}
                %TIME%
                Transaction: {Result: {Success: true}}
              }
            ) {
              Block { Time }
              Transaction { Signature }
              Trade {
                Side { Type AmountInUSD Amount Account { Address } }
                Dex { ProtocolName ProtocolFamily }
                Currency { MintAddress Symbol }
                Market { MarketAddress }
                Price
                PriceInUSD
                Amount
                AmountInUSD
                Account { Address }
              }
            }
          }
        }
        """.replace("%TIME%", time_frag)
        variables = {"market": pair_address, "limit": int(limit), **time_vars}
        data = await self._gql(query, variables)
        rows = ((data.get("Solana") or {}).get("DEXTradeByTokens")) or []
        out: list[Trade] = []
        for r in rows:
            trade = r.get("Trade") or {}
            side = trade.get("Side") or {}
            dex = trade.get("Dex") or {}
            acct = trade.get("Account") or {}
            side_acct = side.get("Account") or {}
            side_type = (side.get("Type") or "").lower()
            out.append(Trade(
                source="bitquery",
                chain=Chain.SOLANA,
                pair_address=pair_address,
                block_time=_parse_dt((r.get("Block") or {}).get("Time")) or datetime.now(timezone.utc),
                tx_hash=(r.get("Transaction") or {}).get("Signature"),
                side=side_type if side_type in ("buy", "sell") else None,
                dex=dex.get("ProtocolName") or dex.get("ProtocolFamily"),
                price_usd=_f(trade.get("PriceInUSD")),
                price_native=_f(trade.get("Price")),
                amount_base=_f(trade.get("Amount")),
                amount_quote=_f(side.get("Amount")),
                amount_usd=_f(trade.get("AmountInUSD")) or _f(side.get("AmountInUSD")),
                maker=acct.get("Address") or side_acct.get("Address"),
                raw=r,
            ))
        return out

    async def _trades_bsc(self, pair_address, since, until, limit) -> list[Trade]:
        time_frag, time_vars = self._time_filter(since, until)
        # EVM DEXTrades filtered to the pair smart contract (Trade.Dex.SmartContract).
        query = """
        query ($pair: String!, $limit: Int!, $since: DateTime, $till: DateTime) {
          EVM(network: bsc, dataset: archive) {
            DEXTrades(
              limit: {count: $limit}
              orderBy: {descending: Block_Time}
              where: {
                Trade: {Dex: {SmartContract: {is: $pair}}}
                %TIME%
                TransactionStatus: {Success: true}
              }
            ) {
              Block { Time }
              Transaction { Hash From }
              Trade {
                Buy {
                  Buyer
                  Currency { SmartContract Symbol }
                  Amount
                  Price
                  PriceInUSD
                }
                Sell {
                  Seller
                  Currency { SmartContract Symbol }
                  Amount
                  Price
                  PriceInUSD
                }
                Dex { ProtocolName ProtocolFamily SmartContract }
              }
            }
          }
        }
        """.replace("%TIME%", time_frag)
        variables = {"pair": normalize_address(Chain.BSC, pair_address),
                     "limit": int(limit), **time_vars}
        data = await self._gql(query, variables)
        rows = ((data.get("EVM") or {}).get("DEXTrades")) or []
        out: list[Trade] = []
        for r in rows:
            trade = r.get("Trade") or {}
            buy = trade.get("Buy") or {}
            sell = trade.get("Sell") or {}
            dex = trade.get("Dex") or {}
            tx = r.get("Transaction") or {}
            # On EVM, Buy = the bought currency leg; treat the bought token as base.
            buy_amount = _f(buy.get("Amount"))
            buy_px_usd = _f(buy.get("PriceInUSD"))
            amount_usd = (buy_amount * buy_px_usd) if (buy_amount is not None and buy_px_usd is not None) else None
            out.append(Trade(
                source="bitquery",
                chain=Chain.BSC,
                pair_address=pair_address,
                block_time=_parse_dt((r.get("Block") or {}).get("Time")) or datetime.now(timezone.utc),
                tx_hash=tx.get("Hash"),
                side="buy",  # DEXTrades rows are oriented as a buy of Buy.Currency
                dex=dex.get("ProtocolName") or dex.get("ProtocolFamily"),
                price_usd=buy_px_usd if buy_px_usd is not None else _f(sell.get("PriceInUSD")),
                price_native=_f(buy.get("Price")) or _f(sell.get("Price")),
                amount_base=buy_amount,
                amount_quote=_f(sell.get("Amount")),
                amount_usd=amount_usd,
                maker=buy.get("Buyer") or tx.get("From"),
                raw=r,
            ))
        return out

    # ---- ohlcv ----

    async def get_ohlcv(self, chain: Chain, pair_address: str, timeframe: str = "1h",
                        limit: int = 1000, before: int | None = None) -> list[OHLCV]:
        chain = Chain.parse(chain)
        if timeframe not in _TIMEFRAMES:
            raise NotSupported
        count, unit = _TIMEFRAMES[timeframe]
        till_iso = _ts_to_iso(before)
        till_clause = "till: $till" if till_iso else ""

        if chain == Chain.SOLANA:
            network_wrap = "Solana(dataset: archive)"
            market_filter = "Trade: {Market: {MarketAddress: {is: $market}}}"
        elif chain == Chain.BSC:
            network_wrap = "EVM(network: bsc, dataset: archive)"
            market_filter = "Trade: {Dex: {SmartContract: {is: $market}}}"
        else:
            raise NotSupported

        # Time-bucketed OHLC: open/close keyed on Block_Slot/Block_Time extremes,
        # high/low on the price extremes, volume = sum of USD side amount, count = trades.
        query = """
        query ($market: String!, $count: Int!, $limit: Int!, $till: DateTime) {
          %NET% {
            DEXTradeByTokens(
              limit: {count: $limit}
              orderBy: {descendingByField: "Block_Timefield"}
              where: {
                %MARKET%
                %TILLBLOCK%
              }
            ) {
              Block {
                Timefield: Time(interval: {in: %UNIT%, count: $count})
              }
              Trade {
                open: PriceInUSD(minimum: Block_Time)
                close: PriceInUSD(maximum: Block_Time)
                high: PriceInUSD(maximum: Trade_PriceInUSD)
                low: PriceInUSD(minimum: Trade_PriceInUSD)
              }
              volume: sum(of: Trade_Side_AmountInUSD)
              count
            }
          }
        }
        """
        till_block = f"Block: {{Time: {{{till_clause}}}}}" if till_iso else ""
        query = (query
                 .replace("%NET%", network_wrap)
                 .replace("%MARKET%", market_filter)
                 .replace("%TILLBLOCK%", till_block)
                 .replace("%UNIT%", unit))
        variables: dict = {"market": (pair_address if chain == Chain.SOLANA
                                      else normalize_address(Chain.BSC, pair_address)),
                           "count": int(count), "limit": int(limit)}
        if till_iso:
            variables["till"] = till_iso
        data = await self._gql(query, variables)
        root_key = "Solana" if chain == Chain.SOLANA else "EVM"
        rows = ((data.get(root_key) or {}).get("DEXTradeByTokens")) or []
        out: list[OHLCV] = []
        for r in rows:
            bucket = _parse_dt((r.get("Block") or {}).get("Timefield"))
            if bucket is None:
                continue
            trade = r.get("Trade") or {}
            o = _f(trade.get("open"))
            c = _f(trade.get("close"))
            hi = _f(trade.get("high"))
            lo = _f(trade.get("low"))
            out.append(OHLCV(
                source="bitquery",
                chain=chain,
                pair_address=pair_address,
                timeframe=timeframe,
                timestamp=bucket,
                open=o if o is not None else 0.0,
                high=hi if hi is not None else (o if o is not None else 0.0),
                low=lo if lo is not None else (o if o is not None else 0.0),
                close=c if c is not None else 0.0,
                volume=_f(r.get("volume")),
                reserve_usd=None,  # not surfaced by the aggregate trade query
                trade_count=_i(r.get("count")),
                raw=r,
            ))
        out.sort(key=lambda b: b.timestamp)
        return out

    # ---- launch ----

    async def get_launch(self, chain: Chain, address: str) -> Launch | None:
        chain = Chain.parse(chain)
        try:
            if chain == Chain.SOLANA:
                return await self._launch_solana(address)
            if chain == Chain.BSC:
                return await self._launch_bsc(address)
        except HttpError as exc:
            if exc.status == 404:
                return None
            raise
        raise NotSupported

    async def _launch_solana(self, mint: str) -> Launch | None:
        # 1) creation (TokenSupplyUpdates carries Dev creator + first mint time);
        # 2) latest pump.fun pool token balance -> bonding-curve progress;
        # 3) PumpSwap create_pool -> migration / graduation.
        query = """
        query ($mint: String!, $pumpfun: String!, $pumpswap: String!) {
          Solana {
            create: TokenSupplyUpdates(
              limit: {count: 1}
              orderBy: {ascending: Block_Time}
              where: {TokenSupplyUpdate: {Currency: {MintAddress: {is: $mint}}}}
            ) {
              Block { Time }
              Transaction { Signer }
              TokenSupplyUpdate {
                Currency { MintAddress Name Symbol Decimals }
              }
            }
            curve: DEXPools(
              limit: {count: 1}
              orderBy: {descending: Block_Time}
              where: {
                Pool: {
                  Dex: {ProgramAddress: {is: $pumpfun}}
                  Market: {BaseCurrency: {MintAddress: {is: $mint}}}
                }
              }
            ) {
              Pool {
                Base { PostAmount }
                Quote { PostAmountInUSD }
                Market { MarketAddress BaseCurrency { MintAddress Name Symbol } }
              }
            }
            migrate: Instructions(
              limit: {count: 1}
              orderBy: {descending: Block_Time}
              where: {
                Instruction: {Program: {Address: {is: $pumpswap}, Method: {is: "create_pool"}}}
                Transaction: {Result: {Success: true}}
              }
            ) {
              Block { Time }
              Instruction {
                Accounts { Address Token { Mint } }
              }
            }
          }
        }
        """
        data = await self._gql(query, {
            "mint": mint, "pumpfun": PUMPFUN_PROGRAM, "pumpswap": PUMPSWAP_PROGRAM,
        })
        sol = data.get("Solana") or {}
        create_rows = sol.get("create") or []
        curve_rows = sol.get("curve") or []
        migrate_rows = sol.get("migrate") or []

        if not create_rows and not curve_rows:
            return None  # not a pump.fun token we can see

        name = symbol = creator = None
        created_at = None
        if create_rows:
            c0 = create_rows[0]
            created_at = _parse_dt((c0.get("Block") or {}).get("Time"))
            creator = (c0.get("Transaction") or {}).get("Signer")
            cur = ((c0.get("TokenSupplyUpdate") or {}).get("Currency")) or {}
            name, symbol = cur.get("Name"), cur.get("Symbol")

        progress = market_cap = None
        if curve_rows:
            pool = (curve_rows[0].get("Pool") or {})
            base_balance = _f((pool.get("Base") or {}).get("PostAmount"))
            quote_usd = _f((pool.get("Quote") or {}).get("PostAmountInUSD"))
            if base_balance is not None:
                left = base_balance - PF_RESERVED_TOKENS
                pct = 100.0 - (left * 100.0 / PF_INITIAL_REAL_TOKEN_RESERVES)
                progress = max(0.0, min(1.0, pct / 100.0))
            if quote_usd is not None:
                market_cap = quote_usd  # quote-side USD reserve ~ marketcap proxy
            mkt = ((pool.get("Market") or {}).get("BaseCurrency")) or {}
            name = name or mkt.get("Name")
            symbol = symbol or mkt.get("Symbol")

        migrated_pool = graduated_at = None
        complete = bool(progress is not None and progress >= 1.0)
        # confirm graduation via a PumpSwap pool whose accounts reference this mint
        for m in migrate_rows:
            accts = ((m.get("Instruction") or {}).get("Accounts")) or []
            if any(((a.get("Token") or {}).get("Mint")) == mint for a in accts):
                graduated_at = _parse_dt((m.get("Block") or {}).get("Time"))
                # the pool/market account is the create_pool account distinct from the mint
                migrated_pool = next(
                    (a.get("Address") for a in accts
                     if ((a.get("Token") or {}).get("Mint")) is None and a.get("Address")),
                    None,
                )
                complete = True
                break

        return Launch(
            source="bitquery",
            chain=Chain.SOLANA,
            address=mint,
            launchpad="pumpfun",
            name=name,
            symbol=symbol,
            creator=creator,
            created_at=created_at,
            market_cap_usd=market_cap,
            bonding_curve_progress=progress,
            complete=complete,
            graduated_at=graduated_at,
            migrated_pool=migrated_pool,
            target_dex="pumpswap",
            raw=data,
        )

    async def _launch_bsc(self, token: str) -> Launch | None:
        addr = normalize_address(Chain.BSC, token)
        # four.meme create (TokenCreate event on the factory), current factory-held
        # balance -> bonding progress, and PairCreated -> PancakeSwap migration.
        query = """
        query ($token: String!, $factory: String!) {
          EVM(network: bsc) {
            create: Events(
              limit: {count: 1}
              orderBy: {ascending: Block_Time}
              where: {
                Transaction: {To: {is: $factory}}
                Log: {Signature: {Name: {is: "TokenCreate"}}}
                Arguments: {includes: {Value: {Address: {is: $token}}}}
              }
            ) {
              Block { Time }
              Transaction { Hash From }
              Arguments {
                Name
                Value {
                  ... on EVM_ABI_Address_Value_Arg { address }
                  ... on EVM_ABI_String_Value_Arg { string }
                  ... on EVM_ABI_BigInt_Value_Arg { bigInteger }
                }
              }
            }
            curve: BalanceUpdates(
              limit: {count: 1}
              where: {
                BalanceUpdate: {Address: {is: $factory}}
                Currency: {SmartContract: {is: $token}}
              }
            ) {
              Currency { Name Symbol SmartContract Decimals }
              balance: sum(of: BalanceUpdate_Amount)
            }
            price: DEXTradeByTokens(
              limit: {count: 1}
              orderBy: {descending: Block_Time}
              where: {Trade: {Currency: {SmartContract: {is: $token}}}}
            ) {
              Trade { PriceInUSD }
            }
            migrate: Events(
              limit: {count: 1}
              orderBy: {descending: Block_Time}
              where: {
                Log: {Signature: {Name: {is: "PairCreated"}}}
                Arguments: {includes: {Value: {Address: {is: $token}}}}
              }
            ) {
              Block { Time }
              Arguments {
                Name
                Value { ... on EVM_ABI_Address_Value_Arg { address } }
              }
            }
          }
        }
        """
        data = await self._gql(query, {"token": addr, "factory": FOURMEME_FACTORY})
        evm = data.get("EVM") or {}
        create_rows = evm.get("create") or []
        curve_rows = evm.get("curve") or []
        price_rows = evm.get("price") or []
        migrate_rows = evm.get("migrate") or []

        if not create_rows and not curve_rows:
            return None  # not a four.meme token we can see

        name = symbol = creator = None
        created_at = None
        if create_rows:
            c0 = create_rows[0]
            created_at = _parse_dt((c0.get("Block") or {}).get("Time"))
            creator = (c0.get("Transaction") or {}).get("From")
            for arg in c0.get("Arguments") or []:
                aname = (arg.get("Name") or "").lower()
                val = arg.get("Value") or {}
                if aname == "creator" and val.get("address"):
                    creator = val.get("address")
                elif aname == "name" and val.get("string"):
                    name = val.get("string")
                elif aname == "symbol" and val.get("string"):
                    symbol = val.get("string")

        progress = market_cap = None
        if curve_rows:
            cr = curve_rows[0]
            balance = _f(cr.get("balance"))
            cur = cr.get("Currency") or {}
            name = name or cur.get("Name")
            symbol = symbol or cur.get("Symbol")
            if balance is not None:
                left = balance - FM_RESERVED_TOKENS
                pct = 100.0 - (left * 100.0 / FM_INITIAL_REAL_TOKEN_RESERVES)
                progress = max(0.0, min(1.0, pct / 100.0))
        if price_rows:
            px = _f(((price_rows[0].get("Trade") or {}).get("PriceInUSD")))
            if px is not None:
                market_cap = px * FM_TOTAL_SUPPLY

        migrated_pool = graduated_at = None
        complete = bool(progress is not None and progress >= 1.0)
        if migrate_rows:
            m0 = migrate_rows[0]
            graduated_at = _parse_dt((m0.get("Block") or {}).get("Time"))
            complete = True
            for arg in m0.get("Arguments") or []:
                if (arg.get("Name") or "").lower() in ("pair", "pool"):
                    migrated_pool = (arg.get("Value") or {}).get("address")
                    break

        return Launch(
            source="bitquery",
            chain=Chain.BSC,
            address=addr,
            launchpad="fourmeme",
            name=name,
            symbol=symbol,
            creator=creator,
            created_at=created_at,
            market_cap_usd=market_cap,
            bonding_curve_progress=progress,
            complete=complete,
            graduated_at=graduated_at,
            migrated_pool=migrated_pool,
            target_dex="pancakeswap",
            raw=data,
        )
