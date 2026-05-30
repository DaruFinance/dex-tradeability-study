"""Historical, survivorship-free TOKEN/POOL UNIVERSE builder — entirely on-chain.

The single most important fix for survivorship bias in a DEX backtest is to know
*every* pool that ever existed, including the ones that are now dead, rugged, or
de-listed. Paid aggregators only surface pools that are currently alive (or that
they bothered to index), so a backtest built on them silently conditions on
survival. This module digs the universe straight from the chain instead:

  - BSC: chunked ``eth_getLogs`` with **no token filter** over each DEX factory,
    so we capture the *creation event* of every pair/pool ever deployed. A
    creation log is immutable and never disappears, so the set is complete back
    to the factory's deploy block (limited only by how deep the RPC will serve).

  - Solana: ``getProgramAccounts`` snapshots each AMM program's pool accounts.
    A snapshot is point-in-time *current* — it lists pools that still have an
    account on-chain. For the truly survivorship-free *historical* set you must
    replay program creates from an archive (Old Faithful / a local validator
    ledger); a live snapshot is the best a public RPC will give and is noted as
    such on every record (``created_at=None``).

This is a ``UniverseBuilder`` (NOT a Provider): it doesn't fit the per-token
Provider capability model — it scans the whole chain, not one address. RPC only;
no third-party API, no API key.
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone

from .chains import Chain
from .http import HttpError
from .models import PoolCreation
from .storage import ParquetStore

log = logging.getLogger("chainscope.universe")

# ---------------------------------------------------------------------------
# BSC factories + event topic0 hashes — VERIFIED (see module REPORT / sources).
# Topic hashes are keccak256 of the event signature; each was confirmed both by
# local keccak256 and against real on-chain logs on BNB Smart Chain.
# ---------------------------------------------------------------------------

# PancakeSwap V2 factory.
#   event PairCreated(address indexed token0, address indexed token1, address pair, uint)
#   token0 = topics[1], token1 = topics[2]; data = [pair (32B), allPairsLength (32B)]
PANCAKE_V2_FACTORY = "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73"
PAIR_CREATED_TOPIC = "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9"

# PancakeSwap V3 factory (same ABI as Uniswap V3).
#   event PoolCreated(address indexed token0, address indexed token1,
#                     uint24 indexed fee, int24 tickSpacing, address pool)
#   token0 = topics[1], token1 = topics[2], fee = topics[3];
#   data = [tickSpacing (32B), pool (32B)]
PANCAKE_V3_FACTORY = "0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865"
POOL_CREATED_TOPIC = "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118"

# four.meme launchpad token factory (bonding-curve memecoin launches on BSC).
#   event TokenCreate(address creator, address token, uint256 requestId,
#                     string name, string symbol, uint256 totalSupply,
#                     uint256 launchTime, uint256 launchFee)
#   ALL params non-indexed (topics == [topic0]); data words: [0]=creator,
#   [1]=token, [2]=requestId, ... (verified by decoding a live log).
# The bonding-curve quote token is WBNB, so we record token1 = WBNB.
FOURMEME_FACTORY = "0x5c952063c7fc8610ffdb798152d69f0b9550762b"
TOKEN_CREATE_TOPIC = "0x396d5e902b675b032348d3d2e9517ee8f0c4a926603fbc075d3d282ff00cad20"
WBNB = "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"

# getLogs range cap. Free public RPCs reject spans much wider than ~5000 blocks.
BSC_CHUNK = 5000
# Max chunks per build() call before we stop and log a (loud, non-silent) truncation
# note. ~3000 chunks * 5000 blocks ~= 15M blocks ~= the full BSC history at the cap,
# but a single public-RPC call will realistically be told to stop long before that.
BSC_MAX_CHUNKS = 3000

# ---------------------------------------------------------------------------
# Solana AMM programs + pool-account layouts (offsets are the base/quote MINT
# pubkeys, taken from each program's SDK/IDL — same constants the indexer uses).
# ---------------------------------------------------------------------------
RAYDIUM_AMM_V4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
PUMPSWAP_AMM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
ORCA_WHIRLPOOL = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"

# (base_mint offset, quote_mint offset, expected account data length)
# Raydium LIQUIDITY_STATE_LAYOUT_V4: baseMint@400, quoteMint@432, total 752 bytes.
# PumpSwap Pool: base_mint@43, quote_mint@75.
# Orca Whirlpool: token_mint_a@101, token_mint_b@181.
SOLANA_PROGRAMS: dict[str, dict] = {
    RAYDIUM_AMM_V4: {"dex": "raydium-v4", "base": 400, "quote": 432, "size": 752},
    PUMPSWAP_AMM: {"dex": "pumpswap", "base": 43, "quote": 75, "size": None},
    ORCA_WHIRLPOOL: {"dex": "orca-whirlpool", "base": 101, "quote": 181, "size": 653},
}

DEX_FOR_FACTORY = {
    PANCAKE_V2_FACTORY.lower(): "pancakeswap-v2",
    PANCAKE_V3_FACTORY.lower(): "pancakeswap-v3",
    FOURMEME_FACTORY.lower(): "fourmeme",
}

# base58 (Bitcoin/Solana alphabet) — needed to encode mint pubkeys from raw bytes.
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(raw: bytes) -> str:
    """Pure-Python base58 encode (preserves leading-zero bytes as '1')."""
    n = int.from_bytes(raw, "big")
    out = ""
    while n > 0:
        n, rem = divmod(n, 58)
        out = _B58_ALPHABET[rem] + out
    pad = 0
    for b in raw:
        if b == 0:
            pad += 1
        else:
            break
    return "1" * pad + out


def _evm_addr(word: str) -> str:
    """Last 20 bytes of a 32-byte ABI/topic word -> lowercase 0x address."""
    return "0x" + word[-40:].lower()


def pools_for_token(creations: list[PoolCreation], token: str) -> list[PoolCreation]:
    """Filter a creation set down to pools that include `token` on either side.

    The cross-section a backtest actually needs for one asset: every venue it
    ever traded on, dead ones included. Matching is case-insensitive on EVM
    addresses and exact on Solana mints."""
    t = (token or "").strip()
    tl = t.lower()
    out = []
    for c in creations:
        t0 = (c.token0 or "")
        t1 = (c.token1 or "")
        if t0 == t or t1 == t or t0.lower() == tl or t1.lower() == tl:
            out.append(c)
    return out


class UniverseBuilder:
    """Build the historical pool/token universe for a chain, on-chain only.

    Construct with the shared (http, settings). ``scan_bsc`` / ``scan_solana``
    return the raw :class:`PoolCreation` list; ``build`` runs a scan and persists
    it to the ``pool_universe`` parquet dataset.
    """

    def __init__(self, http, settings):
        self.http = http
        self.settings = settings
        self._clock: tuple[int, int, float] | None = None  # (block, ts, sec/block)

    # ------------------------------------------------------------------ RPC --

    async def _bsc_rpc(self, method: str, params: list):
        url = self.settings.rpc_url(Chain.BSC)
        resp = await self.http.post_json(
            url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        )
        if isinstance(resp, dict) and resp.get("error"):
            raise HttpError(0, url, str(resp["error"]))
        return resp.get("result") if isinstance(resp, dict) else None

    async def _sol_rpc(self, method: str, params: list):
        url = self.settings.rpc_url(Chain.SOLANA)
        resp = await self.http.post_json(
            url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        )
        if isinstance(resp, dict) and resp.get("error"):
            raise HttpError(0, url, str(resp["error"]))
        return resp.get("result") if isinstance(resp, dict) else None

    # ------------------------------------------------------- BSC block clock --

    async def _latest_block(self) -> int:
        return int(await self._bsc_rpc("eth_blockNumber", []), 16)

    async def _block_ts(self, block: int) -> int:
        b = await self._bsc_rpc("eth_getBlockByNumber", [hex(block), False])
        return int(b["timestamp"], 16)

    async def _clock_calibrate(self) -> tuple[int, int, float]:
        """Derive seconds/block by sampling two recent blocks (same trick as the
        BSC indexer). Lets us approximate created_at for a creation block without
        an eth_getBlockByNumber per pool."""
        if self._clock is not None:
            return self._clock
        latest = await self._latest_block()
        back = max(1, latest - 20000)
        t_latest = await self._block_ts(latest)
        t_back = await self._block_ts(back)
        spb = (t_latest - t_back) / max(1, latest - back)
        self._clock = (latest, t_latest, spb if spb > 0 else 0.75)
        return self._clock

    def _ts_of_block(self, block: int) -> datetime:
        latest, t_latest, spb = self._clock  # type: ignore[misc]
        secs = t_latest - (latest - block) * spb
        return datetime.fromtimestamp(secs, tz=timezone.utc)

    # ----------------------------------------------------- BSC: scan factories --

    async def _get_factory_logs(self, factory: str, topic: str,
                                from_b: int, to_b: int) -> list[dict]:
        """Chunked eth_getLogs over [from_b, to_b] for one factory+topic, no token
        filter. Returns logs sorted by (block, logIndex). Loud truncation note if
        the range exceeds BSC_MAX_CHUNKS — never a silent cap."""
        out: list[dict] = []
        b = from_b
        chunks = 0
        while b <= to_b and chunks < BSC_MAX_CHUNKS:
            hi = min(b + BSC_CHUNK - 1, to_b)
            try:
                res = await self._bsc_rpc("eth_getLogs", [{
                    "address": factory, "topics": [topic],
                    "fromBlock": hex(b), "toBlock": hex(hi),
                }])
                out.extend(res or [])
            except HttpError as exc:
                log.debug("getLogs %s %s-%s failed: %s", factory, b, hi, exc)
            b = hi + 1
            chunks += 1
        if b <= to_b:
            log.warning(
                "universe.scan_bsc: TRUNCATED %s at block %d (covered %d of %d blocks; "
                "BSC_MAX_CHUNKS=%d). Re-run from_block=%d, or point BSC_RPC_URL at a "
                "local archive node, to capture the rest.",
                factory, b, b - from_b, to_b - from_b, BSC_MAX_CHUNKS, b,
            )
        out.sort(key=lambda lg: (int(lg.get("blockNumber", "0x0"), 16),
                                 int(lg.get("logIndex", "0x0"), 16)))
        return out

    def _decode_v2(self, lg: dict) -> PoolCreation | None:
        topics = lg.get("topics") or []
        data = (lg.get("data") or "0x")[2:]
        if len(topics) < 3 or len(data) < 128:
            return None
        token0 = _evm_addr(topics[1])
        token1 = _evm_addr(topics[2])
        pair = _evm_addr(data[0:64])           # 1st data word = pair address
        blk = int(lg["blockNumber"], 16)
        return PoolCreation(
            source="universe", chain=Chain.BSC, dex="pancakeswap-v2",
            pair_address=pair, token0=token0, token1=token1, fee_bps=25.0,
            created_block=blk, created_at=self._ts_of_block(blk),
            creator=None, raw=lg,
        )

    def _decode_v3(self, lg: dict) -> PoolCreation | None:
        topics = lg.get("topics") or []
        data = (lg.get("data") or "0x")[2:]
        if len(topics) < 4 or len(data) < 128:
            return None
        token0 = _evm_addr(topics[1])
        token1 = _evm_addr(topics[2])
        fee = int(topics[3], 16)                # uint24, units of 1e-6
        pool = _evm_addr(data[64:128])          # 2nd data word = pool (1st is tickSpacing)
        blk = int(lg["blockNumber"], 16)
        return PoolCreation(
            source="universe", chain=Chain.BSC, dex="pancakeswap-v3",
            pair_address=pool, token0=token0, token1=token1,
            fee_bps=fee / 100.0,                # 1e-6 units -> bps
            created_block=blk, created_at=self._ts_of_block(blk),
            creator=None, raw=lg,
        )

    def _decode_fourmeme(self, lg: dict) -> PoolCreation | None:
        # All args non-indexed: creator = data word[0], token = data word[1].
        data = (lg.get("data") or "0x")[2:]
        if len(data) < 128:
            return None
        creator = _evm_addr(data[0:64])
        token = _evm_addr(data[64:128])
        blk = int(lg["blockNumber"], 16)
        # four.meme tokens launch on a WBNB bonding curve; record WBNB as token1.
        return PoolCreation(
            source="universe", chain=Chain.BSC, dex="fourmeme",
            pair_address=token, token0=token, token1=WBNB.lower(), fee_bps=None,
            created_block=blk, created_at=self._ts_of_block(blk),
            creator=creator, raw=lg,
        )

    async def scan_bsc(self, from_block: int | None = None, to_block: int | None = None,
                       days: int | None = None) -> list[PoolCreation]:
        """Scan all PancakeSwap V2 + V3 pair/pool creations and four.meme token
        launches over a block range, decoded into PoolCreation records.

        Range selection (in priority order):
          - explicit from_block / to_block, or
          - `days` back from head (via the block clock), or
          - default: the last BSC_CHUNK*8 blocks (a quick recent window).
        """
        latest, t_latest, spb = await self._clock_calibrate()
        to_b = to_block if to_block is not None else latest
        if from_block is not None:
            from_b = from_block
        elif days is not None:
            from_b = max(1, to_b - int(days * 86400 / spb))
        else:
            from_b = max(1, to_b - BSC_CHUNK * 8)

        creations: list[PoolCreation] = []

        v2 = await self._get_factory_logs(PANCAKE_V2_FACTORY, PAIR_CREATED_TOPIC, from_b, to_b)
        for lg in v2:
            rec = self._decode_v2(lg)
            if rec is not None:
                creations.append(rec)
        log.info("scan_bsc: PancakeSwap V2 PairCreated = %d", len(v2))

        v3 = await self._get_factory_logs(PANCAKE_V3_FACTORY, POOL_CREATED_TOPIC, from_b, to_b)
        for lg in v3:
            rec = self._decode_v3(lg)
            if rec is not None:
                creations.append(rec)
        log.info("scan_bsc: PancakeSwap V3 PoolCreated = %d", len(v3))

        fm = await self._get_factory_logs(FOURMEME_FACTORY, TOKEN_CREATE_TOPIC, from_b, to_b)
        for lg in fm:
            rec = self._decode_fourmeme(lg)
            if rec is not None:
                creations.append(rec)
        log.info("scan_bsc: four.meme TokenCreate = %d", len(fm))

        creations.sort(key=lambda c: (c.created_block or 0))
        return creations

    # ----------------------------------------------- Solana: getProgramAccounts --

    async def _scan_solana_program(self, program: str, cfg: dict) -> list[PoolCreation]:
        """getProgramAccounts snapshot of one AMM program's pool accounts.

        We use a dataSlice that spans both mint offsets so we transfer only the
        bytes we decode (the full Raydium state is 752B/account; a snapshot of all
        of them is huge — public RPCs routinely reject the unsliced call). A
        `dataSize` memcmp filter (where known) restricts to the pool-state account
        type. Returns [] (with a logged note) if the public RPC refuses the call.
        """
        base_off, quote_off = cfg["base"], cfg["quote"]
        # one contiguous slice covering [base_off, quote_off+32)
        lo = min(base_off, quote_off)
        hi = max(base_off, quote_off) + 32
        opts: dict = {
            "encoding": "base64",
            "dataSlice": {"offset": lo, "length": hi - lo},
        }
        filters: list[dict] = []
        if cfg.get("size"):
            filters.append({"dataSize": cfg["size"]})
        if filters:
            opts["filters"] = filters

        try:
            res = await self._sol_rpc("getProgramAccounts", [program, opts])
        except HttpError as exc:
            # Public RPCs commonly answer 410/-32010/"excluded from query" or time out
            # for getProgramAccounts on these large programs. Surface it, don't crash.
            log.warning(
                "scan_solana: getProgramAccounts rejected for %s (%s). Public RPCs "
                "throttle/disable this; use a dedicated RPC (Helius/Triton) or a local "
                "node. Returning 0 pools for this program.", cfg["dex"], exc)
            return []

        out: list[PoolCreation] = []
        for acct in (res or []):
            try:
                pubkey = acct.get("pubkey")
                blob = ((acct.get("account") or {}).get("data"))
                if not pubkey or not blob or not isinstance(blob, list):
                    continue
                raw = base64.b64decode(blob[0])
                # offsets are relative to the slice start (lo)
                bm = raw[base_off - lo: base_off - lo + 32]
                qm = raw[quote_off - lo: quote_off - lo + 32]
                if len(bm) < 32 or len(qm) < 32:
                    continue
                out.append(PoolCreation(
                    source="universe", chain=Chain.SOLANA, dex=cfg["dex"],
                    pair_address=pubkey,
                    token0=_b58encode(bm), token1=_b58encode(qm),
                    fee_bps=None, created_block=None,
                    created_at=None,   # snapshot: historical create time unknown (need archive)
                    creator=None,
                    raw={"program": program},
                ))
            except (ValueError, KeyError) as exc:
                log.debug("scan_solana: decode failed for %s: %s", cfg["dex"], exc)
        log.info("scan_solana: %s pools = %d", cfg["dex"], len(out))
        return out

    async def scan_solana(self, programs: tuple[str, ...] = (
            RAYDIUM_AMM_V4, PUMPSWAP_AMM, ORCA_WHIRLPOOL)) -> list[PoolCreation]:
        """CURRENT snapshot of all live pool accounts across the given AMM programs
        via getProgramAccounts, decoding base/quote mints at the known offsets.

        NOTE on survivorship: this is a *current* snapshot — it sees pools that
        still have an on-chain account. It will NOT include pools whose accounts
        were closed. created_at is None because a snapshot carries no creation
        slot. For the true survivorship-free *historical* set you must replay each
        program's pool-init instructions from an archive (Old Faithful / a local
        validator ledger); that backfill is out of scope for a public-RPC scan.
        """
        out: list[PoolCreation] = []
        for program in programs:
            cfg = SOLANA_PROGRAMS.get(program)
            if cfg is None:
                log.warning("scan_solana: no layout for program %s; skipping", program)
                continue
            out.extend(await self._scan_solana_program(program, cfg))
        return out

    # --------------------------------------------------------------- build --

    async def build(self, chain, *, from_block: int | None = None,
                    to_block: int | None = None, days: int | None = None,
                    programs: tuple[str, ...] | None = None,
                    persist: bool = True) -> dict:
        """Run the universe scan for `chain` and (by default) persist it.

        Writes to the ``pool_universe`` parquet dataset partitioned by created_at
        (creation date) so the universe is laid out by *when pools were born*.
        Returns a stats dict.
        """
        c = Chain.parse(chain)
        if c == Chain.BSC:
            creations = await self.scan_bsc(from_block=from_block, to_block=to_block, days=days)
        else:
            kwargs = {"programs": programs} if programs else {}
            creations = await self.scan_solana(**kwargs)

        rows = 0
        if persist and creations:
            rows = ParquetStore().write("pool_universe", creations, time_field="created_at")

        by_dex: dict[str, int] = {}
        for rec in creations:
            by_dex[rec.dex or "unknown"] = by_dex.get(rec.dex or "unknown", 0) + 1
        return {
            "chain": str(c.value),
            "pools_found": len(creations),
            "by_dex": by_dex,
            "rows_written": rows,
            "with_created_at": sum(1 for r in creations if r.created_at is not None),
        }
