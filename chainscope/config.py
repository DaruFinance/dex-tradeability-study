"""Runtime settings: API keys (all optional), RPC endpoints, and the parquet data directory.

Free/no-key providers work out of the box. Setting any of the API keys below
(via environment variable or a .env file) lights up the corresponding richer
paid provider. Nothing here is required to run the free baseline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .chains import Chain, spec

# Recognized API-key environment variables -> the provider they unlock.
KEY_ENV_VARS = {
    "HELIUS_API_KEY": "helius",
    "BIRDEYE_API_KEY": "birdeye",
    "BITQUERY_TOKEN": "bitquery",
    "MORALIS_API_KEY": "moralis",
    "COINGECKO_API_KEY": "coingecko",
    "GOPLUS_APP_KEY": "goplus",
    "GOPLUS_APP_SECRET": "goplus",
    "ETHERSCAN_API_KEY": "etherscan",
}


def _load_dotenv() -> None:
    """Populate os.environ from a .env file without overriding already-set vars."""
    candidates = []
    explicit = os.environ.get("CHAINSCOPE_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parent.parent / ".env")
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except OSError:
            continue


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path.home() / "chainscope_data")
    solana_rpc_url: str = spec(Chain.SOLANA).default_rpc
    bsc_rpc_url: str = spec(Chain.BSC).default_rpc
    coingecko_tier: str = "demo"            # "demo" or "pro"
    http_timeout: float = 30.0
    max_retries: int = 4
    onchain_only: bool = False              # True = use ONLY raw-RPC providers, no gated 3rd-party APIs
    keys: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        keys = {}
        for env_var in KEY_ENV_VARS:
            val = os.environ.get(env_var)
            if val:
                keys[env_var] = val

        solana_rpc = os.environ.get("SOLANA_RPC_URL")
        if not solana_rpc and "HELIUS_API_KEY" in keys:
            solana_rpc = f"https://mainnet.helius-rpc.com/?api-key={keys['HELIUS_API_KEY']}"
        solana_rpc = solana_rpc or spec(Chain.SOLANA).default_rpc

        data_dir = os.environ.get("CHAINSCOPE_DATA_DIR")
        return cls(
            data_dir=Path(data_dir) if data_dir else Path.home() / "chainscope_data",
            solana_rpc_url=solana_rpc,
            bsc_rpc_url=os.environ.get("BSC_RPC_URL", spec(Chain.BSC).default_rpc),
            coingecko_tier=os.environ.get("COINGECKO_API_TIER", "demo").lower(),
            http_timeout=float(os.environ.get("CHAINSCOPE_HTTP_TIMEOUT", "30")),
            max_retries=int(os.environ.get("CHAINSCOPE_MAX_RETRIES", "4")),
            onchain_only=os.environ.get("CHAINSCOPE_ONCHAIN_ONLY", "").lower() in ("1", "true", "yes"),
            keys=keys,
        )

    def get_key(self, env_var: str | None) -> str | None:
        if not env_var:
            return None
        return self.keys.get(env_var)

    def rpc_url(self, chain: "str | Chain") -> str:
        return self.solana_rpc_url if Chain.parse(chain) == Chain.SOLANA else self.bsc_rpc_url

    def enabled_paid_providers(self) -> set[str]:
        return {KEY_ENV_VARS[k] for k in self.keys}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
