"""
Token resolver with 3-tier lookup: local cache, on-chain query, 1inch API.

Provides:
  - resolve_erc20(symbol) -> Optional[Dict]       -- symbol -> {address, decimals, symbol, name}
  - resolve_by_address(address) -> Optional[Dict]  -- address -> {address, decimals, symbol, name}
  - resolve_collection(name_or_alias) -> Optional[Dict] -- collection alias -> {address, name, symbol}
  - known_erc20_symbols() -> List[str]
  - known_collection_aliases() -> List[str]
  - symbol_for_address(address) -> Optional[str]

Tiers:
  - Tier 1: Local cache at data/cache/token_cache.json (~100 ERC-20s + ERC-721 collections)
  - Tier 2: On-chain query via Alchemy (ALCHEMY_API_KEY) — decimals(), symbol(), name()
  - Tier 3: 1inch Token API (ONEINCH_API_KEY) — symbol -> address discovery

Fallback: If no API keys set, cache-only mode. Never raises.
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_CACHE_PATH = _DATA_DIR / "cache" / "token_cache.json"

# Minimal ERC-20 ABI for on-chain queries
_ERC20_MINIMAL_ABI = [
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
]

# Some old contracts (e.g. MKR) return bytes32 instead of string
_ERC20_BYTES32_ABI = [
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "bytes32"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "bytes32"}], "type": "function"},
]


class TokenResolver:
    """Live token resolver with persistent file cache and optional on-chain/API lookups."""

    def __init__(self, cache_path: Optional[str] = None, w3=None):
        self._cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE_PATH

        # In-memory indexes
        self._erc20_by_symbol: Dict[str, Dict] = {}   # uppercase symbol -> {address, decimals, symbol, name}
        self._erc20_by_address: Dict[str, Dict] = {}   # lowercased address -> same dict
        self._erc721_collections: Dict[str, Dict] = {}  # lowercased alias -> {address, name, symbol}

        self._load_cache()

        # Initialize web3 provider (or use shared instance)
        self._w3: Any = w3
        if self._w3 is None:
            api_key = os.getenv("ALCHEMY_API_KEY")
            if api_key:
                try:
                    from web3 import Web3
                    provider_url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
                    self._w3 = Web3(Web3.HTTPProvider(provider_url))
                except Exception:
                    self._w3 = None

        self._oneinch_api_key = os.getenv("ONEINCH_API_KEY")

    # -- Primary API (used by resolvers.py) --------------------------------

    def resolve_erc20(self, symbol: str) -> Optional[Dict]:
        """Symbol -> {address, decimals, symbol, name}. Tier 1 -> Tier 3 -> Tier 2."""
        if not symbol:
            return None
        key = symbol.strip().upper()

        # Tier 1: cache
        if key in self._erc20_by_symbol:
            return self._erc20_by_symbol[key]

        # Tier 3: 1inch API (symbol -> address)
        result = self._query_1inch(key)
        if result:
            # Tier 2: confirm metadata on-chain
            on_chain = self._query_on_chain(result["address"])
            if on_chain:
                result = on_chain
            self._add_erc20(result)
            return result

        return None

    def resolve_by_address(self, address: str) -> Optional[Dict]:
        """Address -> {address, decimals, symbol, name}. Tier 1 -> Tier 2."""
        if not address:
            return None
        key = address.strip().lower()

        # Tier 1: cache
        if key in self._erc20_by_address:
            return self._erc20_by_address[key]

        # Tier 2: on-chain query
        result = self._query_on_chain(address)
        if result:
            self._add_erc20(result)
            return result

        return None

    def resolve_collection(self, name_or_alias: str) -> Optional[Dict]:
        """Collection alias -> {address, name, symbol}. Cache-only."""
        if not name_or_alias:
            return None
        s = name_or_alias.strip()
        if s.startswith("0x") and len(s) == 42:
            return {"address": s}

        # Try lowercased alias key
        key = s.lower().replace(" ", "")
        if key in self._erc721_collections:
            return self._erc721_collections[key]

        # Fuzzy match: check against collection name field
        for alias, info in self._erc721_collections.items():
            if (info.get("name") or alias).lower() == s.lower():
                return info

        return None

    # -- Convenience (used by prompts.py, build_metadata) -------------------

    def known_erc20_symbols(self) -> List[str]:
        """Return all cached ERC-20 symbols."""
        return list(self._erc20_by_symbol.keys())

    def known_collection_aliases(self) -> List[str]:
        """Return all cached ERC-721 collection aliases."""
        return list(self._erc721_collections.keys())

    def symbol_for_address(self, address: str) -> Optional[str]:
        """Reverse lookup: address -> symbol (cache only)."""
        if not address:
            return None
        info = self._erc20_by_address.get(address.lower())
        return info["symbol"] if info else None

    # -- Internals ----------------------------------------------------------

    def _query_on_chain(self, address: str) -> Optional[Dict]:
        """Tier 2: query decimals(), symbol(), name() on-chain via web3."""
        if self._w3 is None:
            return None
        try:
            from web3 import Web3
            addr = Web3.to_checksum_address(address)
            contract = self._w3.eth.contract(address=addr, abi=_ERC20_MINIMAL_ABI)

            decimals = contract.functions.decimals().call()

            # Try string ABI first, fall back to bytes32
            try:
                symbol = contract.functions.symbol().call()
            except Exception:
                contract_b32 = self._w3.eth.contract(address=addr, abi=_ERC20_BYTES32_ABI)
                raw = contract_b32.functions.symbol().call()
                symbol = raw.rstrip(b"\x00").decode("utf-8") if isinstance(raw, bytes) else str(raw)

            try:
                name = contract.functions.name().call()
            except Exception:
                contract_b32 = self._w3.eth.contract(address=addr, abi=_ERC20_BYTES32_ABI)
                raw = contract_b32.functions.name().call()
                name = raw.rstrip(b"\x00").decode("utf-8") if isinstance(raw, bytes) else str(raw)

            return {
                "address": addr,
                "decimals": int(decimals),
                "symbol": symbol,
                "name": name,
            }
        except Exception:
            return None

    def _query_1inch(self, symbol: str) -> Optional[Dict]:
        """Tier 3: 1inch Token API — symbol -> {address, decimals, symbol, name}."""
        if not self._oneinch_api_key:
            return None
        try:
            url = f"https://api.1inch.dev/token/v1.2/1/search?query={symbol}&limit=1"
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {self._oneinch_api_key}")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not data or not isinstance(data, list):
                return None
            # Find exact symbol match
            for token in data:
                if token.get("symbol", "").upper() == symbol.upper():
                    return {
                        "address": token["address"],
                        "decimals": token.get("decimals", 18),
                        "symbol": token["symbol"],
                        "name": token.get("name", token["symbol"]),
                    }
            return None
        except Exception:
            return None

    def _add_erc20(self, info: Dict) -> None:
        """Add a new ERC-20 entry to in-memory indexes and persist cache."""
        sym = info["symbol"].upper()
        self._erc20_by_symbol[sym] = info
        self._erc20_by_address[info["address"].lower()] = info
        self._save_cache()

    def _load_cache(self) -> None:
        """Load cache from JSON file and build in-memory indexes."""
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        for sym, info in data.get("erc20_tokens", {}).items():
            self._erc20_by_symbol[sym.upper()] = info
            self._erc20_by_address[info.get("address", "").lower()] = info

        for alias, info in data.get("erc721_collections", {}).items():
            self._erc721_collections[alias.lower()] = info

    def _save_cache(self) -> None:
        """Persist current state to JSON file."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "erc20_tokens": {info["symbol"]: info for info in self._erc20_by_symbol.values()},
            "erc721_collections": self._erc721_collections,
        }
        self._cache_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
