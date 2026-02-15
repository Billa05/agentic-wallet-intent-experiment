"""
ENS name resolver backed by Alchemy + web3.py with file-based caching.

Provides:
  - resolve(name) -> Optional[str]   -- forward lookup (name -> address)
  - reverse(address) -> Optional[str] -- reverse lookup (address -> name)
  - known_names() -> List[str]        -- all cached forward-lookup names

Cache is persisted to data/cache/ens_cache.json. If ALCHEMY_API_KEY is not set
in the environment, all lookups return None and known_names() returns cached
names only (no network calls).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_CACHE_PATH = _DATA_DIR / "cache" / "ens_cache.json"


class ENSResolver:
    """Live ENS resolver with persistent file cache."""

    def __init__(self, cache_path: Optional[str] = None, w3=None):
        self._cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE_PATH

        # forward: name.eth -> 0x address
        # reverse: 0x address (lowercased) -> name.eth
        self._forward: Dict[str, str] = {}
        self._reverse: Dict[str, str] = {}

        self._load_cache()

        # Initialize web3 provider (use shared instance or create new)
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

    # -- Public API -------------------------------------------------------

    def resolve(self, name: str) -> Optional[str]:
        """Forward lookup: ENS name -> checksummed address."""
        if not name:
            return None
        key = name.strip().lower()
        if not key.endswith(".eth"):
            key = key + ".eth"

        # Check cache first
        if key in self._forward:
            return self._forward[key]

        # Live lookup
        if self._w3 is None:
            return None
        try:
            address = self._w3.ens.address(key)
        except Exception:
            return None
        if address is None:
            return None

        # Store in both directions and persist
        addr_str = str(address)
        self._forward[key] = addr_str
        self._reverse[addr_str.lower()] = key
        self._save_cache()
        return addr_str

    def reverse(self, address: str) -> Optional[str]:
        """Reverse lookup: address -> ENS name."""
        if not address:
            return None
        key = address.strip().lower()

        # Check cache first
        if key in self._reverse:
            return self._reverse[key]

        # Live lookup
        if self._w3 is None:
            return None
        try:
            name = self._w3.ens.name(address)
        except Exception:
            return None
        if name is None:
            return None

        # Store in both directions and persist
        self._reverse[key] = name
        self._forward[name.lower()] = address
        self._save_cache()
        return name

    def known_names(self) -> List[str]:
        """Return all cached forward-lookup names (for LLM prompts)."""
        return list(self._forward.keys())

    # -- Cache persistence ------------------------------------------------

    def _load_cache(self) -> None:
        """Load forward + reverse cache from JSON file."""
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            self._forward = data.get("forward", {})
            self._reverse = data.get("reverse", {})
        except (json.JSONDecodeError, OSError):
            pass

    def _save_cache(self) -> None:
        """Persist forward + reverse cache to JSON file."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"forward": self._forward, "reverse": self._reverse}
        self._cache_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
