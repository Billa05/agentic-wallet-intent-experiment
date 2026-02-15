"""
ABI-driven transaction encoder.

Converts ExecutablePayload (action + arguments) to raw transaction {to, value, data}.

Architecture:
  - Function signatures & types come from **Etherscan-verified ABIs** cached locally
    (see data/fetch_abis.py bootstrap script).
  - All protocol metadata (contract addresses, action→function, standard ABIs) comes
    from JSON playbooks in data/playbooks/ — the single source of truth.
  - PlaybookEngine handles all param_mapping and encoding via JSON playbooks.

To add a new protocol or action:
  1. Create or update a playbook JSON in data/playbooks/
  2. Run `python data/fetch_abis.py` to fetch & cache the ABI from Etherscan
  That's it. No selectors, no type strings, no per-action code.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from eth_abi import encode as abi_encode
from eth_utils import keccak, to_checksum_address


# ─────────────────────────────────────────────────────────────────────
# ABI helpers — generic, work for any function (including struct/tuple params)
# ─────────────────────────────────────────────────────────────────────

def _resolve_abi_type(inp: Dict) -> str:
    """Resolve ABI input type, handling tuple/struct types recursively.

    Etherscan ABIs represent struct params as ``{"type": "tuple", "components": [...]}``.
    The canonical Solidity signature (used for selector computation) and ``eth_abi``
    both need the expanded form, e.g. ``(address,address,uint24,...)``.
    """
    typ = inp.get("type", "")
    if typ == "tuple":
        comps = inp.get("components", [])
        inner = ",".join(_resolve_abi_type(c) for c in comps)
        return f"({inner})"
    if typ == "tuple[]":
        comps = inp.get("components", [])
        inner = ",".join(_resolve_abi_type(c) for c in comps)
        return f"({inner})[]"
    return typ


def compute_selector(abi_entry: Dict) -> str:
    """Compute 4-byte selector from an ABI function entry (the canonical way)."""
    name = abi_entry["name"]
    types = [_resolve_abi_type(inp) for inp in abi_entry.get("inputs", [])]
    sig = f"{name}({','.join(types)})"
    return "0x" + keccak(sig.encode()).hex()[:8]


def encode_from_abi(abi_entry: Dict, values: List[Any]) -> str:
    """
    Encode calldata from an ABI entry + ordered values.
    Returns full calldata: selector + ABI-encoded params.

    Handles tuple/struct parameters: pass a Python tuple as the
    corresponding value, e.g. ``(addr, addr, fee, ...)``.
    """
    types = [_resolve_abi_type(inp) for inp in abi_entry.get("inputs", [])]
    selector = compute_selector(abi_entry)
    if types:
        encoded = abi_encode(types, values)
        return selector + encoded.hex()
    return selector


# ─────────────────────────────────────────────────────────────────────
# Cached ABI loader (reads from data/abi_cache/)
# ─────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ABI_CACHE_DIR = _DATA_DIR / "abi_cache"

_abi_cache: Dict[str, List[Dict]] = {}  # address.lower() -> full ABI


def _load_contract_abi(address: str) -> Optional[List[Dict]]:
    """Load a cached Etherscan ABI for a contract address."""
    key = address.lower()
    if key in _abi_cache:
        return _abi_cache[key]
    cache_file = _ABI_CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    abi = json.loads(cache_file.read_text())
    _abi_cache[key] = abi
    return abi


def _find_function_in_abi(abi: List[Dict], func_name: str) -> Optional[Dict]:
    """Find a function entry by name in an ABI list."""
    for entry in abi:
        if entry.get("type") == "function" and entry.get("name") == func_name:
            return entry
    return None


# ─────────────────────────────────────────────────────────────────────
# Address helper
# ─────────────────────────────────────────────────────────────────────

CRYPTOPUNKS_ADDRESS = "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb"


def _addr(a: str) -> str:
    """Normalise to EIP-55 checksum address (required by eth_abi >= 5)."""
    if not a:
        raise ValueError("_addr received empty/None address")
    s = a if a.startswith("0x") else "0x" + a
    # Must be a full 20-byte hex address (42 chars including '0x' prefix)
    if len(s) != 42:
        raise ValueError(f"_addr received invalid address (length {len(s)}): {s!r}")
    return to_checksum_address(s)


# ─────────────────────────────────────────────────────────────────────
# Main encoder — thin wrapper over PlaybookEngine for backward compat
# ─────────────────────────────────────────────────────────────────────

_playbook_engine = None


def _get_playbook_engine():
    """Lazy-init a PlaybookEngine for the encode_tx path."""
    global _playbook_engine
    if _playbook_engine is None:
        from engine.playbook_engine import PlaybookEngine
        from engine.token_resolver import TokenResolver
        from engine.ens_resolver import ENSResolver
        tr = TokenResolver()
        _playbook_engine = PlaybookEngine(
            token_resolver=tr,
            ens_resolver=ENSResolver(w3=tr._w3),
        )
    return _playbook_engine


def payload_to_raw_tx(
    payload: Dict[str, Any],
    from_address: str,
) -> Optional[Dict[str, Any]]:
    """
    Convert ExecutablePayload dict to raw tx: { chain_id, to, value, data }.

    This is a backward-compatible wrapper around PlaybookEngine.encode_tx().
    """
    engine = _get_playbook_engine()
    return engine.encode_tx(payload, from_address)


# ─────────────────────────────────────────────────────────────────────
# Metadata builder (unchanged — not encoding-related)
# ─────────────────────────────────────────────────────────────────────

def build_metadata(
    payload: Dict[str, Any],
    token_resolver,
    ens_resolver,
) -> Dict[str, Any]:
    """Build action-specific metadata for display and validation."""
    if not payload:
        return {}
    action = payload.get("action")
    args = payload.get("arguments") or {}
    target_contract = payload.get("target_contract")
    meta = {"action": action}

    def symbol_for_address(addr: Optional[str]) -> Optional[str]:
        if not addr:
            return None
        if token_resolver:
            return token_resolver.symbol_for_address(addr)
        return None

    def ens_for_address(addr: Optional[str]) -> Optional[str]:
        if not addr:
            return None
        return ens_resolver.reverse(addr) if ens_resolver else None

    if action == "transfer_native":
        to = args.get("to")
        meta["recipient"] = ens_for_address(to)
        meta["resolved_recipient"] = to
        meta["human_readable_amount"] = args.get("human_readable_amount", "")
        meta["amount_wei"] = args.get("value", "")

    elif action == "transfer_erc20":
        meta["token_address"] = target_contract
        meta["token_symbol"] = symbol_for_address(target_contract) or ""
        meta["recipient"] = ens_for_address(args.get("to"))
        meta["resolved_recipient"] = args.get("to", "")
        meta["human_readable_amount"] = args.get("human_readable_amount", "")
        meta["amount_wei"] = args.get("value", "")

    elif action == "transfer_erc721":
        meta["token_address"] = target_contract
        meta["recipient"] = ens_for_address(args.get("to"))
        meta["resolved_recipient"] = args.get("to", "")
        meta["token_id"] = args.get("tokenId")
        meta["human_readable_amount"] = args.get("human_readable_amount", "")

    else:
        meta["human_readable_amount"] = args.get("human_readable_amount", "")

    return meta
