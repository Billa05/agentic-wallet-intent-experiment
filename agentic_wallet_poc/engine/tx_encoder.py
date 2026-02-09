"""
ABI-driven transaction encoder.

Converts ExecutablePayload (action + arguments) to raw transaction {to, value, data}.

Architecture:
  - Function signatures & types come from **Etherscan-verified ABIs** cached locally
    (see data/fetch_abis.py bootstrap script).
  - protocol_registry.json only stores metadata: action→function name, target contract,
    arg ordering. Zero ABI duplication.
  - standard_abis in the registry are used only for ERC20/ERC721/CryptoPunks transfers
    (these are token standards, not on-chain contracts to fetch).

To add a new protocol or action:
  1. Add contract address + action entry to protocol_registry.json
  2. Run `python data/fetch_abis.py` to fetch & cache the ABI from Etherscan
  3. Add an argument mapping in ACTION_ARG_MAP
  That's it. No selectors, no type strings, no new encode functions.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from eth_abi import encode as abi_encode
from eth_utils import keccak


# ─────────────────────────────────────────────────────────────────────
# ABI helpers — generic, work for any function
# ─────────────────────────────────────────────────────────────────────

def compute_selector(abi_entry: Dict) -> str:
    """Compute 4-byte selector from an ABI function entry (the canonical way)."""
    name = abi_entry["name"]
    types = [inp["type"] for inp in abi_entry.get("inputs", [])]
    sig = f"{name}({','.join(types)})"
    return "0x" + keccak(sig.encode()).hex()[:8]


def encode_from_abi(abi_entry: Dict, values: List[Any]) -> str:
    """
    Encode calldata from an ABI entry + ordered values.
    Returns full calldata: selector + ABI-encoded params.
    """
    types = [inp["type"] for inp in abi_entry.get("inputs", [])]
    selector = compute_selector(abi_entry)
    if types:
        encoded = abi_encode(types, values)
        return selector + encoded.hex()
    return selector


# ─────────────────────────────────────────────────────────────────────
# Registry + cached ABI loader
# ─────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ABI_CACHE_DIR = _DATA_DIR / "abi_cache"
_REGISTRY_PATH = _DATA_DIR / "registries" / "protocol_registry.json"

_registry: Optional[Dict] = None
_abi_cache: Dict[str, List[Dict]] = {}  # address.lower() -> full ABI


def _load_registry() -> Dict:
    global _registry
    if _registry is not None:
        return _registry
    _registry = json.loads(_REGISTRY_PATH.read_text())
    return _registry


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


def _resolve_action(action: str) -> Optional[Dict]:
    """Return the (protocol_dict, action_info) for an action name."""
    reg = _load_registry()
    for _, proto in reg.get("protocols", {}).items():
        action_info = proto.get("actions", {}).get(action)
        if action_info:
            return {"proto": proto, "action_info": action_info}
    return None


def get_action_abi(action: str) -> Optional[Dict]:
    """
    Resolve the ABI entry for a protocol action.

    Flow:
      1. Look up action in protocol_registry.json to get function name + target key
      2. Resolve target key → contract address
      3. Load cached Etherscan ABI for that contract
      4. Find the function by name in the ABI
      5. Return the ABI entry (with canonical types from Etherscan)

    This means selectors and parameter types are NEVER hardcoded — they come
    from Etherscan-verified contract source code.
    """
    resolved = _resolve_action(action)
    if not resolved:
        return None

    proto = resolved["proto"]
    action_info = resolved["action_info"]
    func_name = action_info.get("function")
    target_key = action_info.get("target")

    if not func_name or not target_key:
        return None

    target_addr = proto.get(target_key)
    if not target_addr:
        return None

    abi = _load_contract_abi(target_addr)
    if not abi:
        return None

    return _find_function_in_abi(abi, func_name)


def get_standard_abi(key: str) -> Optional[Dict]:
    """Look up a standard ABI (erc20_transfer, erc721_transferFrom, etc.).
    These are token standards that don't have a specific on-chain contract to fetch."""
    reg = _load_registry()
    return reg.get("standard_abis", {}).get(key)


def get_action_target(action: str) -> Optional[str]:
    """
    Resolve the target contract address for a protocol action.
    Uses the 'target' field in the action config to look up the address
    from the protocol's top-level fields.
    """
    resolved = _resolve_action(action)
    if not resolved:
        return None
    proto = resolved["proto"]
    target_key = resolved["action_info"].get("target")
    return proto.get(target_key) if target_key else None


# ─────────────────────────────────────────────────────────────────────
# Address helper
# ─────────────────────────────────────────────────────────────────────

CRYPTOPUNKS_ADDRESS = "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb"


def _addr(a: str) -> str:
    """Ensure address has 0x prefix."""
    return a if a.startswith("0x") else "0x" + a


# ─────────────────────────────────────────────────────────────────────
# Argument mapping: action → how to extract ordered ABI values from payload args
#
# Each entry maps action name → a callable that takes (args, from_address)
# and returns a list of values in ABI parameter order.
#
# This is the ONLY place that knows how payload fields map to ABI params.
# Adding a new action = adding one entry here + one ABI in the registry.
# ─────────────────────────────────────────────────────────────────────

def _args_aave_supply(args: Dict, from_addr: str) -> List[Any]:
    return [
        _addr(args["asset"]),
        int(args["amount"]),
        _addr(args.get("onBehalfOf", from_addr)),
        int(args.get("referralCode", 0)),
    ]


def _args_aave_withdraw(args: Dict, from_addr: str) -> List[Any]:
    return [
        _addr(args["asset"]),
        int(args["amount"]),
        _addr(args.get("to", from_addr)),
    ]


def _args_aave_borrow(args: Dict, from_addr: str) -> List[Any]:
    return [
        _addr(args["asset"]),
        int(args["amount"]),
        int(args.get("interestRateMode", 2)),
        int(args.get("referralCode", 0)),
        _addr(args.get("onBehalfOf", from_addr)),
    ]


def _args_aave_repay(args: Dict, from_addr: str) -> List[Any]:
    return [
        _addr(args["asset"]),
        int(args["amount"]),
        int(args.get("interestRateMode", 2)),
        _addr(args.get("onBehalfOf", from_addr)),
    ]


def _args_lido_stake(args: Dict, from_addr: str) -> List[Any]:
    return [_addr(args.get("_referral", "0x0000000000000000000000000000000000000000"))]


def _args_lido_unstake(args: Dict, from_addr: str) -> List[Any]:
    amounts_raw = args.get("_amounts") or ([args.get("amount")] if args.get("amount") is not None else [])
    return [
        [int(a) for a in amounts_raw],
        _addr(args.get("_owner", from_addr)),
    ]


def _args_uniswap_swap(args: Dict, from_addr: str) -> List[Any]:
    path = [_addr(p) for p in (args.get("path") or [])]
    deadline = int(args.get("deadline", 0)) or (int(time.time()) + 1200)
    return [
        int(args["amountIn"]),
        int(args.get("amountOutMin", 0)),
        path,
        _addr(args.get("to", from_addr)),
        deadline,
    ]


def _args_curve_add_liquidity(args: Dict, from_addr: str) -> List[Any]:
    amounts = [int(a) for a in (args.get("amounts") or ["0", "0", "0"])[:3]]
    return [amounts, int(args.get("min_mint_amount", 0))]


def _args_curve_remove_liquidity(args: Dict, from_addr: str) -> List[Any]:
    min_amounts = [int(m) for m in (args.get("min_amounts") or ["0", "0", "0"])[:3]]
    return [int(args["amount"]), min_amounts]


ACTION_ARG_MAP = {
    "aave_supply":           _args_aave_supply,
    "aave_withdraw":         _args_aave_withdraw,
    "aave_borrow":           _args_aave_borrow,
    "aave_repay":            _args_aave_repay,
    "lido_stake":            _args_lido_stake,
    "lido_unstake":          _args_lido_unstake,
    "uniswap_swap":          _args_uniswap_swap,
    "curve_add_liquidity":   _args_curve_add_liquidity,
    "curve_remove_liquidity": _args_curve_remove_liquidity,
}


# ─────────────────────────────────────────────────────────────────────
# Main encoder
# ─────────────────────────────────────────────────────────────────────

def payload_to_raw_tx(
    payload: Dict[str, Any],
    from_address: str,
) -> Optional[Dict[str, Any]]:
    """
    Convert ExecutablePayload dict to raw tx: { chain_id, to, value, data }.
    Returns None if the action is unsupported.
    """
    if not payload:
        return None

    action = payload.get("action")
    args = payload.get("arguments") or {}
    target_contract = payload.get("target_contract")
    chain_id = payload.get("chain_id", 1)

    # ── Native ETH transfer ─────────────────────────────────────
    if action == "transfer_native":
        to = args.get("to")
        if not to:
            return None
        return {
            "chain_id": chain_id,
            "to": to,
            "value": str(args.get("value", "0")),
            "data": "0x",
        }

    # ── ERC20 transfer ───────────────────────────────────────────
    if action == "transfer_erc20":
        to_recipient = args.get("to")
        value = args.get("value", "0")
        if not target_contract or not to_recipient:
            return None
        abi_entry = get_standard_abi("erc20_transfer")
        data = encode_from_abi(abi_entry, [_addr(to_recipient), int(value)])
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    # ── ERC721 transfer ──────────────────────────────────────────
    if action == "transfer_erc721":
        to_recipient = args.get("to")
        token_id = args.get("tokenId")
        if not target_contract or not to_recipient or token_id is None or not from_address:
            return None
        if target_contract.lower() == CRYPTOPUNKS_ADDRESS:
            abi_entry = get_standard_abi("cryptopunks_transferPunk")
            data = encode_from_abi(abi_entry, [_addr(to_recipient), int(token_id)])
        else:
            abi_entry = get_standard_abi("erc721_transferFrom")
            data = encode_from_abi(abi_entry, [_addr(from_address), _addr(to_recipient), int(token_id)])
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    # ── Protocol actions (ABI-driven) ────────────────────────────
    arg_mapper = ACTION_ARG_MAP.get(action)
    if arg_mapper is None:
        return None  # Unsupported action

    abi_entry = get_action_abi(action)
    if abi_entry is None:
        return None  # No ABI in registry

    # Build values in ABI parameter order
    try:
        values = arg_mapper(args, from_address)
    except (KeyError, TypeError, ValueError):
        return None

    # Encode: selector (computed from ABI) + params (encoded via eth_abi)
    data = encode_from_abi(abi_entry, values)

    # Resolve target contract from registry if not in payload
    to = target_contract or get_action_target(action)
    if not to:
        return None

    # Value: non-zero for payable calls (e.g., lido_stake)
    value = str(args.get("value", "0"))

    return {"chain_id": chain_id, "to": to, "value": value, "data": data}


# ─────────────────────────────────────────────────────────────────────
# Metadata builder (unchanged — not encoding-related)
# ─────────────────────────────────────────────────────────────────────

def build_metadata(
    payload: Dict[str, Any],
    token_registry: Dict[str, Any],
    ens_registry: Dict[str, str],
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
        addr_lower = addr.lower()
        for symbol, info in (token_registry.get("erc20_tokens") or {}).items():
            if (info.get("address") or "").lower() == addr_lower:
                return symbol
        return None

    def ens_for_address(addr: Optional[str]) -> Optional[str]:
        if not addr:
            return None
        addr_lower = addr.lower()
        for ens_name, a in ens_registry.items():
            if (a or "").lower() == addr_lower:
                return ens_name
        return None

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
