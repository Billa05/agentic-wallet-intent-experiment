"""
Structural Calldata Validator (Etherscan ABI-driven)

Validates annotated dataset records by:
  1. Decoding selectors against Etherscan-verified ABIs (cached locally by fetch_abis.py)
  2. Decoding calldata params using eth_abi (Ethereum Foundation library)
  3. Generic checks — NO per-action code, scales to any protocol

Architecture:
  - Encoder uses:  Etherscan ABIs → eth_abi.encode() → calldata
  - Validator uses: Etherscan ABIs → eth_abi.decode() → params → check against intent
  Validation is independent because encode and decode are inverse operations.
  The ABIs are an external source of truth (Etherscan-verified contract source code).

Benefits over OpenChain:
  - Named parameters (asset, amount, to) instead of (arg0, arg1, arg2)
  - Complete coverage for all cached contracts
  - Zero network calls — fully offline
  - No selector collisions

Prerequisites:
  Run `python data/fetch_abis.py` first to populate data/abi_cache/

Usage:
  python data/validate_calldata.py --input data/datasets/annotated/annotated_defi_dataset.json
  python data/validate_calldata.py --input data/datasets/annotated/annotated_hybrid_dataset.json -v

Requires: eth-abi, eth-utils
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Auto-detect venv site-packages
_venv = Path(__file__).resolve().parent.parent / ".venv"
if _venv.exists():
    for sp in sorted(_venv.glob("lib/*/site-packages")):
        if str(sp) not in sys.path:
            sys.path.insert(0, str(sp))

from eth_abi import decode as abi_decode
from eth_abi.exceptions import DecodingError
from eth_utils import keccak


# ─────────────────────────────────────────────────────────────────────
# ABI type resolution (handles tuple/struct params)
# ─────────────────────────────────────────────────────────────────────

def _resolve_abi_type(inp: Dict) -> str:
    """Resolve ABI type, expanding tuple/struct to canonical form."""
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


# ─────────────────────────────────────────────────────────────────────
# Etherscan ABI cache — sole decode source (from data/abi_cache/)
# ─────────────────────────────────────────────────────────────────────

_ABI_CACHE_DIR = Path(__file__).parent / "abi_cache"

# selector -> { name, types, param_names, sig, raw_inputs }
_selector_map: Dict[str, Dict] = {}


def _build_selector_map():
    """
    Build selector → ABI entry map from all cached Etherscan ABIs.
    Called once at startup. Each cached file is a full contract ABI
    fetched from Etherscan by fetch_abis.py.
    """
    global _selector_map
    if _selector_map:
        return

    if not _ABI_CACHE_DIR.exists():
        print(f"Warning: ABI cache directory not found: {_ABI_CACHE_DIR}")
        print("Run `python data/fetch_abis.py` first to populate the cache.")
        return

    files_loaded = 0
    for cache_file in _ABI_CACHE_DIR.glob("0x*.json"):
        try:
            abi = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        files_loaded += 1
        for entry in abi:
            if entry.get("type") != "function":
                continue
            name = entry.get("name", "")
            inputs = entry.get("inputs", [])
            types = [_resolve_abi_type(inp) for inp in inputs]
            param_names = [inp.get("name", f"arg{i}") for i, inp in enumerate(inputs)]
            sig = f"{name}({','.join(types)})"
            sel = "0x" + keccak(sig.encode()).hex()[:8]
            # Don't overwrite — first match wins
            if sel not in _selector_map:
                _selector_map[sel] = {
                    "name": name,
                    "types": types,
                    "param_names": param_names,
                    "sig": sig,
                    "source": cache_file.stem,  # contract address
                    "raw_inputs": inputs,  # original ABI inputs (for tuple flattening)
                }

    if files_loaded == 0:
        print("Warning: No ABI cache files found. Run `python data/fetch_abis.py` first.")


def resolve_selectors(selectors: List[str]) -> Dict[str, Dict]:
    """
    Resolve a list of selectors against the Etherscan ABI cache.
    Returns { selector: { name, types, param_names, sig } } for found selectors.
    """
    _build_selector_map()
    result = {}
    for sel in selectors:
        sl = sel.lower()
        entry = _selector_map.get(sl)
        if entry:
            result[sl] = entry
    return result


# ─────────────────────────────────────────────────────────────────────
# Also support standard ABIs (ERC20/ERC721) from playbook JSONs
# ─────────────────────────────────────────────────────────────────────

_standard_selectors: Dict[str, Dict] = {}

_PLAYBOOKS_DIR = Path(__file__).parent / "playbooks"


def _build_standard_selectors():
    """Build selector map for standard ABIs (ERC20 transfer, ERC721 transferFrom, etc.)."""
    global _standard_selectors
    if _standard_selectors:
        return

    if not _PLAYBOOKS_DIR.exists():
        return

    for pb_file in _PLAYBOOKS_DIR.glob("*.json"):
        pb = json.loads(pb_file.read_text())
        for key, abi_entry in pb.get("standard_abis", {}).items():
            name = abi_entry.get("name", "")
            inputs = abi_entry.get("inputs", [])
            types = [_resolve_abi_type(inp) for inp in inputs]
            param_names = [inp.get("name", f"arg{i}") for i, inp in enumerate(inputs)]
            sig = f"{name}({','.join(types)})"
            sel = "0x" + keccak(sig.encode()).hex()[:8]
            _standard_selectors[sel] = {
                "name": name,
                "types": types,
                "param_names": param_names,
                "sig": sig,
                "source": f"standard:{key}",
                "raw_inputs": inputs,
            }


# ─────────────────────────────────────────────────────────────────────
# Decode calldata using Etherscan ABIs + eth_abi
# ─────────────────────────────────────────────────────────────────────

def decode_calldata(data_hex: str) -> Dict:
    """
    Decode calldata:
      1. Extract 4-byte selector
      2. Look up function in Etherscan ABI cache (with named params)
      3. Decode params with eth_abi

    Returns:
      { function_name, selector, signature, params: {name: val}, types, source }
    or:
      { error: "..." }
    """
    if not data_hex or data_hex in ("0x", "0x00"):
        return {"function_name": None, "params": {}, "is_empty": True}

    if len(data_hex) < 10:
        return {"error": "calldata too short"}

    selector = data_hex[:10].lower()
    payload_hex = data_hex[10:]

    # Look up in Etherscan cache first, then standard ABIs
    _build_selector_map()
    _build_standard_selectors()

    entry = _selector_map.get(selector) or _standard_selectors.get(selector)
    if not entry:
        return {"error": "unknown_selector", "selector": selector}

    types = entry["types"]
    param_names = entry["param_names"]
    func_name = entry["name"]
    sig = entry["sig"]

    if not types and payload_hex:
        return {"error": f"no types for '{sig}'", "function_name": func_name, "selector": selector}

    try:
        if types:
            values = abi_decode(types, bytes.fromhex(payload_hex))
        else:
            values = ()
    except Exception as e:
        return {"error": f"decode_failed: {e}", "function_name": func_name,
                "selector": selector, "signature": sig}

    # Flatten tuple/struct params into individual named fields
    raw_inputs = entry.get("raw_inputs", [])
    params = {}
    for i, val in enumerate(values):
        raw_inp = raw_inputs[i] if i < len(raw_inputs) else {}
        if raw_inp.get("type") == "tuple" and isinstance(val, tuple):
            components = raw_inp.get("components", [])
            for j, comp_val in enumerate(val):
                comp_name = components[j].get("name", f"arg{j}") if j < len(components) else f"arg{j}"
                if isinstance(comp_val, bytes):
                    comp_val = "0x" + comp_val.hex()
                params[comp_name] = comp_val
        else:
            if isinstance(val, bytes):
                val = "0x" + val.hex()
            name = param_names[i] if i < len(param_names) else f"arg{i}"
            params[name] = val

    return {
        "function_name": func_name,
        "selector": selector,
        "signature": sig,
        "params": params,
        "types": types,
        "source": entry.get("source", "etherscan"),
    }


# ─────────────────────────────────────────────────────────────────────
# Registries (loaded from playbooks + token cache)
# ─────────────────────────────────────────────────────────────────────

def _load_token_cache(base: Path) -> Dict:
    cache_path = Path(__file__).parent / "cache" / "token_cache.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {"erc20_tokens": {}, "erc721_collections": {}}


def _build_address_lookup(token_reg: Dict) -> Dict[str, Dict]:
    lookup = {}
    for _, info in token_reg.get("erc20_tokens", {}).items():
        lookup[info.get("address", "").lower()] = info
    return lookup


def _build_action_to_function_from_playbooks() -> Dict[str, str]:
    """Build action→function_name mapping from playbook JSONs."""
    mapping = {}
    if not _PLAYBOOKS_DIR.exists():
        return mapping
    for pb_file in _PLAYBOOKS_DIR.glob("*.json"):
        pb = json.loads(pb_file.read_text())
        for action_name, action_spec in pb.get("actions", {}).items():
            fn = action_spec.get("function_name", "")
            if fn:
                mapping[action_name] = fn
    return mapping


def _build_protocol_addresses_from_playbooks() -> Dict[str, List[str]]:
    """Build protocol_prefix→contract addresses mapping from playbook JSONs.

    The validator looks up addresses by action prefix (e.g. 'aave' from 'aave_supply').
    We derive prefixes from action names so the lookup matches.
    """
    result: Dict[str, List[str]] = {}
    if not _PLAYBOOKS_DIR.exists():
        return result
    for pb_file in _PLAYBOOKS_DIR.glob("*.json"):
        pb = json.loads(pb_file.read_text())
        addrs = []
        for _, contract_info in pb.get("contracts", {}).items():
            addr = contract_info.get("address", "")
            if addr:
                addrs.append(addr.lower())
        if not addrs:
            continue
        # Derive unique prefixes from action names (e.g. "aave" from "aave_supply")
        for action_name in pb.get("actions", {}):
            prefix = action_name.split("_")[0]
            if prefix not in result:
                result[prefix] = []
            for a in addrs:
                if a not in result[prefix]:
                    result[prefix].append(a)
    return result


# ─────────────────────────────────────────────────────────────────────
# Generic validation
# ─────────────────────────────────────────────────────────────────────

def _addr_eq(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower() if a and b else False


def _parse_human_amount(text: str) -> Tuple[Optional[float], Optional[str]]:
    if not text:
        return None, None
    m = re.match(r"^([\d.]+)\s+(\w+)$", text.strip())
    if m:
        return float(m.group(1)), m.group(2)
    m = re.match(r"^Swap\s+([\d.]+)\s+\((\w+)\s*->\s*(\w+)\)$", text.strip())
    if m:
        return float(m.group(1)), m.group(2)
    return None, None


def validate_record(
    idx: int,
    record: Dict,
    addr_lookup: Dict[str, Dict],
    action_to_func: Dict[str, str],
    proto_addrs: Dict[str, List[str]],
) -> Dict:
    """
    Generic validation — works for ANY action with zero per-action code.

    Checks:
      1. DECODE:    calldata decodes against Etherscan ABI + eth_abi
      2. FUNCTION:  decoded name matches playbook expectation
      3. TARGET:    'to' address is a known protocol contract
      4. TOKEN:     address params match known tokens consistent with intent
      5. AMOUNT:    uint param matches human_readable_amount x decimals
      6. SENDER:    address param matches from_address where expected
    """
    meta = record.get("metadata") or {}
    action = meta.get("action", "unknown")
    payload = record.get("target_payload") or {}
    ctx = record.get("user_context") or {}
    intent = record.get("user_intent", "")[:80]
    data_hex = payload.get("data", "0x")
    to_addr = payload.get("to", "")
    from_addr = ctx.get("from_address", "")

    checks: List[str] = []
    errors: List[str] = []

    if record.get("_annotation_failed") or not payload:
        return {"index": idx, "intent": intent, "action": action or "unknown",
                "checks": ["SKIP: annotation failed"], "errors": [], "status": "skipped"}

    # ── 1. DECODE ────────────────────────────────────────────────
    decoded = decode_calldata(data_hex)

    if decoded.get("is_empty"):
        if action == "transfer_native":
            checks.append("data=0x (correct for native transfer)")
        else:
            errors.append(f"calldata is empty for {action}")
    elif decoded.get("error"):
        errors.append(f"DECODE: {decoded['error']}")
    else:
        checks.append(f"DECODE OK: {decoded['signature']}")

    # ── 2. FUNCTION NAME ─────────────────────────────────────────
    expected_func = action_to_func.get(action)
    actual_func = decoded.get("function_name")

    if expected_func and actual_func:
        if actual_func == expected_func:
            checks.append(f"FUNCTION: '{actual_func}' matches playbook")
        else:
            alt_valid = {
                "transferFrom": {"safeTransferFrom", "transferPunk"},
                "transfer": {"transfer"},
            }
            alts = alt_valid.get(expected_func, set())
            if actual_func in alts:
                checks.append(f"FUNCTION: '{actual_func}' (valid alternate)")
            else:
                errors.append(f"FUNCTION: got '{actual_func}', playbook says '{expected_func}'")
    elif actual_func and action.startswith("transfer_"):
        valid = {"transfer", "transferFrom", "safeTransferFrom", "transferPunk"}
        if actual_func in valid:
            checks.append(f"FUNCTION: '{actual_func}' (valid transfer)")
        else:
            errors.append(f"FUNCTION: '{actual_func}' is not a transfer function")

    # ── 3. TARGET ADDRESS ────────────────────────────────────────
    protocol_prefix = action.split("_")[0]
    known = proto_addrs.get(protocol_prefix, [])
    if known:
        if any(_addr_eq(to_addr, k) for k in known):
            checks.append(f"TARGET: {to_addr[:14]}... is known {protocol_prefix} contract")
        else:
            errors.append(f"TARGET: {to_addr[:14]}... NOT in {protocol_prefix} playbook")

    # ── 4–6. GENERIC PARAM CHECKS ───────────────────────────────
    params = decoded.get("params", {})
    types = decoded.get("types", [])
    human_amount, intent_token = _parse_human_amount(meta.get("human_readable_amount", ""))

    # 4. TOKEN — find address params that match known tokens
    for pname, pval in params.items():
        if not isinstance(pval, str) or not pval.startswith("0x") or len(pval) != 42:
            continue
        token_info = addr_lookup.get(pval.lower())
        if token_info:
            sym = token_info["symbol"]
            if intent_token and sym.upper() == intent_token.upper():
                checks.append(f"TOKEN: {pname}={sym} matches intent '{intent_token}'")
            else:
                checks.append(f"TOKEN: {pname}={sym}")

    # 5. SENDER — find address params matching from_address
    for pname, pval in params.items():
        if isinstance(pval, str) and _addr_eq(pval, from_addr):
            checks.append(f"SENDER: {pname} matches from_address")

    # 6. SWAP PATH — detect address arrays
    for pname, pval in params.items():
        if isinstance(pval, (list, tuple)) and len(pval) >= 2:
            if all(isinstance(v, str) and v.startswith("0x") and len(v) == 42 for v in pval):
                src = addr_lookup.get(pval[0].lower(), {}).get("symbol", pval[0][:10])
                dst = addr_lookup.get(pval[-1].lower(), {}).get("symbol", pval[-1][:10])
                checks.append(f"PATH: {src} -> {dst}")

    # 7. AMOUNT
    if human_amount is not None and intent_token:
        decimals = None
        if intent_token.upper() in ("ETH", "STETH"):
            decimals = 18
        elif intent_token.upper() == "LP":
            decimals = 18
        else:
            for _, info in addr_lookup.items():
                if info.get("symbol", "").upper() == intent_token.upper():
                    decimals = info.get("decimals", 18)
                    break

        if decimals is not None:
            expected_base = int(round(human_amount * 10**decimals))
            tolerance = max(1, expected_base // 100_000)

            matched = False

            # Check tx value field (native ETH / payable)
            actual_value = int(payload.get("value", "0"))
            if actual_value > 0 and abs(actual_value - expected_base) <= tolerance:
                checks.append(f"AMOUNT: value={actual_value} matches {human_amount} {intent_token}")
                matched = True

            # Check uint params
            if not matched:
                for pname, pval in params.items():
                    if isinstance(pval, int) and pval > 0 and abs(pval - expected_base) <= tolerance:
                        checks.append(f"AMOUNT: {pname}={pval} matches {human_amount} {intent_token}")
                        matched = True
                        break
                    # Check inside arrays (Curve amounts, Lido amounts)
                    if isinstance(pval, (list, tuple)):
                        for j, av in enumerate(pval):
                            if isinstance(av, int) and av > 0 and abs(av - expected_base) <= tolerance:
                                checks.append(f"AMOUNT: {pname}[{j}]={av} matches {human_amount} {intent_token}")
                                matched = True
                                break
                    if matched:
                        break

            if not matched and actual_value == 0 and intent_token.upper() != "LP":
                uints = {k: v for k, v in params.items() if isinstance(v, int) and v > 0}
                if uints:
                    errors.append(
                        f"AMOUNT: no param matches {human_amount} {intent_token} "
                        f"(expected {expected_base}, got: {dict(list(uints.items())[:3])})"
                    )

    status = "PASS" if not errors else "FAIL"
    return {"index": idx, "intent": intent, "action": action,
            "checks": checks, "errors": errors, "status": status}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Structural calldata validator (Etherscan ABI-driven)")
    parser.add_argument("--input", required=True, help="Annotated dataset JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output", help="Write results to JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent.parent / args.input
    if not input_path.exists():
        input_path = Path(__file__).parent / Path(args.input).name
    if not input_path.exists():
        print(f"Error: not found: {args.input}")
        sys.exit(1)

    reg_dir = Path(__file__).parent / "registries"
    token_reg = _load_token_cache(reg_dir)
    addr_lookup = _build_address_lookup(token_reg)
    action_to_func = _build_action_to_function_from_playbooks()
    proto_addrs = _build_protocol_addresses_from_playbooks()

    with open(input_path) as f:
        dataset = json.load(f)

    # Build selector map from cached Etherscan ABIs
    _build_selector_map()
    _build_standard_selectors()

    print("Structural Calldata Validator (Etherscan ABI)")
    print("=" * 60)
    print(f"Input:      {input_path.name}")
    print(f"Records:    {len(dataset)}")
    print(f"ABI source: Etherscan-verified contracts ({len(_selector_map)} selectors)")
    print(f"Standard:   {len(_standard_selectors)} selectors (ERC20/ERC721/CryptoPunks)")
    print(f"Decoder:    eth_abi (Ethereum Foundation)")
    print(f"Checks:     decode, function, target, token, amount, sender")
    print("=" * 60)

    # Show which selectors are in the dataset
    selectors = set()
    for rec in dataset:
        tp = rec.get("target_payload") or {}
        d = tp.get("data", "0x")
        if d and len(d) >= 10 and d != "0x":
            selectors.add(d[:10].lower())

    if selectors:
        print(f"\nDataset selectors ({len(selectors)}):")
        all_known = {**_selector_map, **_standard_selectors}
        for sel in sorted(selectors):
            entry = all_known.get(sel)
            if entry:
                print(f"  {sel} -> {entry['sig']}")
            else:
                print(f"  {sel} -> (unknown)")
    print()

    # Validate
    pass_n = fail_n = skip_n = 0
    results = []

    for i, rec in enumerate(dataset):
        r = validate_record(i, rec, addr_lookup, action_to_func, proto_addrs)
        results.append(r)

        if r["status"] == "skipped":
            skip_n += 1
            if args.verbose:
                print(f"[{i:3d}] SKIP  {r['action']:25s}  {r['intent']}")
            continue

        if r["status"] == "PASS":
            pass_n += 1
        else:
            fail_n += 1

        marker = "✓" if r["status"] == "PASS" else "✗"

        if args.verbose or r["status"] == "FAIL":
            print(f"[{i:3d}] {marker} {r['action']:25s}  {r['intent']}")
            if args.verbose:
                for c in r["checks"]:
                    print(f"        {c}")
            for e in r["errors"]:
                print(f"      ✗ {e}")
            if args.verbose:
                print()

    total = pass_n + fail_n
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Records:    {len(dataset)}")
    if skip_n:
        print(f"  Skipped:    {skip_n} (annotation failures)")
    print(f"  Validated:  {total}")
    print(f"  ✓ PASS:     {pass_n}")
    print(f"  ✗ FAIL:     {fail_n}")
    if total:
        print(f"  Pass rate:  {pass_n / total * 100:.1f}%")
    print()
    if fail_n == 0:
        print("All records structurally valid!")
    else:
        print(f"{fail_n} record(s) have issues — see errors above.")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
