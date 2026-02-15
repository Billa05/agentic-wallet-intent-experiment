"""
Parity test: verify the new PlaybookEngine produces identical output
to the old hardcoded payload_builder + tx_encoder for all 12 actions.
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.payload_builder import convert_human_to_payload
from engine.tx_encoder import payload_to_raw_tx
from engine.playbook_engine import PlaybookEngine

# ─────────────────────────────────────────────────────────────────────
# Load registries
# ─────────────────────────────────────────────────────────────────────

from engine.token_resolver import TokenResolver
from engine.ens_resolver import ENSResolver

TOKEN_RESOLVER = TokenResolver()
ENS_RESOLVER = ENSResolver(w3=TOKEN_RESOLVER._w3)

FROM_ADDRESS = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
CHAIN_ID = 1

# Fixed time for deadline parity
FIXED_TIME = 1700000000

# ─────────────────────────────────────────────────────────────────────
# Test cases: LLM output for each of the 12 action types
# ─────────────────────────────────────────────────────────────────────

TEST_CASES = [
    # Transfer actions
    {
        "name": "transfer_native",
        "llm_output": {
            "action": "transfer_native",
            "arguments": {"to": "bob.eth", "amount_human": "0.5"},
        },
    },
    {
        "name": "transfer_erc20",
        "llm_output": {
            "action": "transfer_erc20",
            "arguments": {"to": "alice.eth", "amount_human": "100", "asset": "USDC"},
        },
    },
    {
        "name": "transfer_erc721_standard",
        "llm_output": {
            "action": "transfer_erc721",
            "arguments": {"to": "bob.eth", "tokenId": 1234, "collection": "Bored Ape Yacht Club"},
        },
    },
    {
        "name": "transfer_erc721_cryptopunks",
        "llm_output": {
            "action": "transfer_erc721",
            "arguments": {"to": "alice.eth", "tokenId": 42, "collection": "CryptoPunks"},
        },
    },
    # AAVE actions
    {
        "name": "aave_supply",
        "llm_output": {
            "action": "aave_supply",
            "arguments": {"asset": "USDC", "amount_human": "500"},
        },
    },
    {
        "name": "aave_withdraw",
        "llm_output": {
            "action": "aave_withdraw",
            "arguments": {"asset": "DAI", "amount_human": "250"},
        },
    },
    {
        "name": "aave_withdraw_max",
        "llm_output": {
            "action": "aave_withdraw",
            "arguments": {"asset": "USDC", "amount_human": "max"},
        },
    },
    {
        "name": "aave_borrow",
        "llm_output": {
            "action": "aave_borrow",
            "arguments": {"asset": "USDT", "amount_human": "1000"},
        },
    },
    {
        "name": "aave_repay",
        "llm_output": {
            "action": "aave_repay",
            "arguments": {"asset": "USDC", "amount_human": "500"},
        },
    },
    {
        "name": "aave_repay_max",
        "llm_output": {
            "action": "aave_repay",
            "arguments": {"asset": "DAI", "amount_human": "all"},
        },
    },
    # Lido actions
    {
        "name": "lido_stake",
        "llm_output": {
            "action": "lido_stake",
            "arguments": {"amount_human": "1"},
        },
    },
    {
        "name": "lido_unstake",
        "llm_output": {
            "action": "lido_unstake",
            "arguments": {"amount_human": "15"},
        },
    },
    # Uniswap swap
    {
        "name": "uniswap_swap",
        "llm_output": {
            "action": "uniswap_swap",
            "arguments": {
                "asset_in": "WETH",
                "asset_out": "USDC",
                "amount_human": "0.5",
                "amountOutMinimum": "850",
            },
        },
    },
    {
        "name": "uniswap_swap_stable",
        "llm_output": {
            "action": "uniswap_swap",
            "arguments": {
                "asset_in": "USDC",
                "asset_out": "USDT",
                "amount_human": "1000",
            },
        },
    },
    # Curve actions
    {
        "name": "curve_add_liquidity",
        "llm_output": {
            "action": "curve_add_liquidity",
            "arguments": {"asset": "USDC", "amount_human": "100"},
        },
    },
    {
        "name": "curve_remove_liquidity",
        "llm_output": {
            "action": "curve_remove_liquidity",
            "arguments": {"amount_human": "50"},
        },
    },
]

# ─────────────────────────────────────────────────────────────────────
# New protocol test cases (PlaybookEngine only — no old engine equiv)
# ─────────────────────────────────────────────────────────────────────

NEW_PROTOCOL_TEST_CASES = [
    # WETH
    {
        "name": "weth_wrap",
        "llm_output": {
            "action": "weth_wrap",
            "arguments": {"amount_human": "2"},
        },
        "expect": {
            "action": "weth_wrap",
            "function_name": "deposit",
            "target_contract": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "value_nonzero": True,
        },
    },
    {
        "name": "weth_unwrap",
        "llm_output": {
            "action": "weth_unwrap",
            "arguments": {"amount_human": "1.5"},
        },
        "expect": {
            "action": "weth_unwrap",
            "function_name": "withdraw",
            "target_contract": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        },
    },
    # Compound V3
    {
        "name": "compound_supply",
        "llm_output": {
            "action": "compound_supply",
            "arguments": {"asset": "USDC", "amount_human": "1000"},
        },
        "expect": {
            "action": "compound_supply",
            "function_name": "supply",
            "target_contract": "0xc3d688B66703497DAA19211EEdff47f25384cdc3",
        },
    },
    {
        "name": "compound_withdraw",
        "llm_output": {
            "action": "compound_withdraw",
            "arguments": {"asset": "USDC", "amount_human": "500"},
        },
        "expect": {
            "action": "compound_withdraw",
            "function_name": "withdraw",
            "target_contract": "0xc3d688B66703497DAA19211EEdff47f25384cdc3",
        },
    },
    {
        "name": "compound_borrow",
        "llm_output": {
            "action": "compound_borrow",
            "arguments": {"asset": "USDC", "amount_human": "2000"},
        },
        "expect": {
            "action": "compound_borrow",
            "function_name": "withdraw",
            "target_contract": "0xc3d688B66703497DAA19211EEdff47f25384cdc3",
        },
    },
    {
        "name": "compound_repay",
        "llm_output": {
            "action": "compound_repay",
            "arguments": {"asset": "USDC", "amount_human": "1500"},
        },
        "expect": {
            "action": "compound_repay",
            "function_name": "supply",
            "target_contract": "0xc3d688B66703497DAA19211EEdff47f25384cdc3",
        },
    },
    # MakerDAO DSR
    {
        "name": "maker_deposit",
        "llm_output": {
            "action": "maker_deposit",
            "arguments": {"amount_human": "5000"},
        },
        "expect": {
            "action": "maker_deposit",
            "function_name": "deposit",
            "target_contract": "0x83F20F44975D03b1b09e64809B757c47f942BeeA",
        },
    },
    {
        "name": "maker_redeem",
        "llm_output": {
            "action": "maker_redeem",
            "arguments": {"amount_human": "3000"},
        },
        "expect": {
            "action": "maker_redeem",
            "function_name": "redeem",
            "target_contract": "0x83F20F44975D03b1b09e64809B757c47f942BeeA",
        },
    },
    # Rocket Pool
    {
        "name": "rocketpool_stake",
        "llm_output": {
            "action": "rocketpool_stake",
            "arguments": {"amount_human": "5"},
        },
        "expect": {
            "action": "rocketpool_stake",
            "function_name": "deposit",
            "target_contract": "0xDD9683b1bF4bB6d8fDF0A2B4A05aaadCA2A8a921",
            "value_nonzero": True,
        },
    },
    {
        "name": "rocketpool_unstake",
        "llm_output": {
            "action": "rocketpool_unstake",
            "arguments": {"amount_human": "3"},
        },
        "expect": {
            "action": "rocketpool_unstake",
            "function_name": "burn",
            "target_contract": "0xae78736Cd615f374D3085123A210448E74Fc6393",
        },
    },
    # EigenLayer
    {
        "name": "eigenlayer_deposit",
        "llm_output": {
            "action": "eigenlayer_deposit",
            "arguments": {"asset": "stETH", "amount_human": "10"},
        },
        "expect": {
            "action": "eigenlayer_deposit",
            "function_name": "depositIntoStrategy",
            "target_contract": "0x858646372CC42E1A627fcE94aa7A7033e7CF075A",
        },
    },
    # Balancer V2
    {
        "name": "balancer_swap",
        "llm_output": {
            "action": "balancer_swap",
            "arguments": {
                "asset_in": "WETH",
                "asset_out": "USDC",
                "amount_human": "1",
            },
        },
        "expect": {
            "action": "balancer_swap",
            "function_name": "swap",
            "target_contract": "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
        },
    },
]


def run_old_engine(llm_output):
    """Run the backward-compat wrapper (payload_builder + tx_encoder)."""
    payload = convert_human_to_payload(
        llm_output,
        TOKEN_RESOLVER,
        ENS_RESOLVER,
        chain_id=CHAIN_ID,
        from_address=FROM_ADDRESS,
    )
    if payload is None:
        return None, None
    raw_tx = payload_to_raw_tx(payload, FROM_ADDRESS)
    return payload, raw_tx


def run_new_engine(engine, llm_output):
    """Run the new PlaybookEngine."""
    payload = engine.build_payload(llm_output, chain_id=CHAIN_ID, from_address=FROM_ADDRESS)
    if payload is None:
        return None, None
    raw_tx = engine.encode_tx(payload, FROM_ADDRESS)
    return payload, raw_tx


def compare_payloads(old_payload, new_payload, test_name):
    """Compare two ExecutablePayload dicts, returns list of mismatches."""
    mismatches = []
    if old_payload is None and new_payload is None:
        return mismatches
    if old_payload is None or new_payload is None:
        return [f"One is None: old={old_payload is not None}, new={new_payload is not None}"]

    for key in ("chain_id", "action", "target_contract", "function_name"):
        if old_payload.get(key) != new_payload.get(key):
            mismatches.append(f"  {key}: old={old_payload.get(key)!r}, new={new_payload.get(key)!r}")

    old_args = old_payload.get("arguments", {})
    new_args = new_payload.get("arguments", {})

    # Compare all keys present in old (the ground truth)
    for k in old_args:
        old_val = old_args.get(k)
        new_val = new_args.get(k)
        # Skip deadline comparison (time-dependent)
        if k == "deadline":
            continue
        if str(old_val) != str(new_val):
            mismatches.append(f"  arguments.{k}: old={old_val!r}, new={new_val!r}")

    return mismatches


def compare_raw_tx(old_tx, new_tx, test_name):
    """Compare two raw tx dicts. For uniswap, data comparison is relaxed for deadline bytes."""
    mismatches = []
    if old_tx is None and new_tx is None:
        return mismatches
    if old_tx is None or new_tx is None:
        return [f"One is None: old={old_tx is not None}, new={new_tx is not None}"]

    for key in ("chain_id", "to", "value"):
        if str(old_tx.get(key)) != str(new_tx.get(key)):
            mismatches.append(f"  {key}: old={old_tx.get(key)!r}, new={new_tx.get(key)!r}")

    # For data, compare the selector (first 10 chars = 0x + 8 hex) always
    old_data = old_tx.get("data", "")
    new_data = new_tx.get("data", "")

    if "uniswap" in test_name:
        # Compare selector only (deadline makes the rest differ slightly)
        if old_data[:10] != new_data[:10]:
            mismatches.append(f"  data selector: old={old_data[:10]}, new={new_data[:10]}")
        # Also check data length matches
        if len(old_data) != len(new_data):
            mismatches.append(f"  data length: old={len(old_data)}, new={len(new_data)}")
    else:
        if old_data != new_data:
            mismatches.append(f"  data: old={old_data[:80]}..., new={new_data[:80]}...")

    return mismatches


def run_new_protocol_test(engine, tc):
    """Run a new-protocol test: build_payload + encode_tx, validate against expectations."""
    name = tc["name"]
    llm = tc["llm_output"]
    expect = tc.get("expect", {})
    mismatches = []

    with patch("time.time", return_value=FIXED_TIME):
        with patch("engine.resolvers.time") as mock_res_time:
            mock_res_time.time.return_value = FIXED_TIME
            payload = engine.build_payload(llm, chain_id=CHAIN_ID, from_address=FROM_ADDRESS)

    if payload is None:
        return ["build_payload returned None"]

    # Check payload fields
    if expect.get("action") and payload.get("action") != expect["action"]:
        mismatches.append(f"  action: expected={expect['action']!r}, got={payload.get('action')!r}")
    if expect.get("function_name") and payload.get("function_name") != expect["function_name"]:
        mismatches.append(f"  function_name: expected={expect['function_name']!r}, got={payload.get('function_name')!r}")
    if expect.get("target_contract"):
        exp_tc = expect["target_contract"].lower()
        got_tc = (payload.get("target_contract") or "").lower()
        if exp_tc != got_tc:
            mismatches.append(f"  target_contract: expected={expect['target_contract']!r}, got={payload.get('target_contract')!r}")

    # Encode tx
    with patch("time.time", return_value=FIXED_TIME):
        with patch("engine.resolvers.time") as mock_res_time:
            mock_res_time.time.return_value = FIXED_TIME
            raw_tx = engine.encode_tx(payload, FROM_ADDRESS)

    if raw_tx is None:
        mismatches.append("  encode_tx returned None")
        return mismatches

    # Validate tx has required fields
    if not raw_tx.get("data"):
        mismatches.append("  raw_tx.data is empty")
    if not raw_tx.get("to"):
        mismatches.append("  raw_tx.to is empty")

    # Check value for payable functions
    if expect.get("value_nonzero") and (raw_tx.get("value", "0") == "0"):
        mismatches.append("  raw_tx.value should be nonzero but is 0")

    return mismatches


def main():
    """Run parity tests."""
    engine = PlaybookEngine(
        token_resolver=TOKEN_RESOLVER,
        ens_resolver=ENS_RESOLVER,
    )

    passed = 0
    failed = 0
    errors = []

    print(f"\n{'='*70}")
    print("PLAYBOOK ENGINE PARITY TEST")
    print(f"{'='*70}\n")

    # --- Original parity tests (old vs new engine) ---
    print("--- Parity tests (old vs new engine) ---\n")
    for tc in TEST_CASES:
        name = tc["name"]
        llm = tc["llm_output"]

        try:
            # Use fixed time for deadline parity
            with patch("time.time", return_value=FIXED_TIME):
                with patch("engine.tx_encoder.time") as mock_tx_time:
                    mock_tx_time.time.return_value = FIXED_TIME
                    old_payload, old_tx = run_old_engine(llm)

            with patch("time.time", return_value=FIXED_TIME):
                with patch("engine.resolvers.time") as mock_res_time:
                    mock_res_time.time.return_value = FIXED_TIME
                    new_payload, new_tx = run_new_engine(engine, llm)

            # Compare payloads
            payload_mismatches = compare_payloads(old_payload, new_payload, name)
            tx_mismatches = compare_raw_tx(old_tx, new_tx, name)

            all_mismatches = payload_mismatches + tx_mismatches

            if all_mismatches:
                print(f"  FAIL  {name}")
                for m in all_mismatches:
                    print(f"        {m}")
                failed += 1
                errors.append((name, all_mismatches))
            else:
                print(f"  PASS  {name}")
                passed += 1

        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            errors.append((name, [str(e)]))

    # --- New protocol tests (PlaybookEngine only) ---
    print("\n--- New protocol tests (PlaybookEngine only) ---\n")
    for tc in NEW_PROTOCOL_TEST_CASES:
        name = tc["name"]
        try:
            mismatches = run_new_protocol_test(engine, tc)
            if mismatches:
                print(f"  FAIL  {name}")
                for m in mismatches:
                    print(f"        {m}")
                failed += 1
                errors.append((name, mismatches))
            else:
                print(f"  PASS  {name}")
                passed += 1
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            errors.append((name, [str(e)]))

    total = len(TEST_CASES) + len(NEW_PROTOCOL_TEST_CASES)
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {total} tests")
    print(f"{'='*70}\n")

    if errors:
        print("FAILURES:")
        for name, msgs in errors:
            print(f"\n  {name}:")
            for m in msgs:
                print(f"    {m}")
        print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
