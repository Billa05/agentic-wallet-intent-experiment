"""
Minimal Tenderly simulation: run one transaction and report if it is executable or not.

Aave V3 repay requires overriding multiple contracts (debt token + USDC balance/allowance).
USDC uses upgradeable proxy layout (balance at slot 9, allowance at slot 10).
VariableDebtToken uses ScaledBalanceTokenBase _userState (struct packed in one slot).

Alternative approach: use a real user who has USDC debt (e.g. from Aave subgraph)
and simulate as them with only balance/allowance overrides; or use --simulation-type full
and a very recent --block-number.

Payload: "Can you help me repay my Aave loan of 1200 USDC?" from annotated_defi_dataset.json.

Usage:
  python data/test_tenderly_api.py [--block-number N] [--simulation-type full|quick]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass


def _storage_slot(key_address: str, mapping_slot: int = 0) -> str:
    """keccak256(hexZeroPad(key, 32), hexZeroPad(slot, 32)) for mapping(address => uint256)."""
    from eth_utils import keccak
    key = key_address if key_address.startswith("0x") else "0x" + key_address
    key_hex = key[2:].zfill(64)
    slot_hex = hex(mapping_slot)[2:].zfill(64)
    raw = bytes.fromhex(key_hex) + bytes.fromhex(slot_hex)
    return "0x" + keccak(raw).hex()


def _allowance_slot(owner: str, spender: str, allowances_mapping_slot: int = 1) -> str:
    """Storage slot for ERC20 allowance[owner][spender] (OZ: _allowances at slot 1)."""
    from eth_utils import keccak
    owner = owner if owner.startswith("0x") else "0x" + owner
    spender = spender if spender.startswith("0x") else "0x" + spender
    inner = keccak(bytes.fromhex(owner[2:].zfill(64)) + bytes.fromhex(hex(allowances_mapping_slot)[2:].zfill(64)))
    outer = keccak(bytes.fromhex(spender[2:].zfill(64)) + inner)
    return "0x" + outer.hex()


def build_repay_state_overrides(user_address: str, amount: int = 1200 * 10**6) -> Dict[str, Any]:
    """
    State overrides for Aave V3 repay to succeed.

    Aave V3 repay flow: Pool.repay() -> checks reserve -> VariableDebtToken.burn() -> transferFrom(user).
    We override:
    - USDC: balance (slot 9, upgradeable proxy layout) + allowance to Pool (slot 10).
    - VariableDebtToken: _userState[user] = packed(uint128 scaledBalance, uint128 additionalData) at slot 0;
      and totalScaledSupply (slot 2) so reserve state is consistent.
    """
    pool_address = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
    variable_debt_usdc = "0x72E95b8931767C79bA4EeE721354d6E99a61D004"
    usdc_address = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

    # USDC (upgradeable proxy): balance at slot 9, _allowances at slot 10
    slot_balance = _storage_slot(user_address, 9)
    slot_allowance = _allowance_slot(user_address, pool_address, 10)

    usdc_balance_value = "0x" + format(amount * 10, "064x")
    usdc_allowance_value = "0x" + format(2**255, "064x")

    # VariableDebtToken (ScaledBalanceTokenBase): _userState at slot 0 is mapping -> struct (uint128 balance, uint128 additionalData)
    # Scaled balance: debt = scaled * index / 1e27; use scaled = amount * 0.9e27 so debt >= amount at typical index
    scaled_debt = int(amount * 0.9 * (10**27))
    packed_user_state = scaled_debt  # lower 128 bits = balance, upper 128 = 0
    slot_debt = _storage_slot(user_address, 0)
    debt_value = "0x" + format(packed_user_state, "064x")

    # Total scaled supply (slot 2) non-zero so reserve looks initialized
    total_supply_slot = "0x0000000000000000000000000000000000000000000000000000000000000002"
    total_supply_value = debt_value

    return {
        usdc_address: {
            "storage": {
                slot_balance: usdc_balance_value,
                slot_allowance: usdc_allowance_value,
            },
        },
        variable_debt_usdc: {
            "storage": {
                slot_debt: debt_value,
                total_supply_slot: total_supply_value,
            },
        },
    }


def load_payload_from_dataset(dataset_path: Path, intent_substring: str):
    """Load target_payload and user_context.from for the first row whose intent contains intent_substring."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        if intent_substring in (r.get("user_intent") or ""):
            tp = r.get("target_payload")
            ctx = r.get("user_context") or {}
            if tp and tp.get("to"):
                return {
                    "from": ctx.get("from_address"),
                    "to": tp["to"],
                    "value": str(tp.get("value", "0")),
                    "input": tp.get("data", "0x"),
                }
            break
    return None


def simulate(
    from_address: str,
    to_address: str,
    input_hex: str,
    network_id: str = "1",
    block_number: int = 16533883,
    gas: int = 8_000_000,
    value: str = "0",
    simulation_type: str = "quick",
    state_objects: Optional[Dict[str, Any]] = None,
):
    """POST to Tenderly simulate API. Returns (success: bool, error_message or None)."""
    access_key = os.getenv("TENDERLY_ACCESS_KEY")
    account_slug = os.getenv("TENDERLY_ACCOUNT_SLUG")
    project_slug = os.getenv("TENDERLY_PROJECT_SLUG")
    if not access_key or not account_slug or not project_slug:
        return False, "Missing env: TENDERLY_ACCESS_KEY, TENDERLY_ACCOUNT_SLUG, TENDERLY_PROJECT_SLUG"

    url = f"https://api.tenderly.co/api/v1/account/{account_slug}/project/{project_slug}/simulate"
    payload = {
        "network_id": network_id,
        "block_number": block_number,
        "from": from_address,
        "to": to_address,
        "input": input_hex if input_hex.startswith("0x") else "0x" + input_hex,
        "gas": gas,
        "value": int(value) if value else 0,
        "gas_price": 0,
        "simulation_type": simulation_type,
    }
    if state_objects:
        payload["state_objects"] = state_objects

    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Access-Key": access_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err = json.loads(body)
            msg = err.get("error", {}).get("message", body) or body
        except Exception:
            msg = body
        return False, msg or str(e)

    tx = data.get("transaction") or {}
    status = tx.get("status")
    if status is False or (isinstance(status, str) and str(status).lower() in ("0", "false", "reverted")):
        return False, tx.get("error_message") or "Transaction reverted"
    return True, None


def main():
    parser = argparse.ArgumentParser(description="Simulate one transaction via Tenderly; report if executable.")
    parser.add_argument("--block-number", type=int, default=16533883, help="Block number (use a recent block for more predictable state)")
    parser.add_argument("--simulation-type", choices=("full", "quick", "abi"), default="full", help="Tenderly simulation mode (default: full)")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to annotated_defi_dataset.json")
    args = parser.parse_args()

    dataset_path = args.dataset or (project_root / "data" / "datasets" / "annotated" / "annotated_defi_dataset.json")
    if not dataset_path.exists():
        print("Error: dataset not found:", dataset_path, file=sys.stderr)
        return 1

    payload = load_payload_from_dataset(dataset_path, "repay my Aave loan of 1200 USDC")
    if not payload or not payload.get("from"):
        print("Error: could not find payload for 'repay my Aave loan of 1200 USDC' in dataset", file=sys.stderr)
        return 1

    state_objects = build_repay_state_overrides(payload["from"])
    success, err = simulate(
        from_address=payload["from"],
        to_address=payload["to"],
        input_hex=payload["input"],
        value=payload["value"],
        block_number=args.block_number,
        simulation_type=args.simulation_type,
        state_objects=state_objects,
    )

    if success:
        print("Transaction is executable: yes")
        return 0
    print("Transaction is executable: no")
    if err:
        print("Error:", err)
        # Aave V3 revert reasons (from Errors.sol) for common codes
        aave_errors = {
            "34": "NO_DEBT / COLLATERAL_BALANCE_IS_ZERO – user has no collateral",
            "39": "NO_DEBT_OF_SELECTED_TYPE – user has no USDC debt (or no debt of this interest rate mode) to repay",
            "40": "NO_EXPLICIT_AMOUNT_TO_REPAY_ON_BEHALF",
            "42": "NO_OUTSTANDING_VARIABLE_DEBT – no variable-rate debt to repay",
            "43": "UNDERLYING_BALANCE_ZERO – sender has no USDC balance to repay with",
        }
        err_str = str(err).strip()
        if err_str in aave_errors:
            print("(Aave:", aave_errors[err_str] + ")")
    return 1


if __name__ == "__main__":
    sys.exit(main())
