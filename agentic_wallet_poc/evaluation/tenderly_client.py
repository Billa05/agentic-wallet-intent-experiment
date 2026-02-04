"""
Tenderly Simulation API client.
Simulates a raw transaction (to, value, data) and returns success/failure.
"""

import os
from typing import Dict, Any, Optional

# Optional: use requests if available, else urllib
try:
    import urllib.request
    import urllib.error
    import json as json_module
    _HAS_REQUESTS = False
except ImportError:
    _HAS_REQUESTS = False


def tenderly_simulate(
    from_address: str,
    to_address: str,
    value: str,
    data: str,
    network_id: str = "1",
    block_number: str = "latest",
    gas: int = 8_000_000,
    access_key: Optional[str] = None,
    account_slug: Optional[str] = None,
    project_slug: Optional[str] = None,
    state_objects: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Simulate a transaction via Tenderly API.
    See: https://docs.tenderly.co/simulations/single-simulations
    State overrides: https://docs.tenderly.co/simulations/state-overrides

    Args:
        from_address: Sender address (for simulation can be any address).
        to_address: Recipient / contract address.
        value: Wei as decimal string (e.g. "0" or "1000000000000000000").
        data: Hex-encoded input data (e.g. "0x" or "0xa9059cbb...").
        network_id: Chain ID as string (e.g. "1" for mainnet).
        block_number: "latest" or block number.
        gas: Gas limit for simulation.
        access_key: TENDERLY_ACCESS_KEY (or env TENDERLY_ACCESS_KEY).
        account_slug: TENDERLY_ACCOUNT_SLUG (or env).
        project_slug: TENDERLY_PROJECT_SLUG (or env).
        state_objects: Optional state overrides: { "0xaddr": { "balance": "0x...", "storage": { "0xslot": "0xvalue" } } }.

    Returns:
        Dict with:
          - "success": bool (True if simulation succeeded)
          - "error": optional str (message if failed)
          - "response": optional raw API response for debugging
    """
    access_key = access_key or os.getenv("TENDERLY_ACCESS_KEY")
    account_slug = account_slug or os.getenv("TENDERLY_ACCOUNT_SLUG")
    project_slug = project_slug or os.getenv("TENDERLY_PROJECT_SLUG")

    if not access_key or not account_slug or not project_slug:
        return {
            "success": False,
            "error": "Missing Tenderly config: set TENDERLY_ACCESS_KEY, TENDERLY_ACCOUNT_SLUG, TENDERLY_PROJECT_SLUG",
        }

    url = f"https://api.tenderly.co/api/v1/account/{account_slug}/project/{project_slug}/simulate"
    # Value: API expects number for value (wei). Keep as integer to avoid float.
    value_int = int(value) if value else 0
    # Tenderly API expects block_number as integer; "latest" is not accepted (causes "Bad request input parameters")
    if block_number in ("latest", "pending", "safe", "finalized", "earliest"):
        block_num = 21_000_000  # fallback for "latest" (mainnet); simulation uses recent state
    else:
        block_num = int(block_number)
    payload = {
        "network_id": network_id,
        "block_number": block_num,
        "from": from_address,
        "to": to_address,
        "gas": gas,
        "gas_price": 0,
        "value": value_int,
        "input": data if data.startswith("0x") else "0x" + data,
        "simulation_type": "quick",
    }
    if state_objects:
        payload["state_objects"] = state_objects

    try:
        req = urllib.request.Request(
            url,
            data=json_module.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Access-Key": access_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json_module.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err_json = json_module.loads(body)
            err_msg = err_json.get("error", {}).get("message", body) or body or str(e)
            # Keep raw body in response for debugging "Bad request" etc.
            try:
                err_response = json_module.loads(body)
            except Exception:
                err_response = {"raw": body}
        except Exception:
            err_msg = body or str(e)
            err_response = {"raw": body}
        return {"success": False, "error": err_msg, "response": err_response}
    except Exception as e:
        return {"success": False, "error": str(e), "response": None}

    # Tenderly returns simulation result; transaction failure (revert) may still be 200 OK
    # Check for simulation success (no revert)
    tx_status = (response_data or {}).get("transaction", {}).get("status")
    if tx_status is False or (isinstance(tx_status, str) and tx_status.lower() in ("0", "false", "reverted")):
        error_info = (response_data or {}).get("transaction", {}).get("error_message") or (response_data or {}).get("simulation", {}).get("error")
        return {
            "success": False,
            "error": error_info or "Transaction reverted in simulation",
            "response": response_data,
        }
    return {"success": True, "error": None, "response": response_data}
