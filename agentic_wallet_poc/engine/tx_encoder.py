"""
Convert ExecutablePayload (action + arguments) to raw transaction (to, value, data)
for simulation or execution. Used by Tenderly validation and live execution.
"""

import time
from typing import Dict, Any, Optional, Tuple, List

try:
    from eth_abi import encode
except ImportError:
    encode = None

# ERC20 / ERC721
TRANSFER_SELECTOR = "0xa9059cbb"
TRANSFER_FROM_SELECTOR = "0x23b872dd"

# CryptoPunks (non-standard - uses transferPunk instead of transferFrom)
CRYPTOPUNKS_ADDRESS = "0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB".lower()
CRYPTOPUNKS_TRANSFER_SELECTOR = "0x8b72a2ec"  # transferPunk(address,uint256)
# Aave V3 Pool (first 4 bytes of keccak256(signature))
AAVE_SUPPLY_SELECTOR = "0x617ba037"      # supply(address,uint256,address,uint16)
AAVE_WITHDRAW_SELECTOR = "0x69328dec"   # withdraw(address,uint256,address)
AAVE_BORROW_SELECTOR = "0xa415bcad"      # borrow(address,uint256,uint256,uint16,address)
AAVE_REPAY_SELECTOR = "0x573ade81"      # repay(address,uint256,uint256,address)
# Lido
LIDO_SUBMIT_SELECTOR = "0xa1903eab"     # submit(address)
LIDO_REQUEST_WITHDRAWALS_SELECTOR = "0x7b0a5ee8"  # requestWithdrawals(uint256[],address)
# Uniswap V2 Router
UNISWAP_SWAP_EXACT_TOKENS_SELECTOR = "0x38ed1739"  # swapExactTokensForTokens(uint256,uint256,address[],address,uint256)
# Curve 3pool
CURVE_ADD_LIQUIDITY_SELECTOR = "0x4515cef3"   # add_liquidity(uint256[3],uint256)
CURVE_REMOVE_LIQUIDITY_SELECTOR = "0xecb586a5"  # remove_liquidity(uint256,uint256[3])


def _addr(a: str) -> str:
    return a if a.startswith("0x") else "0x" + a


def _encode_erc20_transfer(to_address: str, value_wei: str) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    encoded = encode(["address", "uint256"], [_addr(to_address), int(value_wei)])
    return TRANSFER_SELECTOR + encoded.hex()


def _encode_erc721_transfer_from(from_address: str, to_address: str, token_id: int) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    encoded = encode(["address", "address", "uint256"], [_addr(from_address), _addr(to_address), token_id])
    return TRANSFER_FROM_SELECTOR + encoded.hex()


def _encode_cryptopunks_transfer(to_address: str, punk_index: int) -> str:
    """Encode CryptoPunks transferPunk(address to, uint punkIndex)."""
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    encoded = encode(["address", "uint256"], [_addr(to_address), punk_index])
    return CRYPTOPUNKS_TRANSFER_SELECTOR + encoded.hex()


def _encode_aave_supply(asset: str, amount: str, on_behalf_of: str, referral_code: int = 0) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    data = encode(
        ["address", "uint256", "address", "uint16"],
        [_addr(asset), int(amount), _addr(on_behalf_of), referral_code],
    )
    return AAVE_SUPPLY_SELECTOR + data.hex()


def _encode_aave_withdraw(asset: str, amount: str, to: str) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    data = encode(["address", "uint256", "address"], [_addr(asset), int(amount), _addr(to)])
    return AAVE_WITHDRAW_SELECTOR + data.hex()


def _encode_aave_borrow(asset: str, amount: str, interest_rate_mode: int, referral_code: int, on_behalf_of: str) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    data = encode(
        ["address", "uint256", "uint256", "uint16", "address"],
        [_addr(asset), int(amount), interest_rate_mode, referral_code, _addr(on_behalf_of)],
    )
    return AAVE_BORROW_SELECTOR + data.hex()


def _encode_aave_repay(asset: str, amount: str, interest_rate_mode: int, on_behalf_of: str) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    data = encode(
        ["address", "uint256", "uint256", "address"],
        [_addr(asset), int(amount), interest_rate_mode, _addr(on_behalf_of)],
    )
    return AAVE_REPAY_SELECTOR + data.hex()


def _encode_lido_submit(referral: str = "0x0000000000000000000000000000000000000000") -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    data = encode(["address"], [_addr(referral)])
    return LIDO_SUBMIT_SELECTOR + data.hex()


def _encode_lido_request_withdrawals(amounts: List[str], owner: str) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    amounts_int = [int(a) for a in amounts]
    data = encode(["uint256[]", "address"], [amounts_int, _addr(owner)])
    return LIDO_REQUEST_WITHDRAWALS_SELECTOR + data.hex()


def _encode_uniswap_swap_exact_tokens_for_tokens(
    amount_in: str, amount_out_min: str, path: List[str], to: str, deadline: int
) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    path_addrs = [_addr(p) for p in path]
    data = encode(
        ["uint256", "uint256", "address[]", "address", "uint256"],
        [int(amount_in), int(amount_out_min), path_addrs, _addr(to), deadline],
    )
    return UNISWAP_SWAP_EXACT_TOKENS_SELECTOR + data.hex()


def _encode_curve_add_liquidity(amounts: List[str], min_mint_amount: str) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    amounts_int = [int(a) for a in amounts]
    data = encode(["uint256[3]", "uint256"], [amounts_int, int(min_mint_amount)])
    return CURVE_ADD_LIQUIDITY_SELECTOR + data.hex()


def _encode_curve_remove_liquidity(amount: str, min_amounts: List[str]) -> str:
    if encode is None:
        raise ImportError("eth_abi is required for tx encoding; pip install eth-abi")
    min_int = [int(m) for m in min_amounts]
    data = encode(["uint256", "uint256[3]"], [int(amount), min_int])
    return CURVE_REMOVE_LIQUIDITY_SELECTOR + data.hex()


def payload_to_raw_tx(
    payload: Dict[str, Any],
    from_address: str,
) -> Optional[Dict[str, Any]]:
    """
    Convert our ExecutablePayload dict to raw tx shape: { to, value, data }.
    value and data are strings (decimal value, 0x-prefixed data).
    Returns None if the action is not supported for encoding yet.
    """
    if not payload:
        return None
    action = payload.get("action")
    args = payload.get("arguments") or {}
    target_contract = payload.get("target_contract")
    chain_id = payload.get("chain_id", 1)

    # transfer_native: send ETH to recipient
    if action == "transfer_native":
        to = args.get("to")
        value = args.get("value", "0")
        if not to:
            return None
        return {
            "chain_id": chain_id,
            "to": to,
            "value": str(value),
            "data": "0x",
        }

    # transfer_erc20: call token.transfer(to, value)
    if action == "transfer_erc20":
        to_recipient = args.get("to")
        value = args.get("value", "0")
        if not target_contract or not to_recipient:
            return None
        data = _encode_erc20_transfer(to_recipient, value)
        return {
            "chain_id": chain_id,
            "to": target_contract,
            "value": "0",
            "data": data,
        }

    # transfer_erc721: call token.transferFrom(from, to, tokenId)
    # Special case: CryptoPunks uses transferPunk(to, punkIndex) instead
    if action == "transfer_erc721":
        to_recipient = args.get("to")
        token_id = args.get("tokenId")
        if not target_contract or not to_recipient or token_id is None:
            return None
        if not from_address:
            return None
        
        # CryptoPunks: use transferPunk(address to, uint punkIndex)
        if target_contract.lower() == CRYPTOPUNKS_ADDRESS:
            data = _encode_cryptopunks_transfer(to_recipient, int(token_id))
        else:
            # Standard ERC721: transferFrom(from, to, tokenId)
            data = _encode_erc721_transfer_from(from_address, to_recipient, int(token_id))
        
        return {
            "chain_id": chain_id,
            "to": target_contract,
            "value": "0",
            "data": data,
        }

    # Aave V3
    if action == "aave_supply":
        asset = args.get("asset")
        amount = args.get("amount")
        on_behalf = args.get("onBehalfOf")
        ref = int(args.get("referralCode", 0))
        if not target_contract or not asset or amount is None or not on_behalf:
            return None
        data = _encode_aave_supply(asset, amount, on_behalf, ref)
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    if action == "aave_withdraw":
        asset = args.get("asset")
        amount = args.get("amount")
        to_addr = args.get("to")
        if not target_contract or not asset or amount is None or not to_addr:
            return None
        data = _encode_aave_withdraw(asset, amount, to_addr)
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    if action == "aave_borrow":
        asset = args.get("asset")
        amount = args.get("amount")
        on_behalf = args.get("onBehalfOf")
        rate_mode = int(args.get("interestRateMode", 2))
        ref = int(args.get("referralCode", 0))
        if not target_contract or not asset or amount is None or not on_behalf:
            return None
        data = _encode_aave_borrow(asset, amount, rate_mode, ref, on_behalf)
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    if action == "aave_repay":
        asset = args.get("asset")
        amount = args.get("amount")
        on_behalf = args.get("onBehalfOf")
        rate_mode = int(args.get("interestRateMode", 2))
        if not target_contract or not asset or amount is None or not on_behalf:
            return None
        data = _encode_aave_repay(asset, amount, rate_mode, on_behalf)
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    # Lido: stake (payable submit) and unstake (requestWithdrawals)
    if action == "lido_stake":
        value_wei = args.get("value", "0")
        if not target_contract:
            return None
        data = _encode_lido_submit()
        return {"chain_id": chain_id, "to": target_contract, "value": str(value_wei), "data": data}

    if action == "lido_unstake":
        amounts = args.get("_amounts") or ([args.get("amount")] if args.get("amount") is not None else None)
        owner = args.get("_owner")
        if not target_contract or not amounts or not owner:
            return None
        data = _encode_lido_request_withdrawals(amounts, owner)
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    # Uniswap V2: swapExactTokensForTokens(amountIn, amountOutMin, path, to, deadline)
    if action == "uniswap_swap":
        amount_in = args.get("amountIn")
        amount_out_min = args.get("amountOutMin", "0")
        path = args.get("path") or []
        to_addr = args.get("to") or from_address
        deadline = int(args.get("deadline", 0)) or (int(time.time()) + 1200)  # now + 20 min
        if not target_contract or amount_in is None or len(path) < 2 or not to_addr:
            return None
        data = _encode_uniswap_swap_exact_tokens_for_tokens(
            amount_in, amount_out_min, path, to_addr, deadline
        )
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    # Curve 3pool
    if action == "curve_add_liquidity":
        amounts = args.get("amounts") or ["0", "0", "0"]
        min_mint = args.get("min_mint_amount", "0")
        if not target_contract or len(amounts) < 3:
            return None
        data = _encode_curve_add_liquidity(amounts[:3], min_mint)
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    if action == "curve_remove_liquidity":
        amount = args.get("amount")
        min_amounts = args.get("min_amounts") or ["0", "0", "0"]
        if not target_contract or amount is None:
            return None
        data = _encode_curve_remove_liquidity(amount, min_amounts[:3])
        return {"chain_id": chain_id, "to": target_contract, "value": "0", "data": data}

    # 1inch: requires API to get tx data; no direct ABI encoding
    return None


def build_metadata(
    payload: Dict[str, Any],
    token_registry: Dict[str, Any],
    ens_registry: Dict[str, str],
) -> Dict[str, Any]:
    """
    Build action-specific metadata for display and validation.
    token_registry/ens_registry used to resolve symbols and ENS names.
    """
    if not payload:
        return {}
    action = payload.get("action")
    args = payload.get("arguments") or {}
    target_contract = payload.get("target_contract")
    meta = {"action": action}

    # Resolve recipient ENS -> address for metadata
    def resolve_ens(name_or_addr: Optional[str]) -> Optional[str]:
        if not name_or_addr or not isinstance(name_or_addr, str):
            return None
        s = name_or_addr.strip().lower()
        if s.startswith("0x"):
            return name_or_addr
        if not s.endswith(".eth"):
            s = s + ".eth"
        return ens_registry.get(s)

    # Resolve token/contract address -> symbol from registry
    def symbol_for_address(addr: Optional[str]) -> Optional[str]:
        if not addr:
            return None
        addr_lower = addr.lower()
        for symbol, info in (token_registry.get("erc20_tokens") or {}).items():
            if (info.get("address") or "").lower() == addr_lower:
                return symbol
        return None

    # Reverse lookup: address -> ENS name from registry
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
