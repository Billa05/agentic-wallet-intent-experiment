"""
Hybrid payload builder: convert LLM output (human-readable params) to ExecutablePayload.

LLMs return intent and parameters (e.g. amount_human, asset symbol); this module
performs Wei/base-unit conversion and transaction construction deterministically.
"""

from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, Optional

# Non-standard NFT contracts that need special handling
CRYPTOPUNKS_ADDRESS = "0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB".lower()


def _eth_to_wei(amount: str) -> str:
    d = Decimal(amount)
    wei = (d * Decimal("1000000000000000000")).to_integral_value(rounding=ROUND_DOWN)
    return str(int(wei))


def _token_to_base(amount: str, decimals: int) -> str:
    d = Decimal(amount)
    scale = Decimal(10) ** decimals
    base = (d * scale).to_integral_value(rounding=ROUND_DOWN)
    return str(int(base))


def _resolve_asset(symbol: str, token_registry: Dict[str, Any]) -> Optional[tuple]:
    """Return (address, decimals) for symbol or None."""
    erc20 = token_registry.get("erc20_tokens", {})
    info = erc20.get(symbol.upper())
    if not info:
        return None
    return (info["address"], info["decimals"])


def _get_protocol_contract(action: str, protocol_registry: Dict[str, Any]) -> Optional[tuple]:
    """Return (target_contract, function_name) for action."""
    protocols = protocol_registry.get("protocols", {})
    if action.startswith("aave_"):
        p = protocols.get("aave", {})
        pool = p.get("pool")
        fn = p.get("actions", {}).get(action, {}).get("function")
        return (pool, fn) if pool and fn else None
    if action.startswith("lido_"):
        p = protocols.get("lido", {})
        act_cfg = p.get("actions", {}).get(action, {})
        target_key = act_cfg.get("target", "steth")
        addr = p.get(target_key)
        fn = act_cfg.get("function")
        return (addr, fn) if addr and fn else None
    if action == "uniswap_swap":
        p = protocols.get("uniswap", {})
        return (p.get("router"), p.get("actions", {}).get("uniswap_swap", {}).get("function"))
    if action == "oneinch_swap":
        p = protocols.get("oneinch", {})
        return (p.get("aggregator"), p.get("actions", {}).get("oneinch_swap", {}).get("function"))
    if action.startswith("curve_"):
        p = protocols.get("curve", {})
        pool = p.get("pool_3crv")
        fn = p.get("actions", {}).get(action, {}).get("function")
        return (pool, fn) if pool and fn else None
    return None


def convert_human_to_payload(
    llm_payload: Dict[str, Any],
    token_registry: Dict[str, Any],
    protocol_registry: Dict[str, Any],
    ens_registry: Dict[str, str],
    chain_id: int = 1,
    from_address: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert LLM output (possibly human-readable) to final ExecutablePayload dict.
    Resolves ENS, asset symbols, and converts amount_human to Wei/base units.
    """
    action = llm_payload.get("action")
    if not action:
        return None
    args = llm_payload.get("arguments") or {}

    # Transfer actions: already expect value in Wei or we convert from amount_human
    if action == "transfer_native":
        value = args.get("value")
        if not value and "amount_human" in args:
            value = _eth_to_wei(args["amount_human"])
        if not value:
            return None
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": None,
            "function_name": None,
            "arguments": {
                "to": _resolve_ens(args.get("to"), ens_registry) or args.get("to"),
                "value": str(value),
                "human_readable_amount": args.get("human_readable_amount") or (f"{args.get('amount_human')} ETH" if args.get("amount_human") else ""),
            },
        }

    if action == "transfer_erc20":
        target = args.get("target_contract")
        if not target and args.get("asset"):
            res = _resolve_asset(args["asset"], token_registry)
            target = res[0] if res else None
        value = args.get("value")
        if not value and "amount_human" in args:
            res = _resolve_asset(args.get("asset", "USDC"), token_registry)
            dec = res[1] if res else 18
            value = _token_to_base(args["amount_human"], dec)
        if not target or not value:
            return None
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target,
            "function_name": "transfer",
            "arguments": {
                "to": _resolve_ens(args.get("to"), ens_registry) or args.get("to"),
                "value": str(value),
                "human_readable_amount": args.get("human_readable_amount") or (f"{args.get('amount_human')} {args.get('asset')}".strip() if (args.get("amount_human") or args.get("asset")) else ""),
            },
        }

    if action == "transfer_erc721":
        target = args.get("target_contract")
        if not target and (args.get("collection") or args.get("asset")):
            coll_name = (args.get("collection") or args.get("asset") or "").strip()
            if coll_name:
                colls = token_registry.get("erc721_collections", {})
                for key, info in colls.items():
                    if key.lower() == coll_name.lower() or (info.get("name") or key).lower() == coll_name.lower():
                        target = info.get("address")
                        break
        if not target:
            return None
        
        # CryptoPunks uses non-standard interface: transferPunk(to, punkIndex)
        if target.lower() == CRYPTOPUNKS_ADDRESS:
            function_name = "transferPunk"
        else:
            function_name = "transferFrom"
        
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target,
            "function_name": function_name,
            "arguments": {
                "to": _resolve_ens(args.get("to"), ens_registry) or args.get("to"),
                "tokenId": args.get("tokenId"),
                "human_readable_amount": args.get("human_readable_amount") or (f"Token #{args.get('tokenId')}" if args.get("tokenId") is not None else ""),
            },
        }

    # DeFi: get contract from registry, convert amounts
    proto = _get_protocol_contract(action, protocol_registry)
    if not proto:
        return None
    target_contract, function_name = proto

    def resolve_addr(a):
        return _resolve_ens(a, ens_registry) if isinstance(a, str) and not a.startswith("0x") else a

    # AAVE supply/withdraw/borrow/repay (amounts in base units; supply/borrow need referralCode; borrow/repay need interestRateMode)
    if action.startswith("aave_"):
        asset_sym = args.get("asset", "USDC")
        res = _resolve_asset(asset_sym, token_registry)
        if not res:
            return None
        asset_addr, decimals = res
        amount_human = args.get("amount_human") or args.get("amount")
        amount = _token_to_base(str(amount_human or "0"), decimals)
        on_behalf = resolve_addr(args.get("onBehalfOf")) or from_address or (list(ens_registry.values())[0] if ens_registry else None)
        if not on_behalf:
            return None
        out = {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "asset": asset_addr,
                "amount": amount,
                "onBehalfOf": on_behalf,
                "human_readable_amount": args.get("human_readable_amount", f"{args.get('amount_human', amount_human)} {asset_sym}"),
            },
        }
        if action == "aave_supply":
            out["arguments"]["referralCode"] = int(args.get("referralCode", 0))
        if action == "aave_withdraw":
            out["arguments"]["to"] = resolve_addr(args.get("to")) or on_behalf
        if action == "aave_borrow":
            out["arguments"]["referralCode"] = int(args.get("referralCode", 0))
            out["arguments"]["interestRateMode"] = int(args.get("interestRateMode", 2))  # 2 = variable
        if action == "aave_repay":
            out["arguments"]["interestRateMode"] = int(args.get("interestRateMode", 2))
        return out

    # Lido stake: value in wei (submit is payable; amount goes in tx.value)
    if action == "lido_stake":
        amount_human = args.get("amount_human") or args.get("value")
        value = _eth_to_wei(str(amount_human or "0"))
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "value": value,
                "human_readable_amount": args.get("human_readable_amount", f"{args.get('amount_human', amount_human)} ETH"),
            },
        }

    # Lido unstake: requestWithdrawals(uint256[] _amounts, address _owner)
    if action == "lido_unstake":
        amount_human = args.get("amount_human") or args.get("amount")
        amount_base = _token_to_base(str(amount_human or "0"), 18)
        owner = resolve_addr(args.get("onBehalfOf")) or args.get("_owner") or from_address
        if not owner:
            return None
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "_amounts": [amount_base],
                "_owner": owner,
                "human_readable_amount": args.get("human_readable_amount", f"{args.get('amount_human', amount_human)} stETH"),
            },
        }

    # Uniswap / 1inch swap: amountIn/amountOutMin in correct token decimals; to required (user address); deadline set in encoder
    if action in ("uniswap_swap", "oneinch_swap"):
        path_raw = args.get("path") or []
        path = []
        for p in path_raw:
            if isinstance(p, str) and not p.startswith("0x") and p.upper() != "ETH":
                r = _resolve_asset(p, token_registry)
                path.append(r[0] if r else p)
            elif isinstance(p, str) and p.upper() == "ETH":
                weth = _resolve_asset("WETH", token_registry)
                path.append(weth[0] if weth else p)
            else:
                path.append(p)
        amount_human = args.get("amount_human")
        amount_in = args.get("amountIn")
        if amount_human and (not amount_in or str(amount_in) == "0"):
            if path and path[0].startswith("0x"):
                # First token is address: get decimals from registry (WETH=18)
                first_addr = path[0].lower()
                dec = 18
                for _sym, info in (token_registry.get("erc20_tokens") or {}).items():
                    if (info.get("address") or "").lower() == first_addr:
                        dec = info.get("decimals", 18)
                        break
                amount_in = _token_to_base(str(amount_human), dec)
            else:
                res = _resolve_asset(args.get("asset_in", "USDC"), token_registry)
                dec = res[1] if res else 6
                amount_in = _token_to_base(str(amount_human), dec)
        amount_out_min = args.get("amountOutMin", "0")
        if path and str(amount_out_min) != "0" and amount_out_min == args.get("amountOutMin"):
            last_addr = path[-1] if isinstance(path[-1], str) else None
            if last_addr and last_addr.startswith("0x"):
                dec_out = 18
                for _sym, info in (token_registry.get("erc20_tokens") or {}).items():
                    if (info.get("address") or "").lower() == last_addr.lower():
                        dec_out = info.get("decimals", 18)
                        break
                amount_out_min = _token_to_base(str(amount_out_min), dec_out)
        to_addr = resolve_addr(args.get("to")) or args.get("to") or from_address
        path_str = " -> ".join(str(p) for p in path_raw) if path_raw else "tokens"
        swap_desc = args.get("human_readable_amount") or (f"Swap {args.get('amount_human', '')} ({path_str})" if args.get("amount_human") else "")
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "amountIn": str(amount_in or "0"),
                "amountOutMin": str(amount_out_min),
                "path": path or [],
                "to": to_addr,
                "human_readable_amount": swap_desc,
            },
        }

    # Curve 3pool: add_liquidity(uint256[3] amounts, uint256 min_mint_amount). Pool order: DAI=0, USDC=1, USDT=2
    if action == "curve_add_liquidity":
        amount_human = args.get("amount_human") or args.get("amount")
        asset_sym = (args.get("asset") or "USDC").upper()
        res = _resolve_asset(asset_sym, token_registry)
        dec = res[1] if res else 6
        amt = _token_to_base(str(amount_human or "0"), dec)
        idx = {"DAI": 0, "USDC": 1, "USDT": 2}.get(asset_sym, 1)
        amounts = ["0", "0", "0"]
        amounts[idx] = amt
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "pool": target_contract,
                "amounts": amounts,
                "min_mint_amount": str(args.get("min_mint_amount", "0")),
                "human_readable_amount": args.get("human_readable_amount") or (f"{args.get('amount_human')} {asset_sym}".strip() if args.get("amount_human") else ""),
            },
        }
    # remove_liquidity(uint256 amount, uint256[3] min_amounts): amount is LP token (18 decimals)
    if action == "curve_remove_liquidity":
        amount_human = args.get("amount_human") or args.get("amount")
        amt = _token_to_base(str(amount_human or "0"), 18)
        min_amounts = args.get("min_amounts", ["0", "0", "0"])
        if isinstance(min_amounts, list) and len(min_amounts) < 3:
            min_amounts = (min_amounts + ["0", "0", "0"])[:3]
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "pool": target_contract,
                "amount": amt,
                "min_amounts": min_amounts if isinstance(min_amounts, list) else ["0", "0", "0"],
                "human_readable_amount": args.get("human_readable_amount") or (f"{args.get('amount_human')} LP" if args.get("amount_human") else ""),
            },
        }

    return None


def _resolve_ens(name_or_addr: Optional[str], ens_registry: Dict[str, str]) -> Optional[str]:
    if not name_or_addr:
        return None
    s = name_or_addr.strip().lower()
    if s.startswith("0x"):
        return name_or_addr
    if not s.endswith(".eth"):
        s = s + ".eth"
    return ens_registry.get(s)
