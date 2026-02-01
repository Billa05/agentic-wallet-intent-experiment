"""
Hybrid payload builder: convert LLM output (human-readable params) to ExecutablePayload.

LLMs return intent and parameters (e.g. amount_human, asset symbol); this module
performs Wei/base-unit conversion and transaction construction deterministically.
"""

from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, Optional


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
                "human_readable_amount": args.get("human_readable_amount", ""),
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
                "human_readable_amount": args.get("human_readable_amount", ""),
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
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target,
            "function_name": "transferFrom",
            "arguments": {
                "to": _resolve_ens(args.get("to"), ens_registry) or args.get("to"),
                "tokenId": args.get("tokenId"),
                "human_readable_amount": args.get("human_readable_amount", ""),
            },
        }

    # DeFi: get contract from registry, convert amounts
    proto = _get_protocol_contract(action, protocol_registry)
    if not proto:
        return None
    target_contract, function_name = proto

    def resolve_addr(a):
        return _resolve_ens(a, ens_registry) if isinstance(a, str) and not a.startswith("0x") else a

    # AAVE supply/withdraw/borrow/repay
    if action.startswith("aave_"):
        asset_sym = args.get("asset", "USDC")
        res = _resolve_asset(asset_sym, token_registry)
        if not res:
            return None
        asset_addr, decimals = res
        amount_human = args.get("amount_human") or args.get("amount")
        if isinstance(amount_human, str) and not amount_human.isdigit():
            amount = _token_to_base(amount_human, decimals)
        else:
            amount = str(amount_human) if amount_human is not None else "0"
        on_behalf = resolve_addr(args.get("onBehalfOf")) or list(ens_registry.values())[0] if ens_registry else None
        out = {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "asset": asset_addr,
                "amount": amount,
                "onBehalfOf": on_behalf,
                "human_readable_amount": args.get("human_readable_amount", f"{args.get('amount_human', amount)} {asset_sym}"),
            },
        }
        if action == "aave_withdraw":
            out["arguments"]["to"] = resolve_addr(args.get("to")) or out["arguments"]["onBehalfOf"]
        return out

    # Lido stake
    if action == "lido_stake":
        amount_human = args.get("amount_human") or args.get("value")
        if isinstance(amount_human, str) and not amount_human.isdigit():
            value = _eth_to_wei(amount_human)
        else:
            value = str(amount_human) if amount_human else "0"
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "value": value,
                "human_readable_amount": args.get("human_readable_amount", f"{args.get('amount_human', '0')} ETH"),
            },
        }

    # Lido unstake: amount_human is in stETH (18 decimals)
    if action == "lido_unstake":
        amount_human = args.get("amount_human") or args.get("amount")
        if isinstance(amount_human, str) and not amount_human.replace(".", "").replace("-", "").isdigit():
            value = _token_to_base(amount_human, 18)
        else:
            value = _token_to_base(str(amount_human or "0"), 18)
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "amount": value,
                "human_readable_amount": args.get("human_readable_amount", f"{args.get('amount_human', '0')} stETH"),
            },
        }

    # Uniswap / 1inch swap: require amountIn, path, to; amountOutMin can be "0"
    if action in ("uniswap_swap", "oneinch_swap"):
        amount_in = args.get("amountIn")
        amount_human = args.get("amount_human")
        if amount_human and (not amount_in or not str(amount_in).isdigit()):
            path = args.get("path") or []
            if path and isinstance(path[0], str) and path[0].startswith("0x"):
                amount_in = _eth_to_wei(amount_human)
            else:
                res = _resolve_asset(args.get("asset_in", "USDC"), token_registry)
                dec = res[1] if res else 6
                amount_in = _token_to_base(amount_human, dec)
        path = args.get("path")
        if isinstance(path, list) and path:
            resolved = []
            for p in path:
                if isinstance(p, str) and not p.startswith("0x"):
                    r = _resolve_asset(p, token_registry)
                    resolved.append(r[0] if r else p)
                else:
                    resolved.append(p)
            path = resolved
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "amountIn": str(amount_in or "0"),
                "amountOutMin": str(args.get("amountOutMin", "0")),
                "path": path or [],
                "to": resolve_addr(args.get("to")) or args.get("to"),
                "human_readable_amount": args.get("human_readable_amount", ""),
            },
        }

    # Curve add/remove liquidity
    if action == "curve_add_liquidity":
        amount_human = args.get("amount_human") or args.get("amount")
        res = _resolve_asset(args.get("asset", "USDC"), token_registry)
        dec = res[1] if res else 6
        amt = _token_to_base(str(amount_human or "0"), dec)
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "pool": target_contract,
                "amounts": [amt, "0", "0"],
                "min_mint_amount": str(args.get("min_mint_amount", "0")),
                "human_readable_amount": args.get("human_readable_amount", ""),
            },
        }
    if action == "curve_remove_liquidity":
        amount_human = args.get("amount_human") or args.get("amount")
        amt = str(amount_human) if amount_human else "0"
        if isinstance(amount_human, str) and not amount_human.isdigit():
            amt = _token_to_base(amount_human, 18)
        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": {
                "pool": target_contract,
                "amount": amt,
                "min_amounts": args.get("min_amounts", ["0", "0", "0"]),
                "human_readable_amount": args.get("human_readable_amount", ""),
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
