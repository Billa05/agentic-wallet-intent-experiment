"""
Generic resolver functions for the playbook engine.

Each resolver is a small, stateless function that transforms a raw LLM output
value into a resolved on-chain value (address, base-unit amount, etc.).
Resolvers are registered in RESOLVER_REGISTRY and dispatched by name from
playbook JSON payload_args entries.
"""

import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional

from engine.ens_resolver import ENSResolver


UINT256_MAX = str(2**256 - 1)
_STABLE_TOKENS = {"USDC", "USDT", "DAI"}


# ─────────────────────────────────────────────────────────────────────
# Resolve context — carries all registries + accumulates resolved values
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ResolveContext:
    """Immutable context passed to all resolvers."""
    token_resolver: Any  # TokenResolver instance (or None)
    ens_resolver: ENSResolver
    from_address: Optional[str]
    chain_id: int
    action: str
    raw_args: Dict[str, Any]
    resolved: Dict[str, Any] = field(default_factory=dict)

    # Caches decimals for resolved tokens so dependent args can look them up
    _decimals_cache: Dict[str, int] = field(default_factory=dict)

    def get_decimals_for(self, key: str) -> int:
        """Get decimals for a previously resolved token address or symbol."""
        if key in self._decimals_cache:
            return self._decimals_cache[key]
        return 18  # fallback


# ─────────────────────────────────────────────────────────────────────
# Individual resolver functions
# ─────────────────────────────────────────────────────────────────────

def resolve_token_address(value: str, ctx: ResolveContext, **kwargs) -> Optional[str]:
    """Symbol -> ERC-20 contract address. Handles ETH -> WETH alias."""
    if not value:
        return None
    s = value.strip()
    if s.startswith("0x") and len(s) == 42:
        # Already an address — look up metadata via resolver
        if ctx.token_resolver:
            info = ctx.token_resolver.resolve_by_address(s)
            if info:
                ctx._decimals_cache[s] = info["decimals"]
                return info["address"]
        return s
    # Handle ETH alias
    if s.upper() == "ETH":
        s = kwargs.get("eth_alias", "WETH")
    if ctx.token_resolver:
        info = ctx.token_resolver.resolve_erc20(s)
        if info:
            addr = info["address"]
            ctx._decimals_cache[addr] = info["decimals"]
            ctx._decimals_cache[s.upper()] = info["decimals"]
            return addr
    return None


def resolve_collection_address(value: str, ctx: ResolveContext, **kwargs) -> Optional[str]:
    """Collection name -> ERC-721 contract address."""
    if not value:
        return None
    s = value.strip()
    if s.startswith("0x") and len(s) == 42:
        return s
    if ctx.token_resolver:
        info = ctx.token_resolver.resolve_collection(s)
        if info:
            return info.get("address")
    return None


def resolve_amount(value: str, ctx: ResolveContext, **kwargs) -> Optional[str]:
    """Human-readable amount -> base units string.
    decimals_from: "$asset" means look up from resolved token, "$native" means 18,
    or an integer literal.
    """
    if value is None:
        return None
    decimals = _resolve_decimals(kwargs.get("decimals_from"), ctx)
    d = Decimal(str(value))
    scale = Decimal(10) ** decimals
    base = (d * scale).to_integral_value(rounding=ROUND_DOWN)
    return str(int(base))


def resolve_amount_or_max(value: str, ctx: ResolveContext, **kwargs) -> Optional[str]:
    """Like resolve_amount but 'max'/'all' -> uint256.max."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("max", "all", "full", "maximum", "everything", "entire"):
        return UINT256_MAX
    return resolve_amount(value, ctx, **kwargs)


def resolve_ens_or_hex(value: Optional[str], ctx: ResolveContext, **kwargs) -> Optional[str]:
    """ENS name -> address, or pass through hex address."""
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    if s.startswith("0x") and len(s) == 42:
        return s
    # Reject incomplete hex
    if s.startswith("0x"):
        return None
    # ENS lookup
    ens_key = s.lower()
    if not ens_key.endswith(".eth"):
        ens_key = ens_key + ".eth"
    return ctx.ens_resolver.resolve(ens_key)


def resolve_fee_tier(value: Any, ctx: ResolveContext, **kwargs) -> int:
    """Resolve Uniswap V3 fee tier. Uses LLM value if provided, else heuristic."""
    if value is not None and str(value) != "":
        try:
            return int(value)
        except (ValueError, TypeError):
            pass
    # Heuristic from playbook config
    heuristic = kwargs.get("fee_heuristic", {})
    stable_tokens = set(heuristic.get("stable_tokens", _STABLE_TOKENS))
    sym_in = (ctx.raw_args.get(kwargs.get("input_symbol_field", "asset_in")) or "").upper()
    sym_out = (ctx.raw_args.get(kwargs.get("output_symbol_field", "asset_out")) or "").upper()
    if sym_in in stable_tokens and sym_out in stable_tokens:
        return heuristic.get("stable_stable_fee", 100)
    return heuristic.get("default_fee", 3000)


def resolve_deadline(value: Any, ctx: ResolveContext, **kwargs) -> int:
    """Current timestamp + buffer seconds."""
    buffer = kwargs.get("buffer_seconds", 1200)
    return int(time.time()) + buffer


def resolve_smart_amount(value: Any, ctx: ResolveContext, **kwargs) -> str:
    """Try to parse as base-unit integer first; fall back to token_to_base."""
    if value is None or str(value).strip() == "":
        return kwargs.get("fallback", "0")
    s = str(value).strip()
    if s == "0":
        return "0"
    try:
        int(s)
        return s  # already in base units
    except (ValueError, TypeError):
        return resolve_amount(s, ctx, **kwargs)


def wrap_in_array(value: str, ctx: ResolveContext, **kwargs) -> List[str]:
    """Resolve a single amount and wrap in a list: amount -> [amount]."""
    if value is None:
        return ["0"]
    decimals = kwargs.get("decimals", 18)
    d = Decimal(str(value))
    scale = Decimal(10) ** decimals
    base = (d * scale).to_integral_value(rounding=ROUND_DOWN)
    return [str(int(base))]


def build_fixed_array(value: Any, ctx: ResolveContext, **kwargs) -> List[str]:
    """Build a fixed-size array with one non-zero slot (e.g. Curve 3pool).
    index_map maps asset symbols to array indices."""
    array_size = kwargs.get("array_size", 3)
    fill_value = kwargs.get("fill_value", "0")
    arr = [fill_value] * array_size

    index_map = kwargs.get("index_map", {})
    asset_field = kwargs.get("asset_field", "asset")
    asset_sym = (ctx.raw_args.get(asset_field) or "USDC").upper()
    idx = index_map.get(asset_sym, 0)

    # Resolve the amount
    llm_field = kwargs.get("llm_field", "amount_human")
    amount_human = ctx.raw_args.get(llm_field) or ctx.raw_args.get("amount")
    if amount_human is not None:
        decimals = _resolve_decimals(kwargs.get("decimals_from"), ctx, asset_sym=asset_sym)
        d = Decimal(str(amount_human))
        scale = Decimal(10) ** decimals
        base = (d * scale).to_integral_value(rounding=ROUND_DOWN)
        arr[idx] = str(int(base))

    return arr


def resolve_constant(value: Any, ctx: ResolveContext, **kwargs) -> Any:
    """Return a constant value from the playbook spec."""
    return kwargs.get("value", value)


def llm_passthrough(value: Any, ctx: ResolveContext, **kwargs) -> Any:
    """Return the LLM field value as-is."""
    return value


def compute_human_readable(value: Any, ctx: ResolveContext, **kwargs) -> str:
    """Render a template string using LLM args.
    Normalizes max-like amounts ('all', 'full', etc.) to 'max'."""
    template = kwargs.get("template", "")
    merged = {**ctx.raw_args}
    # Normalize max-like amount_human values
    ah = merged.get("amount_human", "")
    if isinstance(ah, str) and ah.strip().lower() in ("max", "all", "full", "maximum", "everything", "entire"):
        merged["amount_human"] = "max"
    try:
        return template.format(**merged)
    except (KeyError, IndexError):
        return ""


def resolve_contract_address(value: Any, ctx: ResolveContext, **kwargs) -> Optional[str]:
    """Look up address from the playbook's contracts map (injected via _playbook_contracts)."""
    contract_key = kwargs.get("contract_key", "")
    contracts = kwargs.get("_playbook_contracts", {})
    contract = contracts.get(contract_key, {})
    return contract.get("address")


# EigenLayer strategy addresses for LSTs
_EIGENLAYER_STRATEGIES = {
    "stETH":  "0x93c4b944D05dfe6df7645A86cd2206016c51564D",
    "STETH":  "0x93c4b944D05dfe6df7645A86cd2206016c51564D",
    "rETH":   "0x1BeE69b7dFFfA4E2d53C2a2Df135C388AD25dCD2",
    "RETH":   "0x1BeE69b7dFFfA4E2d53C2a2Df135C388AD25dCD2",
    "cbETH":  "0x54945180dB7943c0ed0FEE7EdaB2Bd24620256bc",
    "CBETH":  "0x54945180dB7943c0ed0FEE7EdaB2Bd24620256bc",
    "wBETH":  "0x7CA911E83dabf90C90dD3De5411a10F1A6112184",
    "WBETH":  "0x7CA911E83dabf90C90dD3De5411a10F1A6112184",
    "swETH":  "0x0Fe4F44beE93503346A3Ac9EE5A26b130a5796d6",
    "SWETH":  "0x0Fe4F44beE93503346A3Ac9EE5A26b130a5796d6",
    "sfrxETH": "0x8CA7A5d6f3acd3A7A8bC468a8CD0FB14B6BD28b6",
    "SFRXETH": "0x8CA7A5d6f3acd3A7A8bC468a8CD0FB14B6BD28b6",
    "osETH":  "0x57ba429517c3473B6d34CA9aCd56c0e735b94c02",
    "OSETH":  "0x57ba429517c3473B6d34CA9aCd56c0e735b94c02",
}


def resolve_eigenlayer_strategy(value: Any, ctx: ResolveContext, **kwargs) -> Optional[str]:
    """Map LST symbol (stETH, rETH, cbETH, etc.) to EigenLayer strategy contract address."""
    if not value:
        return None
    sym = str(value).strip()
    strategy = _EIGENLAYER_STRATEGIES.get(sym)
    if strategy:
        return strategy
    # Try uppercase
    strategy = _EIGENLAYER_STRATEGIES.get(sym.upper())
    return strategy


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _resolve_decimals(spec: Any, ctx: ResolveContext, asset_sym: str = None) -> int:
    """Resolve a decimals_from spec to an integer.
    "$asset" -> look up from raw_args.asset symbol
    "$native" -> 18
    "$tokenIn" / "$tokenOut" -> look up from resolved token address
    int -> literal
    """
    if spec is None:
        return 18
    if isinstance(spec, int):
        return spec
    s = str(spec)
    if s == "$native":
        return 18
    if s.startswith("$"):
        # $asset, $tokenIn, $tokenOut, etc. -> look up the symbol from raw_args
        ref_key = s[1:]  # "asset", "tokenIn", "tokenOut"
        # First check if we already resolved decimals for this
        if ref_key in ctx._decimals_cache:
            return ctx._decimals_cache[ref_key]
        # Otherwise look up the symbol from raw_args and find decimals
        sym = asset_sym or ctx.raw_args.get(ref_key, "")
        if isinstance(sym, str) and not sym.startswith("0x"):
            if sym.upper() == "ETH":
                return 18
            if ctx.token_resolver:
                info = ctx.token_resolver.resolve_erc20(sym)
                if info:
                    ctx._decimals_cache[ref_key] = info["decimals"]
                    return info["decimals"]
        # Maybe it's an already-resolved address
        resolved_addr = ctx.resolved.get(ref_key)
        if resolved_addr and isinstance(resolved_addr, str) and resolved_addr in ctx._decimals_cache:
            return ctx._decimals_cache[resolved_addr]
        return 18
    try:
        return int(s)
    except ValueError:
        return 18


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────

RESOLVER_REGISTRY = {
    "resolve_token_address": resolve_token_address,
    "resolve_collection_address": resolve_collection_address,
    "resolve_amount": resolve_amount,
    "resolve_amount_or_max": resolve_amount_or_max,
    "resolve_ens_or_hex": resolve_ens_or_hex,
    "resolve_fee_tier": resolve_fee_tier,
    "resolve_deadline": resolve_deadline,
    "resolve_smart_amount": resolve_smart_amount,
    "wrap_in_array": wrap_in_array,
    "build_fixed_array": build_fixed_array,
    "constant": resolve_constant,
    "llm_passthrough": llm_passthrough,
    "compute_human_readable": compute_human_readable,
    "resolve_contract_address": resolve_contract_address,
    "resolve_eigenlayer_strategy": resolve_eigenlayer_strategy,
}
