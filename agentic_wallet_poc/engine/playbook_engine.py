"""
Generic playbook engine — replaces per-protocol hardcoded logic.

All protocol-specific knowledge lives in JSON playbook files under data/playbooks/.
This engine reads those playbooks and executes two stages:
  Stage 1 (build_payload): LLM output → ExecutablePayload dict
  Stage 2 (encode_tx):     ExecutablePayload → raw tx {chain_id, to, value, data}

The engine code contains ZERO protocol-specific if/else branches.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from eth_utils import to_checksum_address

from engine.resolvers import (
    RESOLVER_REGISTRY,
    ResolveContext,
)
from engine.tx_encoder import (
    encode_from_abi,
    _load_contract_abi,
    _find_function_in_abi,
    _addr,
)


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PLAYBOOKS_DIR = _DATA_DIR / "playbooks"


class PlaybookEngine:
    """Generic engine driven by JSON playbook files."""

    def __init__(
        self,
        token_resolver=None,
        ens_resolver=None,
        playbooks_dir: Optional[str] = None,
    ):
        self.token_resolver = token_resolver
        self.ens_resolver = ens_resolver
        self._playbooks: Dict[str, Dict] = {}     # action_name -> action_spec
        self._playbook_meta: Dict[str, Dict] = {}  # action_name -> full playbook (for contracts)
        self._standard_abis: Dict[str, Dict] = {} # key -> ABI entry (from playbooks)
        self._load_playbooks(Path(playbooks_dir) if playbooks_dir else _PLAYBOOKS_DIR)

    def _load_playbooks(self, playbooks_dir: Path) -> None:
        """Load all .json playbook files and build action lookup."""
        for pb_file in sorted(playbooks_dir.glob("*.json")):
            pb = json.loads(pb_file.read_text())
            for action_name, action_spec in pb.get("actions", {}).items():
                self._playbooks[action_name] = action_spec
                self._playbook_meta[action_name] = pb
            # Collect standard ABIs (e.g. from transfers.json)
            for key, abi_entry in pb.get("standard_abis", {}).items():
                self._standard_abis[key] = abi_entry

    # ─────────────────────────────────────────────────────────────────
    # Stage 1: LLM output → ExecutablePayload
    # ─────────────────────────────────────────────────────────────────

    def build_payload(
        self,
        llm_output: Dict[str, Any],
        chain_id: int = 1,
        from_address: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convert LLM output to ExecutablePayload dict."""
        action = llm_output.get("action")
        if not action or action not in self._playbooks:
            return None

        action_spec = self._playbooks[action]
        playbook = self._playbook_meta[action]
        args = llm_output.get("arguments") or {}

        ctx = ResolveContext(
            token_resolver=self.token_resolver,
            ens_resolver=self.ens_resolver,
            from_address=from_address,
            chain_id=chain_id,
            action=action,
            raw_args=args,
        )

        # Resolve all payload_args in declaration order
        payload_args_spec = action_spec.get("payload_args", {})
        for arg_name, arg_spec in payload_args_spec.items():
            resolved_value = self._resolve_payload_arg(arg_name, arg_spec, ctx, playbook)
            ctx.resolved[arg_name] = resolved_value

        # Build the output arguments dict (exclude double-underscore internal keys)
        arguments = {}
        for arg_name, resolved_value in ctx.resolved.items():
            if not arg_name.startswith("__"):
                arguments[arg_name] = resolved_value

        # Resolve target_contract
        target_contract = self._resolve_target_contract(action_spec, playbook, ctx)

        # Resolve function_name (may be overridden by function_overrides)
        function_name = action_spec.get("function_name")
        if target_contract and "function_overrides" in action_spec:
            override = action_spec["function_overrides"].get(target_contract.lower())
            if override:
                function_name = override.get("function_name", function_name)

        return {
            "chain_id": chain_id,
            "action": action,
            "target_contract": target_contract,
            "function_name": function_name,
            "arguments": arguments,
        }

    def _resolve_payload_arg(
        self,
        arg_name: str,
        arg_spec: Dict[str, Any],
        ctx: ResolveContext,
        playbook: Dict,
    ) -> Any:
        """Resolve a single payload_args entry using the appropriate resolver."""
        source = arg_spec.get("source", "constant")
        resolver_fn = RESOLVER_REGISTRY.get(source)
        if resolver_fn is None:
            return arg_spec.get("value")

        # Extract the raw value from LLM args
        raw_value = self._extract_llm_value(arg_spec, ctx)

        # For resolvers that don't need a raw value (deadline, constant, etc.)
        if source == "constant":
            return arg_spec.get("value")
        if source == "resolve_contract_address":
            return resolver_fn(raw_value, ctx, _playbook_contracts=playbook.get("contracts", {}), **arg_spec)
        if source == "compute_human_readable":
            # Check if LLM provided a human_readable_amount, use it if available
            existing = ctx.raw_args.get("human_readable_amount")
            if existing:
                return existing
            return resolver_fn(raw_value, ctx, **arg_spec)
        if source == "resolve_deadline":
            return resolver_fn(raw_value, ctx, **arg_spec)
        if source == "build_fixed_array":
            return resolver_fn(raw_value, ctx, **arg_spec)

        # Standard resolver: pass raw value + kwargs from spec
        kwargs = {k: v for k, v in arg_spec.items()
                  if k not in ("source", "llm_field", "fallback_llm_fields", "context_field", "fallback_context")}
        result = resolver_fn(raw_value, ctx, **kwargs)

        # Fallback to context if resolver returned None
        if result is None:
            context_field = arg_spec.get("context_field")
            if context_field == "from_address":
                result = ctx.from_address
            elif context_field:
                result = ctx.raw_args.get(context_field) or ctx.from_address

        return result

    def _extract_llm_value(self, arg_spec: Dict, ctx: ResolveContext) -> Any:
        """Extract a value from LLM args, trying primary field then fallbacks."""
        primary = arg_spec.get("llm_field")
        if primary:
            # Handle array access: path[0], path[-1]
            if "[" in primary:
                base_key, idx_str = primary.rstrip("]").split("[")
                arr = ctx.raw_args.get(base_key)
                if arr and isinstance(arr, list):
                    try:
                        idx = int(idx_str)
                        if abs(idx) <= len(arr):
                            return str(arr[idx])
                    except (ValueError, IndexError):
                        pass
            else:
                val = ctx.raw_args.get(primary)
                if val is not None:
                    return val

        # Try fallback fields
        for fb in arg_spec.get("fallback_llm_fields", []):
            if "[" in fb:
                base_key, idx_str = fb.rstrip("]").split("[")
                arr = ctx.raw_args.get(base_key)
                if arr and isinstance(arr, list):
                    try:
                        idx = int(idx_str)
                        return str(arr[idx])
                    except (ValueError, IndexError):
                        pass
            else:
                val = ctx.raw_args.get(fb)
                if val is not None:
                    return val

        return None

    def _resolve_target_contract(
        self,
        action_spec: Dict,
        playbook: Dict,
        ctx: ResolveContext,
    ) -> Optional[str]:
        """Resolve target_contract from playbook spec."""
        target = action_spec.get("target_contract")
        if not target:
            return None

        # Dynamic sentinels
        if target == "$recipient":
            return ctx.resolved.get("to")
        if target == "$token_address":
            return ctx.resolved.get("__token_address")
        if target == "$collection_address":
            return ctx.resolved.get("__collection_address")

        # Lookup from contracts map
        contracts = playbook.get("contracts", {})
        contract_info = contracts.get(target)
        if contract_info:
            return contract_info.get("address")

        return None

    # ─────────────────────────────────────────────────────────────────
    # Stage 2: ExecutablePayload → raw tx
    # ─────────────────────────────────────────────────────────────────

    def encode_tx(
        self,
        payload: Dict[str, Any],
        from_address: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert ExecutablePayload dict to raw tx: {chain_id, to, value, data}."""
        if not payload:
            return None

        action = payload.get("action")
        if not action or action not in self._playbooks:
            return None

        action_spec = self._playbooks[action]
        args = payload.get("arguments") or {}
        target_contract = payload.get("target_contract")
        chain_id = payload.get("chain_id", 1)

        # Determine function_name, param_mapping, and ABI (may be overridden)
        function_name = payload.get("function_name")
        param_mapping = action_spec.get("param_mapping", [])
        abi_source = action_spec.get("abi_source", "etherscan_cache")
        standard_abi_key = action_spec.get("standard_abi_key")

        # Check function_overrides (CryptoPunks)
        if target_contract and "function_overrides" in action_spec:
            override = action_spec["function_overrides"].get(target_contract.lower())
            if override:
                function_name = override.get("function_name", function_name)
                param_mapping = override.get("param_mapping", param_mapping)
                standard_abi_key = override.get("standard_abi_key", standard_abi_key)

        # No function = no calldata (native transfer)
        if function_name is None:
            to = target_contract or args.get("to")
            if not to:
                return None
            value = self._resolve_tx_value(action_spec, args)
            return {
                "chain_id": chain_id,
                "to": to,
                "value": value,
                "data": "0x",
            }

        # Load ABI entry
        abi_entry = self._get_abi_entry(action, abi_source, standard_abi_key, function_name)
        if abi_entry is None:
            return None

        # Build values from param_mapping
        values = self._build_abi_values(param_mapping, args, from_address)

        # Encode calldata
        try:
            data = encode_from_abi(abi_entry, values)
        except Exception:
            return None

        # Target address
        to = target_contract
        if not to:
            return None

        # Transaction value
        value = self._resolve_tx_value(action_spec, args)

        return {
            "chain_id": chain_id,
            "to": to,
            "value": value,
            "data": data,
        }

    def _get_abi_entry(
        self,
        action: str,
        abi_source: str,
        standard_abi_key: Optional[str],
        function_name: str,
    ) -> Optional[Dict]:
        """Load the ABI entry for encoding directly from playbook data."""
        if abi_source == "standard" and standard_abi_key:
            return self._standard_abis.get(standard_abi_key)
        if abi_source == "etherscan_cache":
            # Resolve contract address from the playbook's contracts section
            action_spec = self._playbooks.get(action)
            playbook = self._playbook_meta.get(action)
            if not action_spec or not playbook:
                return None
            target_key = action_spec.get("target_contract")
            if not target_key:
                return None
            contracts = playbook.get("contracts", {})
            contract_info = contracts.get(target_key, {})
            address = contract_info.get("address")
            if not address:
                return None
            abi = _load_contract_abi(address)
            if not abi:
                return None
            return _find_function_in_abi(abi, function_name)
        return None

    def _build_abi_values(
        self,
        param_mapping: List[Dict],
        args: Dict[str, Any],
        from_address: str,
    ) -> List[Any]:
        """Build ordered list of ABI-encoded values from param_mapping."""
        values = []
        for entry in param_mapping:
            source = entry.get("source")
            coerce = entry.get("coerce", "")

            if source == "struct":
                # Recursively build a tuple from nested fields
                struct_values = []
                for field_entry in entry.get("fields", []):
                    field_val = self._resolve_param_entry(field_entry, args, from_address)
                    struct_values.append(field_val)
                values.append(tuple(struct_values))

            elif source == "arg":
                raw = args.get(entry.get("arg_key"))
                values.append(self._coerce_value(raw, coerce, from_address))

            elif source == "context":
                context_key = entry.get("context_key", "")
                if context_key == "from_address":
                    values.append(self._coerce_value(from_address, coerce, from_address))
                else:
                    values.append(self._coerce_value(None, coerce, from_address))

            elif source == "constant":
                values.append(self._coerce_value(entry.get("value"), coerce, from_address))

        return values

    def _resolve_param_entry(
        self,
        entry: Dict,
        args: Dict[str, Any],
        from_address: str,
    ) -> Any:
        """Resolve a single param_mapping entry to its typed value."""
        source = entry.get("source")
        coerce = entry.get("coerce", "")

        if source == "arg":
            raw = args.get(entry.get("arg_key"))
            return self._coerce_value(raw, coerce, from_address)
        if source == "context":
            context_key = entry.get("context_key", "")
            if context_key == "from_address":
                return self._coerce_value(from_address, coerce, from_address)
            return self._coerce_value(None, coerce, from_address)
        if source == "constant":
            return self._coerce_value(entry.get("value"), coerce, from_address)
        return None

    def _coerce_value(self, value: Any, coerce: str, from_address: str) -> Any:
        """Coerce a resolved value to the type expected by eth_abi."""
        if coerce == "address":
            return _addr(value) if value else _addr(from_address)
        if coerce in ("uint256", "uint24", "uint160"):
            return int(value) if value is not None else 0
        if coerce == "int_array":
            if isinstance(value, list):
                return [int(v) for v in value]
            return [0, 0, 0]
        if coerce == "uint256_array":
            if isinstance(value, list):
                return [int(v) for v in value]
            return []
        return value

    def _resolve_tx_value(self, action_spec: Dict, args: Dict) -> str:
        """Resolve the ETH value to send with the transaction."""
        value_logic = action_spec.get("value_logic", {})
        vtype = value_logic.get("type", "zero")

        if vtype == "zero":
            return "0"
        if vtype == "from_arg":
            source_arg = value_logic.get("source_arg", "value")
            return str(args.get(source_arg, "0"))
        if vtype == "amount_as_value":
            return str(args.get("value", "0"))
        return "0"

    # ─────────────────────────────────────────────────────────────────
    # Utility: get required args for evaluation
    # ─────────────────────────────────────────────────────────────────

    def get_required_payload_args(self) -> Dict[str, List[str]]:
        """Build ACTION_REQUIRED_ARGS dict from playbook specs."""
        result = {}
        for action_name, spec in self._playbooks.items():
            result[action_name] = spec.get("required_payload_args", [])
        return result

    def get_supported_actions(self) -> List[str]:
        """Return list of all action names from loaded playbooks."""
        return list(self._playbooks.keys())
