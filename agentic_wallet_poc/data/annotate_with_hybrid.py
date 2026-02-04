"""
Annotate raw intents with the hybrid translator and output simulation-ready format.

Single step: run translator -> convert to raw tx (to, value, data) -> add metadata
-> write { user_intent, user_context (with from_address), target_payload, metadata,
_tenderly_validated: false, _annotation_failed }.

Then run: python data/run_tenderly_simulation.py --input <path> to update _tenderly_validated.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.llm_translator import LLMTranslator
from engine.tx_encoder import payload_to_raw_tx, build_metadata


def load_raw_intents(path: str) -> List[Dict[str, Any]]:
    """Load raw intents JSON: list of {intent, transaction_type}."""
    p = project_root / path if not path.startswith("/") else Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    return data


def load_registries() -> tuple:
    token_path = project_root / "data" / "registries" / "token_registry.json"
    ens_path = project_root / "data" / "registries" / "ens_registry.json"
    with open(token_path, "r", encoding="utf-8") as f:
        token_registry = json.load(f)
    with open(ens_path, "r", encoding="utf-8") as f:
        ens_data = json.load(f)
        ens_registry = ens_data.get("ens_names", {})
    return token_registry, ens_registry


def annotate_with_hybrid(
    raw_intents_path: str = "data/datasets/intents/raw_intents_defi.json",
    output_path: str = "data/datasets/annotated/annotated_dataset.json",
    chain_id: int = 1,
    delay_seconds: float = 0,
    from_address: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run hybrid translator and output simulation-ready format in one step.
    Output: user_intent, user_context (with from_address), target_payload as
    { chain_id, to, value, data }, metadata, _tenderly_validated: false, _annotation_failed.
    """
    if not Path(raw_intents_path).is_absolute():
        raw_intents_path = str(project_root / raw_intents_path)
    if not Path(output_path).is_absolute():
        output_path = str(project_root / output_path)
    if from_address is None:
        from_address = os.getenv("TENDERLY_FROM_ADDRESS", "0xe2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2")

    token_registry, ens_registry = load_registries()
    raw = load_raw_intents(raw_intents_path)
    translator = LLMTranslator()
    annotated = []

    for i, item in enumerate(raw):
        intent = item.get("intent", "")
        if not intent:
            continue
        if delay_seconds and i > 0:
            time.sleep(delay_seconds)
        failure_info: Dict[str, Any] = {}
        result = translator.translate(intent, chain_id=chain_id, failure_info=failure_info, from_address=from_address)

        user_context = {
            "current_chain_id": chain_id,
            "from_address": from_address,
        }

        if result is None:
            annotated.append({
                "user_intent": intent,
                "user_context": user_context,
                "target_payload": None,
                "metadata": {"action": None},
                "_tenderly_validated": False,
                "_annotation_failed": True,
                "_failure_reason": failure_info.get("message", "Translation returned null."),
                "_failure_stage": failure_info.get("stage", "unknown"),
                **({k: v for k, v in failure_info.items() if k not in ("stage", "message")}),
            })
        else:
            payload_dict = result.target_payload.model_dump(mode="json")
            raw_tx = payload_to_raw_tx(payload_dict, from_address)
            if raw_tx is not None:
                target_payload = {
                    "chain_id": raw_tx.get("chain_id", chain_id),
                    "to": raw_tx["to"],
                    "value": raw_tx["value"],
                    "data": raw_tx["data"],
                }
            else:
                # Unsupported for encoding (e.g. some DeFi) – minimal raw shape
                args = payload_dict.get("arguments") or {}
                target_payload = {
                    "chain_id": chain_id,
                    "to": payload_dict.get("target_contract") or args.get("to"),
                    "value": args.get("value", "0"),
                    "data": "0x",
                } if payload_dict.get("target_contract") or args.get("to") else None
            metadata = build_metadata(payload_dict, token_registry, ens_registry)
            annotated.append({
                "user_intent": intent,
                "user_context": user_context,
                "target_payload": target_payload,
                "metadata": metadata,
                "_tenderly_validated": False,
                "_annotation_failed": False,
            })
        if (i + 1) % 5 == 0:
            print(f"  Annotated {i + 1}/{len(raw)}...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    failed_count = sum(1 for a in annotated if a.get("_annotation_failed"))
    print(f"\n✓ Wrote {len(annotated)} annotated examples to {output_path}")
    if failed_count:
        print(f"  Failed: {failed_count}. Next: run python data/run_tenderly_simulation.py --input {output_path} to update _tenderly_validated.")
    else:
        print(f"  Failed: 0. Next: run python data/run_tenderly_simulation.py --input {output_path} to validate with Tenderly.")
    return annotated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annotate raw intents with hybrid translator (output: raw tx + metadata, _tenderly_validated=false)")
    parser.add_argument("--input", default="data/datasets/intents/raw_intents_defi.json", help="Raw intents JSON")
    parser.add_argument("--output", default="data/datasets/annotated/annotated_dataset_candidate.json", help="Output path")
    parser.add_argument("--chain-id", type=int, default=1, help="Chain ID")
    parser.add_argument("--delay", type=float, default=0, help="Seconds between API calls")
    parser.add_argument("--from-address", default=os.getenv("TENDERLY_FROM_ADDRESS", ""), help="Sender address for user_context (default: env or placeholder)")
    args = parser.parse_args()
    from_addr = args.from_address or os.getenv("TENDERLY_FROM_ADDRESS", "0xe2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2")
    annotate_with_hybrid(
        raw_intents_path=args.input,
        output_path=args.output,
        chain_id=args.chain_id,
        delay_seconds=args.delay,
        from_address=from_addr,
    )
