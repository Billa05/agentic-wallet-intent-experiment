"""
Annotate raw intents with the hybrid translator.

Single step: run translator -> convert to raw tx (to, value, data) -> add metadata
-> write { user_intent, user_context, target_payload, metadata, _annotation_failed }.

Then run: python data/validate_calldata.py --input <path> to validate calldata.
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
from engine.ens_resolver import ENSResolver


def load_raw_intents(path: str) -> List[Dict[str, Any]]:
    """Load raw intents JSON: list of {intent, transaction_type}."""
    p = project_root / path if not path.startswith("/") else Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    return data


def load_registries() -> tuple:
    from engine.token_resolver import TokenResolver
    token_resolver = TokenResolver()
    ens_resolver = ENSResolver(w3=token_resolver._w3)
    return token_resolver, ens_resolver


def annotate_with_hybrid(
    raw_intents_path: str = "data/datasets/intents/raw_intents_defi.json",
    output_path: str = "data/datasets/annotated/annotated_dataset.json",
    chain_id: int = 1,
    delay_seconds: float = 0,
    from_address: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run hybrid translator and output annotated dataset.
    Output: user_intent, user_context (with from_address), target_payload as
    { chain_id, to, value, data }, metadata, _annotation_failed.
    """
    if not Path(raw_intents_path).is_absolute():
        raw_intents_path = str(project_root / raw_intents_path)
    if not Path(output_path).is_absolute():
        output_path = str(project_root / output_path)
    if from_address is None:
        from_address = os.getenv("DEFAULT_FROM_ADDRESS", "0xe2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2")

    token_resolver, ens_resolver = load_registries()
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
            metadata = build_metadata(payload_dict, token_resolver, ens_resolver)
            annotated.append({
                "user_intent": intent,
                "user_context": user_context,
                "target_payload": target_payload,
                "metadata": metadata,
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
        print(f"  Failed: {failed_count}. Next: run python data/validate_calldata.py --input {output_path} to validate.")
    else:
        print(f"  All succeeded. Next: run python data/validate_calldata.py --input {output_path} to validate.")
    return annotated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annotate raw intents with hybrid translator")
    parser.add_argument("--input", default="data/datasets/intents/raw_intents_defi.json", help="Raw intents JSON")
    parser.add_argument("--output", default="data/datasets/annotated/annotated_dataset_candidate.json", help="Output path")
    parser.add_argument("--chain-id", type=int, default=1, help="Chain ID")
    parser.add_argument("--delay", type=float, default=0, help="Seconds between API calls")
    parser.add_argument("--from-address", default=os.getenv("DEFAULT_FROM_ADDRESS", ""), help="Sender address for user_context")
    args = parser.parse_args()
    from_addr = args.from_address or os.getenv("DEFAULT_FROM_ADDRESS", "0xe2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2")
    annotate_with_hybrid(
        raw_intents_path=args.input,
        output_path=args.output,
        chain_id=args.chain_id,
        delay_seconds=args.delay,
        from_address=from_addr,
    )
