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
            if raw_tx is None:
                # Encoding failed — keep semantic payload, raw_tx is null
                raw_tx_out = None
            else:
                raw_tx_out = {
                    "chain_id": raw_tx.get("chain_id", chain_id),
                    "to": raw_tx["to"],
                    "value": raw_tx["value"],
                    "data": raw_tx["data"],
                }
            metadata = build_metadata(payload_dict, token_resolver, ens_resolver)
            annotated.append({
                "user_intent": intent,
                "user_context": user_context,
                "target_payload": payload_dict,
                "raw_tx": raw_tx_out,
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
    parser.add_argument("--input", default=None, help="Raw intents JSON (overrides --protocol)")
    parser.add_argument("--output", default=None, help="Output path (overrides --protocol)")
    parser.add_argument("--protocol", default=None, help="Protocol name (e.g. aave_v3, weth). Reads from intents/{protocol}_intents.json, writes to annotated/{protocol}_annotated.json")
    parser.add_argument("--all", action="store_true", help="Annotate all protocols (scans intents/ dir)")
    parser.add_argument("--chain-id", type=int, default=1, help="Chain ID")
    parser.add_argument("--delay", type=float, default=0, help="Seconds between API calls")
    parser.add_argument("--from-address", default=os.getenv("DEFAULT_FROM_ADDRESS", ""), help="Sender address for user_context")
    args = parser.parse_args()
    from_addr = args.from_address or os.getenv("DEFAULT_FROM_ADDRESS", "0xe2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2")

    if args.all:
        # Annotate all *_intents.json files in the intents dir
        intents_dir = project_root / "data" / "datasets" / "intents"
        annotated_dir = project_root / "data" / "datasets" / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        for intent_file in sorted(intents_dir.glob("*_intents.json")):
            proto = intent_file.stem.replace("_intents", "")
            out_file = annotated_dir / f"{proto}_annotated.json"
            print(f"\n[{proto}]")
            annotate_with_hybrid(
                raw_intents_path=str(intent_file),
                output_path=str(out_file),
                chain_id=args.chain_id,
                delay_seconds=args.delay,
                from_address=from_addr,
            )
    elif args.protocol:
        inp = f"data/datasets/intents/{args.protocol}_intents.json"
        out = f"data/datasets/annotated/{args.protocol}_annotated.json"
        annotate_with_hybrid(
            raw_intents_path=inp,
            output_path=out,
            chain_id=args.chain_id,
            delay_seconds=args.delay,
            from_address=from_addr,
        )
