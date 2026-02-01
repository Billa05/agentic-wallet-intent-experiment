"""
Annotate raw intents with the hybrid translator to create payload dataset.

Loads raw_intents_defi.json (or raw_intents.json), runs the hybrid translator
(engine.llm_translator.LLMTranslator) on each intent, and saves annotated
format like annotated_dataset.json for human validation.
After human validation, save as annotated_dataset.json and run evaluation.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.llm_translator import LLMTranslator


def load_raw_intents(path: str) -> List[Dict[str, Any]]:
    """Load raw intents JSON: list of {intent, transaction_type}."""
    p = project_root / path if not path.startswith("/") else Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    return data


def annotate_with_hybrid(
    raw_intents_path: str = "data/datasets/intents/raw_intents_defi.json",
    output_path: str = "data/datasets/annotated/annotated_dataset.json",
    chain_id: int = 1,
    delay_seconds: float = 0,
) -> List[Dict[str, Any]]:
    """
    Run hybrid translator on each raw intent and save annotated format.
    Output format: list of {user_intent, user_context, target_payload} like annotated_dataset.json.
    """
    if not Path(raw_intents_path).is_absolute():
        raw_intents_path = str(project_root / raw_intents_path)
    if not Path(output_path).is_absolute():
        output_path = str(project_root / output_path)

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
        result = translator.translate(intent, chain_id=chain_id, failure_info=failure_info)
        transaction_type = item.get("transaction_type", "")
        if result is None:
            annotated.append({
                "user_intent": intent,
                "expected_transaction_type": transaction_type,
                "user_context": {"current_chain_id": chain_id, "token_prices": {"ETH": 2500.0}},
                "target_payload": None,
                "_annotation_failed": True,
                "_failure_reason": failure_info.get("message", "Translation returned null."),
                "_failure_stage": failure_info.get("stage", "unknown"),
                **({k: v for k, v in failure_info.items() if k not in ("stage", "message")}),
            })
        else:
            payload_dict = result.target_payload.model_dump(mode="json")
            annotated.append({
                "user_intent": intent,
                "user_context": {"current_chain_id": chain_id, "token_prices": {"ETH": 2500.0}},
                "target_payload": payload_dict,
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
        print(f"  Failed: {failed_count} (see _failure_reason in output). Next: human-validate, fix/remove failed rows, then run evaluation.")
    else:
        print(f"  Failed: 0. Next: human-validate, then run evaluation.")
    return annotated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annotate raw intents with hybrid translator for human validation")
    parser.add_argument("--input", default="data/datasets/intents/raw_intents_defi.json", help="Raw intents JSON (e.g. intents/raw_intents_defi.json or intents/raw_intents.json)")
    parser.add_argument("--output", default="data/datasets/annotated/annotated_dataset_candidate.json", help="Output path for annotated candidate")
    parser.add_argument("--chain-id", type=int, default=1, help="Chain ID")
    parser.add_argument("--delay", type=float, default=0, help="Seconds between API calls (e.g. 12 for rate limits)")
    args = parser.parse_args()
    annotate_with_hybrid(
        raw_intents_path=args.input,
        output_path=args.output,
        chain_id=args.chain_id,
        delay_seconds=args.delay,
    )
