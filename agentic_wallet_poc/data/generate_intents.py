"""
Unified intent generator — replaces both intent_generator.py and defi_intent_generator.py.

Auto-discovers protocols by scanning data/playbooks/*.json.
Groups actions by protocol and generates intents per protocol.

Usage:
  python data/generate_intents.py --protocol aave_v3 --count 6
  python data/generate_intents.py --protocol transfers --count 6
  python data/generate_intents.py --all --count 6
  python data/generate_intents.py --list   # show available protocols
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from litellm import completion

from data.prompts import PromptConfig, create_prompt_for_defi_action, create_prompt_for_transaction_type
from utils.schemas import TransactionType

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PLAYBOOKS_DIR = Path(__file__).parent / "playbooks"
INTENTS_DIR = Path(__file__).parent / "datasets" / "intents"


def discover_protocols() -> Dict[str, List[str]]:
    """Scan playbooks and return {protocol_name: [action_names]}."""
    protocols: Dict[str, List[str]] = {}
    for pb_file in sorted(PLAYBOOKS_DIR.glob("*.json")):
        pb = json.loads(pb_file.read_text())
        protocol = pb.get("protocol", pb_file.stem)
        actions = list(pb.get("actions", {}).keys())
        if actions:
            protocols[protocol] = actions
    return protocols


def _parse_llm_json_array(response_text: str) -> List[str]:
    """Extract JSON array of strings from LLM response."""
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text = text + "]"
    items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError("Response is not a list")
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def generate_intents_for_action(
    action_type: str,
    count: int = 6,
    model_name: str = "gpt-4o",
    defi_style: str = "mixed",
) -> List[Dict[str, Any]]:
    """Generate intents for a single action type. Returns [{intent, transaction_type}]."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set.")

    # Check if it's a transfer type handled by the old prompt system
    transfer_map = {
        "transfer_native": TransactionType.SEND_ETH,
        "transfer_erc20": TransactionType.TRANSFER_ERC20,
        "transfer_erc721": TransactionType.TRANSFER_ERC721,
    }

    if action_type in transfer_map:
        config = PromptConfig(count=count, defi_style=defi_style)
        prompt = create_prompt_for_transaction_type(transfer_map[action_type], config)
    else:
        config = PromptConfig(count=count, include_edge_cases=False, include_negative_examples=False, defi_style=defi_style)
        prompt = create_prompt_for_defi_action(action_type, config)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            response_text = ""
            if hasattr(response, "choices") and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    response_text = choice.message.content or ""
                elif hasattr(choice, "text"):
                    response_text = choice.text or ""
            if not response_text:
                response_text = str(response)

            intents_raw = _parse_llm_json_array(response_text)
            result = [{"intent": s, "transaction_type": action_type} for s in intents_raw[:count]]
            return result
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                print(f"  Warning: {e}. Retrying...")
                time.sleep(2)
                continue
            raise RuntimeError(f"Failed to generate intents for {action_type}: {e}") from e
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise RuntimeError(f"Failed to generate intents for {action_type}: {e}") from e
    return []


def generate_for_protocol(
    protocol: str,
    count_per_action: int = 6,
    model_name: str = "gpt-4o",
    defi_style: str = "mixed",
) -> List[Dict[str, Any]]:
    """Generate intents for all actions in a protocol. Saves to intents dir."""
    protocols = discover_protocols()
    if protocol not in protocols:
        raise ValueError(f"Unknown protocol: {protocol}. Available: {list(protocols.keys())}")

    actions = protocols[protocol]
    all_intents = []

    for action in actions:
        print(f"  Generating {count_per_action} intents for {action}...")
        intents = generate_intents_for_action(action, count=count_per_action, model_name=model_name, defi_style=defi_style)
        all_intents.extend(intents)
        print(f"    -> {len(intents)} intents")

    # Save
    INTENTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTENTS_DIR / f"{protocol}_intents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_intents, f, indent=2, ensure_ascii=False)

    print(f"  Wrote {len(all_intents)} intents to {output_path}")
    return all_intents


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified intent generator (playbook-driven)")
    parser.add_argument("--protocol", type=str, help="Protocol name (e.g. aave_v3, weth, transfers)")
    parser.add_argument("--all", action="store_true", help="Generate for all protocols")
    parser.add_argument("--list", action="store_true", help="List available protocols and actions")
    parser.add_argument("--count", type=int, default=6, help="Intents per action (default: 6)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model via litellm (default: gpt-4o)")
    parser.add_argument("--style", choices=["basic", "advanced", "mixed"], default="mixed", help="Intent style")
    args = parser.parse_args()

    protocols = discover_protocols()

    if args.list:
        print("Available protocols:")
        for proto, actions in protocols.items():
            print(f"  {proto}: {', '.join(actions)}")
        return

    if args.all:
        print(f"Generating intents for all {len(protocols)} protocols ({args.count} per action, style={args.style})")
        print("=" * 60)
        for proto in protocols:
            print(f"\n[{proto}]")
            generate_for_protocol(proto, count_per_action=args.count, model_name=args.model, defi_style=args.style)
        print("\nDone!")
        return

    if args.protocol:
        print(f"Generating intents for protocol: {args.protocol} ({args.count} per action, style={args.style})")
        print("=" * 60)
        generate_for_protocol(args.protocol, count_per_action=args.count, model_name=args.model, defi_style=args.style)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
