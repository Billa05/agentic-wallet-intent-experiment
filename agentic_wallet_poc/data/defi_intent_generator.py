"""
DeFi raw intents generator (like dataset_generator.py).

Generates synthetic natural language DeFi intents using litellm (gpt-4o).
Output: raw_intents_defi.json (intents only; no payloads).
Then use annotate_with_hybrid.py to create payloads; human-validate; run evaluation.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from litellm import completion
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.prompts import PromptConfig, create_prompt_for_defi_action

load_dotenv()

# litellm uses OPENAI_API_KEY for gpt-4o by default
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Number of intent examples per DeFi action type (variable; build like dataset_generator)
DEFI_INTENTS_PER_ACTION = 6

# All DeFi action types we generate intents for
DEFI_ACTION_TYPES = [
    "aave_supply",
    "aave_withdraw",
    "aave_borrow",
    "aave_repay",
    "lido_stake",
    "lido_unstake",
    "uniswap_swap",
    "curve_add_liquidity",
    "curve_remove_liquidity",
]


def generate_defi_intent_examples(
    action_type: str,
    count: int = DEFI_INTENTS_PER_ACTION,
    model_name: str = "gpt-4o",
    config: Optional[PromptConfig] = None,
    defi_style: str = "mixed",
) -> List[Dict[str, Any]]:
    """
    Generate diverse natural language DeFi intent examples using litellm (gpt-4o).
    Returns list of {"intent": str, "transaction_type": action_type}.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    if config is None:
        config = PromptConfig(count=count, defi_style=defi_style)
    else:
        config.count = count
        config.defi_style = defi_style

    prompt = create_prompt_for_defi_action(action_type, config)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            response_text = None
            if hasattr(response, "choices") and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    response_text = choice.message.content
                elif hasattr(choice, "text"):
                    response_text = choice.text
            if response_text:
                response_text = response_text.strip()
            if not response_text:
                response_text = str(response).strip()
                json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0).strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            if not response_text.startswith("["):
                response_text = "[" + response_text
            if not response_text.endswith("]"):
                response_text = response_text + "]"

            intents = json.loads(response_text)
            if not isinstance(intents, list):
                raise ValueError("Response is not a list")

            result = []
            for intent in intents:
                if isinstance(intent, str):
                    result.append({"intent": intent.strip(), "transaction_type": action_type})
                    if len(result) >= count:
                        break
            return result[:count]
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                print(f"Warning: {e}. Retrying...")
                time.sleep(2)
                continue
            raise RuntimeError(f"Failed to generate DeFi intents for {action_type}: {e}") from e
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise RuntimeError(f"Failed to generate DeFi intents for {action_type}: {e}") from e
    return []


def generate_full_defi_dataset(
    intents_per_action: int = DEFI_INTENTS_PER_ACTION,
    output_path: str = "data/datasets/intents/raw_intents_defi.json",
    model_name: str = "gpt-4o",
    defi_style: str = "mixed",
) -> List[Dict[str, Any]]:
    """
    Generate raw DeFi intents for all DEFI_ACTION_TYPES (like generate_full_dataset in dataset_generator).
    Saves to raw_intents_defi.json; format like raw_intents.json: [{"intent": "...", "transaction_type": "aave_supply"}, ...].
    defi_style: "basic" | "advanced" | "mixed" — mixed adds advanced-but-popular variations (slippage, repay max, etc.).
    """
    if not Path(output_path).is_absolute():
        output_path = str(project_root / output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_intents = []
    config = PromptConfig(
        count=intents_per_action,
        include_edge_cases=False,
        include_negative_examples=False,
        defi_style=defi_style,
    )

    for action_type in DEFI_ACTION_TYPES:
        print(f"Generating {intents_per_action} examples for {action_type} (style={defi_style})...")
        intents = generate_defi_intent_examples(
            action_type,
            count=intents_per_action,
            model_name=model_name,
            config=config,
            defi_style=defi_style,
        )
        all_intents.extend(intents)
        print(f"  ✓ {len(intents)} intents")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_intents, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Wrote {len(all_intents)} DeFi intents to {output_path}")
    print("Next: run python data/annotate_with_hybrid.py to create data/datasets/annotated/annotated_dataset_candidate.json for human validation.")
    return all_intents


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate DeFi raw intents (like dataset_generator)")
    parser.add_argument("--count", type=int, default=DEFI_INTENTS_PER_ACTION, help=f"Intents per action (default: {DEFI_INTENTS_PER_ACTION})")
    parser.add_argument("--output", default="data/datasets/intents/raw_intents_defi.json", help="Output JSON path")
    parser.add_argument("--model", default="gpt-4o", help="Model name via litellm (default: gpt-4o)")
    parser.add_argument(
        "--style",
        choices=["basic", "advanced", "mixed"],
        default="mixed",
        help="Intent style: basic (simple only), advanced (mostly advanced/popular), mixed (default)",
    )
    args = parser.parse_args()
    generate_full_defi_dataset(
        intents_per_action=args.count,
        output_path=args.output,
        model_name=args.model,
        defi_style=args.style,
    )
