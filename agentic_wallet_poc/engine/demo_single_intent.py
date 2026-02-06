"""
Step-by-step demo: process a single intent and print every step and its output.

Usage (from agentic_wallet_poc/):
  python engine/demo_single_intent.py
  python engine/demo_single_intent.py "Your intent here"

Default intent: "Could you please facilitate the withdrawal of 15 stETH from Lido?"

Actions that need a sender (e.g. lido_unstake, aave_*) use from_address; set
DEFAULT_FROM_ADDRESS in .env or environment, or a test address is used.
"""

import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Default sender for actions that require it (lido_unstake _owner, aave onBehalfOf, swap to)
DEFAULT_FROM_ADDRESS = os.getenv("DEFAULT_FROM_ADDRESS", "0x70997970C51812dc3A010C7d01b50e0d17dc79C8")

# -----------------------------------------------------------------------------
# Step 0: Load registries
# -----------------------------------------------------------------------------
def step0_load_registries():
    print("\n" + "=" * 70)
    print("STEP 0: Load registries")
    print("=" * 70)

    token_path = project_root / "data" / "registries" / "token_registry.json"
    ens_path = project_root / "data" / "registries" / "ens_registry.json"
    protocol_path = project_root / "data" / "registries" / "protocol_registry.json"

    with open(token_path, "r", encoding="utf-8") as f:
        token_registry = json.load(f)
    with open(ens_path, "r", encoding="utf-8") as f:
        ens_data = json.load(f)
        ens_registry = ens_data.get("ens_names", {})
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol_registry = json.load(f)

    print(f"  token_registry: {len(token_registry.get('erc20_tokens', {}))} ERC-20 tokens")
    print(f"  ens_registry: {list(ens_registry.keys())[:5]}...")
    print(f"  protocol_registry: {list(protocol_registry.get('protocols', {}).keys())}")
    return token_registry, ens_registry, protocol_registry


# -----------------------------------------------------------------------------
# Step 1: Build prompts
# -----------------------------------------------------------------------------
def step1_build_prompts(intent: str, chain_id: int, token_registry, ens_registry, protocol_registry):
    print("\n" + "=" * 70)
    print("STEP 1: Build prompts")
    print("=" * 70)

    from engine.prompts import create_system_prompt, create_user_prompt

    system_prompt = create_system_prompt(token_registry, ens_registry, protocol_registry)
    user_prompt = create_user_prompt(intent, chain_id)

    print("\n  [System prompt] (first 800 chars):")
    print("  " + "-" * 60)
    print("  " + system_prompt[:800].replace("\n", "\n  ") + "\n  ...")
    print("  " + "-" * 60)
    print(f"  Total system prompt length: {len(system_prompt)} chars")

    print("\n  [User prompt]:")
    print("  " + "-" * 60)
    print("  " + user_prompt.replace("\n", "\n  "))
    print("  " + "-" * 60)

    return system_prompt, user_prompt


# -----------------------------------------------------------------------------
# Step 2: Call LLM
# -----------------------------------------------------------------------------
def step2_call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o"):
    print("\n" + "=" * 70)
    print("STEP 2: Call LLM")
    print("=" * 70)

    from litellm import completion

    print(f"  Model: {model}")
    print("  Calling completion(...) ...")
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
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

    print("\n  [Raw LLM response]:")
    print("  " + "-" * 60)
    print("  " + (response_text or "(empty)").replace("\n", "\n  "))
    print("  " + "-" * 60)
    return response_text


# -----------------------------------------------------------------------------
# Step 3: Parse JSON and normalize
# -----------------------------------------------------------------------------
def step3_parse_json(response_text: str, chain_id: int):
    print("\n" + "=" * 70)
    print("STEP 3: Parse JSON and normalize")
    print("=" * 70)

    if not response_text or response_text.lower() == "null":
        print("  -> Response is null or empty. Abort.")
        return None

    # Strip markdown
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        payload_dict = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  -> JSON parse error: {e}")
        return None

    if payload_dict is None:
        print("  -> Parsed JSON is null.")
        return None

    payload_dict["chain_id"] = chain_id
    print("  [Parsed payload (after adding chain_id)]:")
    print("  " + json.dumps(payload_dict, indent=2).replace("\n", "\n  "))
    return payload_dict


# -----------------------------------------------------------------------------
# Step 4: convert_human_to_payload (always: LLM only classifies, we build in code)
# -----------------------------------------------------------------------------
def step4_convert_human_to_payload(
    payload_dict: dict,
    token_registry,
    protocol_registry,
    ens_registry,
    chain_id: int,
    from_address: str | None = None,
):
    print("\n" + "=" * 70)
    print("STEP 4: convert_human_to_payload (payload builder)")
    print("=" * 70)

    from engine.payload_builder import convert_human_to_payload

    action = payload_dict.get("action")
    print(f"  action: {action}")
    print("  Always run payload builder (LLM only returns intent + human params; we construct everything in code).")
    print("  Calling convert_human_to_payload(..., chain_id, from_address) ...")
    built = convert_human_to_payload(
        payload_dict,
        token_registry,
        protocol_registry,
        ens_registry,
        chain_id=chain_id,
        from_address=from_address,
    )
    if built is None:
        print("  -> convert_human_to_payload returned None (e.g. unknown action/asset or missing args).")
        return None
    print("  [Payload builder output]:")
    print("  " + json.dumps(built, indent=2).replace("\n", "\n  "))
    return built


# -----------------------------------------------------------------------------
# Step 5: ExecutablePayload validation
# -----------------------------------------------------------------------------
def step5_validate_payload(payload_dict: dict):
    print("\n" + "=" * 70)
    print("STEP 5: ExecutablePayload validation")
    print("=" * 70)

    from utils.schemas import ExecutablePayload

    try:
        payload = ExecutablePayload(**payload_dict)
        print("  -> ExecutablePayload(**payload_dict) OK")
        print("  [Final ExecutablePayload (model_dump)]:")
        out = payload.model_dump(mode="json")
        print("  " + json.dumps(out, indent=2).replace("\n", "\n  "))
        return payload
    except Exception as e:
        print(f"  -> Validation error: {e}")
        return None


# -----------------------------------------------------------------------------
# Step 6: AnnotatedIntent
# -----------------------------------------------------------------------------
def step6_annotated_intent(intent: str, chain_id: int, payload):
    print("\n" + "=" * 70)
    print("STEP 6: Build AnnotatedIntent")
    print("=" * 70)

    from utils.schemas import AnnotatedIntent, UserContext

    user_context = UserContext(current_chain_id=chain_id, token_prices={"ETH": 2500.0})
    annotated = AnnotatedIntent(
        user_intent=intent,
        user_context=user_context,
        target_payload=payload,
    )
    print("  AnnotatedIntent created.")
    print("  [Summary]: user_intent, user_context.current_chain_id, target_payload.action + arguments")
    out = annotated.model_dump(mode="json")
    print("  " + json.dumps(out, indent=2).replace("\n", "\n  "))
    return annotated


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    intent = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Could you please facilitate the withdrawal of 15 stETH from Lido?"
    )
    chain_id = 1

    print("\n" + "#" * 70)
    print("# DEMO: Single intent pipeline (every step and output)")
    print("#" * 70)
    print(f"\n  Intent: \"{intent}\"")
    print(f"  Chain ID: {chain_id}")
    print(f"  From address (for DeFi/sender): {DEFAULT_FROM_ADDRESS}")

    token_registry, ens_registry, protocol_registry = step0_load_registries()
    system_prompt, user_prompt = step1_build_prompts(
        intent, chain_id, token_registry, ens_registry, protocol_registry
    )
    response_text = step2_call_llm(system_prompt, user_prompt)
    if not response_text:
        print("\n  Pipeline stopped: no LLM response.")
        return 1

    payload_dict = step3_parse_json(response_text, chain_id)
    if payload_dict is None:
        print("\n  Pipeline stopped: could not parse JSON.")
        return 1

    payload_dict = step4_convert_human_to_payload(
        payload_dict,
        token_registry,
        protocol_registry,
        ens_registry,
        chain_id,
        from_address=DEFAULT_FROM_ADDRESS,
    )
    if payload_dict is None:
        print("\n  Pipeline stopped: payload builder returned None.")
        return 1

    payload = step5_validate_payload(payload_dict)
    if payload is None:
        print("\n  Pipeline stopped: ExecutablePayload validation failed.")
        return 1

    annotated = step6_annotated_intent(intent, chain_id, payload)
    print("\n" + "=" * 70)
    print("DONE. Pipeline succeeded.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
