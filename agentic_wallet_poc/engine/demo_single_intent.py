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

    from engine.token_resolver import TokenResolver
    from engine.ens_resolver import ENSResolver
    token_resolver = TokenResolver()
    ens_resolver = ENSResolver(w3=token_resolver._w3)

    print(f"  token_resolver: {len(token_resolver.known_erc20_symbols())} ERC-20 tokens")
    print(f"  ens_resolver: {ens_resolver.known_names()[:5]}...")
    return token_resolver, ens_resolver


# -----------------------------------------------------------------------------
# Step 1: Build prompts
# -----------------------------------------------------------------------------
def step1_build_prompts(intent: str, chain_id: int, token_resolver, ens_resolver, supported_actions):
    print("\n" + "=" * 70)
    print("STEP 1: Build prompts")
    print("=" * 70)

    from engine.prompts import create_system_prompt, create_user_prompt

    system_prompt = create_system_prompt(token_resolver, ens_resolver, supported_actions=supported_actions)
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
    engine,
    chain_id: int,
    from_address: str | None = None,
):
    print("\n" + "=" * 70)
    print("STEP 4: PlaybookEngine.build_payload (generic playbook engine)")
    print("=" * 70)

    action = payload_dict.get("action")
    print(f"  action: {action}")
    print("  LLM only returns intent + human params; playbook engine resolves everything.")
    built = engine.build_payload(payload_dict, chain_id=chain_id, from_address=from_address)
    if built is None:
        print("  -> PlaybookEngine.build_payload returned None (unknown action/asset or missing args).")
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
# Step 6: Encode to raw transaction (calldata generation)
# -----------------------------------------------------------------------------
def step6_encode_to_raw_tx(payload, engine, from_address: str):
    print("\n" + "=" * 70)
    print("STEP 6: Encode to raw transaction (calldata generation)")
    print("=" * 70)

    payload_dict = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else dict(payload)
    print(f"  action: {payload_dict.get('action')}")
    print(f"  from_address: {from_address}")
    print("  Calling PlaybookEngine.encode_tx(...) ...")

    raw_tx = engine.encode_tx(payload_dict, from_address)
    if raw_tx is None:
        print("  -> payload_to_raw_tx returned None (unsupported action or missing ABI).")
        return None

    print("\n  [Raw Transaction]:")
    print("  " + "-" * 60)
    print(f"    chain_id : {raw_tx.get('chain_id')}")
    print(f"    to       : {raw_tx.get('to')}")
    print(f"    value    : {raw_tx.get('value')} wei")
    data = raw_tx.get("data", "")
    if data and data != "0x":
        print(f"    data     : {data[:10]}...{data[-8:]}  ({(len(data) - 2) // 2} bytes)")
        print(f"    selector : {data[:10]}")
    else:
        print(f"    data     : {data}")
    print("  " + "-" * 60)

    print("\n  [Full raw_tx dict]:")
    print("  " + json.dumps(raw_tx, indent=2).replace("\n", "\n  "))
    return raw_tx


# -----------------------------------------------------------------------------
# Step 7: AnnotatedIntent
# -----------------------------------------------------------------------------
def step7_annotated_intent(intent: str, chain_id: int, payload, raw_tx: dict | None):
    print("\n" + "=" * 70)
    print("STEP 7: Build AnnotatedIntent")
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

    # Attach the raw tx to the output for a complete picture
    if raw_tx:
        out["raw_tx"] = raw_tx
        print("  (raw_tx attached to output)")

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

    token_resolver, ens_resolver = step0_load_registries()

    from engine.playbook_engine import PlaybookEngine
    engine = PlaybookEngine(
        token_resolver=token_resolver,
        ens_resolver=ens_resolver,
    )

    system_prompt, user_prompt = step1_build_prompts(
        intent, chain_id, token_resolver, ens_resolver, engine.get_supported_actions()
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
        engine,
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

    raw_tx = step6_encode_to_raw_tx(payload, engine, from_address=DEFAULT_FROM_ADDRESS)
    if raw_tx is None:
        print("\n  Pipeline stopped: calldata encoding failed.")
        return 1

    annotated = step7_annotated_intent(intent, chain_id, payload, raw_tx)
    print("\n" + "=" * 70)
    print("DONE. Pipeline succeeded — raw transaction ready for signing.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
