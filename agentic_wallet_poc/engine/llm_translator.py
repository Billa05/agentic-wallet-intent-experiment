"""
LLM-based Intent Translator for Agentic Wallet Intent Translation System

Uses litellm to support multiple LLM providers (OpenAI, Anthropic, Google, etc.)
to translate natural language intents into executable blockchain transactions.
"""

import json
import sys
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, MutableMapping

from litellm import completion
from dotenv import load_dotenv
import os
# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schemas import (
    ExecutablePayload,
    AnnotatedIntent,
    UserContext,
)
from engine.prompts import create_system_prompt, create_user_prompt
from engine.payload_builder import convert_human_to_payload

load_dotenv()


class LLMTranslator:
    """
    LLM-based intent translator using litellm (supports multiple providers).
    Converts natural language to ExecutablePayload using structured output.
    
    Supported providers: Groq, OpenAI, Anthropic, Google, Cohere, etc.
    Model format: "provider/model-name" (e.g., "groq/llama-3.1-8b-instant", "gpt-4o", "claude-3-5-sonnet-20241022")
    """
    
    def __init__(
        self, 
        token_registry_path: str = "data/registries/token_registry.json", 
        ens_registry_path: str = "data/registries/ens_registry.json", 
        protocol_registry_path: str = "data/registries/protocol_registry.json",
        model: str = "gpt-4o"
    ):
        """
        Initialize the LLM translator.
        
        Args:
            token_registry_path: Path to token registry JSON file
            ens_registry_path: Path to ENS registry JSON file
            protocol_registry_path: Path to protocol registry (DeFi contracts)
            model: Model identifier (litellm format)
        """
        self.model = model
        self.token_registry = self._load_token_registry(token_registry_path)
        self.ens_registry = self._load_ens_registry(ens_registry_path)
        self.protocol_registry = self._load_protocol_registry(protocol_registry_path)
        
        self._validate_api_keys()
    
    def _load_token_registry(self, registry_path: str) -> Dict[str, Any]:
        """Load token registry from JSON file."""
        if not Path(registry_path).is_absolute():
            registry_path = project_root / registry_path
        registry_path = Path(registry_path)
        if not registry_path.exists():
            return {"erc20_tokens": {}, "erc721_collections": {}}
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _validate_api_keys(self) -> None:
        """Validate that at least one LLM provider API key is set."""
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "google": os.getenv("GEMINI_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
        }
        model_lower = self.model.lower()
        if model_lower.startswith("groq/"):
            if not api_keys["groq"]:
                raise ValueError("GROQ_API_KEY not found in environment variables")
        elif model_lower.startswith("gpt") or model_lower.startswith("o1"):
            if not api_keys["openai"]:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        elif model_lower.startswith("claude"):
            if not api_keys["anthropic"]:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        elif model_lower.startswith("gemini"):
            if not api_keys["google"]:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables")
        elif model_lower.startswith("command") or model_lower.startswith("cohere"):
            if not api_keys["cohere"]:
                raise ValueError("COHERE_API_KEY not found in environment variables")
        else:
            print(f"WARNING: Unknown model '{self.model}'. Make sure appropriate API key is set.", file=sys.stderr)
    
    def _load_ens_registry(self, registry_path: str) -> Dict[str, str]:
        """Load ENS registry from JSON file."""
        if not Path(registry_path).is_absolute():
            registry_path = project_root / registry_path
        registry_path = Path(registry_path)
        if not registry_path.exists():
            return {}
        with open(registry_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("ens_names", {})
    
    def _load_protocol_registry(self, registry_path: str) -> Dict[str, Any]:
        """Load protocol registry (DeFi contracts) from JSON file."""
        if not Path(registry_path).is_absolute():
            registry_path = project_root / registry_path
        registry_path = Path(registry_path)
        if not registry_path.exists():
            return {}
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def translate(
        self,
        user_intent: str,
        chain_id: int = 1,
        failure_info: Optional[MutableMapping[str, Any]] = None,
    ) -> Optional[AnnotatedIntent]:
        """
        Main entry point. Translates natural language intent to AnnotatedIntent using LLM.
        If translation fails and failure_info is provided (e.g. a dict), it is populated with
        stage, message, and raw_llm_response (when available) for diagnostics.
        """
        def set_fail(stage: str, message: str, raw: Optional[str] = None) -> None:
            if failure_info is not None:
                failure_info["stage"] = stage
                failure_info["message"] = message
                if raw is not None:
                    failure_info["raw_llm_response"] = raw[:4000] if len(raw) > 4000 else raw

        try:
            system_prompt = create_system_prompt(
                self.token_registry, self.ens_registry, self.protocol_registry
            )
            user_prompt = create_user_prompt(user_intent, chain_id)
            
            max_retries = 3
            response_text = None
            
            for attempt in range(max_retries):
                try:
                    response = completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                    )
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        choice = response.choices[0]
                        if hasattr(choice, 'message'):
                            response_text = choice.message.content
                        elif hasattr(choice, 'text'):
                            response_text = choice.text
                    if response_text:
                        response_text = response_text.strip()
                        break
                    else:
                        set_fail("llm_no_response", "LLM returned no or empty content.")
                        return None
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = (
                        "429" in error_str or "rate_limit" in error_str.lower() or
                        "quota" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str or
                        "RateLimitError" in type(e).__name__
                    )
                    if is_rate_limit and attempt < max_retries - 1:
                        retry_seconds = 60
                        delay_match = re.search(r'retry[_\s]?after[_\s]?(\d+(?:\.\d+)?)\s*(?:s|seconds?)?', error_str, re.IGNORECASE)
                        if delay_match:
                            retry_seconds = float(delay_match.group(1)) + 2
                        print(f"DEBUG: Rate limit hit. Waiting {retry_seconds:.1f}s...", file=sys.stderr)
                        time.sleep(retry_seconds)
                        continue
                    else:
                        print(f"DEBUG: LLM API error: {e}", file=sys.stderr)
                        set_fail("llm_api_error", str(e))
                        return None
            
            if not response_text:
                set_fail("llm_no_response", "No response text after retries.")
                return None
            
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            if response_text.lower() == "null" or response_text == "":
                set_fail("llm_empty_or_null", "LLM response was null or empty.", raw=response_text)
                return None
            
            try:
                payload_dict = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON parse error: {e}", file=sys.stderr)
                set_fail("llm_json_parse", str(e), raw=response_text)
                return None
            
            if payload_dict is None:
                set_fail("llm_empty_or_null", "Parsed JSON was null.", raw=response_text)
                return None
            
            payload_dict["chain_id"] = chain_id
            
            # LLM only classifies + extracts human params; we always build the full payload in code
            built = convert_human_to_payload(
                payload_dict,
                self.token_registry,
                self.protocol_registry,
                self.ens_registry,
                chain_id=chain_id,
            )
            if built is None:
                set_fail("payload_builder", "convert_human_to_payload returned null (e.g. unknown action/asset or missing args).", raw=response_text)
                return None
            payload_dict = built
            
            try:
                payload = ExecutablePayload(**payload_dict)
            except Exception as e:
                print(f"DEBUG: ExecutablePayload validation error: {e}", file=sys.stderr)
                set_fail("payload_validation", str(e), raw=response_text)
                return None
            
            user_context = UserContext(current_chain_id=chain_id, token_prices={"ETH": 2500.0})
            annotated = AnnotatedIntent(user_intent=user_intent, user_context=user_context, target_payload=payload)
            return annotated
            
        except Exception as e:
            print(f"DEBUG: Unexpected error in translate(): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            set_fail("unexpected", str(e))
            return None


if __name__ == "__main__":
    translator = LLMTranslator()
    for intent in ["send 0.5 eth to alice.eth", "Supply 100 USDC to AAVE"]:
        result = translator.translate(intent)
        print(f"Intent: {intent} -> {'OK' if result else 'FAIL'}")
