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
from typing import Optional, Dict, Any

from litellm import completion
from dotenv import load_dotenv
import os
# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schemas import (
    ExecutablePayload,
    AnnotatedIntent,
    UserContext
)
from models.prompts import create_system_prompt, create_user_prompt

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
        model: str = "groq/llama-3.1-8b-instant"
    ):
        """
        Initialize the LLM translator.
        
        Args:
            token_registry_path: Path to token registry JSON file
            ens_registry_path: Path to ENS registry JSON file
            model: Model identifier (litellm format)
                   Examples:
                   - "groq/llama-3.1-8b-instant" (Groq - fast, default)
                   - "groq/llama-3.1-70b-versatile" (Groq - more capable)
                   - "gpt-4o" (OpenAI)
                   - "claude-3-5-sonnet-20241022" (Anthropic)
                   - "gemini/gemini-2.0-flash-exp" (Google)
                   - "gpt-3.5-turbo" (OpenAI)
        """
        self.model = model
        self.token_registry = self._load_token_registry(token_registry_path)
        self.ens_registry = self._load_ens_registry(ens_registry_path)
        
        # Validate that at least one API key is set
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
        
        # Check if model provider has API key
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
            # For unknown models, just warn
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
    
    
    def translate(self, user_intent: str, chain_id: int = 1) -> Optional[AnnotatedIntent]:
        """
        Main entry point. Translates natural language intent to AnnotatedIntent using LLM.
        
        Args:
            user_intent: Natural language transaction intent
            chain_id: Chain ID (default: 1 = Ethereum Mainnet)
            
        Returns:
            AnnotatedIntent if successful, None if translation failed
        """
        try:
            # Create prompts with token registry and ENS registry
            system_prompt = create_system_prompt(self.token_registry, self.ens_registry)
            user_prompt = create_user_prompt(user_intent, chain_id)
            
            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Call LLM API using litellm with retry logic for rate limits
            max_retries = 3
            response_text = None
            
            for attempt in range(max_retries):
                try:
                    # Use litellm's unified completion interface
                    response = completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,  # Low temperature for consistent, structured output
                    )
                    
                    # Extract response text (litellm standardizes response format)
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        choice = response.choices[0]
                        if hasattr(choice, 'message'):
                            response_text = choice.message.content
                        elif hasattr(choice, 'text'):
                            response_text = choice.text
                    
                    if response_text:
                        response_text = response_text.strip()
                        break  # Success, exit retry loop
                    else:
                        print(f"DEBUG: Empty response from LLM for intent '{user_intent[:50]}...'", file=sys.stderr)
                        return None
                        
                except Exception as e:
                    error_str = str(e)
                    error_type = type(e).__name__
                    
                    # Check if it's a rate limit error (429 or rate limit related)
                    is_rate_limit = (
                        "429" in error_str or 
                        "rate_limit" in error_str.lower() or 
                        "quota" in error_str.lower() or
                        "RESOURCE_EXHAUSTED" in error_str or
                        "RateLimitError" in error_type
                    )
                    
                    if is_rate_limit and attempt < max_retries - 1:
                        # Extract retry delay from error message if available
                        retry_seconds = 60  # Default: wait 60 seconds
                        
                        # Try to extract suggested delay from error message
                        delay_match = re.search(r'retry[_\s]?after[_\s]?(\d+(?:\.\d+)?)\s*(?:s|seconds?)?', error_str, re.IGNORECASE)
                        if delay_match:
                            retry_seconds = float(delay_match.group(1)) + 2  # Add 2 second buffer
                        else:
                            # Try other patterns
                            delay_match = re.search(r'retry in (\d+(?:\.\d+)?)\s*s', error_str, re.IGNORECASE)
                            if delay_match:
                                retry_seconds = float(delay_match.group(1)) + 2
                            else:
                                delay_match = re.search(r'retrydelay["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)', error_str, re.IGNORECASE)
                                if delay_match:
                                    retry_seconds = float(delay_match.group(1)) + 2
                        
                        print(f"DEBUG: Rate limit hit for intent '{user_intent[:50]}...'. Waiting {retry_seconds:.1f}s before retry {attempt + 1}/{max_retries}...", file=sys.stderr)
                        time.sleep(retry_seconds)
                        continue
                    else:
                        # Non-rate-limit error or max retries reached
                        if is_rate_limit:
                            print(f"DEBUG: Rate limit exceeded after {max_retries} attempts for intent '{user_intent[:50]}...'", file=sys.stderr)
                        else:
                            print(f"DEBUG: LLM API error for intent '{user_intent[:50]}...': {error_type}: {e}", file=sys.stderr)
                        return None
            
            if not response_text:
                print(f"DEBUG: No response text extracted for intent '{user_intent[:50]}...'", file=sys.stderr)
                return None
            
            # Parse JSON response
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Handle null response
            if response_text.lower() == "null" or response_text == "":
                return None
            
            # Parse JSON
            try:
                payload_dict = json.loads(response_text)
            except json.JSONDecodeError as e:
                # Log JSON parsing errors for debugging
                print(f"DEBUG: JSON parse error for intent '{user_intent[:50]}...': {e}", file=sys.stderr)
                print(f"DEBUG: Response text: {response_text[:200]}...", file=sys.stderr)
                return None
            
            # Validate and create ExecutablePayload
            if payload_dict is None:
                return None
            
            # Ensure chain_id is set
            payload_dict["chain_id"] = chain_id
            
            # Create ExecutablePayload
            try:
                payload = ExecutablePayload(**payload_dict)
            except Exception as e:
                # Log validation errors for debugging
                print(f"DEBUG: ExecutablePayload validation error for intent '{user_intent[:50]}...': {e}", file=sys.stderr)
                print(f"DEBUG: Payload dict: {json.dumps(payload_dict, indent=2)[:500]}...", file=sys.stderr)
                return None
            
            # Create user context
            user_context = UserContext(
                current_chain_id=chain_id,
                token_prices={"ETH": 2500.0}  # Default price
            )
            
            # Create annotated intent
            annotated = AnnotatedIntent(
                user_intent=user_intent,
                user_context=user_context,
                target_payload=payload
            )
            
            return annotated
            
        except Exception as e:
            # Log unexpected errors for debugging
            print(f"DEBUG: Unexpected error in translate() for intent '{user_intent[:50]}...': {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get JSON schema for structured output."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["transfer_native", "transfer_erc20", "transfer_erc721"]
                },
                "target_contract": {
                    "type": ["string", "null"],
                    "description": "Contract address (null for native ETH)"
                },
                "function_name": {
                    "type": ["string", "null"],
                    "description": "Function name (null for native ETH)"
                },
                "arguments": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient address (checksummed)"
                        },
                        "value": {
                            "type": "string",
                            "description": "Amount in Wei/base units"
                        },
                        "human_readable_amount": {
                            "type": "string",
                            "description": "Human-readable amount for UI"
                        },
                        "tokenId": {
                            "type": "integer",
                            "description": "NFT token ID (for ERC-721)"
                        }
                    },
                    "required": ["to", "value", "human_readable_amount"]
                }
            },
            "required": ["action", "arguments"]
        }


if __name__ == "__main__":
    # Test the LLM translator
    translator = LLMTranslator()
    
    test_intents = [
        "send 0.5 eth to alice.eth",
        "transfer 100 USDC to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        "send my bored ape #1234 to bob.eth",
    ]
    
    print("LLM Translator Test")
    print("=" * 60)
    
    for intent in test_intents:
        print(f"\nIntent: {intent}")
        result = translator.translate(intent)
        if result:
            print(f"✓ Success")
            print(f"  Action: {result.target_payload.action}")
            print(f"  Target Contract: {result.target_payload.target_contract}")
            print(f"  Arguments: {result.target_payload.arguments}")
        else:
            print("✗ Failed to translate")
