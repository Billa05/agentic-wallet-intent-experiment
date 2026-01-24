"""
LLM-based Intent Translator for Agentic Wallet Intent Translation System

Uses Google Gemini API to translate natural language intents into
executable blockchain transactions with structured output.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from google import genai
from dotenv import load_dotenv
import os
# Removed eth_utils - testing raw LLM output without post-processing

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schemas import (
    TransactionType,
    ExecutablePayload,
    ActionType,
    AnnotatedIntent,
    UserContext
)
from models.prompts import create_system_prompt, create_user_prompt

# Load environment variables
load_dotenv()

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLIENT = None
if GEMINI_API_KEY:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)


class LLMTranslator:
    """
    LLM-based intent translator using Google Gemini.
    Converts natural language to ExecutablePayload using structured output.
    """
    
    def __init__(self, token_registry_path: str = "data/registries/token_registry.json", ens_registry_path: str = "data/registries/ens_registry.json", model_name: str = "gemini-2.5-flash"):
        """
        Initialize the LLM translator.
        
        Args:
            token_registry_path: Path to token registry JSON file
            ens_registry_path: Path to ENS registry JSON file
            model_name: Gemini model name to use
        """
        self.model_name = model_name
        self.token_registry = self._load_token_registry(token_registry_path)
        self.ens_registry = self._load_ens_registry(ens_registry_path)
        
        if not GEMINI_CLIENT:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    def _load_token_registry(self, registry_path: str) -> Dict[str, Any]:
        """Load token registry from JSON file."""
        if not Path(registry_path).is_absolute():
            registry_path = project_root / registry_path
        
        registry_path = Path(registry_path)
        
        if not registry_path.exists():
            return {"erc20_tokens": {}, "erc721_collections": {}}
        
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
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
            # Create prompts
            system_prompt = create_system_prompt(self.token_registry, self.ens_registry)
            user_prompt = create_user_prompt(user_intent, chain_id)
            
            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Call Gemini API (matching dataset_generator format)
            try:
                response = GEMINI_CLIENT.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt
                )
            except Exception as e:
                # Log API errors for debugging
                import sys
                print(f"DEBUG: Gemini API error for intent '{user_intent[:50]}...': {e}", file=sys.stderr)
                return None
            
            # Extract JSON from response (matching dataset_generator format)
            response_text = None
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content'):
                    if hasattr(candidate.content, 'parts') and len(candidate.content.parts) > 0:
                        response_text = candidate.content.parts[0].text.strip()
                    elif hasattr(candidate.content, 'text'):
                        response_text = candidate.content.text.strip()
            
            if not response_text:
                # Fallback: try to extract from string representation
                response_str = str(response)
                import re
                json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', response_str, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0).strip()
            
            if not response_text:
                import sys
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
                import sys
                print(f"DEBUG: JSON parse error for intent '{user_intent[:50]}...': {e}", file=sys.stderr)
                print(f"DEBUG: Response text: {response_text[:200]}...", file=sys.stderr)
                return None
            
            # Validate and create ExecutablePayload
            if payload_dict is None:
                return None
            
            # Ensure chain_id is set
            payload_dict["chain_id"] = chain_id
            
            # No post-processing - accept LLM output as-is for pure capability testing
            # The LLM should output correct addresses from the registry based on the prompt
            # We only validate that the JSON structure is correct, not the content
            
            # Create ExecutablePayload
            try:
                payload = ExecutablePayload(**payload_dict)
            except Exception as e:
                # Log validation errors for debugging
                import sys
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
            
        except Exception:
            # Silently fail - errors will be caught in evaluation
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
