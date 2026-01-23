"""
LLM-based Intent Translator for Agentic Wallet Intent Translation System

Uses Google Gemini API to translate natural language intents into
executable blockchain transactions with structured output.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from decimal import Decimal, ROUND_DOWN

from google import genai
from dotenv import load_dotenv
import os

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
    
    def __init__(self, token_registry_path: str = "data/token_registry.json", model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the LLM translator.
        
        Args:
            token_registry_path: Path to token registry JSON file
            model_name: Gemini model name to use
        """
        self.model_name = model_name
        self.token_registry = self._load_token_registry(token_registry_path)
        
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
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with schema and token registry information."""
        # Build token registry info
        tokens_info = []
        for symbol, info in self.token_registry.get("erc20_tokens", {}).items():
            tokens_info.append(f"- {symbol}: {info['address']} (decimals: {info['decimals']})")
        
        collections_info = []
        for name, info in self.token_registry.get("erc721_collections", {}).items():
            collections_info.append(f"- {name}: {info['address']}")
        
        return f"""You are a blockchain transaction translator. Convert natural language intents into executable blockchain transactions.

CRITICAL REQUIREMENTS:
1. All amounts must be in Wei/base units (integers as strings)
2. ETH: Multiply by 10^18 (e.g., 0.5 ETH = "500000000000000000")
3. ERC-20 tokens: Multiply by 10^decimals (USDC/USDT = 6, DAI/WETH = 18)
4. All addresses must be checksummed (EIP-55 format)
5. Chain ID is always 1 (Ethereum Mainnet)

TOKEN REGISTRY:
ERC-20 Tokens:
{chr(10).join(tokens_info) if tokens_info else "- None"}

ERC-721 Collections:
{chr(10).join(collections_info) if collections_info else "- None"}

OUTPUT SCHEMA:
{{
  "action": "transfer_native" | "transfer_erc20" | "transfer_erc721",
  "target_contract": null | "0x...",  // null for native ETH, address for tokens/NFTs
  "function_name": null | "transfer" | "transferFrom",
  "arguments": {{
    "to": "0x...",  // checksummed recipient address
    "value": "1234567890",  // Wei/base units as STRING
    "human_readable_amount": "0.5 ETH"  // for UI display
  }}
}}

For ERC-721, also include:
  "arguments": {{
    "to": "0x...",
    "tokenId": 1234,
    "human_readable_amount": "Token #1234"
  }}

IMPORTANT:
- If intent is ambiguous or missing required info, return null
- For ENS names (e.g., "alice.eth"), generate a deterministic address using SHA256 hash
- Always validate addresses are 0x + 40 hex characters
- Amounts must be exact - no rounding errors"""
    
    def _create_user_prompt(self, intent: str, chain_id: int = 1) -> str:
        """Create user prompt with the intent to translate."""
        return f"""Translate this user intent into a blockchain transaction:

Intent: "{intent}"
Chain ID: {chain_id} (Ethereum Mainnet)

Return ONLY valid JSON matching the schema above. If the intent cannot be translated, return null."""
    
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
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(user_intent, chain_id)
            
            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Call Gemini API (matching dataset_generator format)
            try:
                response = GEMINI_CLIENT.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt
                )
            except Exception as e:
                # Silently fail - errors will be caught in evaluation
                return None
            
            # Extract JSON from response (matching dataset_generator format)
            response_text = None
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'content'):
                    if hasattr(response.candidates[0].content, 'parts'):
                        if response.candidates[0].content.parts:
                            response_text = response.candidates[0].content.parts[0].text
                    elif hasattr(response.candidates[0].content, 'text'):
                        response_text = response.candidates[0].content.text
            
            if not response_text:
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
            except json.JSONDecodeError:
                # Silently fail - errors will be caught in evaluation
                return None
            
            # Validate and create ExecutablePayload
            if payload_dict is None:
                return None
            
            # Ensure chain_id is set
            payload_dict["chain_id"] = chain_id
            
            # Create ExecutablePayload
            try:
                payload = ExecutablePayload(**payload_dict)
            except Exception:
                # Silently fail - errors will be caught in evaluation
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
