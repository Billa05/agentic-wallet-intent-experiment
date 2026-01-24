"""
Baseline Intent Translator for Agentic Wallet Intent Translation System

Rule-based system that extracts parameters from natural language
and converts them to executable blockchain payloads.

Reuses extraction functions from dataset_annotator.py and adds classification logic.
"""

import re
import sys
from pathlib import Path
from typing import Optional

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

# Import extraction functions from dataset_annotator
from data.dataset_annotator import (
    load_token_registry,
    load_ens_registry,
    normalize_address,
    extract_eth_addresses,
    extract_amount,
    extract_token_info,
    extract_nft_info,
    extract_parameters,
    create_executable_payload
)


class ExtractionError(Exception):
    """Exception raised when required parameters cannot be extracted."""
    pass


class BaselineTranslator:
    """
    Rule-based intent translator.
    Converts natural language to ExecutablePayload.
    
    Reuses extraction logic from dataset_annotator and adds classification.
    """
    
    def __init__(self, token_registry_path: str = "data/registries/token_registry.json", 
                 ens_registry_path: str = "data/registries/ens_registry.json"):
        """
        Initialize the baseline translator.
        
        Args:
            token_registry_path: Path to token registry JSON file
            ens_registry_path: Path to ENS registry JSON file
        """
        # Load registries (using annotator functions)
        load_token_registry(token_registry_path)
        load_ens_registry(ens_registry_path)
        
        # NFT collection keywords for classification
        self.nft_keywords = [
            'nft', 'bored ape', 'boredape', 'bayc', 'cryptopunk', 'cryptopunks', 'punk',
            'mutant ape', 'mutantape', 'mayc', 'azuki', 'doodles', 'doodle'
        ]
    
    def _classify_intent(self, text: str) -> Optional[TransactionType]:
        """
        Classify transaction type from text.
        
        Priority:
        1. NFT keywords (highest priority)
        2. ERC-20 token keywords
        3. ETH keywords
        
        Args:
            text: Natural language intent text
            
        Returns:
            TransactionType or None if can't classify
        """
        text_lower = text.lower()
        
        # Check for NFT keywords first
        for keyword in self.nft_keywords:
            if keyword in text_lower:
                return TransactionType.TRANSFER_ERC721
        
        # Check for ERC-20 tokens (but not ETH)
        # First check for specific token symbols (USDC, USDT, DAI, WETH)
        if re.search(r'\b(usdc|usdt|dai|weth)\b', text_lower):
            return TransactionType.TRANSFER_ERC20
        
        # Check for generic "token" or "tokens" (but not ETH)
        # Only match if "token" appears and "eth"/"ether" doesn't appear
        if re.search(r'\b(token|tokens)\b', text_lower):
            if not re.search(r'\b(eth|ether)\b', text_lower):
                return TransactionType.TRANSFER_ERC20
        
        # Check for ETH
        if re.search(r'\b(eth|ether)\b', text_lower):
            return TransactionType.SEND_ETH
        
        return None
    
    
    def translate(self, user_intent: str, chain_id: int = 1) -> Optional[AnnotatedIntent]:
        """
        Main entry point. Translates natural language intent to AnnotatedIntent.
        
        Args:
            user_intent: Natural language transaction intent
            chain_id: Chain ID (default: 1 = Ethereum Mainnet)
            
        Returns:
            AnnotatedIntent if successful, None if translation failed
        """
        try:
            # Step 1: Classify intent (unique to translator)
            tx_type = self._classify_intent(user_intent)
            if tx_type is None:
                return None
            
            # Step 2: Extract parameters using annotator functions
            params = extract_parameters(user_intent, tx_type)
            
            # Step 3: Validate address extraction (only use registry, no fallback)
            # If address not found or not in registry, translation fails
            if not params.get('to_address'):
                return None
            
            # Step 4: Validate required parameters
            if tx_type == TransactionType.SEND_ETH:
                if not params.get('amount'):
                    return None
            elif tx_type == TransactionType.TRANSFER_ERC20:
                if not params.get('amount') or not params.get('token_address'):
                    return None
            elif tx_type == TransactionType.TRANSFER_ERC721:
                if not params.get('contract_address') or params.get('token_id') is None:
                    return None
            
            # Step 5: Create executable payload using annotator function
            params['_intent'] = user_intent  # Store intent for token symbol detection
            payload = create_executable_payload(tx_type, params, chain_id)
            
            if not payload:
                return None
            
            # Step 6: Create user context
            user_context = UserContext(
                current_chain_id=chain_id,
                token_prices={"ETH": 2500.0}  # Default price
            )
            
            # Step 7: Create annotated intent
            annotated = AnnotatedIntent(
                user_intent=user_intent,
                user_context=user_context,
                target_payload=payload
            )
            
            return annotated
            
        except ExtractionError:
            return None
        except Exception as e:
            print(f"Error translating intent: {e}")
            return None


if __name__ == "__main__":
    translator = BaselineTranslator()
    
    test_intents = [
        "send 0.5 eth to alice.eth",
        "transfer 100 USDC to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        "send my bored ape #1234 to bob.eth",
    ]
    
    for intent in test_intents:
        result = translator.translate(intent)
        print(f"\nIntent: {intent}")
        if result:
            print(f"Action: {result.target_payload.action}")
            print(f"Arguments: {result.target_payload.arguments}")
        else:
            print("Failed to translate")
