"""
Baseline Intent Translator for Agentic Wallet Intent Translation System

Rule-based system that extracts parameters from natural language
and converts them to executable blockchain payloads.
"""

import json
import re
import hashlib
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from decimal import Decimal, ROUND_DOWN

from eth_utils import to_checksum_address, is_address
from web3 import Web3

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


class ExtractionError(Exception):
    """Exception raised when required parameters cannot be extracted."""
    pass


class BaselineTranslator:
    """
    Rule-based intent translator.
    Converts natural language to ExecutablePayload.
    """
    
    def __init__(self, token_registry_path: str = "data/token_registry.json"):
        """
        Initialize the baseline translator.
        
        Args:
            token_registry_path: Path to token registry JSON file
        """
        # Load token registry
        self.token_registry = self._load_token_registry(token_registry_path)
        
        # Compile regex patterns
        self.patterns = {
            # Ethereum address: 0x followed by 40 hex characters
            'eth_address': re.compile(r'0x[a-fA-F0-9]{40}'),
            
            # ENS name: word.eth pattern
            'ens_name': re.compile(r'\b[a-z0-9]+\.eth\b', re.IGNORECASE),
            
            # ETH amount: number followed by eth/ether
            'eth_amount': re.compile(r'(\d+\.?\d*)\s*(?:eth|ether)', re.IGNORECASE),
            
            # Token amount: number followed by token symbol
            'token_amount': re.compile(r'(\d+\.?\d*)\s*(usdc|usdt|dai|weth|token|tokens)', re.IGNORECASE),
            
            # NFT ID: #1234 or "token id 1234" patterns
            'nft_id': re.compile(r'(?:#|token\s+id\s+)(\d+)', re.IGNORECASE),
        }
        
        # NFT collection keywords
        self.nft_keywords = [
            'nft', 'bored ape', 'boredape', 'bayc', 'cryptopunk', 'cryptopunks', 'punk',
            'mutant ape', 'mutantape', 'mayc', 'azuki', 'doodles', 'doodle'
        ]
        
        # ERC-20 token keywords
        self.erc20_keywords = ['usdc', 'usdt', 'dai', 'weth', 'token', 'tokens']
    
    def _load_token_registry(self, registry_path: str) -> Dict[str, Any]:
        """
        Load token registry from JSON file.
        
        Args:
            registry_path: Path to token registry file
            
        Returns:
            Dictionary with token registry data
        """
        # Handle relative paths
        if not Path(registry_path).is_absolute():
            registry_path = project_root / registry_path
        
        registry_path = Path(registry_path)
        
        if not registry_path.exists():
            print(f"Warning: Token registry not found at {registry_path}")
            return {"erc20_tokens": {}, "erc721_collections": {}}
        
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
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
            # Classify intent
            tx_type = self._classify_intent(user_intent)
            if tx_type is None:
                return None
            
            # Extract payload based on type
            if tx_type == TransactionType.SEND_ETH:
                payload = self._extract_eth_transfer(user_intent)
            elif tx_type == TransactionType.TRANSFER_ERC20:
                payload = self._extract_erc20_transfer(user_intent)
            elif tx_type == TransactionType.TRANSFER_ERC721:
                payload = self._extract_erc721_transfer(user_intent)
            else:
                return None
            
            # Set chain_id
            payload.chain_id = chain_id
            
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
            
        except ExtractionError:
            return None
        except Exception as e:
            print(f"Error translating intent: {e}")
            return None
    
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
    
    def _extract_address(self, text: str) -> str:
        """
        Extract and normalize Ethereum address from text.
        
        Priority:
        1. 0x address
        2. ENS name (converted to deterministic address)
        
        Args:
            text: Text to search for address
            
        Returns:
            Checksummed Ethereum address
            
        Raises:
            ExtractionError: If no address found
        """
        # Try to find 0x address first
        eth_match = self.patterns['eth_address'].search(text)
        if eth_match:
            address = eth_match.group(0)
            if is_address(address):
                return to_checksum_address(address)
        
        # Try to find ENS name
        ens_match = self.patterns['ens_name'].search(text)
        if ens_match:
            ens_name = ens_match.group(0).lower()
            # Generate deterministic address from ENS name
            return self._ens_to_address(ens_name)
        
        # Try to find placeholder names (e.g., "to alice", "to bob")
        placeholder_match = re.search(r'\bto\s+([a-z]+)\b', text.lower())
        if placeholder_match:
            name = placeholder_match.group(1)
            # Skip common words that aren't names
            if name not in ['the', 'my', 'your', 'their', 'our', 'this', 'that']:
                return self._ens_to_address(f"{name}.eth")
        
        raise ExtractionError("No valid address found in intent")
    
    def _ens_to_address(self, ens_name: str) -> str:
        """
        Convert ENS name to deterministic Ethereum address.
        
        Uses SHA256 hash of the name to generate a deterministic address.
        
        Args:
            ens_name: ENS name (e.g., "alice.eth")
            
        Returns:
            Checksummed Ethereum address
        """
        # Remove .eth if present for hashing
        name_clean = ens_name.replace('.eth', '').lower()
        
        # Generate deterministic address using SHA256
        hash_obj = hashlib.sha256(name_clean.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Use first 40 chars of hash as address
        address = "0x" + hash_hex[:40]
        
        # Ensure it's a valid address format
        if is_address(address):
            return to_checksum_address(address)
        else:
            # Fallback: use Web3 to generate from hash
            w3 = Web3()
            hash_bytes = w3.keccak(text=name_clean)
            address = "0x" + hash_bytes.hex()[:40]
            return to_checksum_address(address)
    
    def _extract_eth_amount(self, text: str) -> Tuple[Decimal, str]:
        """
        Extract ETH amount from text.
        
        Args:
            text: Text to search for amount
            
        Returns:
            Tuple of (amount_decimal, amount_string)
            
        Raises:
            ExtractionError: If no amount found
        """
        match = self.patterns['eth_amount'].search(text)
        if match:
            amount_str = match.group(1)
            try:
                amount = Decimal(amount_str)
                if amount <= 0:
                    raise ExtractionError("ETH amount must be positive")
                return amount, amount_str
            except (ValueError, Exception):
                raise ExtractionError(f"Invalid ETH amount: {amount_str}")
        
        raise ExtractionError("No ETH amount found in intent")
    
    def _extract_token_info(self, text: str) -> Tuple[str, Decimal, int]:
        """
        Extract token symbol, amount, and decimals from text.
        
        Args:
            text: Text to search for token information
            
        Returns:
            Tuple of (token_symbol, amount_decimal, decimals)
            
        Raises:
            ExtractionError: If token info cannot be extracted
        """
        match = self.patterns['token_amount'].search(text)
        if not match:
            raise ExtractionError("No token amount found in intent")
        
        amount_str = match.group(1)
        token_symbol = match.group(2).upper()
        
        # Handle generic "token" or "tokens" - try to infer from context
        if token_symbol in ['TOKEN', 'TOKENS']:
            # Try to find specific token in text
            for symbol in ['USDC', 'USDT', 'DAI', 'WETH']:
                if symbol.lower() in text.lower():
                    token_symbol = symbol
                    break
            else:
                raise ExtractionError("Cannot determine token type")
        
        # Lookup token in registry
        tokens = self.token_registry.get("erc20_tokens", {})
        token_info = tokens.get(token_symbol)
        
        if not token_info:
            raise ExtractionError(f"Token {token_symbol} not found in registry")
        
        try:
            amount = Decimal(amount_str)
            if amount <= 0:
                raise ExtractionError("Token amount must be positive")
            decimals = token_info.get("decimals", 18)
            return token_symbol, amount, decimals
        except (ValueError, Exception) as e:
            raise ExtractionError(f"Invalid token amount: {amount_str}")
    
    def _extract_nft_info(self, text: str) -> Tuple[str, int]:
        """
        Extract NFT collection address and token ID from text.
        
        Args:
            text: Text to search for NFT information
            
        Returns:
            Tuple of (collection_address, token_id)
            
        Raises:
            ExtractionError: If NFT info cannot be extracted
        """
        # Extract token ID
        nft_id_match = self.patterns['nft_id'].search(text)
        if not nft_id_match:
            raise ExtractionError("No NFT token ID found in intent")
        
        token_id = int(nft_id_match.group(1))
        
        # Find collection name
        text_lower = text.lower()
        collections = self.token_registry.get("erc721_collections", {})
        
        # Try to match collection name
        collection_info = None
        for collection_key, info in collections.items():
            # Check if collection name or alias appears in text
            if collection_key in text_lower or info.get("name", "").lower() in text_lower:
                collection_info = info
                break
        
        if not collection_info:
            raise ExtractionError("NFT collection not found in registry")
        
        collection_address = collection_info.get("address")
        if not collection_address:
            raise ExtractionError("Collection address not found")
        
        return collection_address, token_id
    
    def _extract_eth_transfer(self, text: str) -> ExecutablePayload:
        """
        Extract ETH transfer parameters.
        
        Args:
            text: Natural language intent text
            
        Returns:
            ExecutablePayload for ETH transfer
            
        Raises:
            ExtractionError: If required parameters cannot be extracted
        """
        # Extract address
        to_address = self._extract_address(text)
        
        # Extract amount
        amount_decimal, amount_str = self._extract_eth_amount(text)
        
        # Convert to Wei
        wei_amount = (amount_decimal * Decimal(10 ** 18)).to_integral_value(rounding=ROUND_DOWN)
        wei_str = str(int(wei_amount))
        
        # Create payload
        payload = ExecutablePayload(
            chain_id=1,  # Will be overridden by translate()
            action=ActionType.TRANSFER_NATIVE,
            target_contract=None,
            function_name=None,
            arguments={
                "to": to_address,
                "value": wei_str,
                "human_readable_amount": f"{amount_str} ETH"
            }
        )
        
        return payload
    
    def _extract_erc20_transfer(self, text: str) -> ExecutablePayload:
        """
        Extract ERC-20 transfer parameters.
        
        Args:
            text: Natural language intent text
            
        Returns:
            ExecutablePayload for ERC-20 transfer
            
        Raises:
            ExtractionError: If required parameters cannot be extracted
        """
        # Extract address
        to_address = self._extract_address(text)
        
        # Extract token info
        token_symbol, amount_decimal, decimals = self._extract_token_info(text)
        
        # Get token address from registry
        tokens = self.token_registry.get("erc20_tokens", {})
        token_info = tokens.get(token_symbol)
        if not token_info:
            raise ExtractionError(f"Token {token_symbol} not found in registry")
        
        token_address = token_info.get("address")
        if not token_address:
            raise ExtractionError(f"Token address not found for {token_symbol}")
        
        # Convert to base units
        scale = Decimal(10) ** decimals
        base_units = (amount_decimal * scale).to_integral_value(rounding=ROUND_DOWN)
        base_units_str = str(int(base_units))
        
        # Create payload
        payload = ExecutablePayload(
            chain_id=1,  # Will be overridden by translate()
            action=ActionType.TRANSFER_ERC20,
            target_contract=token_address,
            function_name="transfer",
            arguments={
                "to": to_address,
                "value": base_units_str,
                "human_readable_amount": f"{amount_decimal} {token_symbol}"
            }
        )
        
        return payload
    
    def _extract_erc721_transfer(self, text: str) -> ExecutablePayload:
        """
        Extract ERC-721 transfer parameters.
        
        Args:
            text: Natural language intent text
            
        Returns:
            ExecutablePayload for ERC-721 transfer
            
        Raises:
            ExtractionError: If required parameters cannot be extracted
        """
        # Extract address
        to_address = self._extract_address(text)
        
        # Extract NFT info
        collection_address, token_id = self._extract_nft_info(text)
        
        # Create payload
        payload = ExecutablePayload(
            chain_id=1,  # Will be overridden by translate()
            action=ActionType.TRANSFER_ERC721,
            target_contract=collection_address,
            function_name="transferFrom",
            arguments={
                "to": to_address,
                "tokenId": token_id,
                "human_readable_amount": f"Token #{token_id}"
            }
        )
        
        return payload


if __name__ == "__main__":
    translator = BaselineTranslator()
    
    test_intents = [
        "send 0.5 eth to alice.eth",
        "transfer 100 USDC to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        "send my bored ape #1234 to bob.eth",
    ]
    
    print("Baseline Translator Test")
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
