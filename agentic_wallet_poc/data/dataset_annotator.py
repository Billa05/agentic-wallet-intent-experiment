"""
Dataset Annotator for Agentic Wallet Intent Translation System

Loads raw intents, extracts parameters, and creates structured annotations
with interactive validation and correction interface.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from eth_utils import to_checksum_address, is_address
from web3 import Web3

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schemas import (
    TransactionType,
    SendETHTransaction,
    TransferERC20Transaction,
    TransferERC721Transaction,
    BaseTransaction,
    ActionType,
    UserContext,
    ExecutablePayload,
    AnnotatedIntent
)
from decimal import Decimal

# Placeholder address mappings (for ENS-like names)
PLACEHOLDER_ADDRESSES: Dict[str, str] = {}

# Token registry (loaded from file)
TOKEN_REGISTRY: Dict[str, Any] = {}


def load_token_registry(registry_path: str = "data/token_registry.json") -> Dict[str, Any]:
    """
    Load token registry from JSON file.
    
    Args:
        registry_path: Path to token registry JSON file
        
    Returns:
        Dictionary with token registry data
    """
    global TOKEN_REGISTRY
    
    # Handle relative paths
    if not os.path.isabs(registry_path):
        registry_path = str(project_root / registry_path)
    
    if os.path.exists(registry_path):
        with open(registry_path, 'r', encoding='utf-8') as f:
            TOKEN_REGISTRY = json.load(f)
        print(f"✓ Loaded token registry from {registry_path}")
    else:
        print(f"⚠ Warning: Token registry not found at {registry_path}")
        TOKEN_REGISTRY = {
            "erc20_tokens": {},
            "erc721_collections": {}
        }
    
    return TOKEN_REGISTRY


def lookup_erc20_token(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Look up ERC-20 token information from registry.
    
    Args:
        symbol: Token symbol (e.g., "USDC", "USDT")
        
    Returns:
        Token info dict with address and decimals, or None
    """
    if not TOKEN_REGISTRY:
        load_token_registry()
    
    symbol_upper = symbol.upper()
    tokens = TOKEN_REGISTRY.get("erc20_tokens", {})
    
    if symbol_upper in tokens:
        return tokens[symbol_upper]
    
    return None


def lookup_erc721_collection(collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Look up ERC-721 collection information from registry.
    
    Args:
        collection_name: Collection name (e.g., "boredape", "cryptopunk")
        
    Returns:
        Collection info dict with address, or None
    """
    if not TOKEN_REGISTRY:
        load_token_registry()
    
    collection_lower = collection_name.lower().replace(' ', '')
    collections = TOKEN_REGISTRY.get("erc721_collections", {})
    
    if collection_lower in collections:
        return collections[collection_lower]
    
    return None


def generate_random_address(seed: str = None) -> str:
    """
    Generate a random but valid Ethereum address.
    
    Args:
        seed: Optional seed string for deterministic generation
        
    Returns:
        Checksummed Ethereum address
    """
    if seed:
        # Use seed to generate deterministic address
        w3 = Web3()
        # Create a simple hash-based address from seed
        hash_bytes = w3.keccak(text=seed)
        address = "0x" + hash_bytes.hex()[:40]
    else:
        # Generate random address
        w3 = Web3()
        account = w3.eth.account.create()
        address = account.address
    
    return to_checksum_address(address)


def normalize_address(address_str: str) -> Optional[str]:
    """
    Normalize an Ethereum address to checksum format.
    Handles both full addresses and placeholder names.
    
    Args:
        address_str: Address string (0x..., ENS name, or placeholder)
        
    Returns:
        Checksummed address or None if invalid
    """
    if not address_str:
        return None
    
    address_str = address_str.strip()
    
    # Check if it's a placeholder/ENS name
    if address_str.endswith('.eth') or not address_str.startswith('0x'):
        # Generate or retrieve placeholder address
        if address_str not in PLACEHOLDER_ADDRESSES:
            PLACEHOLDER_ADDRESSES[address_str] = generate_random_address(address_str)
        return PLACEHOLDER_ADDRESSES[address_str]
    
    # Validate and checksum real address
    if is_address(address_str):
        try:
            return to_checksum_address(address_str)
        except Exception:
            return None
    
    return None


def extract_eth_addresses(text: str) -> List[str]:
    """
    Extract Ethereum addresses from text.
    
    Args:
        text: Text to search for addresses
        
    Returns:
        List of address strings found
    """
    addresses = []
    
    # Extract 0x addresses (40 hex chars)
    hex_pattern = r'0x[a-fA-F0-9]{40}'
    hex_addresses = re.findall(hex_pattern, text)
    addresses.extend(hex_addresses)
    
    # Extract ENS-like names
    ens_pattern = r'\b[a-z0-9]+\.eth\b'
    ens_names = re.findall(ens_pattern, text, re.IGNORECASE)
    addresses.extend(ens_names)
    
    # Extract placeholder names (common names after "to")
    placeholder_pattern = r'\bto\s+([a-z]+)\b'
    placeholders = re.findall(placeholder_pattern, text.lower())
    for p in placeholders:
        if p not in ['the', 'my', 'your', 'their', 'this', 'that']:
            addresses.append(p + '.eth')  # Treat as ENS-like
    
    return list(set(addresses))  # Remove duplicates


def extract_amount(text: str, token_type: str = "ETH") -> Optional[str]:
    """
    Extract amount from text.
    
    Args:
        text: Text to search for amounts
        token_type: Type of token (ETH, ERC20, etc.)
        
    Returns:
        Amount as string or None
    """
    # Pattern for decimal numbers
    amount_patterns = [
        r'(\d+\.?\d*)\s*(?:eth|ether|wei)',  # ETH amounts
        r'(\d+\.?\d*)\s*(?:usdc|usdt|dai|weth|token|tokens)',  # Token amounts
        r'amount[=:]\s*(\d+\.?\d*)',  # amount= or amount:
        r'(\d+\.?\d*)',  # Any decimal number (fallback)
    ]
    
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the first reasonable match
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                try:
                    amount = float(match)
                    if amount > 0:
                        return str(amount)
                except ValueError:
                    continue
    
    return None


def extract_token_info(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract token address and decimals from text using token registry.
    
    Args:
        text: Text to search
        
    Returns:
        Tuple of (token_address, decimals) or (None, None)
    """
    text_lower = text.lower()
    
    # Check for known token symbols and look up in registry
    token_symbol = None
    decimals = 18  # Default
    token_address = None
    
    # Try to match token symbols
    if 'usdc' in text_lower:
        token_symbol = 'USDC'
    elif 'usdt' in text_lower:
        token_symbol = 'USDT'
    elif 'dai' in text_lower:
        token_symbol = 'DAI'
    elif 'weth' in text_lower:
        token_symbol = 'WETH'
    
    # Look up token in registry
    if token_symbol:
        token_info = lookup_erc20_token(token_symbol)
        if token_info:
            token_address = normalize_address(token_info['address'])
            decimals = token_info.get('decimals', 18)
    
    # If no registry match, try to extract address from text
    # (in case user provided explicit contract address)
    if not token_address:
        addresses = extract_eth_addresses(text)
        if addresses:
            # If multiple addresses, first might be token, last is recipient
            if len(addresses) > 1:
                token_address = normalize_address(addresses[0])
            # If only one address and we have a symbol, it's likely the recipient
            # (token address should come from registry)
            elif not token_symbol:
                # No symbol and one address - might be token address
                token_address = normalize_address(addresses[0])
    
    return token_address, decimals


def extract_nft_info(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract NFT contract address and token ID from text using token registry.
    
    Args:
        text: Text to search
        
    Returns:
        Tuple of (contract_address, token_id) or (None, None)
    """
    # Extract token ID (patterns like #1234, token id 1234, etc.)
    token_id_patterns = [
        r'#(\d+)',
        r'token\s+id\s+(\d+)',
        r'nft\s+#(\d+)',
        r'(\d{3,})',  # 3+ digit number (likely token ID)
    ]
    
    token_id = None
    for pattern in token_id_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                token_id = int(matches[0])
                break
            except (ValueError, IndexError):
                continue
    
    # Extract contract address from registry or text
    addresses = extract_eth_addresses(text)
    contract_address = None
    
    # Check for collection names and look up in registry
    collection_patterns = [
        r'(bored\s+ape|boredape|bayc)',
        r'(cryptopunk|crypto\s+punk|punk)',
        r'(mutant\s+ape|mutantape|mayc)',
        r'(azuki)',
        r'(doodles)',
    ]
    
    collection_name = None
    for pattern in collection_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            collection_name = match.group(1).lower().replace(' ', '')
            break
    
    # Look up collection in registry
    if collection_name:
        collection_info = lookup_erc721_collection(collection_name)
        if collection_info:
            contract_address = normalize_address(collection_info['address'])
    
    # If no registry match, try to extract address from text
    if not contract_address and addresses:
        contract_address = normalize_address(addresses[0])
    
    return contract_address, token_id


def extract_parameters(intent: str, transaction_type: TransactionType) -> Dict[str, Any]:
    """
    Extract all parameters from an intent text.
    
    Args:
        intent: Natural language intent text
        transaction_type: Type of transaction
        
    Returns:
        Dictionary with extracted parameters
    """
    addresses = extract_eth_addresses(intent)
    to_address = normalize_address(addresses[-1]) if addresses else None
    
    params = {
        'to_address': to_address,
        'from_address': None,  # Usually inferred from wallet context
    }
    
    if transaction_type == TransactionType.SEND_ETH:
        amount = extract_amount(intent, "ETH")
        params['amount'] = amount
        
    elif transaction_type == TransactionType.TRANSFER_ERC20:
        amount = extract_amount(intent, "ERC20")
        token_address, decimals = extract_token_info(intent)
        params['amount'] = amount
        params['token_address'] = token_address
        params['decimals'] = decimals
        
    elif transaction_type == TransactionType.TRANSFER_ERC721:
        contract_address, token_id = extract_nft_info(intent)
        params['contract_address'] = contract_address
        params['token_id'] = token_id
    
    return params


def create_executable_payload(
    transaction_type: TransactionType,
    parameters: Dict[str, Any],
    chain_id: int = 1
) -> Optional[ExecutablePayload]:
    """
    Create an executable payload from parameters with Wei/base unit amounts.
    
    Args:
        transaction_type: Type of transaction
        parameters: Extracted parameters
        chain_id: Chain ID (default 1 = Ethereum Mainnet)
        
    Returns:
        ExecutablePayload object or None if invalid
    """
    try:
        if transaction_type == TransactionType.SEND_ETH:
            if not parameters.get('to_address') or not parameters.get('amount'):
                return None
            
            # Convert ETH amount to Wei
            eth_amount = Decimal(str(parameters['amount']))
            wei_per_eth = Decimal("1000000000000000000")  # 1e18
            value_wei = str(int((eth_amount * wei_per_eth).to_integral_value()))
            
            # Build human-readable amount string
            human_readable = f"{parameters['amount']} ETH"
            
            return ExecutablePayload(
                chain_id=chain_id,
                action=ActionType.TRANSFER_NATIVE,
                target_contract=None,
                function_name=None,
                arguments={
                    "to": to_checksum_address(parameters['to_address']),
                    "value": value_wei,  # Wei as string to prevent overflow
                    "human_readable_amount": human_readable
                }
            )
        
        elif transaction_type == TransactionType.TRANSFER_ERC20:
            if not parameters.get('to_address') or not parameters.get('amount') or not parameters.get('token_address'):
                return None
            
            # Convert token amount to base units
            token_amount = Decimal(str(parameters['amount']))
            decimals = parameters.get('decimals', 18)
            scale = Decimal(10) ** int(decimals)
            value_base_units = str(int((token_amount * scale).to_integral_value()))
            
            # Detect token symbol for human-readable
            token_symbol = "TOKEN"  # Default
            intent_lower = parameters.get('_intent', '').lower() if '_intent' in parameters else ''
            if 'usdc' in intent_lower:
                token_symbol = "USDC"
            elif 'usdt' in intent_lower:
                token_symbol = "USDT"
            elif 'dai' in intent_lower:
                token_symbol = "DAI"
            elif 'weth' in intent_lower:
                token_symbol = "WETH"
            
            human_readable = f"{parameters['amount']} {token_symbol}"
            
            return ExecutablePayload(
                chain_id=chain_id,
                action=ActionType.TRANSFER_ERC20,
                target_contract=to_checksum_address(parameters['token_address']),
                function_name="transfer",
                arguments={
                    "to": to_checksum_address(parameters['to_address']),
                    "value": value_base_units,  # Base units as string
                    "human_readable_amount": human_readable
                }
            )
        
        elif transaction_type == TransactionType.TRANSFER_ERC721:
            if not parameters.get('to_address') or not parameters.get('contract_address') or parameters.get('token_id') is None:
                return None
            
            return ExecutablePayload(
                chain_id=chain_id,
                action=ActionType.TRANSFER_ERC721,
                target_contract=to_checksum_address(parameters['contract_address']),
                function_name="transferFrom",  # or "safeTransferFrom"
                arguments={
                    "to": to_checksum_address(parameters['to_address']),
                    "tokenId": parameters['token_id'],
                    "human_readable_amount": f"Token #{parameters['token_id']}"
                }
            )
    
    except Exception as e:
        print(f"Error creating executable payload: {e}")
        return None


def create_structured_transaction(
    transaction_type: TransactionType,
    parameters: Dict[str, Any]
) -> Optional[BaseTransaction]:
    """
    Create a structured transaction object from parameters.
    
    Args:
        transaction_type: Type of transaction
        parameters: Extracted parameters
        
    Returns:
        Structured transaction object or None if invalid
    """
    try:
        if transaction_type == TransactionType.SEND_ETH:
            if not parameters.get('to_address') or not parameters.get('amount'):
                return None
            return SendETHTransaction(
                to_address=parameters['to_address'],
                from_address=parameters.get('from_address'),
                amount=parameters['amount']
            )
        
        elif transaction_type == TransactionType.TRANSFER_ERC20:
            if not parameters.get('to_address') or not parameters.get('amount') or not parameters.get('token_address'):
                return None
            return TransferERC20Transaction(
                to_address=parameters['to_address'],
                from_address=parameters.get('from_address'),
                token_address=parameters['token_address'],
                amount=parameters['amount'],
                decimals=parameters.get('decimals', 18)
            )
        
        elif transaction_type == TransactionType.TRANSFER_ERC721:
            if not parameters.get('to_address') or not parameters.get('contract_address') or parameters.get('token_id') is None:
                return None
            return TransferERC721Transaction(
                to_address=parameters['to_address'],
                from_address=parameters.get('from_address'),
                contract_address=parameters['contract_address'],
                token_id=parameters['token_id']
            )
    
    except Exception as e:
        print(f"Error creating transaction: {e}")
        return None


def load_raw_intents(file_path: str) -> List[Dict[str, Any]]:
    """Load raw intents from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_annotation(annotation: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate an annotation has all required fields for the new format.
    
    Args:
        annotation: Annotation dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    required_fields = ['user_intent', 'user_context', 'target_payload']
    for field in required_fields:
        if field not in annotation:
            errors.append(f"Missing required field: {field}")
    
    if 'target_payload' in annotation and annotation['target_payload'] is None:
        errors.append("target_payload is None (validation failed)")
    
    # Validate target_payload structure
    if 'target_payload' in annotation and annotation['target_payload']:
        payload = annotation['target_payload']
        payload_required = ['chain_id', 'action', 'arguments']
        for field in payload_required:
            if field not in payload:
                errors.append(f"Missing required field in target_payload: {field}")
        
        # Validate value is a string (Wei/base units) for native/ERC-20 transfers
        if 'arguments' in payload:
            args = payload['arguments']
            if 'value' in args:
                value = args['value']
                if not isinstance(value, str):
                    errors.append("target_payload.arguments.value must be a string (Wei/base units)")
                else:
                    try:
                        int(value)  # Must be a valid integer string
                    except (ValueError, TypeError):
                        errors.append("target_payload.arguments.value must be a valid integer string")
            # Validate tokenId for ERC-721
            if 'tokenId' in args:
                token_id = args['tokenId']
                if not isinstance(token_id, int):
                    errors.append("target_payload.arguments.tokenId must be an integer")
    
    return len(errors) == 0, errors


def interactive_annotate(
    intent: str,
    transaction_type: TransactionType,
    auto_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Interactive annotation interface for a single intent.
    
    Args:
        intent: Original intent text
        transaction_type: Transaction type
        auto_params: Auto-extracted parameters
        
    Returns:
        Final annotation dictionary
    """
    print("\n" + "="*70)
    print(f"Intent: {intent}")
    print(f"Type: {transaction_type.value}")
    print("-"*70)
    print("Auto-extracted parameters:")
    for key, value in auto_params.items():
        print(f"  {key}: {value}")
    print("-"*70)
    
    # Ask for confirmation/correction
    response = input("\nAccept these parameters? (y/n/edit): ").strip().lower()
    
    if response == 'y' or response == '':
        # Accept auto-extracted
        final_params = auto_params
    elif response == 'edit':
        # Manual editing
        final_params = {}
        print("\nEnter parameters (press Enter to keep auto-extracted value):")
        
        if transaction_type == TransactionType.SEND_ETH:
            to_addr = input(f"To address [{auto_params.get('to_address', '')}]: ").strip()
            final_params['to_address'] = normalize_address(to_addr) if to_addr else auto_params.get('to_address')
            amount = input(f"Amount (ETH) [{auto_params.get('amount', '')}]: ").strip()
            final_params['amount'] = amount if amount else auto_params.get('amount')
            final_params['from_address'] = auto_params.get('from_address')
        
        elif transaction_type == TransactionType.TRANSFER_ERC20:
            to_addr = input(f"To address [{auto_params.get('to_address', '')}]: ").strip()
            final_params['to_address'] = normalize_address(to_addr) if to_addr else auto_params.get('to_address')
            token_addr = input(f"Token address [{auto_params.get('token_address', '')}]: ").strip()
            final_params['token_address'] = normalize_address(token_addr) if token_addr else auto_params.get('token_address')
            amount = input(f"Amount [{auto_params.get('amount', '')}]: ").strip()
            final_params['amount'] = amount if amount else auto_params.get('amount')
            decimals = input(f"Decimals [{auto_params.get('decimals', 18)}]: ").strip()
            final_params['decimals'] = int(decimals) if decimals else auto_params.get('decimals', 18)
            final_params['from_address'] = auto_params.get('from_address')
        
        elif transaction_type == TransactionType.TRANSFER_ERC721:
            to_addr = input(f"To address [{auto_params.get('to_address', '')}]: ").strip()
            final_params['to_address'] = normalize_address(to_addr) if to_addr else auto_params.get('to_address')
            contract_addr = input(f"Contract address [{auto_params.get('contract_address', '')}]: ").strip()
            final_params['contract_address'] = normalize_address(contract_addr) if contract_addr else auto_params.get('contract_address')
            token_id = input(f"Token ID [{auto_params.get('token_id', '')}]: ").strip()
            final_params['token_id'] = int(token_id) if token_id else auto_params.get('token_id')
            final_params['from_address'] = auto_params.get('from_address')
    else:
        # Skip this one
        return None
    
    # Create executable payload
    final_params['_intent'] = intent  # Store intent for token symbol detection
    executable_payload = create_executable_payload(transaction_type, final_params, chain_id=1)
    
    if not executable_payload:
        print("ERROR: Could not create valid executable payload!")
        retry = input("Retry editing? (y/n): ").strip().lower()
        if retry == 'y':
            return interactive_annotate(intent, transaction_type, final_params)
        return None
    
    # Create user context
    user_context = UserContext(
        current_chain_id=1,
        token_prices={"ETH": 2500.00}  # Default price, can be updated
    )
    
    # Create annotated intent
    annotated_intent = AnnotatedIntent(
        user_intent=intent,
        user_context=user_context,
        target_payload=executable_payload
    )
    
    annotation = annotated_intent.model_dump(mode='json')
    
    is_valid, errors = validate_annotation(annotation)
    if not is_valid:
        print(f"Validation errors: {errors}")
        retry = input("Retry editing? (y/n): ").strip().lower()
        if retry == 'y':
            return interactive_annotate(intent, transaction_type, final_params)
        return None
    
    return annotation


def annotate_dataset(
    raw_intents_path: str = "data/raw_intents.json",
    output_path: str = "data/annotated_dataset.json",
    interactive: bool = True
) -> List[Dict[str, Any]]:
    """
    Annotate the entire dataset.
    
    Args:
        raw_intents_path: Path to raw intents JSON
        output_path: Path to save annotated dataset
        interactive: Whether to use interactive mode
        
    Returns:
        List of annotated examples
    """
    # Handle relative paths
    if not os.path.isabs(raw_intents_path):
        raw_intents_path = str(project_root / raw_intents_path)
    if not os.path.isabs(output_path):
        output_path = str(project_root / output_path)
    
    # Load token registry first
    load_token_registry()
    
    print("Loading raw intents...")
    raw_intents = load_raw_intents(raw_intents_path)
    print(f"Loaded {len(raw_intents)} intents")
    
    annotated = []
    skipped = []
    
    for i, item in enumerate(raw_intents, 1):
        intent = item['intent']
        transaction_type = TransactionType(item['transaction_type'])
        
        print(f"\n[{i}/{len(raw_intents)}] Processing intent...")
        
        # Auto-extract parameters
        auto_params = extract_parameters(intent, transaction_type)
        
        if interactive:
            annotation = interactive_annotate(intent, transaction_type, auto_params)
            if annotation:
                annotated.append(annotation)
                print("✓ Annotation saved")
            else:
                skipped.append(i)
                print("⊘ Annotation skipped")
        else:
            # Non-interactive: auto-annotate and validate
            auto_params['_intent'] = intent  # Store intent for token symbol detection
            executable_payload = create_executable_payload(transaction_type, auto_params, chain_id=1)
            if executable_payload:
                # Create user context
                user_context = UserContext(
                    current_chain_id=1,
                    token_prices={"ETH": 2500.00}
                )
                
                # Create annotated intent
                annotated_intent = AnnotatedIntent(
                    user_intent=intent,
                    user_context=user_context,
                    target_payload=executable_payload
                )
                
                annotation = annotated_intent.model_dump(mode='json')
                is_valid, errors = validate_annotation(annotation)
                if is_valid:
                    annotated.append(annotation)
                    print("✓ Auto-annotated")
                else:
                    print(f"⊘ Validation failed: {errors}")
                    skipped.append(i)
            else:
                print("⊘ Could not create executable payload")
                skipped.append(i)
    
    # Save annotated dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print(f"Annotation complete!")
    print(f"  Annotated: {len(annotated)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Saved to: {output_path}")
    print("="*70)
    
    return annotated


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Annotate transaction intent dataset")
    parser.add_argument(
        '--input',
        default='data/raw_intents.json',
        help='Path to raw intents JSON file'
    )
    parser.add_argument(
        '--output',
        default='data/annotated_dataset.json',
        help='Path to save annotated dataset'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode (auto-annotate all)'
    )
    
    args = parser.parse_args()
    
    print("Agentic Wallet Intent Translation System - Dataset Annotator")
    print("="*70)
    
    annotated = annotate_dataset(
        raw_intents_path=args.input,
        output_path=args.output,
        interactive=not args.non_interactive
    )
    
    print(f"\nSuccessfully annotated {len(annotated)} examples!")


if __name__ == "__main__":
    main()
