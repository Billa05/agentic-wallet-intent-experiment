"""
Dataset Generator for Agentic Wallet Intent Translation System

Generates synthetic natural language transaction intents using Google Gemini API.
Creates diverse examples with varied phrasings, formality levels, and styles.
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from google import genai
from dotenv import load_dotenv

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schemas import TransactionType

# Load environment variables
load_dotenv()


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    count: int
    include_edge_cases: bool = True
    include_negative_examples: bool = False
    include_context_variations: bool = True
    amount_range: str = "varied"  # "micro", "small", "medium", "large", "varied"
    
    def __post_init__(self):
        """Validate amount_range."""
        valid_ranges = ["micro", "small", "medium", "large", "varied"]
        if self.amount_range not in valid_ranges:
            raise ValueError(f"amount_range must be one of {valid_ranges}")

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLIENT = None
if GEMINI_API_KEY:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)


def _validate_intent_example(intent: str, transaction_type: TransactionType) -> bool:
    """
    Validate that a generated intent example contains necessary information.
    
    Args:
        intent: The natural language intent text
        transaction_type: The expected transaction type
        
    Returns:
        True if the intent appears valid, False otherwise
    """
    intent_lower = intent.lower()
    
    # Check for address-like patterns (0x... or ENS-like names)
    has_address = bool(
        re.search(r'0x[a-fA-F0-9]{40}', intent) or
        re.search(r'\b[a-z0-9]+\.eth\b', intent_lower) or
        re.search(r'\bto\s+[a-z]+\b', intent_lower)  # "to alice", "to bob"
    )
    
    if transaction_type == TransactionType.SEND_ETH:
        # Must have ETH amount and address
        has_eth = bool(
            re.search(r'\beth\b', intent_lower) or
            re.search(r'\d+\.?\d*\s*eth', intent_lower, re.IGNORECASE)
        )
        return has_address and has_eth
    
    elif transaction_type == TransactionType.TRANSFER_ERC20:
        # Must have token name/symbol and amount
        has_token = bool(
            re.search(r'\b(usdc|usdt|dai|weth|token|tokens)\b', intent_lower) or
            re.search(r'\d+\s+(token|tokens)', intent_lower)
        )
        has_amount = bool(re.search(r'\d+\.?\d*', intent))
        return has_address and has_token and has_amount
    
    elif transaction_type == TransactionType.TRANSFER_ERC721:
        # Must have NFT identifier (token ID, collection name, or #number)
        has_nft_id = bool(
            re.search(r'#\d+', intent) or
            re.search(r'token\s+id\s+\d+', intent_lower) or
            re.search(r'\b(bored\s+ape|nft|nft\s+#\d+)\b', intent_lower, re.IGNORECASE)
        )
        return has_address and has_nft_id
    
    return False


def _get_amount_guidance(amount_range: str) -> str:
    """Get amount guidance based on range."""
    guidance = {
        "micro": "Focus on very small amounts: 0.0001, 0.001, 0.00001 ETH (gas-like amounts)",
        "small": "Focus on small amounts: 0.01, 0.05, 0.1, 0.25 ETH",
        "medium": "Focus on medium amounts: 0.5, 1, 2.5, 5 ETH",
        "large": "Focus on large amounts: 10, 50, 100, 500 ETH",
        "varied": """Include a wide range:
- Micro: 0.0001, 0.001 ETH (gas-like amounts)
- Small: 0.01, 0.05, 0.1 ETH
- Medium: 0.5, 1, 2.5 ETH
- Large: 10, 50, 100 ETH
- Edge cases: 0, 0.0, very large numbers like 1000000
- High precision: 1.234567890123456789 ETH (max precision)"""
    }
    return guidance.get(amount_range, guidance["varied"])


def _get_edge_cases_section(include: bool) -> str:
    """Get edge cases section for prompts."""
    if not include:
        return ""
    
    return """
CHALLENGING VARIATIONS TO INCLUDE:
- Ambiguous amounts: "send like half an ETH", "send around 100 bucks worth", "send a bit of ETH"
- Multiple ways to say amounts: "0.5", "0.50", ".5", "half", "500000000000000000 wei", "1e18 wei"
- Incomplete info (for testing): "send some ETH to alice", "transfer tokens to bob.eth"
- Non-standard ENS: "send to vitalik.eth", "send to mydomain.crypto", "send to test.xyz"
- Mixed languages/slang: "yo fam send 1 eth to homie.eth asap", "pls send 0.5 eth to friend.eth"
- Questions as intents: "can you send 1 ETH to alice.eth?", "could you transfer 100 USDC to bob?"
- Conditional phrasing: "I want to send 1 ETH to bob.eth", "I need to transfer 50 DAI to alice"
- Multi-step (should be single transaction only): "send 1 ETH to alice and 2 ETH to bob" (generate as separate examples)
"""


def _get_context_variations_section(include: bool) -> str:
    """Get context variations section for prompts."""
    if not include:
        return ""
    
    return """
CONTEXT VARIATIONS TO INCLUDE:
- With reasons: "send 1 ETH to alice.eth for the concert tickets", "transfer 100 USDC to bob.eth for payment"
- With urgency: "URGENT: transfer 100 USDC to bob.eth immediately", "send 5 ETH to alice.eth ASAP"
- With confirmations: "please confirm before sending 5 ETH to 0x...", "double check then send 1 ETH to bob.eth"
- With history: "send another 1 ETH to alice.eth like yesterday", "transfer 50 DAI to bob.eth again"
- With emotions: "send 0.5 ETH to charity.eth please", "transfer 100 USDC to friend.eth thanks!"
"""


def _get_negative_examples_section(include: bool) -> str:
    """Get negative examples section for prompts."""
    if not include:
        return ""
    
    return """
ALSO GENERATE 3-5 INVALID or UNSUPPORTED intents (mark with "[INVALID]" prefix):
- Swaps: "swap 1 ETH for USDC" (not supported in MVP)
- Cross-chain: "send 1 ETH to Polygon address 0x...", "bridge 100 USDC to BSC"
- Ambiguous: "send money to my friend", "transfer tokens somewhere"
- Missing critical info: "transfer tokens" (no amount, no recipient), "send ETH" (no recipient)
- Wrong network: "send 100 SOL to alice.eth", "transfer BTC to 0x..."
- Unsupported protocols: "stake 10 ETH", "lend 100 USDC", "provide liquidity"
- Invalid addresses: "send 1 ETH to 0x123" (too short), "send 1 ETH to invalid.eth"
"""


def _create_prompt_for_transaction_type(
    transaction_type: TransactionType, 
    config: PromptConfig
) -> str:
    """
    Create a detailed prompt for Gemini to generate diverse intent examples.
    
    Args:
        transaction_type: The type of transaction to generate examples for
        config: Prompt configuration
        
    Returns:
        Formatted prompt string for Gemini
    """
    total_count = config.count
    if config.include_negative_examples:
        # Reserve some count for negative examples
        valid_count = max(1, total_count - 3)
        negative_count = total_count - valid_count
    else:
        valid_count = total_count
        negative_count = 0
    
    base_instructions = f"""Generate {valid_count} diverse natural language examples of users requesting blockchain transactions.

REQUIREMENTS:
- Each example must be a single line of natural language text
- Use VARIED language styles: formal, casual, abbreviated, with typos, slang, technical terms
- Include realistic Ethereum addresses (0x followed by 40 hex characters) OR ENS names (e.g., alice.eth, bob.eth, vitalik.eth)
- Vary the phrasing: "send", "transfer", "move", "give", "pay", "disburse", "forward", etc.
- Include amounts, addresses, and all necessary details
- Make examples realistic - users might misspell words, use abbreviations, or be informal

OUTPUT FORMAT:
Return ONLY a JSON array of strings, where each string is one intent example.
Example format: ["intent 1", "intent 2", "intent 3"]

"""
    
    # Add edge cases section
    base_instructions += _get_edge_cases_section(config.include_edge_cases)
    
    # Add context variations section
    base_instructions += _get_context_variations_section(config.include_context_variations)
    
    # Add negative examples section
    if config.include_negative_examples and negative_count > 0:
        base_instructions += f"\n{_get_negative_examples_section(True)}"
        base_instructions += f"\nGenerate {negative_count} negative examples marked with [INVALID] prefix.\n"
    
    if transaction_type == TransactionType.SEND_ETH:
        amount_guidance = _get_amount_guidance(config.amount_range)
        
        return base_instructions + f"""
TRANSACTION TYPE: Send Ethereum (ETH) native token

Each example must include:
- Amount in ETH (e.g., 0.5, 1.2, 0.001)
- Recipient address (0x... or ENS name)
- Action words: "send", "transfer", "pay", "give", "disburse", "forward", etc.

AMOUNT VARIATIONS TO INCLUDE:
{amount_guidance}

Also include these amount formats:
- Decimal: "0.5", "1.234", "0.001"
- Spelled out: "half an ETH", "one ETH", "quarter ETH"
- Wei format: "500000000000000000 wei", "1e18 wei" (less common but valid)
- With units: "0.5 ETH", "0.5 Ether", "0.5 ethereum"

Example styles to vary:
- Formal: "Please send 0.5 ETH to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
- Casual: "send 0.5 eth to alice.eth"
- Abbreviated: "tx 0.5 ETH -> 0xABC...", "0.5 ETH -> alice.eth"
- With typos: "send 0.5 ethh to alice", "tranfer 1 eth to bob"
- Technical: "Initiate transfer of 0.5 Ether to address 0x..."
- Question form: "can you send 1 ETH to alice.eth?", "could you transfer 0.5 ETH to bob?"
- Conditional: "I want to send 1 ETH to alice.eth", "I need to transfer 2 ETH to bob.eth"

Generate diverse examples with different amounts, different address formats (full 0x addresses and ENS names), 
and varied language styles. Include some examples with ambiguous amounts like "send some ETH" or "send around 0.5 ETH".
"""
    
    elif transaction_type == TransactionType.TRANSFER_ERC20:
        amount_guidance = _get_amount_guidance(config.amount_range)
        
        return base_instructions + f"""
TRANSACTION TYPE: Transfer ERC-20 tokens (like USDC, USDT, DAI, WETH)

Each example must include:
- Token name/symbol (USDC, USDT, DAI, WETH, or generic "tokens")
- Amount of tokens
- Recipient address (0x... or ENS name)

SUPPORTED TOKENS (use these symbols):
- USDC (USD Coin, 6 decimals)
- USDT (Tether USD, 6 decimals)
- DAI (Dai Stablecoin, 18 decimals)
- WETH (Wrapped Ether, 18 decimals)

AMOUNT VARIATIONS TO INCLUDE:
{amount_guidance}

Token amount formats:
- Standard: "100 USDC", "50.5 DAI", "1.234 WETH"
- Without decimals: "100", "50", "1" (assume token units)
- Ambiguous: "send some USDC", "transfer around 100 tokens"
- Large numbers: "1000000 USDC", "500000 DAI"

Example styles to vary:
- Formal: "Transfer 100 USDC tokens to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
- Casual: "send 50 usdc to bob.eth", "yo send 100 dai to alice"
- Abbreviated: "tx 100 tokens -> 0xABC...", "100 USDC -> bob.eth"
- With typos: "transfer 100 usdc toknes to alice", "send 50 usdt to bob"
- Technical: "Execute ERC-20 transfer of 100 USDC to address 0x..."
- Question form: "can you send 100 USDC to alice.eth?", "transfer 50 DAI to bob?"
- With contract: "transfer 100 USDC using contract 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48 to 0x..."

Generate diverse examples with different tokens (USDC, USDT, DAI, WETH), 
different amounts, different address formats, and varied language styles.
Include some examples with ambiguous amounts or missing explicit amounts.
"""
    
    elif transaction_type == TransactionType.TRANSFER_ERC721:
        return base_instructions + """
TRANSACTION TYPE: Transfer ERC-721 NFT (Non-Fungible Token)

Each example must include:
- NFT identifier using ONE of these formats:
  * Token ID only: "#1234", "token 5678", "id 9999", "nft #1234"
  * Collection + ID: "Bored Ape #1234", "CryptoPunk 5678", "Mutant Ape #2143"
  * Contract + ID: "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D token 1234" (less common)
- Recipient address (0x... or ENS name)

SUPPORTED COLLECTIONS (use these exact names/variations):
- Bored Ape Yacht Club: "bored ape", "boredape", "bayc", "ape", "Bored Ape Yacht Club"
- CryptoPunks: "cryptopunk", "cryptopunks", "punk", "CryptoPunk", "CryptoPunks"
- Mutant Ape Yacht Club: "mutant ape", "mutantape", "mayc", "Mutant Ape"
- Azuki: "azuki", "Azuki"
- Doodles: "doodle", "doodles", "Doodles"

IMPORTANT: Token IDs should be realistic:
- Bored Ape: 1-10000 range
- CryptoPunk: 1-10000 range
- Mutant Ape: 1-20000 range
- Azuki: 1-10000 range
- Doodles: 1-10000 range

Token ID formats to include:
- Hash format: "#1234", "nft #5678"
- Word format: "token 1234", "token id 5678", "id 9999"
- Spelled: "token number 1234", "nft number 5678"
- Without prefix: "1234" (when collection name is clear)

Example styles to vary:
- Formal: "Transfer my Bored Ape NFT #1234 to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
- Casual: "send my bored ape #1234 to alice.eth", "yo send my cryptopunk 7804 to bob"
- Abbreviated: "tx nft #5678 -> 0xABC...", "bayc #1234 -> alice.eth"
- With typos: "transfer nft #1234 to bob", "send my bored apee #5678 to alice"
- Technical: "Execute ERC-721 transfer of token ID 1234 to address 0x..."
- Question form: "can you send my Bored Ape #1234 to alice.eth?", "transfer cryptopunk 7804 to bob?"
- Without collection: "send nft #1234 to alice.eth", "transfer token 5678 to bob.eth"
- With collection only: "send my bored ape to alice.eth" (missing token ID - edge case)

Generate diverse examples with different NFT identifiers (token IDs, collection names),
different address formats, and varied language styles. Use the supported collection names above.
Include some examples with missing token IDs or ambiguous collection names for edge case testing.
"""
    
    else:
        raise ValueError(f"Unknown transaction type: {transaction_type}")


def list_available_models() -> List[str]:
    """
    List available Gemini models that support generateContent.
    
    Returns:
        List of available model names
    """
    if not GEMINI_CLIENT:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    try:
        # Common model names - the new API may not have list_models easily accessible
        return ["gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    except Exception as e:
        print(f"Warning: Could not list models: {e}")
        return ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]


def generate_intent_examples(
    transaction_type: TransactionType, 
    count: int = 10,
    model_name: str = "gemini-2.5-flash",
    config: Optional[PromptConfig] = None
) -> List[Dict[str, Any]]:
    """
    Generate diverse natural language transaction intent examples using Gemini API.
    
    Args:
        transaction_type: Type of transaction (SEND_ETH, TRANSFER_ERC20, TRANSFER_ERC721)
        count: Number of examples to generate
        model_name: Gemini model to use (default: "gemini-2.5-flash")
                    Common options: "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"
        config: Optional prompt configuration. If None, uses defaults with count.
        
    Returns:
        List of dictionaries with 'intent' (text) and 'transaction_type' keys
        
    Raises:
        ValueError: If API key is not configured
        RuntimeError: If generation fails or validation fails
    """
    if not GEMINI_CLIENT:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    
    # Create config if not provided
    if config is None:
        config = PromptConfig(count=count)
    else:
        # Update count if provided separately
        config.count = count
    
    try:
        prompt = _create_prompt_for_transaction_type(transaction_type, config)
        
        print(f"Generating {count} examples for {transaction_type.value}...")
        
        # Generate with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = GEMINI_CLIENT.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                # Extract text from response - new API structure
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
                    # Fallback: convert entire response to string
                    response_text = str(response).strip()
                    # Try to extract JSON if it's embedded
                    import re
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(0).strip()
                
                # Try to parse JSON array from response
                # Sometimes Gemini wraps it in markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                # Remove leading/trailing brackets if not present
                if not response_text.startswith("["):
                    response_text = "[" + response_text
                if not response_text.endswith("]"):
                    response_text = response_text + "]"
                
                intents = json.loads(response_text)
                
                if not isinstance(intents, list):
                    raise ValueError("Response is not a list")
                
                # Validate and filter intents
                validated_intents = []
                negative_intents = []
                
                for intent in intents:
                    if not isinstance(intent, str):
                        continue
                    
                    intent_clean = intent.strip()
                    
                    # Check if it's a negative example
                    if intent_clean.startswith("[INVALID]"):
                        # Remove the prefix and add as negative example
                        intent_clean = intent_clean.replace("[INVALID]", "").strip()
                        negative_intents.append({
                            'intent': intent_clean,
                            'transaction_type': transaction_type.value
                        })
                        continue
                    
                    # Validate positive examples
                    if _validate_intent_example(intent_clean, transaction_type):
                        validated_intents.append({
                            'intent': intent_clean,
                            'transaction_type': transaction_type.value
                        })
                        # Stop once we have enough valid examples (if not including negatives)
                        if not config.include_negative_examples and len(validated_intents) >= config.count:
                            break
                    else:
                        print(f"Warning: Skipping invalid intent: {intent_clean[:50]}...")
                
                # Add negative examples if configured
                if config.include_negative_examples:
                    # Fill remaining slots with negative examples
                    remaining = config.count - len(validated_intents)
                    if remaining > 0:
                        # Strip is_valid field from negative intents before adding
                        for neg_intent in negative_intents[:remaining]:
                            validated_intents.append({
                                'intent': neg_intent['intent'],
                                'transaction_type': neg_intent['transaction_type']
                            })
                
                # Limit to exactly the requested count
                validated_intents = validated_intents[:config.count]
                
                # For validation threshold, only count positive examples
                # (negative examples are only added if include_negative_examples is True)
                positive_count = len(validated_intents) - (len(negative_intents) if config.include_negative_examples else 0)
                if positive_count < config.count * 0.7:  # Require at least 70% valid positive examples
                    if attempt < max_retries - 1:
                        print(f"Only {positive_count}/{config.count} valid examples. Retrying...")
                        time.sleep(2)  # Brief delay before retry
                        continue
                    else:
                        print(f"Warning: Only {positive_count}/{config.count} valid examples generated")
                
                return validated_intents
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"JSON parsing failed. Retrying... Error: {e}")
                    time.sleep(2)
                    continue
                else:
                    raise RuntimeError(f"Failed to parse JSON response after {max_retries} attempts: {e}")
            except Exception as e:
                error_str = str(e)
                # Check if it's a model not found error
                if "404" in error_str and "not found" in error_str.lower():
                    available_models = list_available_models()
                    raise ValueError(
                        f"Model '{model_name}' not found. Available models: {', '.join(available_models)}\n"
                        f"Try using one of: gemini-1.5-flash, gemini-1.5-pro, or gemini-pro"
                    ) from e
                if attempt < max_retries - 1:
                    print(f"Generation failed. Retrying... Error: {e}")
                    time.sleep(2)
                    continue
                else:
                    raise RuntimeError(f"Failed to generate examples after {max_retries} attempts: {e}")
        
    except ValueError:
        # Re-raise ValueError (model not found) without wrapping
        raise
    except Exception as e:
        raise RuntimeError(f"Error generating intent examples: {e}")


def generate_full_dataset(
    eth_count: int = 10,
    erc20_count: int = 5,
    erc721_count: int = 5,
    output_path: str = "data/raw_intents.json",
    append: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate a complete dataset with examples for all transaction types.
    
    Args:
        eth_count: Number of SEND_ETH examples
        erc20_count: Number of TRANSFER_ERC20 examples
        erc721_count: Number of TRANSFER_ERC721 examples
        output_path: Path to save the generated dataset
        append: If True, append to existing file; if False, overwrite
        
    Returns:
        List of all generated intent examples
    """
    # Handle relative paths
    if not os.path.isabs(output_path):
        output_path = str(project_root / output_path)
    
    # Load existing intents if appending
    existing_intents = []
    if append and os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_intents = json.load(f)
            print(f"✓ Loaded {len(existing_intents)} existing intents")
        except Exception as e:
            print(f"⚠ Warning: Could not load existing intents: {e}")
            existing_intents = []
    
    all_intents = existing_intents.copy() if append else []
    new_intents = []
    
    # Generate ETH examples
    print("\n" + "="*60)
    print("Generating SEND_ETH examples...")
    print("="*60)
    eth_intents = generate_intent_examples(TransactionType.SEND_ETH, eth_count)
    new_intents.extend(eth_intents)
    all_intents.extend(eth_intents)
    print(f"✓ Generated {len(eth_intents)} SEND_ETH examples")
    
    # Generate ERC-20 examples
    print("\n" + "="*60)
    print("Generating TRANSFER_ERC20 examples...")
    print("="*60)
    erc20_intents = generate_intent_examples(TransactionType.TRANSFER_ERC20, erc20_count)
    new_intents.extend(erc20_intents)
    all_intents.extend(erc20_intents)
    print(f"✓ Generated {len(erc20_intents)} TRANSFER_ERC20 examples")
    
    # Generate ERC-721 examples
    print("\n" + "="*60)
    print("Generating TRANSFER_ERC721 examples...")
    print("="*60)
    erc721_intents = generate_intent_examples(TransactionType.TRANSFER_ERC721, erc721_count)
    new_intents.extend(erc721_intents)
    all_intents.extend(erc721_intents)
    print(f"✓ Generated {len(erc721_intents)} TRANSFER_ERC721 examples")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_intents, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"✓ Dataset saved to {output_path}")
    if append:
        print(f"  - Existing: {len(existing_intents)}")
        print(f"  - New: {len(new_intents)}")
    print(f"✓ Total examples: {len(all_intents)}")
    print("="*60)
    
    return all_intents


def main():
    """Main function to generate the dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction intent dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 30 balanced examples (10 of each type)
  python data/dataset_generator.py --count 30
  
  # Generate 60 balanced examples and append to existing
  python data/dataset_generator.py --count 60 --append
  
  # Generate custom counts for each type (advanced)
  python data/dataset_generator.py --eth-count 20 --erc20-count 15 --erc721-count 10
        """
    )
    
    # Primary argument: single count for balanced dataset
    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help='Total number of examples to generate (divided equally among 3 types). '
             'For balanced ML training, use this instead of individual counts. '
             'Example: --count 30 generates 10 SEND_ETH, 10 TRANSFER_ERC20, 10 TRANSFER_ERC721'
    )
    
    # Advanced arguments: individual counts (mutually exclusive with --count)
    parser.add_argument(
        '--eth-count',
        type=int,
        default=None,
        help='Number of SEND_ETH examples (advanced: use --count for balanced dataset)'
    )
    parser.add_argument(
        '--erc20-count',
        type=int,
        default=None,
        help='Number of TRANSFER_ERC20 examples (advanced: use --count for balanced dataset)'
    )
    parser.add_argument(
        '--erc721-count',
        type=int,
        default=None,
        help='Number of TRANSFER_ERC721 examples (advanced: use --count for balanced dataset)'
    )
    
    parser.add_argument(
        '--output',
        default='data/raw_intents.json',
        help='Output file path (default: data/raw_intents.json)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing file instead of overwriting'
    )
    
    args = parser.parse_args()
    
    # Determine counts: prioritize --count for balanced dataset
    if args.count is not None:
        # Balanced generation: divide count equally among 3 types
        if args.count % 3 != 0:
            print(f"Warning: --count {args.count} is not divisible by 3. "
                  f"Will generate {args.count // 3} of each type (total: {(args.count // 3) * 3})")
        count_per_type = args.count // 3
        eth_count = count_per_type
        erc20_count = count_per_type
        erc721_count = count_per_type
        
        # Check if individual counts were also specified (conflict)
        if args.eth_count is not None or args.erc20_count is not None or args.erc721_count is not None:
            print("Warning: --count specified with individual counts. Using --count for balanced generation.")
    else:
        # Use individual counts or defaults
        eth_count = args.eth_count if args.eth_count is not None else 10
        erc20_count = args.erc20_count if args.erc20_count is not None else 5
        erc721_count = args.erc721_count if args.erc721_count is not None else 5
    
    print("Agentic Wallet Intent Translation System - Dataset Generator")
    print("="*60)
    
    if not GEMINI_CLIENT:
        print("ERROR: GEMINI_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return
    
    print(f"Generation plan:")
    print(f"  - SEND_ETH: {eth_count} examples")
    print(f"  - TRANSFER_ERC20: {erc20_count} examples")
    print(f"  - TRANSFER_ERC721: {erc721_count} examples")
    print(f"  - Total: {eth_count + erc20_count + erc721_count} examples")
    if args.count is not None:
        print(f"  (Balanced dataset from --count {args.count})")
    print("="*60)
    
    try:
        dataset = generate_full_dataset(
            eth_count=eth_count,
            erc20_count=erc20_count,
            erc721_count=erc721_count,
            output_path=args.output,
            append=args.append
        )
        
        print("\nDataset generation completed successfully!")
        print(f"Generated {len(dataset)} total examples")
        
    except Exception as e:
        print(f"\nError generating dataset: {e}")
        raise


if __name__ == "__main__":
    main()
