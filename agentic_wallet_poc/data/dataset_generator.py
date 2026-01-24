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
from pathlib import Path
from typing import List, Dict, Any, Optional

from google import genai
from dotenv import load_dotenv

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schemas import TransactionType
from data.prompts import (
    PromptConfig,
    create_prompt_for_transaction_type
)

# Load environment variables
load_dotenv()

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLIENT = None
if GEMINI_API_KEY:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)


def validate_intent_example(intent: str, transaction_type: TransactionType) -> bool:
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
        prompt = create_prompt_for_transaction_type(transaction_type, config)
        
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
                    if validate_intent_example(intent_clean, transaction_type):
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
    output_path: str = "data/datasets/raw_intents.json",
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
  # Default: Generate 60 balanced examples (20 of each type)
  python data/dataset_generator.py
  
  # Generate 30 balanced examples (10 of each type)
  python data/dataset_generator.py --total 30
  
  # Generate 90 balanced examples and append to existing
  python data/dataset_generator.py --total 90 --append
  
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
    
    # Alias for --count: total examples
    parser.add_argument(
        '--total',
        type=int,
        default=None,
        help='Total number of examples to generate (alias for --count). '
             'Divided equally among 3 types. Default: 60 (20 per type)'
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
        default='data/datasets/raw_intents.json',
        help='Output file path (default: data/datasets/raw_intents.json)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing file instead of overwriting'
    )
    
    args = parser.parse_args()
    
    # Handle --total and --count (--total takes precedence if both specified)
    total_count = args.total if args.total is not None else args.count
    
    # Determine counts: prioritize --total/--count for balanced dataset
    if total_count is not None:
        # Balanced generation: divide count equally among 3 types
        if total_count % 3 != 0:
            print(f"Warning: --total/--count {total_count} is not divisible by 3. "
                  f"Will generate {total_count // 3} of each type (total: {(total_count // 3) * 3})")
        count_per_type = total_count // 3
        eth_count = count_per_type
        erc20_count = count_per_type
        erc721_count = count_per_type
        
        # Check if individual counts were also specified (conflict)
        if args.eth_count is not None or args.erc20_count is not None or args.erc721_count is not None:
            print("Warning: --total/--count specified with individual counts. Using --total/--count for balanced generation.")
    else:
        # Use individual counts or defaults (20 each = 60 total)
        eth_count = args.eth_count if args.eth_count is not None else 20
        erc20_count = args.erc20_count if args.erc20_count is not None else 20
        erc721_count = args.erc721_count if args.erc721_count is not None else 20
    
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
    if total_count is not None:
        print(f"  (Balanced dataset from --total/--count {total_count})")
    elif args.eth_count is None and args.erc20_count is None and args.erc721_count is None:
        print(f"  (Default: 60 examples, 20 per type)")
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
