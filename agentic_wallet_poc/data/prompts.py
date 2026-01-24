"""
Prompt templates for dataset generation.

Contains all prompt-related functions for generating diverse natural language
transaction intents using LLM APIs.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schemas import TransactionType


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


def get_amount_guidance(amount_range: str) -> str:
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


def get_edge_cases_section(include: bool) -> str:
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


def get_context_variations_section(include: bool) -> str:
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


def get_negative_examples_section(include: bool) -> str:
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


def load_ens_registry(registry_path: str = "data/registries/ens_registry.json") -> dict:
    """Load ENS registry from JSON file."""
    import json
    import os
    
    if not os.path.isabs(registry_path):
        registry_path = project_root / registry_path
    
    registry_path = Path(registry_path)
    
    if not registry_path.exists():
        return {}
    
    with open(registry_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("ens_names", {})


def create_prompt_for_transaction_type(
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
    # Load ENS registry
    ens_registry = load_ens_registry()
    ens_names_list = list(ens_registry.keys()) if ens_registry else []
    ens_names_str = ", ".join(ens_names_list) if ens_names_list else "alice.eth, bob.eth, friend.eth"
    
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
- Include realistic Ethereum addresses (0x followed by 40 hex characters) OR ENS names
- Vary the phrasing: "send", "transfer", "move", "give", "pay", "disburse", "forward", etc.
- Include amounts, addresses, and all necessary details
- Make examples realistic - users might misspell words, use abbreviations, or be informal

ENS NAMES TO USE (only use these):
{ens_names_str}

If using ENS names, ONLY use the names listed above. Do not create new ENS names.

OUTPUT FORMAT:
Return ONLY a JSON array of strings, where each string is one intent example.
Example format: ["intent 1", "intent 2", "intent 3"]

"""
    
    # Add edge cases section
    base_instructions += get_edge_cases_section(config.include_edge_cases)
    
    # Add context variations section
    base_instructions += get_context_variations_section(config.include_context_variations)
    
    # Add negative examples section
    if config.include_negative_examples and negative_count > 0:
        base_instructions += f"\n{get_negative_examples_section(True)}"
        base_instructions += f"\nGenerate {negative_count} negative examples marked with [INVALID] prefix.\n"
    
    if transaction_type == TransactionType.SEND_ETH:
        amount_guidance = get_amount_guidance(config.amount_range)
        
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
        amount_guidance = get_amount_guidance(config.amount_range)
        
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
