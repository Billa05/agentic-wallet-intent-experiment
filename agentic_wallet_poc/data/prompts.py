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
    defi_style: str = "mixed"  # "basic", "advanced", "mixed" - for DeFi intent richness

    def __post_init__(self):
        """Validate amount_range and defi_style."""
        valid_ranges = ["micro", "small", "medium", "large", "varied"]
        if self.amount_range not in valid_ranges:
            raise ValueError(f"amount_range must be one of {valid_ranges}")
        valid_styles = ["basic", "advanced", "mixed"]
        if self.defi_style not in valid_styles:
            raise ValueError(f"defi_style must be one of {valid_styles}")


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


def load_ens_resolver():
    """Create an ENSResolver instance for dataset generation prompts."""
    from engine.ens_resolver import ENSResolver
    return ENSResolver()


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
    # Load ENS resolver for known names
    ens_resolver = load_ens_resolver()
    ens_names_list = ens_resolver.known_names() if ens_resolver else []
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


def _get_defi_advanced_guidance(action_type: str, style: str) -> str:
    """Return advanced-but-popular variation guidance for DeFi intents (same underlying action)."""
    if style == "basic":
        return ""
    advanced_blocks = {
        "aave_supply": """
ADVANCED BUT POPULAR variations (still same action: supply to Aave):
- With purpose: "Supply 500 USDC to Aave to earn yield", "Deposit 1k DAI in Aave for lending APY"
- Specific pool/protocol: "Supply 200 USDT to Aave v3 mainnet", "Deposit into Aave USDC market"
- Round/realistic amounts: 100, 250, 500, 1000, 2500, 5000, 10000 USDC/DAI/USDT
""",
        "aave_withdraw": """
ADVANCED BUT POPULAR variations (still same action: withdraw from Aave):
- With reason: "Withdraw 1000 USDC from Aave to my wallet", "Pull 500 DAI out of Aave for a payment"
- "Max" style: "Withdraw all my USDC from Aave", "Pull out my full DAI balance from Aave"
- Realistic amounts and assets: USDC, DAI, USDT, WETH
""",
        "aave_borrow": """
ADVANCED BUT POPULAR variations (still same action: borrow from Aave):
- With use case: "Borrow 2000 USDC from Aave for expenses", "I need to borrow 500 DAI from Aave"
- Rate mention: "Borrow 1000 USDT from Aave (variable)", "Borrow 500 DAI from Aave stable rate"
- Realistic amounts: 500, 1000, 2000, 5000, 10000
""",
        "aave_repay": """
ADVANCED BUT POPULAR variations (still same action: repay on Aave):
- Repay max / full debt: "Repay my full USDC debt on Aave", "Repay max DAI on Aave", "Pay off my Aave USDT loan"
- With reason: "Repay 1200 USDC on Aave to improve my health factor", "Repay 500 DAI on Aave to free collateral"
- Partial repay with specific amount: "Repay 2000 USDC on my Aave loan"
""",
        "lido_stake": """
ADVANCED BUT POPULAR variations (still same action: stake ETH on Lido):
- With yield context: "Stake 5 ETH on Lido to earn staking rewards", "Put 10 ETH into Lido for stETH yield"
- Realistic amounts: 0.5, 1, 2, 5, 10, 20, 50 ETH
- Casual/formal mix: "Wrap my ETH with Lido", "Deposit 3 ETH to Lido"
""",
        "lido_unstake": """
ADVANCED BUT POPULAR variations (still same action: unstake/withdraw stETH from Lido):
- Request withdrawal: "Request withdrawal of 15 stETH from Lido", "Unstake 8 stETH from Lido"
- With reason: "Withdraw 20 stETH from Lido to my wallet", "Redeem 10 stETH for ETH via Lido"
- Realistic amounts in stETH: 1, 5, 10, 15, 25, 50
""",
        "uniswap_swap": """
ADVANCED BUT POPULAR variations (still same action: swap on Uniswap V3):
- Min amount out / slippage: "Swap 1 ETH for USDC on Uniswap, accept at least 3200", "Swap 500 USDC to DAI with 0.5% slippage"
- Common pairs: WETH/USDC, WETH/USDT, USDC/DAI, WETH/DAI
- Realistic amounts: 0.1, 0.5, 1, 2 WETH or 100, 500, 1000, 5000 USDC/DAI
- Phrasing: "Exchange X for Y on Uniswap", "Swap X to Y via Uniswap", "Trade X for Y on Uniswap V3"
""",
        "curve_add_liquidity": """
ADVANCED BUT POPULAR variations (still same action: add liquidity to Curve):
- Specific pool: "Add 1000 USDC to Curve 3pool", "Add liquidity to the Curve tri-pool: 500 DAI"
- With purpose: "Add 2000 USDC to Curve 3pool for LP fees", "Deposit 500 USDT into Curve 3pool"
- Single-asset add: "Add 1500 USDC to Curve 3pool", "Add 500 DAI to Curve"
""",
        "curve_remove_liquidity": """
ADVANCED BUT POPULAR variations (still same action: remove liquidity from Curve):
- By amount: "Remove 500 USDC worth of liquidity from Curve 3pool", "Withdraw 300 DAI from Curve 3pool"
- Pull out: "Pull my liquidity from Curve 3pool: 1000 USDC", "Remove 500 USDT from Curve pool"
- Realistic amounts: 200, 500, 1000, 2000 (in one of USDC/USDT/DAI or LP terms)
""",
        "weth_wrap": """
ADVANCED BUT POPULAR variations (still same action: wrap ETH to WETH):
- With purpose: "Wrap 2 ETH so I can use it in DeFi", "Convert my ETH to WETH for a swap"
- Realistic amounts: 0.1, 0.5, 1, 2, 5, 10, 20 ETH
""",
        "weth_unwrap": """
ADVANCED BUT POPULAR variations (still same action: unwrap WETH to ETH):
- With purpose: "Unwrap 3 WETH back to ETH for gas", "Convert my WETH back to native ETH"
- Realistic amounts: 0.1, 0.5, 1, 2, 5, 10 WETH
""",
        "compound_supply": """
ADVANCED BUT POPULAR variations (still same action: supply to Compound V3):
- With purpose: "Supply 5000 USDC to Compound to earn interest", "Lend 2000 USDC on Compound V3 for yield"
- Specific market: "Deposit into Compound USDC market", "Supply to Compound V3 Comet"
- Realistic amounts: 500, 1000, 2500, 5000, 10000 USDC
""",
        "compound_withdraw": """
ADVANCED BUT POPULAR variations (still same action: withdraw from Compound V3):
- Max style: "Withdraw all my USDC from Compound", "Pull out everything from Compound V3"
- With reason: "Withdraw 2000 USDC from Compound for expenses"
- Realistic amounts: 500, 1000, 2000, 5000
""",
        "compound_borrow": """
ADVANCED BUT POPULAR variations (still same action: borrow from Compound V3):
- With purpose: "Borrow 3000 USDC from Compound against my collateral"
- Realistic amounts: 500, 1000, 2000, 5000, 10000
""",
        "compound_repay": """
ADVANCED BUT POPULAR variations (still same action: repay on Compound V3):
- Max style: "Repay my full Compound loan", "Pay off all my Compound debt"
- Partial: "Repay 2000 USDC on Compound V3"
""",
        "maker_deposit": """
ADVANCED BUT POPULAR variations (still same action: deposit DAI to MakerDAO DSR):
- With yield: "Deposit 10000 DAI into Maker DSR to earn savings rate", "Put DAI into sDAI for yield"
- Realistic amounts: 1000, 5000, 10000, 50000 DAI
""",
        "maker_redeem": """
ADVANCED BUT POPULAR variations (still same action: redeem sDAI for DAI):
- With reason: "Redeem 5000 sDAI to get my DAI back", "Convert sDAI to DAI for a payment"
- Realistic amounts: 1000, 5000, 10000 sDAI
""",
        "rocketpool_stake": """
ADVANCED BUT POPULAR variations (still same action: stake ETH on Rocket Pool):
- With yield: "Stake 10 ETH on Rocket Pool for rETH yield", "Get rETH by staking 5 ETH on Rocket Pool"
- Realistic amounts: 0.5, 1, 2, 5, 10, 20, 50 ETH
""",
        "rocketpool_unstake": """
ADVANCED BUT POPULAR variations (still same action: unstake/burn rETH):
- With reason: "Burn 5 rETH to get ETH back from Rocket Pool", "Unstake my rETH"
- Realistic amounts: 0.5, 1, 2, 5, 10 rETH
""",
        "eigenlayer_deposit": """
ADVANCED BUT POPULAR variations (still same action: deposit LST into EigenLayer):
- With context: "Restake 10 stETH on EigenLayer for extra yield", "Deposit rETH into EigenLayer strategy"
- Different LSTs: stETH, rETH, cbETH
- Realistic amounts: 1, 5, 10, 20, 50
""",
        "balancer_swap": """
ADVANCED BUT POPULAR variations (still same action: swap on Balancer V2):
- Common pairs: WETH/USDC, WETH/DAI, BAL/WETH
- Phrasing: "Trade X for Y on Balancer", "Exchange X to Y via Balancer V2"
- Realistic amounts: 0.5, 1, 2 WETH or 500, 1000, 5000 USDC/DAI
""",
    }
    block = advanced_blocks.get(action_type, "")
    if style == "advanced" and block:
        return "\nFocus MOST examples on the advanced variations below. Keep each as a single transaction.\n" + block
    if style == "mixed" and block:
        return "\nInclude a MIX of simple phrasings AND these advanced-but-popular variations (same action):\n" + block
    return block


def create_prompt_for_defi_action(
    action_type: str,
    config: PromptConfig,
) -> str:
    """
    Create a prompt for generating DeFi intent examples (AAVE, Lido, Uniswap, Curve).
    Used by dataset generator for synthetic DeFi intents.
    """
    style = getattr(config, "defi_style", "mixed")
    base = f"""Generate {config.count} diverse natural language examples of users requesting this DeFi action.

REQUIREMENTS:
- Each example must be a single line of natural language text
- Use VARIED language: formal, casual, "I want to", "please", "can you", etc.
- Include amounts and asset names (USDC, USDT, DAI, WETH, ETH, stETH) where relevant
- Output ONLY a JSON array of strings. Example: ["intent 1", "intent 2"]

"""
    prompts = {
        "aave_supply": base + """
ACTION: Supply/deposit assets to Aave.
Each example must include: amount, asset (USDC, USDT, DAI, WETH), and intent to supply/deposit to Aave.
Phrasings: "Supply X USDC to AAVE", "Deposit X in Aave", "Put X USDC into Aave pool", "Stake X USDC in Aave"
""",
        "aave_withdraw": base + """
ACTION: Withdraw assets from Aave.
Include: amount, asset, optional recipient. "Withdraw X USDC from AAVE", "Pull X out of Aave"
""",
        "aave_borrow": base + """
ACTION: Borrow assets from Aave.
Include: amount, asset. "Borrow X USDC from AAVE", "I want to borrow X DAI from Aave"
""",
        "aave_repay": base + """
ACTION: Repay borrowed assets on Aave.
Include: amount, asset. "Repay X USDC on AAVE", "Repay my Aave loan: X USDC"
""",
        "lido_stake": base + """
ACTION: Stake ETH on Lido (receive stETH).
Include: amount in ETH. "Stake X ETH on Lido", "Stake X ETH in Lido pool", "Put X ETH into Lido"
""",
        "lido_unstake": base + """
ACTION: Unstake/withdraw stETH from Lido.
Include: amount in stETH. "Unstake X stETH from Lido", "Request withdrawal of X stETH from Lido"
""",
        "uniswap_swap": base + """
ACTION: Swap tokens on Uniswap (V3 single-hop).
Include: amount, input token, output token. "Swap X ETH for USDC on Uniswap", "Swap X USDC to DAI via Uniswap"
Common pairs: WETH/USDC, WETH/USDT, USDC/DAI, WETH/DAI
""",
        "curve_add_liquidity": base + """
ACTION: Add liquidity to Curve pool (e.g. 3pool).
Include: amount, asset (USDC, USDT, DAI). "Add X USDC to Curve 3pool", "Add liquidity: X USDC to Curve pool"
""",
        "curve_remove_liquidity": base + """
ACTION: Remove liquidity from Curve pool.
Include: amount. "Remove X USDC liquidity from Curve 3pool", "Remove X from Curve pool"
""",
        # ─── New protocols ───
        "weth_wrap": base + """
ACTION: Wrap ETH into WETH.
Include: amount in ETH. "Wrap X ETH", "Convert X ETH to WETH", "Wrap my ETH"
Phrasings: "wrap", "convert ETH to WETH", "deposit ETH for WETH", "get WETH"
""",
        "weth_unwrap": base + """
ACTION: Unwrap WETH back to ETH.
Include: amount in WETH. "Unwrap X WETH", "Convert X WETH to ETH", "Unwrap my WETH"
Phrasings: "unwrap", "convert WETH to ETH", "withdraw WETH for ETH", "redeem WETH"
""",
        "compound_supply": base + """
ACTION: Supply/deposit assets to Compound (V3 / Comet).
Include: amount, asset (USDC, WETH, WBTC). "Supply X USDC to Compound", "Deposit X USDC in Compound V3"
Phrasings: "supply to Compound", "lend on Compound", "deposit into Compound"
""",
        "compound_withdraw": base + """
ACTION: Withdraw assets from Compound (V3 / Comet).
Include: amount, asset. "Withdraw X USDC from Compound", "Pull X out of Compound V3"
Support "max"/"all" to withdraw full balance.
""",
        "compound_borrow": base + """
ACTION: Borrow assets from Compound (V3 / Comet).
Include: amount, asset. "Borrow X USDC from Compound", "I need to borrow X USDC on Compound V3"
""",
        "compound_repay": base + """
ACTION: Repay borrowed assets on Compound (V3 / Comet).
Include: amount, asset. "Repay X USDC on Compound", "Pay back my Compound loan"
Support "max"/"all" to repay full debt.
""",
        "maker_deposit": base + """
ACTION: Deposit DAI into MakerDAO DSR (receive sDAI savings token).
Include: amount in DAI. "Deposit X DAI into Maker savings", "Put X DAI into DSR for sDAI"
Phrasings: "deposit DAI in Maker", "save DAI with MakerDAO", "convert DAI to sDAI", "DSR deposit"
""",
        "maker_redeem": base + """
ACTION: Redeem sDAI for DAI from MakerDAO DSR.
Include: amount in sDAI. "Redeem X sDAI for DAI", "Withdraw X from Maker savings"
Phrasings: "redeem sDAI", "convert sDAI to DAI", "withdraw from DSR", "get my DAI back from Maker"
""",
        "rocketpool_stake": base + """
ACTION: Stake ETH via Rocket Pool (receive rETH).
Include: amount in ETH. "Stake X ETH on Rocket Pool", "Deposit X ETH in Rocket Pool for rETH"
Phrasings: "stake on Rocket Pool", "deposit to Rocket Pool", "get rETH", "liquid stake with RPL"
""",
        "rocketpool_unstake": base + """
ACTION: Burn rETH to unstake from Rocket Pool (receive ETH).
Include: amount in rETH. "Unstake X rETH from Rocket Pool", "Burn X rETH for ETH"
Phrasings: "unstake rETH", "redeem rETH", "burn rETH", "withdraw from Rocket Pool"
""",
        "eigenlayer_deposit": base + """
ACTION: Deposit LST (liquid staking token) into EigenLayer for restaking.
Include: amount, LST asset (stETH, rETH, cbETH). "Restake X stETH on EigenLayer", "Deposit X rETH into EigenLayer"
Phrasings: "restake on EigenLayer", "deposit into EigenLayer strategy", "EigenLayer restaking"
Supported LSTs: stETH, rETH, cbETH, swETH, sfrxETH, osETH
""",
        "balancer_swap": base + """
ACTION: Swap tokens via Balancer V2.
Include: amount, input token, output token. "Swap X WETH for USDC on Balancer", "Trade X DAI for USDC via Balancer"
Common pairs: WETH/USDC, WETH/DAI, BAL/WETH, USDC/DAI
Phrasings: "swap on Balancer", "trade via Balancer", "exchange on Balancer V2"
""",
    }
    if action_type not in prompts:
        raise ValueError(f"Unknown DeFi action type: {action_type}. Known: {list(prompts.keys())}")
    out = prompts[action_type] + _get_defi_advanced_guidance(action_type, style)
    return out
