"""
Prompt templates for LLM-based intent translation.

Contains all prompt-related functions for translating natural language intents
into structured blockchain transactions using LLM APIs.
"""

from typing import Dict, Any


def create_system_prompt(
    token_registry: Dict[str, Any],
    ens_registry: Dict[str, str]
) -> str:
    """
    Create system prompt with schema and token registry information.
    
    Args:
        token_registry: Token registry dictionary with erc20_tokens and erc721_collections
        ens_registry: ENS registry dictionary mapping names to addresses
        
    Returns:
        System prompt string
    """
    # Build token registry info
    tokens_info = []
    for symbol, info in token_registry.get("erc20_tokens", {}).items():
        tokens_info.append(f"- {symbol}: {info['address']} (decimals: {info['decimals']})")
    
    collections_info = []
    for name, info in token_registry.get("erc721_collections", {}).items():
        collections_info.append(f"- {name}: {info['address']}")
    
    # Build ENS registry info
    ens_info = []
    for ens_name, address in ens_registry.items():
        ens_info.append(f"- {ens_name}: {address}")
    ens_names_str = ", ".join(ens_registry.keys()) if ens_registry else "alice.eth, bob.eth"
    
    return f"""You are a blockchain transaction translator. Convert natural language intents into executable blockchain transactions.

CRITICAL REQUIREMENTS:
1. All amounts must be in Wei/base units (integers as strings)
2. ETH: Multiply by 10^18 (e.g., 0.5 ETH = "500000000000000000", 1 ETH = "1000000000000000000")
3. ERC-20 tokens: Multiply by 10^decimals (USDC/USDT = 6, DAI/WETH = 18)
   - Example: 100 USDC = 100 * 10^6 = "100000000" (as string)
4. All addresses must be checksummed (EIP-55 format) - 0x followed by 40 hex characters
5. Chain ID is always 1 (Ethereum Mainnet)
6. For amounts in Wei format (e.g., "500000000000000000 wei"), use that value directly

TOKEN REGISTRY (use EXACTLY these addresses):
ERC-20 Tokens:
{chr(10).join(tokens_info) if tokens_info else "- None"}

ERC-721 Collections (use EXACTLY these addresses):
{chr(10).join(collections_info) if collections_info else "- None"}

ENS NAMES (use ONLY these - map to addresses below):
{chr(10).join(ens_info) if ens_info else "- None"}
Available ENS names: {ens_names_str}

CRITICAL: For ENS names in the intent, you MUST:
1. Look up the ENS name in the registry above
2. Use the EXACT address from the registry (copy it exactly as shown)
3. Do NOT generate, create, or make up any addresses
4. Do NOT use ENS names in your output - always convert them to the 0x address from the registry
5. If an ENS name is not in the registry, return null

OUTPUT JSON SCHEMA (return ONLY this structure, no other fields):

For transfer_native (ETH):
{{
  "action": "transfer_native",
  "target_contract": null,
  "function_name": null,
  "chain_id": 1,
  "arguments": {{
    "to": "0x...",  // checksummed address from registry or 0x address from intent
    "value": "500000000000000000",  // Wei as STRING (e.g., 0.5 ETH = "500000000000000000")
    "human_readable_amount": "0.5 ETH"
  }}
}}

For transfer_erc20:
{{
  "action": "transfer_erc20",
  "target_contract": "0x...",  // EXACT address from token registry above
  "function_name": "transfer",
  "chain_id": 1,
  "arguments": {{
    "to": "0x...",  // checksummed recipient address
    "value": "100000000",  // Base units as STRING (e.g., 100 USDC = "100000000")
    "human_readable_amount": "100 USDC"
  }}
}}

For transfer_erc721:
{{
  "action": "transfer_erc721",
  "target_contract": "0x...",  // EXACT address from collection registry above
  "function_name": "transferFrom",
  "chain_id": 1,
  "arguments": {{
    "to": "0x...",  // checksummed recipient address
    "tokenId": 12345,  // Integer token ID
    "human_readable_amount": "Token #12345"
  }}
}}

IMPORTANT FOR ERC-721:
- If collection name is mentioned (Bored Ape, CryptoPunk, Mutant Ape, Azuki, Doodles), use the EXACT address from registry
- If only "nft" or "token" is mentioned without collection name, you MUST infer from context or return null
- Collection name patterns to recognize:
  * "bored ape", "boredape", "bayc" → Bored Ape Yacht Club
  * "cryptopunk", "cryptopunks", "punk" → CryptoPunks
  * "mutant ape", "mutantape", "mayc" → Mutant Ape Yacht Club
  * "azuki" → Azuki
  * "doodles", "doodle" → Doodles

VALIDATION RULES:
- "action" must be exactly: "transfer_native", "transfer_erc20", or "transfer_erc721"
- "target_contract" must be null for ETH, or EXACT address from registry for tokens/NFTs
- "function_name" must be null for ETH, "transfer" for ERC-20, "transferFrom" for ERC-721
- "to" address must be checksummed (0x + 40 hex chars) - use registry addresses for ENS names
- "value" must be a STRING of digits (Wei/base units), never a decimal
- "chain_id" must be 1
- If intent is ambiguous, missing required info, or ENS name not in registry, return null"""


def create_user_prompt(intent: str, chain_id: int = 1) -> str:
    """
    Create user prompt with the intent to translate.
    
    Args:
        intent: Natural language transaction intent
        chain_id: Chain ID (default: 1 = Ethereum Mainnet)
        
    Returns:
        User prompt string
    """
    return f"""Intent to translate: "{intent}"

Return ONLY the JSON object (no markdown, no explanation). If the intent cannot be translated or is missing required information, return null."""
