"""
Prompt templates for LLM-based intent translation.

The LLM's job is ONLY to understand the intent and classify it: output an action type
and human-readable parameters (amounts, symbols, ENS names). All address resolution,
Wei/base-unit conversion, and transaction construction are done in code (payload_builder).
"""

from typing import Dict, Any, List, Optional


def create_system_prompt(
    token_resolver,
    ens_resolver=None,
    supported_actions: Optional[List[str]] = None,
) -> str:
    """
    Create system prompt for intent classification and parameter extraction only.
    Does NOT ask for Wei, contract addresses, or function names—code does that.

    token_resolver: TokenResolver instance (uses known_erc20_symbols() / known_collection_aliases()).
    ens_resolver: ENSResolver instance (uses known_names() for prompt).
    supported_actions: list of action names from playbooks (e.g. PlaybookEngine.get_supported_actions()).
    If provided, appends a summary line of DeFi actions to the prompt.
    """
    # List supported symbols/names for reference (no addresses—we resolve in code)
    erc20_symbols = token_resolver.known_erc20_symbols() if token_resolver else []
    erc721_names = token_resolver.known_collection_aliases() if token_resolver else []
    ens_names = ens_resolver.known_names() if ens_resolver else []

    base = """You are an intent classifier and parameter extractor for blockchain transactions.
Your job is ONLY to:
1. Classify the user's intent into one of the supported actions below.
2. Extract human-readable parameters from the text (amounts as numbers/strings, asset symbols, ENS names or 0x addresses as the user wrote them).

You must NOT:
- Look up or output contract addresses
- Convert amounts to Wei or base units
- Output function names or target_contract
We do all of that in code. Just return action + arguments with human-readable values only.

Supported asset symbols (use as-is in "asset"): """ + ", ".join(erc20_symbols or ["USDC", "USDT", "DAI", "WETH", "stETH"]) + """
Supported NFT collections (use as-is in "collection"): """ + ", ".join(erc721_names or ["Bored Ape Yacht Club", "CryptoPunks"]) + """
Supported ENS names (pass through as user wrote, e.g. alice.eth): """ + ", ".join(ens_names or ["alice.eth", "bob.eth"]) + """

OUTPUT: A single JSON object with "action" and "arguments". No other fields. No markdown.
If the intent is ambiguous, unsupported, or missing required info, return null.

SCHEMA BY ACTION:

transfer_native (send ETH):
  arguments: { "to": "<ENS or 0x address as user said>", "amount_human": "<ETH amount e.g. 0.5 or 1>" }

transfer_erc20 (send token):
  arguments: { "to": "<ENS or 0x>", "amount_human": "<amount e.g. 100>", "asset": "<symbol e.g. USDC>" }

transfer_erc721 (send NFT):
  arguments: { "to": "<ENS or 0x>", "tokenId": <integer>, "collection": "<collection name or symbol>" }

lido_stake (stake ETH for stETH):
  arguments: { "amount_human": "<ETH amount e.g. 1>" }

lido_unstake (withdraw stETH from Lido):
  arguments: { "amount_human": "<stETH amount e.g. 15>" }

aave_supply:
  arguments: { "asset": "<symbol e.g. USDC>", "amount_human": "<amount>", "onBehalfOf": "<address or ENS>" }
aave_withdraw:
  arguments: { "asset": "<symbol>", "amount_human": "<amount or 'max'>", "to": "<address or ENS>" }
  If the user says "withdraw all", "withdraw everything", "withdraw full balance", set amount_human to "max".
aave_borrow:
  arguments: { "asset": "<symbol>", "amount_human": "<amount>", "onBehalfOf": "<address or ENS>" }
aave_repay:
  arguments: { "asset": "<symbol>", "amount_human": "<amount or 'max'>", "onBehalfOf": "<address or ENS>" }
  If the user says "repay all", "repay full debt", "repay everything", "pay off", set amount_human to "max".

uniswap_swap (Uniswap V3 single-hop):
  arguments: { "amount_human": "<input amount>", "asset_in": "<input token symbol e.g. WETH>", "asset_out": "<output token symbol e.g. USDC>", "to": "<address or ENS>" }
  Optional: "amountOutMinimum" as string if user specifies min output / slippage; otherwise we use "0".
  Optional: "fee" as integer if user specifies pool fee tier (100, 500, 3000, 10000); otherwise auto-detected.

curve_add_liquidity:
  arguments: { "amount_human": "<amount>", "asset": "<symbol e.g. USDC>" }
curve_remove_liquidity:
  arguments: { "amount_human": "<LP token amount>" }

Return ONLY valid JSON: { "action": "<action>", "arguments": { ... } }. No explanation. If unclear, return null.
"""

    # Filter to DeFi actions (exclude transfers — those are always available)
    defi_actions = [a for a in (supported_actions or []) if not a.startswith("transfer_")]
    if defi_actions:
        base += "\nDeFi actions you may classify: " + ", ".join(defi_actions) + "\n"
    return base


def create_user_prompt(intent: str, chain_id: int = 1) -> str:
    """
    Create user prompt with the intent to classify and extract from.
    """
    return f"""Classify this intent and extract parameters (human-readable only):

"{intent}"

Return ONLY the JSON object: {{ "action": "...", "arguments": {{ ... }} }}. No markdown. If you cannot classify or required info is missing, return null."""
