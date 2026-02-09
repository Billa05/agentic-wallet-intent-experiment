"""
Pydantic schemas for the Agentic Wallet Intent Translation System.

These models represent structured blockchain transaction intents derived from
natural language. They include strong validation (especially Ethereum address
format) and helpers for converting human-readable amounts into on-chain units.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict, field_validator


_ETH_ADDRESS_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")


def _require_eth_address(value: str) -> str:
    """
    Validate that `value` is an Ethereum address in the strict format:
    `0x` followed by exactly 40 hex characters.
    """
    if not _ETH_ADDRESS_REGEX.fullmatch(value):
        raise ValueError("Invalid Ethereum address (expected 0x + 40 hex chars)")
    return value


class TransactionType(str, Enum):
    """
    Supported transaction intent types.
    """

    SEND_ETH = "SEND_ETH"
    TRANSFER_ERC20 = "TRANSFER_ERC20"
    TRANSFER_ERC721 = "TRANSFER_ERC721"


class BaseTransaction(BaseModel):
    """
    Base schema shared by all transaction intents.

    Attributes:
        transaction_type: Discriminator describing the intent category.
        from_address: Optional sender address (may be inferred by wallet context).
        to_address: Recipient address.
        timestamp: UTC timestamp when the intent was produced.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_type: TransactionType
    from_address: Optional[str] = None
    to_address: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("to_address")
    @classmethod
    def _validate_to_address(cls, v: str) -> str:
        return _require_eth_address(v)

    @field_validator("from_address")
    @classmethod
    def _validate_from_address(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _require_eth_address(v)


class SendETHTransaction(BaseTransaction):
    """
    Ethereum native token transfer intent.

    Attributes:
        amount: Amount in ETH as a string (e.g. "0.5").
    """

    transaction_type: TransactionType = Field(default=TransactionType.SEND_ETH)
    amount: str

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, v: str) -> str:
        try:
            d = Decimal(v)
        except (InvalidOperation, TypeError):
            raise ValueError("amount must be a decimal string") from None
        if d <= 0:
            raise ValueError("amount must be > 0")
        return v

    def amount_wei(self) -> int:
        """
        Convert the human-readable ETH `amount` into Wei.

        Returns:
            Integer Wei amount, rounded down to the nearest Wei.
        """
        wei_per_eth = Decimal("1000000000000000000")  # 1e18
        wei = (Decimal(self.amount) * wei_per_eth).to_integral_value(rounding=ROUND_DOWN)
        return int(wei)


class TransferERC20Transaction(BaseTransaction):
    """
    ERC-20 token transfer intent.

    Attributes:
        token_address: ERC-20 contract address.
        amount: Human-readable token amount as a string (e.g. "12.34").
        decimals: Token decimals (default 18).
    """

    transaction_type: TransactionType = Field(default=TransactionType.TRANSFER_ERC20)
    token_address: str
    amount: str
    decimals: int = 18

    @field_validator("token_address")
    @classmethod
    def _validate_token_address(cls, v: str) -> str:
        return _require_eth_address(v)

    @field_validator("decimals")
    @classmethod
    def _validate_decimals(cls, v: int) -> int:
        if not isinstance(v, int):
            raise ValueError("decimals must be an integer")
        if v < 0 or v > 255:
            raise ValueError("decimals must be in [0, 255]")
        return v

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, v: str) -> str:
        try:
            d = Decimal(v)
        except (InvalidOperation, TypeError):
            raise ValueError("amount must be a decimal string") from None
        if d <= 0:
            raise ValueError("amount must be > 0")
        return v

    def amount_base_units(self) -> int:
        """
        Convert the human-readable token `amount` into base units (uint256).

        Example:
            amount="1.5", decimals=6 -> 1500000

        Returns:
            Integer base-unit amount, rounded down to the smallest unit.
        """
        scale = Decimal(10) ** int(self.decimals)
        base_units = (Decimal(self.amount) * scale).to_integral_value(rounding=ROUND_DOWN)
        return int(base_units)


class TransferERC721Transaction(BaseTransaction):
    """
    ERC-721 NFT transfer intent.

    Attributes:
        contract_address: ERC-721 contract address.
        token_id: Token ID (uint256).
    """

    transaction_type: TransactionType = Field(default=TransactionType.TRANSFER_ERC721)
    contract_address: str
    token_id: int

    @field_validator("contract_address")
    @classmethod
    def _validate_contract_address(cls, v: str) -> str:
        return _require_eth_address(v)

    @field_validator("token_id")
    @classmethod
    def _validate_token_id(cls, v: int) -> int:
        if not isinstance(v, int):
            raise ValueError("token_id must be an integer")
        if v < 0:
            raise ValueError("token_id must be >= 0")
        return v


class ActionType(str, Enum):
    """
    Action types for executable payloads.
    Includes simple transfers and DeFi actions (AAVE, Lido, Uniswap, Curve).
    """
    # Simple transfers
    TRANSFER_NATIVE = "transfer_native"
    TRANSFER_ERC20 = "transfer_erc20"
    TRANSFER_ERC721 = "transfer_erc721"
    # AAVE V3
    AAVE_SUPPLY = "aave_supply"
    AAVE_WITHDRAW = "aave_withdraw"
    AAVE_BORROW = "aave_borrow"
    AAVE_REPAY = "aave_repay"
    # Lido
    LIDO_STAKE = "lido_stake"
    LIDO_UNSTAKE = "lido_unstake"
    # Uniswap V2
    UNISWAP_SWAP = "uniswap_swap"
    # Curve
    CURVE_ADD_LIQUIDITY = "curve_add_liquidity"
    CURVE_REMOVE_LIQUIDITY = "curve_remove_liquidity"


# Golden schema: required arguments per action type (for dataset and evaluation).
# All amounts in arguments must be Wei/base units as strings.
ACTION_REQUIRED_ARGS: Dict[str, list] = {
    "transfer_native": ["to", "value", "human_readable_amount"],
    "transfer_erc20": ["to", "value", "human_readable_amount"],
    "transfer_erc721": ["to", "tokenId", "human_readable_amount"],
    "aave_supply": ["asset", "amount", "onBehalfOf", "human_readable_amount"],
    "aave_withdraw": ["asset", "amount", "to", "human_readable_amount"],
    "aave_borrow": ["asset", "amount", "onBehalfOf", "human_readable_amount"],
    "aave_repay": ["asset", "amount", "onBehalfOf", "human_readable_amount"],
    "lido_stake": ["value", "human_readable_amount"],
    "lido_unstake": ["amount", "human_readable_amount"],
    "uniswap_swap": ["amountIn", "amountOutMin", "path", "to", "human_readable_amount"],
    "curve_add_liquidity": ["pool", "amounts", "min_mint_amount", "human_readable_amount"],
    "curve_remove_liquidity": ["pool", "amount", "min_amounts", "human_readable_amount"],
}


class UserContext(BaseModel):
    """
    User context information for transaction execution.
    
    Attributes:
        current_chain_id: The chain ID where the transaction will execute (1 = Ethereum Mainnet)
        token_prices: Optional dictionary of token prices for UI/validation purposes
    """
    current_chain_id: int = Field(default=1, description="Chain ID (1 = Ethereum Mainnet)")
    token_prices: Optional[Dict[str, float]] = Field(default=None, description="Token prices for UI display")


class ExecutablePayload(BaseModel):
    """
    Production-ready executable payload that maps directly to wallet software requirements.
    
    This schema ensures:
    - All amounts are in Wei/base units (integers as strings to prevent overflow)
    - Chain context is explicit (prevents replay attacks)
    - Structure maps directly to ethers.js/viem transaction formats
    - Human-readable amounts preserved for UI confirmation
    
    Attributes:
        chain_id: Explicit chain ID (1 = Ethereum Mainnet)
        action: Action type (transfer_native, transfer_erc20, transfer_erc721)
        target_contract: Contract address (null for native ETH)
        function_name: Function to call (null for native ETH, "transfer" for ERC-20/721)
        arguments: Transaction arguments with value in Wei/base units
    """
    model_config = ConfigDict(extra="forbid")
    
    chain_id: int = Field(description="Chain ID (1 = Ethereum Mainnet)")
    action: ActionType = Field(description="Action type")
    target_contract: Optional[str] = Field(default=None, description="Contract address (null for native ETH)")
    function_name: Optional[str] = Field(default=None, description="Function name (null for native ETH)")
    arguments: Dict[str, Any] = Field(description="Transaction arguments")
    
    @field_validator("target_contract")
    @classmethod
    def _validate_target_contract(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _require_eth_address(v)
    
    @field_validator("chain_id")
    @classmethod
    def _validate_chain_id(cls, v: int) -> int:
        if v not in [1, 5, 11155111]:  # Mainnet, Goerli, Sepolia
            # Allow other chain IDs but warn
            if v < 1:
                raise ValueError("chain_id must be positive")
        return v


class AnnotatedIntent(BaseModel):
    """
    Complete annotated intent with user context and executable payload.
    
    This is the "golden schema" that proves understanding of blockchain execution:
    - user_intent: Original natural language text
    - user_context: Chain context and token prices
    - target_payload: Executable payload ready for wallet signing
    
    Attributes:
        user_intent: Original natural language intent text
        user_context: User context (chain_id, token_prices)
        target_payload: Executable payload ready for transaction execution
    """
    model_config = ConfigDict(extra="forbid")
    
    user_intent: str = Field(description="Original natural language intent")
    user_context: UserContext = Field(description="User context (chain_id, token_prices)")
    target_payload: ExecutablePayload = Field(description="Executable payload for transaction")

