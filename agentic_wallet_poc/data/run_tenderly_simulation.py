"""
Run Tenderly simulation on an already-annotated dataset and update _tenderly_validated.

Features:
- Token registry: balance/allowance slots for common ERC20s (add new tokens easily)
- ERC721 registry: ownership slot for common NFT collections
- Auto-retry: detects balance/ownership errors and retries with different slots

Usage:
  python data/run_tenderly_simulation.py --input data/datasets/annotated/annotated_hybrid_dataset.json
  python data/run_tenderly_simulation.py --input ... --output ...
  python data/run_tenderly_simulation.py --input ... --max-retries 3

Requires: TENDERLY_ACCESS_KEY, TENDERLY_ACCOUNT_SLUG, TENDERLY_PROJECT_SLUG in env.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# =============================================================================
# TOKEN REGISTRY - Add new tokens here (one line per token)
# =============================================================================
# Format: "address": {"balance_slot": N, "allowance_slot": M, "name": "..."}
# To find slots: check contract source on Etherscan or use storage layout tools

ERC20_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Stablecoins
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": {"balance_slot": 9, "allowance_slot": 10, "name": "USDC"},
    "0xdac17f958d2ee523a2206206994597c13d831ec7": {"balance_slot": 2, "allowance_slot": 4, "name": "USDT"},
    "0x6b175474e89094c44da98b954eedeac495271d0f": {"balance_slot": 2, "allowance_slot": 3, "name": "DAI"},
    "0x4fabb145d64652a948d72533023f6e7a623c7c53": {"balance_slot": 1, "allowance_slot": 2, "name": "BUSD"},
    "0x8e870d67f660d95d5be530380d0ec0bd388289e1": {"balance_slot": 1, "allowance_slot": 2, "name": "USDP"},
    "0x0000000000085d4780b73119b644ae5ecd22b376": {"balance_slot": 14, "allowance_slot": 15, "name": "TUSD"},
    "0x853d955acef822db058eb8505911ed77f175b99e": {"balance_slot": 0, "allowance_slot": 1, "name": "FRAX"},
    
    # Wrapped ETH
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": {"balance_slot": 3, "allowance_slot": 4, "name": "WETH"},
    
    # Major tokens
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": {"balance_slot": 0, "allowance_slot": 1, "name": "WBTC"},
    "0x514910771af9ca656af840dff83e8264ecf986ca": {"balance_slot": 1, "allowance_slot": 2, "name": "LINK"},
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": {"balance_slot": 4, "allowance_slot": 5, "name": "UNI"},
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": {"balance_slot": 0, "allowance_slot": 1, "name": "AAVE"},
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": {"balance_slot": 1, "allowance_slot": 2, "name": "MKR"},
    "0xc00e94cb662c3520282e6f5717214004a7f26888": {"balance_slot": 2, "allowance_slot": 3, "name": "COMP"},
    "0x6982508145454ce325ddbe47a25d4ec3d2311933": {"balance_slot": 0, "allowance_slot": 1, "name": "PEPE"},
    "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce": {"balance_slot": 0, "allowance_slot": 1, "name": "SHIB"},
    
    # Lido
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": {"balance_slot": 0, "allowance_slot": 1, "name": "stETH"},
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0": {"balance_slot": 0, "allowance_slot": 1, "name": "wstETH"},
}

# ERC721 collections - storage slots for owners, balances, approvals
# OpenZeppelin ERC721 layout: _owners=2, _balances=3, _tokenApprovals=4, _operatorApprovals=5
# ERC721Enumerable adds: _ownedTokens=6, _ownedTokensIndex=7, _allTokens=8, _allTokensIndex=9
ERC721_REGISTRY: Dict[str, Dict[str, Any]] = {
    # BAYC uses OpenZeppelin 3.x ERC721Enumerable
    # Storage: Ownable._owner(0), ERC721._name(1), _symbol(2), _owners(3), _balances(4), 
    #          _tokenApprovals(5), _operatorApprovals(6), then Enumerable slots...
    # Note: Ownable shifts everything by 1 in OZ 3.x
    "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d": {
        "owners_slot": 3, "balances_slot": 4, "approvals_slot": 5,
        "all_tokens_index_slot": 10,  # ERC721Enumerable (shifted by 1)
        "name": "BAYC", "standard": True, "enumerable": True
    },
    "0x60e4d786628fea6478f785a6d7e704777c86a7c6": {
        "owners_slot": 2, "balances_slot": 3, "approvals_slot": 4,
        "all_tokens_index_slot": 9,
        "name": "MAYC", "standard": True, "enumerable": True
    },
    "0xba30e5f9bb24caa003e9f2f0497ad287fdf95623": {
        "owners_slot": 2, "balances_slot": 3, "approvals_slot": 4,
        "name": "Bored Ape Kennel Club", "standard": True
    },
    "0xed5af388653567af2f388e6224dc7c4b3241c544": {
        "owners_slot": 2, "balances_slot": 3, "approvals_slot": 4,
        "name": "Azuki", "standard": True
    },
    "0x49cf6f5d44e70224e2e23fdcdd2c053f30ada28b": {
        "owners_slot": 2, "balances_slot": 3, "approvals_slot": 4,
        "name": "CloneX", "standard": True
    },
    "0x8a90cab2b38dba80c64b7734e58ee1db38b8992e": {
        "owners_slot": 2, "balances_slot": 3, "approvals_slot": 4,
        "name": "Doodles", "standard": True
    },
    "0x23581767a106ae21c074b2276d25e5c3e136a68b": {
        "owners_slot": 2, "balances_slot": 3, "approvals_slot": 4,
        "name": "Moonbirds", "standard": True
    },
    "0x34d85c9cdeb23fa97cb08333b511ac86e1c4e258": {
        "owners_slot": 2, "balances_slot": 3, "approvals_slot": 4,
        "name": "Otherdeed", "standard": True
    },
    "0x7bd29408f11d2bfc23c34f18275bbf23bb716bc7": {
        "owners_slot": 2, "balances_slot": 3, "approvals_slot": 4,
        "name": "Meebits", "standard": True
    },
    
    # CryptoPunks: non-standard interface (transferPunk)
    # punkIndexToAddress at slot 10, balanceOf at slot 13
    "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb": {
        "owners_slot": 10, "balances_slot": 13,
        "name": "CryptoPunks", "standard": "cryptopunks"
    },
}

# Default slots for unknown ERC721 contracts (OpenZeppelin layout)
DEFAULT_ERC721_BALANCES_SLOT = 3
DEFAULT_ERC721_APPROVALS_SLOT = 4

# Common balance slot fallbacks to try (in order)
DEFAULT_ERC20_SLOTS = [0, 2, 1, 3, 4, 5, 9, 51]
DEFAULT_ERC721_OWNER_SLOTS = [2, 3, 4, 5, 6]

# Error patterns that indicate balance/ownership issues (for auto-retry)
BALANCE_ERROR_PATTERNS = [
    r"insufficient.?balance",
    r"transfer amount exceeds balance",
    r"exceeds balance",
    r"not enough",
    r"balance too low",
    r"Dai/insufficient-balance",
    r"ERC20: transfer amount",
]

OWNERSHIP_ERROR_PATTERNS = [
    r"not owner",
    r"caller is not owner",
    r"ERC721: transfer caller is not owner nor approved",
    r"ERC721: transfer from incorrect owner",
    r"not the owner",
    r"transfer of token that is not own",
]

# ERC721Enumerable slot combinations to try: (owners, balances, approvals, allTokensIndex)
# Different OpenZeppelin versions have different layouts
ERC721_SLOT_COMBINATIONS = [
    (2, 3, 4, 9),    # Standard OZ 4.x ERC721Enumerable
    (3, 4, 5, 10),   # OZ 3.x with Ownable first
    (6, 7, 8, 13),   # Some older contracts
    (2, 3, 4, None), # Standard ERC721 (non-enumerable)
    (3, 4, 5, None), # OZ 3.x ERC721
]


# =============================================================================
# Slot Discovery Cache (file-based for persistence)
# =============================================================================

CACHE_FILE = project_root / "data" / ".slot_cache.json"

def _load_cache() -> Dict[str, Any]:
    """Load discovered slots from cache file."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_cache(cache: Dict[str, Any]) -> None:
    """Save discovered slots to cache file."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass  # Cache is optional, don't fail

def get_cached_slot(key: str) -> Optional[Any]:
    """Get a cached slot value."""
    cache = _load_cache()
    return cache.get(key)

def set_cached_slot(key: str, value: Any) -> None:
    """Cache a discovered slot value."""
    cache = _load_cache()
    cache[key] = value
    _save_cache(cache)


# =============================================================================
# Storage Slot Computation
# =============================================================================

def _storage_slot(key: str, mapping_slot: int) -> str:
    """
    Compute storage slot for mapping(address => uint256).
    slot = keccak256(abi.encode(key, mapping_slot))
    """
    from eth_utils import keccak
    key = key.lower()
    if key.startswith("0x"):
        key = key[2:]
    key_padded = key.zfill(64)
    slot_padded = hex(mapping_slot)[2:].zfill(64)
    raw = bytes.fromhex(key_padded + slot_padded)
    return "0x" + keccak(raw).hex()


def _storage_slot_uint(key: int, mapping_slot: int) -> str:
    """Compute storage slot for mapping(uint256 => address) like ERC721 _owners."""
    from eth_utils import keccak
    key_padded = hex(key)[2:].zfill(64)
    slot_padded = hex(mapping_slot)[2:].zfill(64)
    raw = bytes.fromhex(key_padded + slot_padded)
    return "0x" + keccak(raw).hex()


def _allowance_slot(owner: str, spender: str, allowance_base_slot: int) -> str:
    """
    Compute storage slot for allowance[owner][spender].
    slot = keccak256(abi.encode(spender, keccak256(abi.encode(owner, allowance_base_slot))))
    """
    from eth_utils import keccak
    # First level: mapping(owner => ...)
    inner = _storage_slot(owner, allowance_base_slot)
    # Second level: mapping(... => spender)
    spender = spender.lower()
    if spender.startswith("0x"):
        spender = spender[2:]
    spender_padded = spender.zfill(64)
    inner_padded = inner[2:]  # remove 0x
    raw = bytes.fromhex(spender_padded + inner_padded)
    return "0x" + keccak(raw).hex()


# =============================================================================
# State Override Builders
# =============================================================================

def build_erc20_balance_override(
    token_address: str,
    holder: str,
    balance_slot: int,
    amount: int = 10**30,
) -> Dict[str, Any]:
    """Build storage override for ERC20 balanceOf(holder)."""
    slot = _storage_slot(holder, balance_slot)
    return {
        token_address: {
            "storage": {
                slot: "0x" + format(amount, "064x"),
            }
        }
    }


def build_erc721_owner_override(
    nft_address: str,
    token_id: int,
    owner: str,
    owners_slot: int,
    balances_slot: Optional[int] = None,
    approvals_slot: Optional[int] = None,
    all_tokens_index_slot: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build storage override for ERC721:
    - _owners[tokenId] = owner
    - _balances[owner] = 1 (so transfer doesn't underflow)
    - _tokenApprovals[tokenId] = owner (so approval check passes)
    - _allTokensIndex[tokenId] = 1 (for ERC721Enumerable - makes token "exist")
    """
    owner_clean = owner.lower()
    if owner_clean.startswith("0x"):
        owner_clean = owner_clean[2:]
    
    storage = {}
    
    # Set _owners[tokenId] = owner
    owner_slot_key = _storage_slot_uint(token_id, owners_slot)
    storage[owner_slot_key] = "0x" + owner_clean.zfill(64)
    
    # Set _balances[owner] = 1 (to prevent underflow on transfer)
    if balances_slot is not None:
        balance_slot_key = _storage_slot(owner, balances_slot)
        storage[balance_slot_key] = "0x" + format(1, "064x")
    
    # Set _tokenApprovals[tokenId] = owner (auto-approve for the owner)
    if approvals_slot is not None:
        approval_slot_key = _storage_slot_uint(token_id, approvals_slot)
        storage[approval_slot_key] = "0x" + owner_clean.zfill(64)
    
    # For ERC721Enumerable: set _allTokensIndex[tokenId] = 1 (token must "exist")
    if all_tokens_index_slot is not None:
        all_tokens_slot_key = _storage_slot_uint(token_id, all_tokens_index_slot)
        # Set to 1 (non-zero means token exists in enumeration; index 0 would be valid too but 1 is safer)
        storage[all_tokens_slot_key] = "0x" + format(1, "064x")
    
    return {
        nft_address: {
            "storage": storage,
        }
    }


def build_state_overrides(
    from_address: str,
    action: Optional[str],
    value_wei: str,
    token_address: Optional[str] = None,
    token_id: Optional[int] = None,
    balance_slot_override: Optional[int] = None,
    owners_slot_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build comprehensive state overrides for simulation.
    
    Args:
        from_address: Sender address
        action: Transaction type (transfer_native, transfer_erc20, transfer_erc721)
        value_wei: ETH value being sent
        token_address: Token contract address (for ERC20/ERC721)
        token_id: Token ID (for ERC721)
        balance_slot_override: Force specific balance slot (for retry)
        owners_slot_override: Force specific owners slot (for retry)
    """
    overrides: Dict[str, Any] = {}
    
    # Always give sender enough ETH for value + gas
    eth_needed = int(value_wei) if value_wei else 0
    eth_balance = max(eth_needed, 10**18) + 10 * 10**18  # 10 ETH buffer
    overrides[from_address.lower()] = {"balance": hex(eth_balance)}
    
    if action == "transfer_erc20" and token_address:
        token_lower = token_address.lower()
        registry_entry = ERC20_REGISTRY.get(token_lower)
        
        if balance_slot_override is not None:
            balance_slot = balance_slot_override
        elif registry_entry:
            balance_slot = registry_entry["balance_slot"]
        else:
            balance_slot = DEFAULT_ERC20_SLOTS[0]  # Default to slot 0
        
        erc20_override = build_erc20_balance_override(
            token_lower, from_address, balance_slot, 10**30
        )
        overrides.update(erc20_override)
    
    if action == "transfer_erc721" and token_address and token_id is not None:
        token_lower = token_address.lower()
        registry_entry = ERC721_REGISTRY.get(token_lower)
        
        # Check if completely unsupported (standard=False with no owners_slot)
        if registry_entry and registry_entry.get("standard") is False and registry_entry.get("owners_slot") is None:
            # Truly non-standard NFT with no known storage layout
            return overrides
        
        if owners_slot_override is not None:
            owners_slot = owners_slot_override
        elif registry_entry and registry_entry.get("owners_slot") is not None:
            owners_slot = registry_entry["owners_slot"]
        else:
            owners_slot = DEFAULT_ERC721_OWNER_SLOTS[0]
        
        # Get balances slot (to prevent underflow on transfer)
        if registry_entry and registry_entry.get("balances_slot") is not None:
            balances_slot = registry_entry["balances_slot"]
        else:
            balances_slot = DEFAULT_ERC721_BALANCES_SLOT
        
        # Get approvals slot (for _tokenApprovals[tokenId])
        if registry_entry and registry_entry.get("approvals_slot") is not None:
            approvals_slot = registry_entry["approvals_slot"]
        else:
            approvals_slot = DEFAULT_ERC721_APPROVALS_SLOT
        
        # Get all_tokens_index slot for ERC721Enumerable (makes token "exist")
        all_tokens_index_slot = None
        if registry_entry and registry_entry.get("all_tokens_index_slot") is not None:
            all_tokens_index_slot = registry_entry["all_tokens_index_slot"]
        
        erc721_override = build_erc721_owner_override(
            token_lower, token_id, from_address, owners_slot, balances_slot,
            approvals_slot, all_tokens_index_slot
        )
        overrides.update(erc721_override)
    
    return overrides


# =============================================================================
# Tenderly API
# =============================================================================

def tenderly_simulate(
    from_address: str,
    to_address: str,
    value: str,
    data: str,
    state_objects: Optional[Dict[str, Any]] = None,
    network_id: str = "1",
    block_number: int = 21_000_000,
    simulation_type: str = "full",
) -> Dict[str, Any]:
    """POST to Tenderly simulate API with optional state overrides."""
    import urllib.request
    import urllib.error
    
    access_key = os.getenv("TENDERLY_ACCESS_KEY")
    account_slug = os.getenv("TENDERLY_ACCOUNT_SLUG")
    project_slug = os.getenv("TENDERLY_PROJECT_SLUG")
    if not access_key or not account_slug or not project_slug:
        return {"success": False, "error": "Missing Tenderly env vars"}
    
    url = f"https://api.tenderly.co/api/v1/account/{account_slug}/project/{project_slug}/simulate"
    payload = {
        "network_id": network_id,
        "block_number": block_number,
        "from": from_address,
        "to": to_address,
        "gas": 8_000_000,
        "gas_price": 0,
        "value": int(value) if value else 0,
        "input": data if data.startswith("0x") else "0x" + data,
        "simulation_type": simulation_type,
    }
    if state_objects:
        payload["state_objects"] = state_objects
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Access-Key": access_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err_json = json.loads(body)
            err_msg = err_json.get("error", {}).get("message", body) or body
        except Exception:
            err_msg = body
        return {"success": False, "error": err_msg or str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    tx = response_data.get("transaction") or {}
    status = tx.get("status")
    if status is False or (isinstance(status, str) and str(status).lower() in ("0", "false", "reverted")):
        return {
            "success": False,
            "error": tx.get("error_message") or "Transaction reverted",
            "response": response_data,
        }
    return {"success": True, "error": None, "response": response_data}


# =============================================================================
# Dynamic Slot Discovery
# =============================================================================

def discover_erc20_balance_slot(
    token_address: str,
    holder: str,
    network_id: str = "1",
    block_number: int = 21_000_000,
) -> Optional[int]:
    """
    Discover ERC20 balance slot by trial-and-error.
    Tests a simple transfer call with different slot overrides.
    """
    token_lower = token_address.lower()
    cache_key = f"erc20:{token_lower}"
    
    # Check cache first
    cached = get_cached_slot(cache_key)
    if cached is not None:
        return cached
    
    # Build a minimal transfer call: transfer(address, 1)
    # 0xa9059cbb = transfer(address,uint256)
    dummy_recipient = "0x" + "1" * 40
    transfer_data = "0xa9059cbb" + dummy_recipient[2:].zfill(64) + format(1, "064x")
    
    for slot in DEFAULT_ERC20_SLOTS:
        state = build_erc20_balance_override(token_lower, holder, slot, 10**30)
        state[holder.lower()] = {"balance": hex(10**18)}  # ETH for gas
        
        result = tenderly_simulate(
            from_address=holder,
            to_address=token_lower,
            value="0",
            data=transfer_data,
            state_objects=state,
            network_id=network_id,
            block_number=block_number,
        )
        
        if result.get("success"):
            set_cached_slot(cache_key, slot)
            return slot
    
    return None


def discover_erc721_slots(
    nft_address: str,
    token_id: int,
    owner: str,
    network_id: str = "1",
    block_number: int = 21_000_000,
) -> Optional[Tuple[int, int, int, Optional[int]]]:
    """
    Discover ERC721 storage slots by trial-and-error.
    Returns (owners_slot, balances_slot, approvals_slot, all_tokens_index_slot) or None.
    """
    nft_lower = nft_address.lower()
    cache_key = f"erc721:{nft_lower}"
    
    # Check cache first
    cached = get_cached_slot(cache_key)
    if cached is not None:
        return tuple(cached) if cached else None
    
    # Build a transferFrom call
    # 0x23b872dd = transferFrom(address,address,uint256)
    dummy_recipient = "0x" + "2" * 40
    transfer_data = (
        "0x23b872dd" +
        owner[2:].lower().zfill(64) +
        dummy_recipient[2:].zfill(64) +
        format(token_id, "064x")
    )
    
    for owners_slot, balances_slot, approvals_slot, all_tokens_idx in ERC721_SLOT_COMBINATIONS:
        override = build_erc721_owner_override(
            nft_lower, token_id, owner, owners_slot, balances_slot,
            approvals_slot, all_tokens_idx
        )
        override[owner.lower()] = {"balance": hex(10**18)}  # ETH for gas
        
        result = tenderly_simulate(
            from_address=owner,
            to_address=nft_lower,
            value="0",
            data=transfer_data,
            state_objects=override,
            network_id=network_id,
            block_number=block_number,
        )
        
        if result.get("success"):
            slots = [owners_slot, balances_slot, approvals_slot, all_tokens_idx]
            set_cached_slot(cache_key, slots)
            return (owners_slot, balances_slot, approvals_slot, all_tokens_idx)
    
    return None


def get_erc20_slots(token_address: str, holder: str, network_id: str, block_number: int) -> int:
    """
    Get ERC20 balance slot using hybrid approach:
    1. Hardcoded registry (fast)
    2. Cache (discovered before)
    3. Dynamic discovery (slow but works for any token)
    4. Default fallback
    """
    token_lower = token_address.lower()
    
    # 1. Check hardcoded registry
    if token_lower in ERC20_REGISTRY:
        return ERC20_REGISTRY[token_lower]["balance_slot"]
    
    # 2. Check cache
    cached = get_cached_slot(f"erc20:{token_lower}")
    if cached is not None:
        return cached
    
    # 3. Try dynamic discovery
    discovered = discover_erc20_balance_slot(token_lower, holder, network_id, block_number)
    if discovered is not None:
        return discovered
    
    # 4. Default fallback
    return DEFAULT_ERC20_SLOTS[0]


def get_erc721_slots(
    nft_address: str,
    token_id: int,
    owner: str,
    network_id: str,
    block_number: int,
) -> Tuple[int, int, int, Optional[int]]:
    """
    Get ERC721 storage slots using hybrid approach:
    1. Hardcoded registry
    2. Cache
    3. Dynamic discovery
    4. Default fallback
    """
    nft_lower = nft_address.lower()
    
    # 1. Check hardcoded registry
    if nft_lower in ERC721_REGISTRY:
        entry = ERC721_REGISTRY[nft_lower]
        return (
            entry.get("owners_slot", 2),
            entry.get("balances_slot", DEFAULT_ERC721_BALANCES_SLOT),
            entry.get("approvals_slot", DEFAULT_ERC721_APPROVALS_SLOT),
            entry.get("all_tokens_index_slot"),
        )
    
    # 2. Check cache
    cached = get_cached_slot(f"erc721:{nft_lower}")
    if cached is not None:
        return tuple(cached)
    
    # 3. Try dynamic discovery
    discovered = discover_erc721_slots(nft_lower, token_id, owner, network_id, block_number)
    if discovered is not None:
        return discovered
    
    # 4. Default fallback (standard OZ ERC721)
    return (2, DEFAULT_ERC721_BALANCES_SLOT, DEFAULT_ERC721_APPROVALS_SLOT, None)


# =============================================================================
# Auto-Retry Logic
# =============================================================================

def matches_error_pattern(error: str, patterns: List[str]) -> bool:
    """Check if error matches any of the regex patterns."""
    if not error:
        return False
    error_lower = error.lower()
    for pattern in patterns:
        if re.search(pattern, error_lower, re.IGNORECASE):
            return True
    return False


def simulate_with_retry(
    from_address: str,
    to_address: str,
    value: str,
    data: str,
    action: Optional[str],
    token_address: Optional[str],
    token_id: Optional[int],
    network_id: str,
    block_number: int,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Simulate with auto-retry on balance/ownership errors.
    Uses hybrid approach: registry → cache → discovery → fallback
    """
    # For ERC721, try dynamic discovery first if not in registry
    if action == "transfer_erc721" and token_address and token_id is not None:
        token_lower = token_address.lower()
        
        # Use hybrid slot discovery
        owners_slot, balances_slot, approvals_slot, all_tokens_idx = get_erc721_slots(
            token_lower, token_id, from_address, network_id, block_number
        )
        
        # Build state override with discovered slots
        state_objects = {from_address.lower(): {"balance": hex(10 * 10**18)}}
        erc721_override = build_erc721_owner_override(
            token_lower, token_id, from_address, owners_slot, balances_slot,
            approvals_slot, all_tokens_idx
        )
        state_objects.update(erc721_override)
        
        result = tenderly_simulate(
            from_address, to_address, value, data,
            state_objects, network_id, block_number
        )
        
        # If failed and slots were from discovery, try other combinations
        if not result.get("success") and token_lower not in ERC721_REGISTRY:
            for combo in ERC721_SLOT_COMBINATIONS:
                if combo == (owners_slot, balances_slot, approvals_slot, all_tokens_idx):
                    continue  # Already tried
                
                o_slot, b_slot, a_slot, ati_slot = combo
                state_objects = {from_address.lower(): {"balance": hex(10 * 10**18)}}
                erc721_override = build_erc721_owner_override(
                    token_lower, token_id, from_address, o_slot, b_slot, a_slot, ati_slot
                )
                state_objects.update(erc721_override)
                
                result = tenderly_simulate(
                    from_address, to_address, value, data,
                    state_objects, network_id, block_number
                )
                
                if result.get("success"):
                    # Cache the working combination
                    set_cached_slot(f"erc721:{token_lower}", [o_slot, b_slot, a_slot, ati_slot])
                    return result
        
        return result
    
    # For ERC20, use hybrid discovery with retry
    if action == "transfer_erc20":
        token_lower = (token_address or "").lower()
        
        # Get initial slot from hybrid approach
        initial_slot = get_erc20_slots(token_lower, from_address, network_id, block_number)
        slots_to_try = [initial_slot] + [s for s in DEFAULT_ERC20_SLOTS if s != initial_slot]
        
        last_error = None
        attempts = min(len(slots_to_try), max_retries)
        
        for slot in slots_to_try[:attempts]:
            state_objects = build_state_overrides(
                from_address, action, value, token_address, token_id,
                balance_slot_override=slot
            )
            
            result = tenderly_simulate(
                from_address, to_address, value, data,
                state_objects, network_id, block_number
            )
            
            if result.get("success"):
                # Cache the working slot if not from registry
                if token_lower not in ERC20_REGISTRY:
                    set_cached_slot(f"erc20:{token_lower}", slot)
                return result
            
            error = result.get("error", "")
            last_error = error
            
            # Only retry on balance-related errors
            if not matches_error_pattern(error, BALANCE_ERROR_PATTERNS):
                return result
        
        return {"success": False, "error": last_error or "All retry attempts failed"}
    
    # For native transfers, use simple overrides
    state_objects = build_state_overrides(from_address, action, value)
    return tenderly_simulate(
        from_address, to_address, value, data,
        state_objects, network_id, block_number
    )


# =============================================================================
# Dataset Processing
# =============================================================================

def run_simulation_on_dataset(
    rows: List[Dict[str, Any]],
    default_from_address: str,
    network_id: str = "1",
    block_number: int = 21_000_000,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Run Tenderly simulation for each row with auto-retry."""
    out = []
    for i, row in enumerate(rows):
        row = dict(row)
        target = row.get("target_payload")
        metadata = row.get("metadata") or {}
        from_addr = (row.get("user_context") or {}).get("from_address") or default_from_address
        
        if not target or not target.get("to"):
            row["_tenderly_validated"] = False
            row.pop("_tenderly_error", None)
            out.append(row)
            continue
        
        action = metadata.get("action")
        token_address = metadata.get("token_address") or target.get("to")
        token_id = metadata.get("token_id")
        value_wei = str(target.get("value", "0"))
        
        result = simulate_with_retry(
            from_address=from_addr,
            to_address=target["to"],
            value=value_wei,
            data=target.get("data", "0x"),
            action=action,
            token_address=token_address if action != "transfer_native" else None,
            token_id=int(token_id) if token_id is not None else None,
            network_id=network_id,
            block_number=block_number,
            max_retries=max_retries,
        )
        
        row["_tenderly_validated"] = result.get("success", False)
        if result.get("success"):
            row.pop("_tenderly_error", None)
        else:
            row["_tenderly_error"] = result.get("error") or "Simulation failed"
        out.append(row)
        
        if (i + 1) % 5 == 0:
            print(f"  Simulated {i + 1}/{len(rows)}...")
    
    return out


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Tenderly simulation with token registry and auto-retry"
    )
    parser.add_argument(
        "--input",
        default="data/datasets/annotated/annotated_hybrid_dataset.json",
        help="Input annotated JSON",
    )
    parser.add_argument("--output", default="", help="Output path (default: same as input)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input file")
    parser.add_argument(
        "--from-address",
        default=os.getenv("TENDERLY_FROM_ADDRESS", "0xe2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2"),
        help="Default sender if not in user_context",
    )
    parser.add_argument("--network-id", default="1", help="Chain ID")
    parser.add_argument("--block-number", type=int, default=21_000_000, help="Block number for simulation")
    parser.add_argument("--max-retries", type=int, default=3, help="Max slot retry attempts")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1
    
    if (
        not os.getenv("TENDERLY_ACCESS_KEY")
        or not os.getenv("TENDERLY_ACCOUNT_SLUG")
        or not os.getenv("TENDERLY_PROJECT_SLUG")
    ):
        print("Error: set TENDERLY_ACCESS_KEY, TENDERLY_ACCOUNT_SLUG, TENDERLY_PROJECT_SLUG in env.")
        return 1
    
    output_path = Path(args.output) if args.output else input_path
    if not output_path.is_absolute():
        output_path = project_root / output_path
    if args.in_place:
        output_path = input_path
    
    print(f"Loading {input_path}...")
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        print("Error: expected JSON array")
        return 1
    
    # Separate annotation failures from simulatable rows
    annotation_failures = [r for r in rows if r.get("_annotation_failed")]
    simulatable_rows = [r for r in rows if not r.get("_annotation_failed")]
    
    print(f"Running Tenderly simulation for {len(rows)} rows...")
    print(f"  Annotation failures (skipped): {len(annotation_failures)}")
    print(f"  Simulatable rows: {len(simulatable_rows)}")
    print(f"  Token registry: {len(ERC20_REGISTRY)} ERC20s, {len(ERC721_REGISTRY)} NFT collections")
    print(f"  Auto-retry: up to {args.max_retries} slot attempts on balance/ownership errors")
    
    updated = run_simulation_on_dataset(
        rows,
        default_from_address=args.from_address,
        network_id=args.network_id,
        block_number=args.block_number,
        max_retries=args.max_retries,
    )
    
    # Calculate metrics excluding annotation failures
    simulated = [r for r in updated if not r.get("_annotation_failed")]
    validated = sum(1 for r in simulated if r.get("_tenderly_validated"))
    sim_failures = [r for r in simulated if not r.get("_tenderly_validated")]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\n✓ Wrote to {output_path}")
    print(f"\n=== Results (excluding {len(annotation_failures)} annotation failures) ===")
    print(f"  Simulated: {len(simulated)}")
    print(f"  Validated: {validated} ({100*validated/len(simulated):.1f}%)" if simulated else "  Validated: 0")
    print(f"  Failed:    {len(sim_failures)}")
    
    # Show simulation failures
    if sim_failures:
        print("\nSimulation Failures:")
        for r in sim_failures:
            intent = r.get("user_intent", "")[:50]
            error = r.get("_tenderly_error", "unknown")[:60]
            print(f"  - {intent}... → {error}")
    
    # Show annotation failures separately
    if annotation_failures:
        print(f"\nAnnotation Failures ({len(annotation_failures)} skipped):")
        for r in annotation_failures:
            intent = r.get("user_intent", "")[:50]
            reason = r.get("_failure_reason", "unknown")[:40]
            print(f"  - {intent}... → {reason}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
