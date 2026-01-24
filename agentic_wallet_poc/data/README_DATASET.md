# Dataset Generation Guide

## Quick Start

**Recommended for ML (balanced dataset):**
```bash
# Generate 30 balanced examples (10 of each type)
python data/dataset_generator.py --count 30

# Append more examples
python data/dataset_generator.py --count 60 --append
```

**Why balanced?** Equal class distribution prevents ML bias and improves metrics.

## Command Line Options

```bash
python data/dataset_generator.py [OPTIONS]

Primary (Recommended):
  --count INT          Total examples (divided equally: count/3 per type)
  
Advanced:
  --eth-count INT      SEND_ETH examples
  --erc20-count INT    TRANSFER_ERC20 examples  
  --erc721-count INT   TRANSFER_ERC721 examples

General:
  --output PATH        Output file (default: data/datasets/raw_intents.json)
  --append             Append to existing file
```

## Examples

```bash
# Balanced: 30 examples (10 each)
python data/dataset_generator.py --count 30

# Custom distribution
python data/dataset_generator.py --eth-count 20 --erc20-count 15 --erc721-count 10

# Default: 10 ETH, 5 ERC20, 5 ERC721
python data/dataset_generator.py
```

## Prerequisites

1. Set up API key:
```bash
cp .env.example .env
# Add: GEMINI_API_KEY=your_key_here
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Enhanced Features

The generator automatically includes:
- **Edge cases**: Ambiguous amounts, incomplete info, typos, slang
- **Context variations**: Urgency, confirmations, reasons, emotions
- **Amount ranges**: Micro to large, various formats (decimal, wei, spelled)
- **Real collections**: Bored Ape, CryptoPunk, Mutant Ape, Azuki, Doodles
- **Token registry**: Uses real contract addresses from `token_registry.json`

## Dataset Files

- **`datasets/raw_intents.json`**: Natural language intents (input)
  ```json
  {"intent": "send 0.5 ETH to alice.eth", "transaction_type": "SEND_ETH"}
  ```

- **`datasets/annotated_dataset.json`**: Structured executable payloads (output)
  ```json
  {
    "user_intent": "send 0.5 ETH to alice.eth",
    "user_context": {"current_chain_id": 1, "token_prices": {"ETH": 2500.0}},
    "target_payload": {
      "chain_id": 1,
      "action": "transfer_native",
      "arguments": {
        "to": "0x...",
        "value": "500000000000000000",
        "human_readable_amount": "0.5 ETH"
      }
    }
  }
  ```

- **`datasets/edge_cases.json`**: 31 hand-crafted adversarial examples
- **`datasets/edge.json`**: Annotated edge cases (validation results)
- **`datasets/train_set.json`**: Training dataset (80% split)
- **`datasets/test_set.json`**: Test dataset (20% split)
- **`registries/token_registry.json`**: Real ERC-20/ERC-721 contract addresses
- **`registries/ens_registry.json`**: ENS name to address mappings

## Complete Workflow

```bash
# 1. Generate raw intents
python data/dataset_generator.py --count 30

# 2. Annotate to structured format
python data/dataset_annotator.py --non-interactive

# 3. Review annotated_dataset.json
```

## Recommended Sizes

- **PoC**: 30 examples (10 each)
- **Training**: 90-300 examples (30-100 each)
- **Production**: 900+ examples (300+ each)

## Edge Cases

### Hand-Crafted Edge Cases

The `edge_cases.json` file contains 31 adversarial examples covering:
- Amount extremes (very small/large, base units)
- Missing information (amount, recipient, token ID)
- Ambiguity (generic tokens, only address)
- Complex scenarios (multiple transactions, mixed types)
- Format variations (reversed order, corrections)
- Advanced features (chain/gas specs, DeFi params)

### Validating Edge Cases

Test how the annotator handles edge cases:

```bash
# Annotate edge cases to see which ones can be processed
python data/dataset_annotator.py --input data/datasets/edge_cases.json --output data/datasets/edge.json --non-interactive
```

This generates `edge.json` showing:
- Which edge cases can be successfully annotated
- Which ones fail validation (missing required info)
- How the system handles challenging inputs

**Note:** Some edge cases (like "missing_amount" or "missing_recipient") are intentionally incomplete and may fail annotation - this is expected and useful for testing error handling.
