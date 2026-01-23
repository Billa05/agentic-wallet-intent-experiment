"""
Dataset Splitter for Agentic Wallet Intent Translation System

Splits the annotated dataset into training and test sets with stratified sampling.
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def load_annotated_dataset(input_path: str) -> List[Dict[str, Any]]:
    """
    Load the annotated dataset from JSON file.
    
    Args:
        input_path: Path to annotated_dataset.json
        
    Returns:
        List of annotated intents
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is invalid JSON
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Annotated dataset not found at {input_path}\n"
            f"Please run: python data/dataset_annotator.py --non-interactive"
        )
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list of annotations, got {type(data)}")
    
    return data


def get_transaction_type(annotation: Dict[str, Any]) -> str:
    """
    Extract transaction type from annotation.
    
    Args:
        annotation: Annotated intent dictionary
        
    Returns:
        Transaction type string (e.g., "transfer_native", "transfer_erc20", "transfer_erc721")
    """
    payload = annotation.get('target_payload', {})
    action = payload.get('action', 'unknown')
    
    # Map action types to transaction types
    action_to_type = {
        'transfer_native': 'SEND_ETH',
        'transfer_erc20': 'TRANSFER_ERC20',
        'transfer_erc721': 'TRANSFER_ERC721'
    }
    
    return action_to_type.get(action, 'UNKNOWN')


def stratified_split(
    data: List[Dict[str, Any]],
    test_size: float = 0.2,
    random_seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train and test sets with stratified sampling.
    
    Args:
        data: List of annotated intents
        test_size: Proportion of data for test set (default: 0.2 = 20%)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_set, test_set)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Group by transaction type
    by_type = defaultdict(list)
    for item in data:
        tx_type = get_transaction_type(item)
        by_type[tx_type].append(item)
    
    train_set = []
    test_set = []
    
    # Split each transaction type separately
    for tx_type, items in by_type.items():
        # Shuffle items for this type
        shuffled = items.copy()
        random.shuffle(shuffled)
        
        # Calculate split point
        n_test = max(1, int(len(shuffled) * test_size))
        n_train = len(shuffled) - n_test
        
        # Split
        test_items = shuffled[:n_test]
        train_items = shuffled[n_test:]
        
        train_set.extend(train_items)
        test_set.extend(test_items)
    
    # Final shuffle of combined sets
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    return train_set, test_set


def print_statistics(
    train_set: List[Dict[str, Any]],
    test_set: List[Dict[str, Any]]
):
    """
    Print statistics about the train/test split.
    
    Args:
        train_set: Training set
        test_set: Test set
    """
    def count_by_type(data: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = defaultdict(int)
        for item in data:
            tx_type = get_transaction_type(item)
            counts[tx_type] += 1
        return dict(counts)
    
    train_counts = count_by_type(train_set)
    test_counts = count_by_type(test_set)
    
    print("\n" + "="*60)
    print("Dataset Split Statistics")
    print("="*60)
    print(f"\nTraining Set: {len(train_set)} examples")
    for tx_type in sorted(train_counts.keys()):
        count = train_counts[tx_type]
        pct = (count / len(train_set) * 100) if train_set else 0
        print(f"  {tx_type:20s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nTest Set: {len(test_set)} examples")
    for tx_type in sorted(test_counts.keys()):
        count = test_counts[tx_type]
        pct = (count / len(test_set) * 100) if test_set else 0
        print(f"  {tx_type:20s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nTotal: {len(train_set) + len(test_set)} examples")
    print(f"Split: {len(train_set) / (len(train_set) + len(test_set)) * 100:.1f}% train, "
          f"{len(test_set) / (len(train_set) + len(test_set)) * 100:.1f}% test")
    print("="*60)


def save_dataset(data: List[Dict[str, Any]], output_path: str):
    """
    Save dataset to JSON file.
    
    Args:
        data: List of annotated intents
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(data)} examples to {output_path}")


def main():
    """Main function to split the dataset."""
    parser = argparse.ArgumentParser(
        description="Split annotated dataset into train and test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default split (80/20 from data/annotated_dataset.json)
  python data/split_dataset.py
  
  # Custom input/output paths
  python data/split_dataset.py --input data/annotated_dataset.json --output-train data/train.json --output-test data/test.json
  
  # Custom test size (30% test, 70% train)
  python data/split_dataset.py --test-size 0.3
        """
    )
    
    parser.add_argument(
        '--input',
        default='data/annotated_dataset.json',
        help='Input annotated dataset file (default: data/annotated_dataset.json)'
    )
    
    parser.add_argument(
        '--output-train',
        default='data/train_set.json',
        help='Output file for training set (default: data/train_set.json)'
    )
    
    parser.add_argument(
        '--output-test',
        default='data/test_set.json',
        help='Output file for test set (default: data/test_set.json)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set (default: 0.2 = 20%%)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate test_size
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    
    print("Agentic Wallet Intent Translation System - Dataset Splitter")
    print("="*60)
    
    try:
        # Load dataset
        print(f"Loading annotated dataset from {args.input}...")
        data = load_annotated_dataset(args.input)
        print(f"✓ Loaded {len(data)} annotated examples")
        
        # Split dataset
        print(f"\nSplitting dataset (test_size={args.test_size:.1%}, seed={args.random_seed})...")
        train_set, test_set = stratified_split(data, test_size=args.test_size, random_seed=args.random_seed)
        
        # Print statistics
        print_statistics(train_set, test_set)
        
        # Save datasets
        print("\nSaving datasets...")
        save_dataset(train_set, args.output_train)
        save_dataset(test_set, args.output_test)
        
        print("\n✓ Dataset split completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    
    return 0


if __name__ == "__main__":
    exit(main())
