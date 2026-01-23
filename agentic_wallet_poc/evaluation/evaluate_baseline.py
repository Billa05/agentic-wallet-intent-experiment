"""
Evaluation Script for Baseline Translator

Evaluates the baseline translator on the test dataset.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.baseline_translator import BaselineTranslator
from evaluation.metrics import (
    evaluate_dataset,
    print_evaluation_report,
    save_results
)


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate baseline translator on test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on default test set
  python evaluation/evaluate_baseline.py
  
  # Evaluate on custom test set
  python evaluation/evaluate_baseline.py --test-set data/test_set.json
  
  # Save results to file
  python evaluation/evaluate_baseline.py --output results/baseline_metrics.json
        """
    )
    
    parser.add_argument(
        '--test-set',
        default='data/test_set.json',
        help='Path to test dataset JSON file (default: data/test_set.json)'
    )
    
    parser.add_argument(
        '--output',
        default=None,
        help='Path to save evaluation results JSON (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress during evaluation'
    )
    
    args = parser.parse_args()
    
    print("Agentic Wallet Intent Translation System - Baseline Evaluation")
    print("=" * 60)
    
    # Load test dataset
    test_path = Path(args.test_set)
    if not test_path.exists():
        print(f"❌ Error: Test dataset not found at {test_path}")
        print(f"   Please run: python data/split_dataset.py")
        return 1
    
    print(f"Loading test dataset from {test_path}...")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"✓ Loaded {len(test_data)} test examples")
    
    # Initialize translator
    print("\nInitializing baseline translator...")
    translator = BaselineTranslator()
    print("✓ Translator ready")
    
    # Evaluate
    print("\nEvaluating translator on test dataset...")
    print("=" * 60)
    
    metrics = evaluate_dataset(translator, test_data, verbose=args.verbose)
    
    # Print report
    print_evaluation_report(metrics)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results(metrics, str(output_path))
    
    print("\n✓ Evaluation completed!")
    
    return 0


if __name__ == "__main__":
    exit(main())
