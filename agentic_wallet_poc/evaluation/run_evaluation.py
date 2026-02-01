#!/usr/bin/env python3
"""
Evaluation Runner for Agentic Wallet Intent Translation System

Main script to evaluate translator on annotated dataset.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.llm_translator import LLMTranslator
from evaluation.metrics import (
    evaluate_dataset,
    print_evaluation_report,
    save_results,
    EvaluationResult
)

# Try to import tqdm for progress bar, fallback to simple print
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def analyze_errors(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Analyze patterns in failed predictions.
    
    Args:
        results: List of EvaluationResult objects
        
    Returns:
        Dictionary with error analysis
    """
    total_errors = sum(1 for r in results if not r.exact_match)
    
    error_by_type = {
        "prediction_failed": 0,
        "wrong_action": 0,
        "wrong_address": 0,
        "wrong_amount": 0,
        "wrong_contract": 0,
        "wrong_arguments": 0,
    }
    
    failed_intents = []
    
    for result in results:
        if result.exact_match:
            continue
        
        # Categorize error (order matters: most specific first)
        if result.error or result.predicted_action is None:
            error_type = "prediction_failed"
        elif not result.correct_action:
            error_type = "wrong_action"
        elif not result.correct_contract:
            error_type = "wrong_contract"
        elif not result.correct_address:
            error_type = "wrong_address"
        elif not getattr(result, "correct_arguments", True):
            error_type = "wrong_arguments"
        elif not result.correct_amount:
            error_type = "wrong_amount"
        else:
            error_type = "unknown"
        
        error_by_type[error_type] += 1
        
        # Collect failed intent
        failed_intents.append({
            "intent": result.intent,
            "error_type": error_type,
            "predicted_action": result.predicted_action,
            "true_action": result.true_action
        })
    
    return {
        "total_errors": total_errors,
        "error_by_type": error_by_type,
        "failed_intents": failed_intents
    }


def print_error_analysis(error_analysis: Dict[str, Any]) -> None:
    """Print formatted error analysis."""
    if error_analysis["total_errors"] == 0:
        print("\n✓ No errors found - perfect score!")
        return
    
    print("\nError Analysis:")
    print(f"  Total errors: {error_analysis['total_errors']}")
    
    error_types = error_analysis["error_by_type"]
    if any(error_types.values()):
        print("  Breakdown:")
        for error_type, count in error_types.items():
            if count > 0:
                label = error_type.replace("_", " ").title()
                print(f"    - {label}: {count}")
    
    # Show failed intents
    failed_intents = error_analysis["failed_intents"]
    if failed_intents:
        print(f"\n  Failed intents ({len(failed_intents)}):")
        for i, item in enumerate(failed_intents[:10], 1):  # Show first 10
            intent = item["intent"]
            error_type = item["error_type"]
            if len(intent) > 60:
                intent = intent[:57] + "..."
            print(f"    {i}. \"{intent}\" ({error_type})")
        
        if len(failed_intents) > 10:
            print(f"    ... and {len(failed_intents) - 10} more")


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load JSON file and return data.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of dictionaries from JSON
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is invalid JSON
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    
    return data


def run_evaluation(
    test_data_path: Path,
    output_path: Path,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation on annotated dataset using the hybrid (LLM) translator.
    
    Args:
        test_data_path: Path to annotated dataset JSON
        output_path: Path to save results
        verbose: Print detailed progress
        
    Returns:
        Evaluation metrics dictionary
    """
    print(f"Loading annotated dataset from {test_data_path}...")
    test_data = load_json_file(test_data_path)
    print(f"Loaded {len(test_data)} test examples\n")
    
    print("Initializing hybrid translator (LLMTranslator)...")
    translator = LLMTranslator()
    model_name = translator.model
    sanitized_model = model_name.replace('/', '_').replace(':', '_').replace(' ', '_')
    translator_name = f"Hybrid ({model_name})"
    if output_path.name == "results.json" or output_path.name.startswith("results_"):
        output_path = output_path.parent / f"results_{sanitized_model}.json"
    print(f"✓ {translator_name} translator ready\n")
    
    print("Running evaluation...")
    if verbose:
        print()  # Empty line for verbose output
    metrics = evaluate_dataset(translator, test_data, verbose=verbose)
    
    print()  # Empty line after evaluation
    
    # Print report
    print_evaluation_report(metrics)
    
    # Analyze errors
    detailed_results = metrics.get("detailed_results", [])
    error_analysis = analyze_errors(detailed_results)
    print_error_analysis(error_analysis)
    
    # Add error analysis to metrics before saving
    metrics["error_analysis"] = error_analysis
    
    # Save results
    save_results(metrics, str(output_path))
    print(f"\nResults saved to {output_path}")
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate translator on annotated dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on default annotated dataset
  python evaluation/run_evaluation.py
  
  # Evaluate with verbose output
  python evaluation/run_evaluation.py --verbose
  
  # Custom dataset and output
  python evaluation/run_evaluation.py --test-data data/datasets/annotated/custom_dataset.json --output evaluation/results/custom.json
  
  # Also evaluate on edge cases
  python evaluation/run_evaluation.py --include-edge-cases
        """
    )
    
    parser.add_argument(
        '--test-data',
        default='data/datasets/annotated/annotated_dataset.json',
        help='Path to annotated dataset JSON (default: data/datasets/annotated/annotated_dataset.json)'
    )
    
    parser.add_argument(
        '--output',
        default='evaluation/results/results.json',
        help='Path to save results JSON (default: evaluation/results/results.json)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed per-example results'
    )
    
    parser.add_argument(
        '--include-edge-cases',
        action='store_true',
        help='Also evaluate on edge_cases.json'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Agentic Wallet Intent Translation System")
    print("           Evaluation Runner")
    print("=" * 60)
    print()
    
    # Convert paths
    test_data_path = Path(args.test_data)
    output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run evaluation (hybrid translator only)
        metrics = run_evaluation(
            test_data_path=test_data_path,
            output_path=output_path,
            verbose=args.verbose,
        )
        
        # Optional: Evaluate on edge cases
        if args.include_edge_cases:
            edge_cases_path = project_root / "data" / "datasets" / "edgecases" / "edge_cases.json"
            if edge_cases_path.exists():
                print("\n" + "=" * 60)
                print("Evaluating on Edge Cases")
                print("=" * 60)
                
                edge_output_path = output_path.parent / f"{output_path.stem}_edge_cases{output_path.suffix}"
                run_evaluation(
                    test_data_path=edge_cases_path,
                    output_path=edge_output_path,
                    verbose=args.verbose
                )
            else:
                print(f"\n⚠ Edge cases file not found at {edge_cases_path}")
        
        print("\n" + "=" * 60)
        print("✓ Evaluation completed successfully!")
        print("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Raw intents have been annotated with the hybrid translator (run: python data/annotate_with_hybrid.py)")
        print("  2. After human validation, annotated dataset exists at: data/datasets/annotated/annotated_dataset.json")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
