"""
Evaluation Metrics for Agentic Wallet Intent Translation System

Measures accuracy of intent translation models against ground truth annotations.
Supports transfer actions and DeFi actions (AAVE, Lido, Uniswap, 1inch, Curve).
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
from collections import Counter, defaultdict

try:
    from utils.schemas import ACTION_REQUIRED_ARGS
except ImportError:
    ACTION_REQUIRED_ARGS = {}


def _normalize_arg_value(v: Any) -> Any:
    """Normalize argument value for comparison (e.g. list order, str)."""
    if v is None:
        return None
    if isinstance(v, list):
        return tuple(_normalize_arg_value(x) for x in v)
    if isinstance(v, str) and v.lower() in ("null", ""):
        return None
    return v


def _arguments_match(
    pred_args: Dict[str, Any],
    true_args: Dict[str, Any],
    action: str,
) -> bool:
    """Check if predicted arguments match ground truth for this action type."""
    required = ACTION_REQUIRED_ARGS.get(action, list(pred_args.keys()) or list(true_args.keys()))
    for key in required:
        if key not in true_args:
            continue
        pv = _normalize_arg_value(pred_args.get(key))
        tv = _normalize_arg_value(true_args.get(key))
        if pv != tv:
            return False
    return True


def _action_has_recipient(action: str) -> bool:
    """True if this action type has a 'to' (recipient) field."""
    return "to" in ACTION_REQUIRED_ARGS.get(action, [])


@dataclass
class EvaluationResult:
    """Results for a single prediction evaluation."""
    intent: str
    predicted_action: Optional[str]
    true_action: str
    correct_action: bool
    correct_address: bool
    correct_amount: bool
    correct_contract: bool
    correct_arguments: bool
    exact_match: bool
    error: Optional[str] = None
    ground_truth: Optional[Dict[str, Any]] = None  # Full ground truth annotation
    predicted: Optional[Dict[str, Any]] = None  # Full predicted annotation


def evaluate_single(
    predicted: Optional[Dict[str, Any]], 
    ground_truth: Dict[str, Any],
    intent: str
) -> EvaluationResult:
    """
    Evaluate a single prediction against ground truth.
    
    Args:
        predicted: The predicted AnnotatedIntent as dict (or None if prediction failed)
        ground_truth: The ground truth AnnotatedIntent as dict
        intent: The original user intent string
    
    Returns:
        EvaluationResult with all accuracy flags
    """
    # Handle failed predictions
    if predicted is None:
        true_payload = ground_truth.get("target_payload") or {}
        true_action = true_payload.get("action", "unknown")
        return EvaluationResult(
            intent=intent,
            predicted_action=None,
            true_action=true_action,
            correct_action=False,
            correct_address=False,
            correct_amount=False,
            correct_contract=False,
            correct_arguments=False,
            exact_match=False,
            error="Prediction failed",
            ground_truth=ground_truth,
            predicted=None
        )
    
    # Extract payloads (guard against None for failed-annotation rows)
    pred_payload = predicted.get("target_payload") or {}
    true_payload = ground_truth.get("target_payload") or {}
    pred_args = pred_payload.get("arguments") or {}
    true_args = true_payload.get("arguments") or {}
    
    # Compare action types
    pred_action = pred_payload.get("action")
    true_action = true_payload.get("action", "unknown")
    correct_action = (pred_action == true_action)
    
    # Address: N/A when action has no "to"; otherwise compare (case-insensitive)
    has_recipient = _action_has_recipient(true_action)
    if not has_recipient:
        correct_address = True  # N/A
    else:
        pred_to = (pred_args.get("to") or "").lower()
        true_to = (true_args.get("to") or "").lower()
        correct_address = bool(pred_to and true_to and pred_to == true_to)
    
    # Primary amount (for backward compat / reporting)
    if true_action == "transfer_erc721":
        pred_token_id = pred_args.get("tokenId")
        true_token_id = true_args.get("tokenId")
        correct_amount = (pred_token_id == true_token_id) if (pred_token_id is not None and true_token_id is not None) else False
    else:
        pred_value = pred_args.get("value", pred_args.get("amount", pred_args.get("amountIn", "")))
        true_value = true_args.get("value", true_args.get("amount", true_args.get("amountIn", "")))
        correct_amount = (str(pred_value) == str(true_value)) if (pred_value is not None and true_value is not None) else False
    
    # Arguments: all required fields must match (Wei/base as exact strings)
    correct_arguments = _arguments_match(pred_args, true_args, true_action)
    
    # Compare target contracts (case-insensitive)
    pred_contract = pred_payload.get("target_contract")
    true_contract = true_payload.get("target_contract")
    if pred_contract is None or pred_contract == "null":
        pred_contract = None
    if true_contract is None or true_contract == "null":
        true_contract = None
    if pred_contract is None and true_contract is None:
        correct_contract = True
    elif pred_contract is not None and true_contract is not None:
        correct_contract = (str(pred_contract).lower() == str(true_contract).lower())
    else:
        correct_contract = False
    
    # Exact match: action, contract, (address when applicable), and all arguments
    exact_match = (
        correct_action
        and correct_contract
        and correct_address
        and correct_arguments
    )
    
    return EvaluationResult(
        intent=intent,
        predicted_action=pred_action,
        true_action=true_action,
        correct_action=correct_action,
        correct_address=correct_address,
        correct_amount=correct_amount,
        correct_contract=correct_contract,
        correct_arguments=correct_arguments,
        exact_match=exact_match,
        error=None,
        ground_truth=ground_truth,
        predicted=predicted
    )


def evaluate_dataset(
    translator,  # LLMTranslator (hybrid) or any translator with translate(intent, chain_id)
    test_data: List[Dict[str, Any]],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate translator on entire test dataset.
    
    Args:
        translator: Translator instance (e.g. engine.llm_translator.LLMTranslator) with translate(intent, chain_id)
        test_data: List of annotated intents (ground truth)
        verbose: If True, print progress
    
    Returns:
        Dictionary with aggregate metrics and detailed results
    """
    detailed_results = []
    action_counts = Counter()
    action_exact_matches = defaultdict(int)
    
    total_examples = len(test_data)
    successful_predictions = 0
    failed_predictions = 0
    
    # Track accuracy metrics
    correct_action_count = 0
    correct_address_count = 0
    correct_amount_count = 0
    correct_contract_count = 0
    correct_arguments_count = 0
    exact_match_count = 0
    
    # Check if translator is LLM-based (for rate limiting)
    is_llm = hasattr(translator, 'model') or hasattr(translator, 'model_name')
    
    for i, ground_truth in enumerate(test_data):
        intent = ground_truth.get("user_intent", "")
        chain_id = ground_truth.get("user_context", {}).get("current_chain_id", 1)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{total_examples}...")
        
        # Add small delay for LLM to avoid hitting rate limits too quickly
        # Free tier: 5 requests/minute, so we add 12 second delay between requests
        if is_llm and i > 0:  # Skip delay for first request
            import time
            time.sleep(12)  # 12 seconds = 5 requests per minute
        
        # Get prediction
        try:
            # Call translate method
            predicted = translator.translate(intent, chain_id=chain_id)
            if predicted is not None:
                # Convert Pydantic model to dict
                predicted = predicted.model_dump(mode='json')
                successful_predictions += 1
            else:
                failed_predictions += 1
                if verbose:
                    print(f"  [{i + 1}/{total_examples}] Translation returned None for: {intent[:60]}...")
        except Exception as e:
            predicted = None
            failed_predictions += 1
            if verbose:
                print(f"  [{i + 1}/{total_examples}] Exception translating: {intent[:60]}...")
                print(f"    Error: {e}")
            else:
                # Always log exceptions, even if not verbose
                import sys
                print(f"DEBUG: Exception translating intent {i + 1}: {e}", file=sys.stderr)
        
        # Evaluate
        result = evaluate_single(predicted, ground_truth, intent)
        detailed_results.append(result)
        
        # Update counters
        if result.correct_action:
            correct_action_count += 1
        if result.correct_address:
            correct_address_count += 1
        if result.correct_amount:
            correct_amount_count += 1
        if result.correct_contract:
            correct_contract_count += 1
        if result.correct_arguments:
            correct_arguments_count += 1
        if result.exact_match:
            exact_match_count += 1
        
        # Per-action metrics
        true_action = result.true_action
        action_counts[true_action] += 1
        if result.exact_match:
            action_exact_matches[true_action] += 1
    
    # Calculate accuracies
    action_accuracy = correct_action_count / total_examples if total_examples > 0 else 0.0
    address_accuracy = correct_address_count / total_examples if total_examples > 0 else 0.0
    amount_accuracy = correct_amount_count / total_examples if total_examples > 0 else 0.0
    contract_accuracy = correct_contract_count / total_examples if total_examples > 0 else 0.0
    arguments_accuracy = correct_arguments_count / total_examples if total_examples > 0 else 0.0
    exact_match_accuracy = exact_match_count / total_examples if total_examples > 0 else 0.0
    
    # Per-action breakdown (all actions present in dataset)
    per_action_accuracy = {}
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        exact_matches = action_exact_matches.get(action, 0)
        exact_match_rate = exact_matches / count if count > 0 else 0.0
        per_action_accuracy[action] = {"count": count, "exact_match": exact_match_rate}
    
    # Per-protocol breakdown (prefix: aave_, lido_, uniswap_, oneinch_, curve_)
    protocol_prefixes = ("aave_", "lido_", "uniswap_", "oneinch_", "curve_")
    per_protocol_accuracy = {}
    for prefix in protocol_prefixes:
        count = sum(c for a, c in action_counts.items() if a.startswith(prefix))
        if count == 0:
            continue
        exact_matches = sum(action_exact_matches.get(a, 0) for a in action_counts if a.startswith(prefix))
        per_protocol_accuracy[prefix.rstrip("_")] = {
            "count": count,
            "exact_match": exact_matches / count if count > 0 else 0.0,
        }
    
    return {
        "total_examples": total_examples,
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "action_accuracy": action_accuracy,
        "address_accuracy": address_accuracy,
        "amount_accuracy": amount_accuracy,
        "contract_accuracy": contract_accuracy,
        "arguments_accuracy": arguments_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "per_action_accuracy": per_action_accuracy,
        "per_protocol_accuracy": per_protocol_accuracy,
        "detailed_results": detailed_results
    }


def print_evaluation_report(metrics: Dict[str, Any]) -> None:
    """Print a formatted evaluation report to console."""
    print("=" * 60)
    print("                    EVALUATION REPORT")
    print("=" * 60)
    
    # Dataset Statistics
    print("\nDataset Statistics:")
    print(f"  Total examples:        {metrics['total_examples']:4d}")
    print(f"  Successful predictions: {metrics['successful_predictions']:4d}")
    print(f"  Failed predictions:    {metrics['failed_predictions']:4d}")
    
    # Overall Accuracy
    print("\nOverall Accuracy:")
    print(f"  Action classification: {metrics['action_accuracy'] * 100:5.1f}%")
    print(f"  Address extraction:    {metrics['address_accuracy'] * 100:5.1f}%")
    print(f"  Amount conversion:    {metrics['amount_accuracy'] * 100:5.1f}%")
    print(f"  Contract lookup:      {metrics['contract_accuracy'] * 100:5.1f}%")
    print(f"  Arguments match:      {metrics.get('arguments_accuracy', 0) * 100:5.1f}%")
    print("  " + "-" * 55)
    print(f"  EXACT MATCH:          {metrics['exact_match_accuracy'] * 100:5.1f}%")
    
    # Per-Action Breakdown (all actions in dataset)
    print("\nPer-Action Breakdown:")
    per_action = metrics.get("per_action_accuracy", {})
    for action in sorted(per_action.keys()):
        stats = per_action[action]
        count = stats.get("count", 0)
        exact_match = stats.get("exact_match", 0.0)
        print(f"  {action:25s}: {exact_match * 100:5.1f}% exact match ({count:3d} examples)")
    
    # Per-Protocol Breakdown
    per_protocol = metrics.get("per_protocol_accuracy", {})
    if per_protocol:
        print("\nPer-Protocol Breakdown:")
        for protocol in sorted(per_protocol.keys()):
            stats = per_protocol[protocol]
            count = stats.get("count", 0)
            exact_match = stats.get("exact_match", 0.0)
            print(f"  {protocol:15s}: {exact_match * 100:5.1f}% exact match ({count:3d} examples)")
    
    print("\n" + "=" * 60)


def save_results(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        metrics: Evaluation metrics dictionary
        output_path: Path to save JSON file
    """
    # Convert EvaluationResult dataclasses to dicts
    metrics_copy = metrics.copy()
    
    if "detailed_results" in metrics_copy:
        detailed_results = metrics_copy["detailed_results"]
        metrics_copy["detailed_results"] = [asdict(result) for result in detailed_results]
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_copy, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Evaluation Metrics Module")
    print("=" * 60)
    print("\nThis module provides:")
    print("  - evaluate_single(): Evaluate one prediction")
    print("  - evaluate_dataset(): Evaluate entire dataset")
    print("  - print_evaluation_report(): Print formatted report")
    print("  - save_results(): Save results to JSON")
    print("\nUsage:")
    print("  from evaluation.metrics import evaluate_dataset, print_evaluation_report")
    print("  from engine.llm_translator import LLMTranslator")
    print("  ")
    print("  translator = LLMTranslator()")
    print("  metrics = evaluate_dataset(translator, test_data)")
    print("  print_evaluation_report(metrics)")
