"""
Evaluation Metrics for Agentic Wallet Intent Translation System

Measures accuracy of intent translation models against ground truth annotations.
Supports transfer actions and DeFi actions (AAVE, Lido, Uniswap, Curve).
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
    
    # Per-protocol breakdown (auto-discovered from action names)
    protocol_prefixes = set()
    for a in action_counts.keys():
        protocol_prefixes.add(a.split("_")[0] + "_")
    per_protocol_accuracy = {}
    for prefix in sorted(protocol_prefixes):
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


# ─────────────────────────────────────────────────────────────────────
# Offline scoring — score already-annotated datasets (zero LLM calls)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ScoringResult:
    """Full scoring result for a single annotated record."""
    intent: str
    action: str
    protocol: str
    annotation_failed: bool
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    # Calldata structural checks
    calldata_decode: Optional[bool] = None
    calldata_function: Optional[bool] = None
    calldata_target: Optional[bool] = None
    calldata_token: Optional[bool] = None
    calldata_amount: Optional[bool] = None
    calldata_sender: Optional[bool] = None
    calldata_status: Optional[str] = None
    calldata_errors: Optional[List[str]] = None
    # Per-argument presence
    per_argument: Optional[Dict[str, bool]] = None


def _get_raw_tx(record: Dict) -> Optional[Dict]:
    """Extract raw tx dict from either new or old annotated format."""
    if record.get("raw_tx"):
        return record["raw_tx"]
    tp = record.get("target_payload") or {}
    if "data" in tp and "action" not in tp:
        return tp  # old format: raw tx in target_payload
    return None


def _get_action(record: Dict) -> str:
    """Extract action from either format."""
    meta = record.get("metadata") or {}
    if meta.get("action"):
        return meta["action"]
    tp = record.get("target_payload") or {}
    return tp.get("action", "unknown")


def _extract_calldata_flags(validation: Dict) -> Dict[str, Optional[bool]]:
    """Parse validate_record() output into per-check booleans."""
    flags = {
        "decode": None, "function": None, "target": None,
        "token": None, "amount": None, "sender": None,
    }
    for text in validation.get("checks", []):
        upper = text.upper()
        for key in flags:
            if upper.startswith(key.upper()):
                flags[key] = True
                break
    for text in validation.get("errors", []):
        upper = text.upper()
        for key in flags:
            if upper.startswith(key.upper()):
                flags[key] = False
                break
    return flags


def score_annotated_record(
    idx: int,
    record: Dict[str, Any],
    addr_lookup: Dict,
    action_to_func: Dict,
    proto_addrs: Dict,
) -> ScoringResult:
    """Score a single annotated record offline (no LLM calls)."""
    intent = record.get("user_intent", "")[:80]
    action = _get_action(record)
    protocol = action.split("_")[0] if action != "unknown" else "unknown"
    failed = record.get("_annotation_failed", False)

    if failed:
        return ScoringResult(
            intent=intent,
            action=action,
            protocol=protocol,
            annotation_failed=True,
            failure_stage=record.get("_failure_stage"),
            failure_reason=record.get("_failure_reason"),
        )

    # Per-argument check: which required args are present?
    tp = record.get("target_payload") or {}
    arguments = tp.get("arguments") or {}
    required = ACTION_REQUIRED_ARGS.get(action, [])
    per_arg = {}
    for key in required:
        if key == "human_readable_amount":
            continue  # display-only, skip
        val = arguments.get(key)
        per_arg[key] = val is not None and str(val) != ""

    # Run structural calldata validation
    from data.validate_calldata import validate_record as _validate_record
    validation = _validate_record(idx, record, addr_lookup, action_to_func, proto_addrs)
    flags = _extract_calldata_flags(validation)

    return ScoringResult(
        intent=intent,
        action=action,
        protocol=protocol,
        annotation_failed=False,
        calldata_decode=flags["decode"],
        calldata_function=flags["function"],
        calldata_target=flags["target"],
        calldata_token=flags["token"],
        calldata_amount=flags["amount"],
        calldata_sender=flags["sender"],
        calldata_status=validation.get("status"),
        calldata_errors=validation.get("errors", []),
        per_argument=per_arg,
    )


def score_annotated_dataset(
    records: List[Dict[str, Any]],
    addr_lookup: Dict,
    action_to_func: Dict,
    proto_addrs: Dict,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Score an entire annotated dataset offline. Returns aggregate metrics."""
    detailed = []
    total = len(records)
    ok_count = 0
    failed_count = 0
    failure_stages: Counter = Counter()
    calldata_pass = 0
    calldata_fail = 0
    calldata_skip = 0
    check_names = ["decode", "function", "target", "token", "amount", "sender"]
    check_pass: Counter = Counter()
    check_fail: Counter = Counter()
    per_protocol: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "ok": 0, "calldata_pass": 0})
    per_action: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "ok": 0, "calldata_pass": 0})
    arg_present: Counter = Counter()
    arg_required: Counter = Counter()

    for i, record in enumerate(records):
        r = score_annotated_record(i, record, addr_lookup, action_to_func, proto_addrs)
        detailed.append(r)

        per_protocol[r.protocol]["total"] += 1
        per_action[r.action]["total"] += 1

        if r.annotation_failed:
            failed_count += 1
            if r.failure_stage:
                failure_stages[r.failure_stage] += 1
            else:
                failure_stages["unknown"] += 1
            if verbose:
                print(f"  [{i:3d}] FAIL  {r.action:25s}  stage={r.failure_stage}  {r.intent}")
            continue

        ok_count += 1
        per_protocol[r.protocol]["ok"] += 1
        per_action[r.action]["ok"] += 1

        if r.calldata_status == "PASS":
            calldata_pass += 1
            per_protocol[r.protocol]["calldata_pass"] += 1
            per_action[r.action]["calldata_pass"] += 1
        elif r.calldata_status == "FAIL":
            calldata_fail += 1
            if verbose:
                print(f"  [{i:3d}] FAIL  {r.action:25s}  {r.calldata_errors}  {r.intent}")
        else:
            calldata_skip += 1

        for cn in check_names:
            val = getattr(r, f"calldata_{cn}", None)
            if val is True:
                check_pass[cn] += 1
            elif val is False:
                check_fail[cn] += 1

        if r.per_argument:
            for k, present in r.per_argument.items():
                arg_required[k] += 1
                if present:
                    arg_present[k] += 1

    calldata_checks = {}
    for cn in check_names:
        calldata_checks[cn] = {"pass": check_pass[cn], "fail": check_fail[cn]}

    argument_coverage = {}
    for k in sorted(arg_required.keys()):
        argument_coverage[k] = {"present": arg_present[k], "required": arg_required[k]}

    return {
        "total": total,
        "annotated_ok": ok_count,
        "annotated_failed": failed_count,
        "failure_stages": dict(failure_stages),
        "calldata_pass": calldata_pass,
        "calldata_fail": calldata_fail,
        "calldata_skip": calldata_skip,
        "calldata_checks": calldata_checks,
        "per_protocol": {k: dict(v) for k, v in sorted(per_protocol.items())},
        "per_action": {k: dict(v) for k, v in sorted(per_action.items())},
        "argument_coverage": argument_coverage,
        "detailed_results": detailed,
    }


def print_scoring_report(metrics: Dict[str, Any]) -> None:
    """Print formatted offline scoring report."""
    total = metrics["total"]
    ok = metrics["annotated_ok"]
    failed = metrics["annotated_failed"]

    print(f"\n{'='*60}")
    print("            OFFLINE SCORING REPORT")
    print(f"{'='*60}")

    print(f"\nDataset:")
    print(f"  Total records:       {total:4d}")
    print(f"  Annotated OK:        {ok:4d}")
    print(f"  Annotation failed:   {failed:4d}")

    fs = metrics.get("failure_stages", {})
    if fs:
        print(f"\nFailure Stage Attribution ({failed} failures):")
        for stage, count in sorted(fs.items(), key=lambda x: -x[1]):
            print(f"  {stage:25s}  {count}")

    cp = metrics["calldata_pass"]
    cf = metrics["calldata_fail"]
    cs = metrics["calldata_skip"]
    validated = cp + cf
    print(f"\nCalldata Structural Validation ({ok} annotated records):")
    print(f"  PASS:    {cp:4d}" + (f"  ({cp/validated*100:.1f}%)" if validated else ""))
    print(f"  FAIL:    {cf:4d}" + (f"  ({cf/validated*100:.1f}%)" if validated else ""))
    if cs:
        print(f"  SKIP:    {cs:4d}")

    checks = metrics.get("calldata_checks", {})
    if checks:
        print(f"\n  Check breakdown:")
        for cn, vals in checks.items():
            p, f_ = vals["pass"], vals["fail"]
            t = p + f_
            pct = f"{p/t*100:.0f}%" if t else "n/a"
            print(f"    {cn.upper():10s}  {p:4d}/{t:<4d}  ({pct})")

    pp = metrics.get("per_protocol", {})
    if pp:
        print(f"\nPer-Protocol:")
        for proto, stats in pp.items():
            t = stats["total"]
            o = stats["ok"]
            cp_ = stats["calldata_pass"]
            rate = f"{cp_/o*100:.1f}%" if o else "n/a"
            print(f"  {proto:20s}  {t:3d} total, {o:3d} ok, {cp_:3d} calldata pass ({rate})")

    pa = metrics.get("per_action", {})
    if pa:
        print(f"\nPer-Action:")
        for action, stats in pa.items():
            t = stats["total"]
            o = stats["ok"]
            cp_ = stats["calldata_pass"]
            rate = f"{cp_/o*100:.1f}%" if o else "n/a"
            print(f"  {action:25s}  {t:3d} total, {o:3d} ok, {cp_:3d} pass ({rate})")

    ac = metrics.get("argument_coverage", {})
    if ac:
        print(f"\nArgument Coverage:")
        for arg, vals in ac.items():
            p, r = vals["present"], vals["required"]
            pct = f"{p/r*100:.0f}%" if r else "n/a"
            print(f"  {arg:25s}  {p:4d}/{r:<4d}  ({pct})")

    print(f"\n{'='*60}")


def save_scoring_results(metrics: Dict[str, Any], output_path: str) -> None:
    """Save offline scoring results to JSON."""
    metrics_copy = metrics.copy()
    if "detailed_results" in metrics_copy:
        metrics_copy["detailed_results"] = [asdict(r) for r in metrics_copy["detailed_results"]]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_copy, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("=" * 60)
    print("\nLive evaluation:")
    print("  from evaluation.metrics import evaluate_dataset, print_evaluation_report")
    print("\nOffline scoring:")
    print("  python evaluation/score_dataset.py --input data/datasets/annotated/weth_annotated.json")
