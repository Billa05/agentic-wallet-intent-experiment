"""
Offline dataset scorer — scores already-annotated datasets with zero LLM calls.

Produces stage-attributed metrics with calldata structural validation,
per-protocol and per-action breakdowns, and argument coverage analysis.

Usage:
  python evaluation/score_dataset.py --input data/datasets/annotated/weth_annotated.json
  python evaluation/score_dataset.py --input data/datasets/annotated/ --all
  python evaluation/score_dataset.py --input data/datasets/dataset.json -v --output evaluation/results/scoring.json
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.metrics import (
    score_annotated_dataset,
    print_scoring_report,
    save_scoring_results,
)
from data.validate_calldata import (
    _load_token_cache,
    _build_address_lookup,
    _build_action_to_function_from_playbooks,
    _build_protocol_addresses_from_playbooks,
    _build_selector_map,
    _build_standard_selectors,
)


def _load_registries():
    """Load validation registries once (shared across all files)."""
    token_reg = _load_token_cache(project_root / "data" / "registries")
    addr_lookup = _build_address_lookup(token_reg)
    action_to_func = _build_action_to_function_from_playbooks()
    proto_addrs = _build_protocol_addresses_from_playbooks()
    _build_selector_map()
    _build_standard_selectors()
    return addr_lookup, action_to_func, proto_addrs


def _load_records(path: Path) -> list:
    """Load annotated records from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Offline dataset scorer (zero LLM calls)")
    parser.add_argument("--input", required=True, help="Annotated JSON file or directory")
    parser.add_argument("--all", action="store_true", help="Score all *_annotated.json in directory")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-record details")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / args.input

    print("Loading validation registries...")
    addr_lookup, action_to_func, proto_addrs = _load_registries()

    if args.all:
        # Score all *_annotated.json in the directory
        if input_path.is_file():
            input_path = input_path.parent
        files = sorted(input_path.glob("*_annotated.json"))
        if not files:
            print(f"No *_annotated.json files found in {input_path}")
            sys.exit(1)

        all_records = []
        for f in files:
            records = _load_records(f)
            all_records.extend(records)
            print(f"  Loaded {len(records):4d} records from {f.name}")

        print(f"\nScoring {len(all_records)} records from {len(files)} files...")
        metrics = score_annotated_dataset(
            all_records, addr_lookup, action_to_func, proto_addrs, verbose=args.verbose,
        )
        print_scoring_report(metrics)

    elif input_path.is_file():
        records = _load_records(input_path)
        print(f"Loaded {len(records)} records from {input_path.name}")
        print(f"Scoring...")
        metrics = score_annotated_dataset(
            records, addr_lookup, action_to_func, proto_addrs, verbose=args.verbose,
        )
        print_scoring_report(metrics)

    else:
        print(f"Error: {input_path} is not a file. Use --all to score a directory.")
        sys.exit(1)

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = project_root / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        save_scoring_results(metrics, str(out))


if __name__ == "__main__":
    main()
