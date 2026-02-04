"""
Run Tenderly simulation on an already-annotated dataset and update _tenderly_validated.

Expects input from annotate_with_hybrid.py (target_payload = { chain_id, to, value, data }).
Only updates _tenderly_validated and _tenderly_error; does not change format.

Usage:
  python data/run_tenderly_simulation.py --input data/datasets/annotated/annotated_hybrid_dataset.json
  python data/run_tenderly_simulation.py --input ... --output ...  # write to different file
  python data/run_tenderly_simulation.py --input ... --in-place   # overwrite input file

Requires: TENDERLY_ACCESS_KEY, TENDERLY_ACCOUNT_SLUG, TENDERLY_PROJECT_SLUG in env.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.tenderly_client import tenderly_simulate


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    return data


def run_simulation_on_dataset(
    rows: List[Dict[str, Any]],
    from_address: str,
    network_id: str = "1",
) -> List[Dict[str, Any]]:
    """Run Tenderly simulation for each row that has target_payload; set _tenderly_validated and _tenderly_error."""
    out = []
    for i, row in enumerate(rows):
        row = dict(row)
        target = row.get("target_payload")
        from_addr = (row.get("user_context") or {}).get("from_address") or from_address

        if not target or not target.get("to"):
            row["_tenderly_validated"] = False
            if row.get("_tenderly_error"):
                del row["_tenderly_error"]
        else:
            result = tenderly_simulate(
                from_address=from_addr,
                to_address=target["to"],
                value=target.get("value", "0"),
                data=target.get("data", "0x"),
                network_id=network_id,
                access_key=os.getenv("TENDERLY_ACCESS_KEY"),
                account_slug=os.getenv("TENDERLY_ACCOUNT_SLUG"),
                project_slug=os.getenv("TENDERLY_PROJECT_SLUG"),
            )
            row["_tenderly_validated"] = result.get("success", False)
            if result.get("success"):
                if row.get("_tenderly_error"):
                    del row["_tenderly_error"]
            else:
                row["_tenderly_error"] = result.get("error") or "Simulation failed"
        out.append(row)
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run Tenderly simulation and update _tenderly_validated on annotated dataset")
    parser.add_argument("--input", default="data/datasets/annotated/annotated_hybrid_dataset.json", help="Input annotated JSON (from annotate_with_hybrid.py)")
    parser.add_argument("--output", default="", help="Output path (default: overwrite --input unless --in-place is used)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input file with updated _tenderly_validated")
    parser.add_argument("--from-address", default=os.getenv("TENDERLY_FROM_ADDRESS", "0xe2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2"), help="Sender for simulation if not in user_context")
    parser.add_argument("--network-id", default="1", help="Chain ID for simulation")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    if not os.getenv("TENDERLY_ACCESS_KEY") or not os.getenv("TENDERLY_ACCOUNT_SLUG") or not os.getenv("TENDERLY_PROJECT_SLUG"):
        print("Error: set TENDERLY_ACCESS_KEY, TENDERLY_ACCOUNT_SLUG, TENDERLY_PROJECT_SLUG in env.")
        return 1

    output_path = Path(args.output) if args.output else input_path
    if not output_path.is_absolute():
        output_path = project_root / output_path
    if args.in_place:
        output_path = input_path

    print(f"Loading {input_path}...")
    rows = load_json(input_path)
    print(f"Running Tenderly simulation for {len(rows)} rows...")
    updated = run_simulation_on_dataset(rows, from_address=args.from_address, network_id=args.network_id)
    validated = sum(1 for r in updated if r.get("_tenderly_validated"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Updated _tenderly_validated. Wrote to {output_path}")
    print(f"  _tenderly_validated: True = {validated}, False = {len(updated) - validated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
