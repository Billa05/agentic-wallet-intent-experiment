"""
Merge all per-protocol annotated datasets into a single dataset.json.

Reads all *_annotated.json files from data/datasets/annotated/,
adds _protocol provenance field, and merges into data/datasets/dataset.json.

Usage:
  python data/merge_datasets.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

project_root = Path(__file__).parent.parent
ANNOTATED_DIR = project_root / "data" / "datasets" / "annotated"
OUTPUT_PATH = project_root / "data" / "datasets" / "dataset.json"


def merge():
    """Merge all annotated datasets into a single file."""
    all_records = []
    stats = Counter()

    annotated_files = sorted(ANNOTATED_DIR.glob("*_annotated.json"))
    if not annotated_files:
        print(f"No *_annotated.json files found in {ANNOTATED_DIR}")
        sys.exit(1)

    for f in annotated_files:
        protocol = f.stem.replace("_annotated", "")
        records = json.loads(f.read_text())
        if not isinstance(records, list):
            print(f"  Warning: {f.name} is not a list, skipping")
            continue
        for r in records:
            r["_protocol"] = protocol
        all_records.extend(records)
        stats[protocol] = len(records)
        print(f"  {protocol:25s} — {len(records)} records")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(all_records, fp, indent=2, ensure_ascii=False)

    total = len(all_records)
    failed = sum(1 for r in all_records if r.get("_annotation_failed"))
    print(f"\nMerged {total} records from {len(annotated_files)} files -> {OUTPUT_PATH}")
    print(f"  Passed: {total - failed}, Failed: {failed}")
    print("\nPer-protocol breakdown:")
    for proto, count in sorted(stats.items()):
        print(f"  {proto:25s}  {count}")


if __name__ == "__main__":
    merge()
