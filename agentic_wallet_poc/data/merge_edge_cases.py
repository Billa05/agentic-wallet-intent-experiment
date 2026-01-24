"""
Merge edge cases into the main raw_intents.json file
"""
import json
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    
    # Load existing intents
    raw_intents_path = project_root / "data" / "datasets" / "raw_intents.json"
    edge_cases_path = project_root / "data" / "datasets" / "edge_cases.json"
    
    with open(raw_intents_path, 'r', encoding='utf-8') as f:
        existing_intents = json.load(f)
    
    with open(edge_cases_path, 'r', encoding='utf-8') as f:
        edge_cases = json.load(f)
    
    # Remove edge_case_type field for consistency (keep it in separate file for reference)
    edge_cases_clean = []
    for case in edge_cases:
        case_copy = {k: v for k, v in case.items() if k != 'edge_case_type'}
        edge_cases_clean.append(case_copy)
    
    # Merge
    all_intents = existing_intents + edge_cases_clean
    
    # Save
    with open(raw_intents_path, 'w', encoding='utf-8') as f:
        json.dump(all_intents, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Merged {len(edge_cases)} edge cases into raw_intents.json")
    print(f"✓ Total intents: {len(all_intents)}")
    print(f"  - Original: {len(existing_intents)}")
    print(f"  - Edge cases: {len(edge_cases)}")
    print(f"\nEdge case types added:")
    edge_types = {}
    for case in edge_cases:
        edge_type = case.get('edge_case_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    for edge_type, count in sorted(edge_types.items()):
        print(f"  - {edge_type}: {count}")

if __name__ == "__main__":
    main()
