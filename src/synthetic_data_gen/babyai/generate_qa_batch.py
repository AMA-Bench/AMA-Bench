"""
Batch QA Generation for BabyAI Trajectories

This script reads trajectory files and generates QA pairs for each one.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from babyai_qa_generator import generate_qa_for_trajectory


def process_json_file(json_file: Path, target_qa_count: int = 12, seed: Optional[int] = None) -> Dict[str, Any]:
    """Process a single JSON trajectory file and add QA pairs."""
    with open(json_file, 'r', encoding='utf-8') as f:
        trajectory_data = json.load(f)
    
    # Generate QA pairs
    qa_pairs = generate_qa_for_trajectory(trajectory_data, target_count=target_qa_count, seed=seed)
    
    # Update trajectory data with QA pairs
    trajectory_data["qa_pairs"] = qa_pairs
    
    # Save back to file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
    
    return {
        "file": json_file.name,
        "episode_id": trajectory_data.get("episode_id", "unknown"),
        "num_qa": len(qa_pairs),
        "num_turns": len(trajectory_data.get("trajectory", []))
    }


def process_jsonl_file(jsonl_file: Path, target_qa_count: int = 12, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Process a JSONL file with multiple trajectories."""
    results = []
    updated_trajectories = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                trajectory_data = json.loads(line)
                qa_pairs = generate_qa_for_trajectory(trajectory_data, target_count=target_qa_count, seed=seed)
                trajectory_data["qa_pairs"] = qa_pairs
                updated_trajectories.append(trajectory_data)
                
                results.append({
                    "episode_id": trajectory_data.get("episode_id", "unknown"),
                    "num_qa": len(qa_pairs),
                    "num_turns": len(trajectory_data.get("trajectory", []))
                })
    
    # Write back to file
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for traj in updated_trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs for BabyAI trajectories"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing trajectory JSON/JSONL files"
    )
    parser.add_argument(
        "--target_qa_count",
        type=int,
        default=12,
        help="Target number of QA pairs per trajectory (default: 12, A+B+C=10, D=2)"
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.json",
        help="File pattern to match (default: *.json, use *.jsonl for JSONL files)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for QA generation reproducibility (optional, uses episode_id hash if not provided)",
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    # Find all matching files
    if args.file_pattern.endswith(".jsonl"):
        files = list(input_dir.glob("**/*.jsonl"))
    else:
        files = list(input_dir.glob("**/*.json"))
        # Exclude JSONL files
        files = [f for f in files if not f.name.endswith(".jsonl")]
    
    if not files:
        print(f"No {args.file_pattern} files found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to process")
    print(f"Target QA count per trajectory: {args.target_qa_count}")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            if file_path.suffix == ".jsonl":
                results = process_jsonl_file(file_path, args.target_qa_count, seed=args.seed)
                all_results.extend(results)
            else:
                result = process_json_file(file_path, args.target_qa_count, seed=args.seed)
                all_results.append(result)
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print("QA GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total trajectories processed: {len(all_results)}")
    
    if all_results:
        total_qa = sum(r["num_qa"] for r in all_results)
        avg_qa = total_qa / len(all_results)
        print(f"Total QA pairs generated: {total_qa}")
        print(f"Average QA pairs per trajectory: {avg_qa:.1f}")
        
        # Count by QA count ranges
        qa_ranges = {
            "0": 0,
            "1-5": 0,
            "6-10": 0,
            "11-15": 0,
            "16+": 0
        }
        for r in all_results:
            qa_count = r["num_qa"]
            if qa_count == 0:
                qa_ranges["0"] += 1
            elif qa_count <= 5:
                qa_ranges["1-5"] += 1
            elif qa_count <= 10:
                qa_ranges["6-10"] += 1
            elif qa_count <= 15:
                qa_ranges["11-15"] += 1
            else:
                qa_ranges["16+"] += 1
        
        print("\nQA count distribution:")
        for range_name, count in qa_ranges.items():
            print(f"  {range_name}: {count}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
