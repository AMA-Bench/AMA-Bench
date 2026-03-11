"""
Convert BabyAI trajectories to per-episode JSON files for dataset/.
Automatically generates QA pairs if missing or empty.

Supports both:
- Directory of JSON files (current output format from batch_trajetory_gen.py)
- JSONL file (legacy format)

Usage:
    # From project root (recommended): output goes to dataset/babyai/
    python memory_data_generation/babyai/convert_to_dataset.py

    # From babyai/:
    python convert_to_dataset.py --output ../../dataset/babyai

    python convert_to_dataset.py --input babyai_out_batch --output dataset/babyai
    python convert_to_dataset.py --input babyai_out_batch/all_trajectories.jsonl --output dataset/babyai
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

# Import QA generator
import sys
from pathlib import Path as PathLib

# Add the script's directory to path to ensure import works from both project root and babyai directory
_script_dir = PathLib(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    from babyai_qa_generator import generate_qa_for_trajectory
except ImportError as e:
    # If import still fails, QA generation will be disabled
    print(f"⚠️  Warning: Could not import babyai_qa_generator: {e}")
    print("   QA generation will be disabled. Make sure babyai_qa_generator.py is in the same directory.")
    generate_qa_for_trajectory = None


def convert(
    input_path: Path,
    output_dir: Path,
    *,
    source: str = "babyai",
    prefix: str = "task_babyai",
    target_qa_count: int = 12,
    auto_generate_qa: bool = True,
    seed: Optional[int] = None,
) -> int:
    """
    Read trajectories from JSONL file or directory of JSON files, write one JSON per episode.
    Ensures "source": "babyai", removes "action_space" if present.
    Automatically generates QA pairs if missing or empty.

    Args:
        input_path: Path to input JSONL file or directory of JSON files
        output_dir: Output directory for JSON files
        source: Source identifier (default: "babyai")
        prefix: Prefix for output filenames (default: "task_babyai")
        target_qa_count: Target number of QA pairs to generate (default: 12)
        auto_generate_qa: Whether to auto-generate QA pairs if missing (default: True)
        seed: Random seed for QA generation reproducibility (optional)

    Returns:
        Number of files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    count = 0
    qa_generated_count = 0
    
    # Determine if input is a file or directory
    input_path = Path(input_path)
    if input_path.is_file():
        # Load from JSONL file
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON error at line {line_num}: {e}")
                    continue
                
                count, qa_generated_count = _process_episode(
                    data, output_dir, prefix, source, target_qa_count,
                    auto_generate_qa, count, qa_generated_count, seed
                )
    elif input_path.is_dir():
        # Load from directory of JSON files
        input_files = sorted(input_path.glob("*.json"))
        for input_file in input_files:
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    count, qa_generated_count = _process_episode(
                        data, output_dir, prefix, source, target_qa_count,
                        auto_generate_qa, count, qa_generated_count, seed
                    )
            except Exception as e:
                print(f"⚠️  Warning: Failed to process {input_file}: {e}")
                continue
    else:
        raise FileNotFoundError(f"Input must be a file or directory: {input_path}")

    if auto_generate_qa and qa_generated_count > 0:
        print(f"\n📝 Generated QA pairs for {qa_generated_count} trajectories")

    return count


def _process_episode(
    data: dict,
    output_dir: Path,
    prefix: str,
    source: str,
    target_qa_count: int,
    auto_generate_qa: bool,
    count: int,
    qa_generated_count: int,
    seed: Optional[int] = None,
) -> tuple[int, int]:
    """Process a single episode and write to output."""
    data["source"] = source
    data.pop("action_space", None)

    # Auto-generate QA pairs if missing or empty
    if auto_generate_qa and generate_qa_for_trajectory is not None:
        qa_pairs = data.get("qa_pairs", [])
        if not qa_pairs or (isinstance(qa_pairs, list) and len(qa_pairs) == 0):
            try:
                generated_qa = generate_qa_for_trajectory(data, target_count=target_qa_count, seed=seed)
                data["qa_pairs"] = generated_qa
                qa_generated_count += 1
            except Exception as e:
                print(f"⚠️  Warning: Failed to generate QA: {e}")
                if "qa_pairs" not in data:
                    data["qa_pairs"] = []
    elif auto_generate_qa and generate_qa_for_trajectory is None:
        if "qa_pairs" not in data:
            data["qa_pairs"] = []

    # Use sequential numbering starting from 1: task_babyai_1.json, task_babyai_2.json, ...
    count += 1
    out_path = output_dir / f"{prefix}_{count}.json"
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
    print(f"✅ {out_path.name}")
    
    return count, qa_generated_count


def main() -> None:
    # Determine if running from project root or babyai directory
    _script_dir = PathLib(__file__).parent
    _cwd = PathLib.cwd()
    
    # Check if we're in the babyai directory (script's parent)
    if _cwd == _script_dir:
        # Running from babyai/ directory
        default_input = "babyai_out_batch"
        default_output = "../../dataset/babyai"
    else:
        # Running from project root
        default_input = "memory_data_generation/babyai/babyai_out_batch"
        default_output = "dataset/babyai"
    
    ap = argparse.ArgumentParser(
        description="Convert BabyAI trajectories (JSONL file or directory of JSON files) to dataset JSON files. "
                    "Automatically generates QA pairs if missing or empty.",
    )
    ap.add_argument(
        "--input", "-i",
        type=str,
        default=default_input,
        help=f"Input JSONL (default: {default_input})",
    )
    ap.add_argument(
        "--output", "-o",
        type=str,
        default=default_output,
        help=f"Output directory (default: {default_output})",
    )
    ap.add_argument(
        "--target-qa-count",
        type=int,
        default=12,
        help="Target number of QA pairs to generate per trajectory (default: 12, A+B+C=10, D=2)",
    )
    ap.add_argument(
        "--no-auto-qa",
        action="store_true",
        help="Disable automatic QA generation (use existing QA pairs only)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for QA generation reproducibility (optional, uses episode_id hash if not provided)",
    )
    args = ap.parse_args()

    # Resolve paths relative to current working directory
    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()

    try:
        n = convert(
            inp, 
            out, 
            target_qa_count=args.target_qa_count,
            auto_generate_qa=not args.no_auto_qa,
            seed=args.seed
        )
        print(f"\n🎉 Wrote {n} files to {out.absolute()}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Run batch_trajetory_gen.py first, then analyze_trajectories.py if needed.")


if __name__ == "__main__":
    main()
