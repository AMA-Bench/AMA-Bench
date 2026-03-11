"""
Analyze BabyAI trajectory generation output.

Usage:
    python analyze_trajectories.py
    python analyze_trajectories.py --input babyai_out_batch/all_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_trajectories(path: Path) -> List[Dict]:
    """Load trajectories from a JSONL file or a directory of JSON files."""
    out: List[Dict] = []
    path = Path(path)
    
    if path.is_file():
        # Load from JSONL file
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    elif path.is_dir():
        # Load from directory of JSON files
        for json_file in sorted(path.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception as e:
                print(f"⚠️  Warning: Failed to load {json_file}: {e}")
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    
    return out


def analyze_trajectories(trajectories: List[Dict]) -> Dict:
    """Compute stats: by task_type, difficulty, token_bin, success, etc."""
    token_bins = {
        "4K": (0, 4096),
        "8K": (4097, 8192),
        "16K": (8193, 16384),
        "32K": (16385, 32768),
        "64K": (32769, 65536),
        "128K": (65537, 131072),
    }

    stats: Dict = {
        "total_count": len(trajectories),
        "by_task_type": defaultdict(int),
        "by_difficulty": defaultdict(int),
        "by_token_bin": defaultdict(int),
        "by_task_type_and_bin": defaultdict(lambda: defaultdict(int)),
        "by_difficulty_and_bin": defaultdict(lambda: defaultdict(int)),
        "success_count": 0,
        "fail_count": 0,
        "total_tokens": 0,
        "total_turns": 0,
        "min_tokens": float("inf"),
        "max_tokens": 0,
        "min_turns": float("inf"),
        "max_turns": 0,
    }

    for t in trajectories:
        task_type = t.get("task_type", "unknown")
        difficulty = t.get("difficulty", "unknown")
        tokens = int(t.get("total_tokens", 0))
        turns = int(t.get("num_turns", 0))
        state = t.get("state", "fail")

        stats["by_task_type"][task_type] += 1
        stats["by_difficulty"][difficulty] += 1

        if state == "success":
            stats["success_count"] += 1
        else:
            stats["fail_count"] += 1

        bin_name = None
        for name, (lo, hi) in token_bins.items():
            if lo <= tokens <= hi:
                bin_name = name
                break
        if bin_name:
            stats["by_token_bin"][bin_name] += 1
            stats["by_task_type_and_bin"][task_type][bin_name] += 1
            stats["by_difficulty_and_bin"][difficulty][bin_name] += 1
        else:
            stats["by_token_bin"]["out_of_range"] += 1

        stats["total_tokens"] += tokens
        stats["total_turns"] += turns
        if tokens < stats["min_tokens"]:
            stats["min_tokens"] = tokens
        if tokens > stats["max_tokens"]:
            stats["max_tokens"] = tokens
        if turns < stats["min_turns"]:
            stats["min_turns"] = turns
        if turns > stats["max_turns"]:
            stats["max_turns"] = turns

    n = len(trajectories)
    stats["avg_tokens"] = (stats["total_tokens"] / n) if n else 0
    stats["avg_turns"] = (stats["total_turns"] / n) if n else 0
    stats["success_rate"] = (stats["success_count"] / n * 100) if n else 0

    return stats


def print_analysis(stats: Dict, title: str = "BabyAI Analysis") -> None:
    """Print a formatted report."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")

    print("Overall:")
    print(f"  Total: {stats['total_count']}")
    print(f"  Success: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_count']})")
    print(f"  Fail: {stats['fail_count']}")

    print("\nToken stats:")
    print(f"  Average: {stats['avg_tokens']:,.0f}")
    print(f"  Range: {stats['min_tokens']:,} - {stats['max_tokens']:,}")

    print("\nTurn stats:")
    print(f"  Average: {stats['avg_turns']:.1f}")
    print(f"  Range: {stats['min_turns']} - {stats['max_turns']}")

    print("\nBy task_type (env_id):")
    for k in sorted(stats["by_task_type"].keys()):
        c = stats["by_task_type"][k]
        pct = (c / stats["total_count"] * 100) if stats["total_count"] else 0
        print(f"  {k:45s} {c:4d} ({pct:5.1f}%)")

    print("\nBy token bin (target ≥10 per bin):")
    for name in ["4K", "8K", "16K", "32K", "64K", "128K", "out_of_range"]:
        c = stats["by_token_bin"].get(name, 0)
        pct = (c / stats["total_count"] * 100) if stats["total_count"] else 0
        ok = "✓" if (c >= 10 or name == "out_of_range") else "✗"
        print(f"  {ok} {name:15s} {c:4d} ({pct:5.1f}%)")

    print(f"\n{'='*70}\n")


def check_requirements(stats: Dict, min_per_bin: int = 10) -> bool:
    """Check targets: 10 per token bin (configurable)."""
    print(f"\n{'='*70}")
    print(f"{'REQUIREMENTS CHECK':^70}")
    print(f"{'='*70}\n")

    all_ok = True

    print("1. Total count:")
    if stats["total_count"] >= 1:
        print(f"   ✓ {stats['total_count']} trajectories")
    else:
        print("   ✗ No trajectories")
        all_ok = False

    print(f"\n2. Token bins (target ≥{min_per_bin} per bin):")
    for name in ["4K", "8K", "16K", "32K", "64K", "128K"]:
        c = stats["by_token_bin"].get(name, 0)
        if c >= min_per_bin:
            print(f"   ✓ {name:10s} {c}")
        else:
            print(f"   ✗ {name:10s} {c} (target ≥{min_per_bin})")
            all_ok = False

    print(f"\n{'='*70}")
    if all_ok:
        print(f"{'🎉 ALL REQUIREMENTS MET':^70}")
    else:
        print(f"{'❌ SOME REQUIREMENTS NOT MET':^70}")
    print(f"{'='*70}\n")

    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze BabyAI trajectories (JSONL).")
    ap.add_argument(
        "--input", "-i",
        type=str,
        default="babyai_out_batch",
        help="Input path: JSONL file or directory of JSON files (default: babyai_out_batch)",
    )
    ap.add_argument(
        "--min-per-bin",
        type=int,
        default=10,
        help="Target minimum count per token bin (default: 10)",
    )
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Error: not found: {path}")
        print("Run: python batch_trajetory_gen.py --difficulty <level> --random_ratio <ratio> --observation_format <format> --traj_per_bin <N> --output_dir babyai_out_batch")
        return

    trajs = load_trajectories(path)
    if not trajs:
        print("No trajectories in file.")
        return

    stats = analyze_trajectories(trajs)
    print_analysis(stats, "BabyAI Trajectory Analysis")
    check_requirements(stats, min_per_bin=args.min_per_bin)


if __name__ == "__main__":
    main()
