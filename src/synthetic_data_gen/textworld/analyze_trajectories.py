

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_trajectories(jsonl_file: Path) -> List[Dict]:
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def analyze_trajectories(trajectories: List[Dict]) -> Dict:
    """Analyze trajectories and return statistics."""
    
    # Token bins
    token_bins = {
        "4K": (0, 4096),
        "8K": (4097, 8192),
        "16K": (8193, 16384),
        "32K": (16385, 32768),
        "64K": (32769, 65536),
        "128K": (65537, 131072),
    }
    
    stats = {
        "total_count": len(trajectories),
        "by_game_type": defaultdict(int),
        "by_token_bin": defaultdict(int),
        "by_game_and_bin": defaultdict(lambda: defaultdict(int)),
        "success_count": 0,
        "fail_count": 0,
        "total_tokens": 0,
        "total_turns": 0,
        "min_tokens": float("inf"),
        "max_tokens": 0,
        "min_turns": float("inf"),
        "max_turns": 0,
    }
    
    for traj in trajectories:
        game_type = traj["task_type"]
        tokens = traj["total_tokens"]
        turns = traj["num_turns"]
        state = traj["state"]
        
        # Count by game type
        stats["by_game_type"][game_type] += 1
        
        # Count by success/fail
        if state == "success":
            stats["success_count"] += 1
        else:
            stats["fail_count"] += 1
        
        # Find token bin
        bin_name = None
        for name, (min_tok, max_tok) in token_bins.items():
            if min_tok <= tokens <= max_tok:
                bin_name = name
                break
        
        if bin_name:
            stats["by_token_bin"][bin_name] += 1
            stats["by_game_and_bin"][game_type][bin_name] += 1
        else:
            stats["by_token_bin"]["out_of_range"] += 1
        
        # Accumulate stats
        stats["total_tokens"] += tokens
        stats["total_turns"] += turns
        stats["min_tokens"] = min(stats["min_tokens"], tokens)
        stats["max_tokens"] = max(stats["max_tokens"], tokens)
        stats["min_turns"] = min(stats["min_turns"], turns)
        stats["max_turns"] = max(stats["max_turns"], turns)
    
    # Calculate averages
    if len(trajectories) > 0:
        stats["avg_tokens"] = stats["total_tokens"] / len(trajectories)
        stats["avg_turns"] = stats["total_turns"] / len(trajectories)
        stats["success_rate"] = stats["success_count"] / len(trajectories) * 100
    else:
        stats["avg_tokens"] = 0
        stats["avg_turns"] = 0
        stats["success_rate"] = 0
    
    return stats


def print_analysis(stats: Dict, title: str = "Analysis Results"):
    """Print analysis results in a formatted way."""
    
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")
    
    print(f"📊 Overall Statistics:")
    print(f"  Total trajectories: {stats['total_count']}")
    print(f"  Success rate: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_count']})")
    print(f"  Fail count: {stats['fail_count']}")
    
    print(f"\n📈 Token Statistics:")
    print(f"  Average: {stats['avg_tokens']:,.0f} tokens")
    print(f"  Range: {stats['min_tokens']:,} - {stats['max_tokens']:,} tokens")
    
    print(f"\n🔄 Turn Statistics:")
    print(f"  Average: {stats['avg_turns']:.1f} turns")
    print(f"  Range: {stats['min_turns']} - {stats['max_turns']} turns")
    
    print(f"\n🎮 By Game Type:")
    for game_type in sorted(stats["by_game_type"].keys()):
        count = stats["by_game_type"][game_type]
        percentage = count / stats["total_count"] * 100 if stats["total_count"] > 0 else 0
        print(f"  {game_type:20s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n📦 By Token Bin (Overall):")
    bin_order = ["4K", "8K", "16K", "32K", "64K", "128K", "out_of_range"]
    for bin_name in bin_order:
        count = stats["by_token_bin"].get(bin_name, 0)
        percentage = count / stats["total_count"] * 100 if stats["total_count"] > 0 else 0
        status = "✓" if count >= 10 else "✗" if bin_name != "out_of_range" else " "
        print(f"  {status} {bin_name:15s}: {count:3d} ({percentage:5.1f}%) {'[TARGET: ≥10]' if bin_name != 'out_of_range' else ''}")
    
    print(f"\n📦 By Token Bin (Per Game Type):")
    for game_type in sorted(stats["by_game_and_bin"].keys()):
        print(f"\n  {game_type}:")
        for bin_name in ["4K", "8K", "16K", "32K", "64K", "128K"]:
            count = stats["by_game_and_bin"][game_type].get(bin_name, 0)
            total_for_game = stats["by_game_type"][game_type]
            percentage = count / total_for_game * 100 if total_for_game > 0 else 0
            print(f"    {bin_name:10s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n{'='*70}\n")


def check_requirements(stats: Dict) -> bool:
    """Check if requirements are met."""
    
    print(f"\n{'='*70}")
    print(f"{'REQUIREMENTS CHECK':^70}")
    print(f"{'='*70}\n")
    
    all_passed = True
    
    # Check total count
    print("1️⃣  Total Trajectory Count:")
    if stats["total_count"] >= 180:
        print(f"   ✓ PASS: {stats['total_count']} trajectories (target: 180)")
    else:
        print(f"   ✗ FAIL: {stats['total_count']} trajectories (target: 180)")
        all_passed = False
    
    # Check by game type
    print("\n2️⃣  Game Type Distribution:")
    target_per_game = 60
    for game_type in ["coin_collector", "cooking", "treasure_hunter"]:
        count = stats["by_game_type"].get(game_type, 0)
        if count >= target_per_game:
            print(f"   ✓ PASS: {game_type:20s} has {count} trajectories (target: {target_per_game})")
        else:
            print(f"   ✗ FAIL: {game_type:20s} has {count} trajectories (target: {target_per_game})")
            all_passed = False
    
    # Check token bins
    print("\n3️⃣  Token Bin Coverage (Overall):")
    min_per_bin = 10
    for bin_name in ["4K", "8K", "16K", "32K", "64K", "128K"]:
        count = stats["by_token_bin"].get(bin_name, 0)
        if count >= min_per_bin:
            print(f"   ✓ PASS: {bin_name:10s} has {count} trajectories (target: ≥{min_per_bin})")
        else:
            print(f"   ✗ FAIL: {bin_name:10s} has {count} trajectories (target: ≥{min_per_bin})")
            all_passed = False
    
    print(f"\n{'='*70}")
    if all_passed:
        print(f"{'🎉 ALL REQUIREMENTS MET! 🎉':^70}")
    else:
        print(f"{'❌ SOME REQUIREMENTS NOT MET':^70}")
    print(f"{'='*70}\n")
    
    return all_passed


def main():
    """Main entry point."""
    output_dir = Path("tw_out_batch")
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Please run batch_generate_trajectories.py first.")
        return
    
    # Analyze overall trajectories
    all_traj_file = output_dir / "all_trajectories.jsonl"
    if all_traj_file.exists():
        print(f"Loading trajectories from: {all_traj_file}")
        trajectories = load_trajectories(all_traj_file)
        
        if len(trajectories) == 0:
            print("No trajectories found!")
            return
        
        stats = analyze_trajectories(trajectories)
        print_analysis(stats, "Overall Analysis")
        
        # Check requirements
        all_passed = check_requirements(stats)
        
        # Analyze per game type
        for game_type in ["coin_collector", "cooking", "treasure_hunter"]:
            game_file = output_dir / f"{game_type}_trajectories.jsonl"
            if game_file.exists():
                game_trajs = load_trajectories(game_file)
                game_stats = analyze_trajectories(game_trajs)
                print_analysis(game_stats, f"{game_type.replace('_', ' ').title()} Analysis")
    else:
        print(f"Error: Trajectory file not found: {all_traj_file}")


if __name__ == "__main__":
    main()
