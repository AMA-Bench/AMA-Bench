"""
Meta-analysis script to scientifically evaluate LLM-as-judge performance.

This script analyzes multiple judge evaluations to assess:
1. Inter-judge reliability (agreement between different judges)
2. Task-specific performance patterns
3. Consistency and potential biases
4. Comparative judge quality
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_agreement_metrics(
    results1: List[Dict], results2: List[Dict]
) -> Dict[str, float]:
    """
    Calculate agreement metrics between two judge evaluations.

    Returns:
        - agreement_rate: Simple percentage agreement
        - cohen_kappa: Cohen's kappa coefficient (accounts for chance agreement)
        - pearson_r: Pearson correlation coefficient
    """
    # Match results by episode_id and question
    matched_pairs = []

    # Create lookup dict for results2
    results2_lookup = {(r["episode_id"], r["question"]): r["score"] for r in results2}

    for r1 in results1:
        key = (r1["episode_id"], r1["question"])
        if key in results2_lookup:
            matched_pairs.append((r1["score"], results2_lookup[key]))

    if not matched_pairs:
        return {"agreement_rate": 0.0, "cohen_kappa": 0.0, "pearson_r": 0.0}

    scores1, scores2 = zip(*matched_pairs)
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    # Agreement rate
    agreement_rate = np.mean(scores1 == scores2)

    # Cohen's Kappa
    n = len(scores1)
    po = agreement_rate  # Observed agreement
    p1_yes = np.mean(scores1 == 1.0)
    p2_yes = np.mean(scores2 == 1.0)
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)  # Expected agreement by chance
    cohen_kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0

    # Pearson correlation
    pearson_r, _ = stats.pearsonr(scores1, scores2)

    return {
        "agreement_rate": agreement_rate,
        "cohen_kappa": cohen_kappa,
        "pearson_r": pearson_r,
        "n_samples": n,
    }


def analyze_task_difficulty(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze task difficulty based on judge scores.
    Tasks with lower average scores are considered harder.
    """
    task_stats = eval_results["by_task_type"]

    # Sort tasks by difficulty (ascending accuracy)
    sorted_tasks = sorted(task_stats.items(), key=lambda x: x[1]["accuracy"])

    return {
        "easiest_tasks": [
            {"task": t[0], "accuracy": t[1]["accuracy"], "count": t[1]["count"]}
            for t in sorted_tasks[-3:]
        ],
        "hardest_tasks": [
            {"task": t[0], "accuracy": t[1]["accuracy"], "count": t[1]["count"]}
            for t in sorted_tasks[:3]
        ],
        "task_variance": np.var([t[1]["accuracy"] for t in sorted_tasks]),
    }


def calculate_strictness_score(eval_results: Dict[str, Any]) -> float:
    """
    Calculate judge strictness (lower accuracy = stricter judge).
    """
    return eval_results["overall"]["accuracy"]


def analyze_all_judges(result_files: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Comprehensive analysis of all judge evaluations.

    Args:
        result_files: List of (judge_name, file_path) tuples

    Returns:
        Comprehensive analysis dictionary
    """
    # Load all results
    all_results = {}
    for judge_name, file_path in result_files:
        print(f"Loading {judge_name} results from {file_path}...")
        all_results[judge_name] = load_evaluation_results(file_path)

    analysis = {
        "judges": {},
        "inter_judge_agreement": {},
        "task_difficulty_consensus": {},
        "strictness_ranking": {},
        "recommendations": [],
    }

    # Analyze each judge individually
    for judge_name, results in all_results.items():
        analysis["judges"][judge_name] = {
            "overall_accuracy": results["overall"]["accuracy"],
            "total_questions": results["overall"]["total_questions"],
            "task_difficulty_analysis": analyze_task_difficulty(results),
            "config": results["config"],
        }

    # Calculate pairwise agreement between judges
    judge_names = list(all_results.keys())
    for i, judge1 in enumerate(judge_names):
        for judge2 in judge_names[i + 1 :]:
            pair_key = f"{judge1} vs {judge2}"
            agreement = calculate_agreement_metrics(
                all_results[judge1]["results"], all_results[judge2]["results"]
            )
            analysis["inter_judge_agreement"][pair_key] = agreement

    # Rank judges by strictness (lower accuracy = stricter)
    strictness = {
        judge: calculate_strictness_score(results)
        for judge, results in all_results.items()
    }
    analysis["strictness_ranking"] = dict(
        sorted(strictness.items(), key=lambda x: x[1])
    )

    # Analyze task difficulty consensus
    # Check which tasks all judges agree are hard/easy
    task_scores_by_judge = defaultdict(list)
    for judge_name, results in all_results.items():
        for task, stats in results["by_task_type"].items():
            task_scores_by_judge[task].append(
                {"judge": judge_name, "accuracy": stats["accuracy"]}
            )

    for task, scores in task_scores_by_judge.items():
        accuracies = [s["accuracy"] for s in scores]
        analysis["task_difficulty_consensus"][task] = {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "coefficient_of_variation": (
                np.std(accuracies) / np.mean(accuracies)
                if np.mean(accuracies) > 0
                else 0
            ),
            "judge_scores": scores,
        }

    # Generate recommendations
    avg_kappa = np.mean(
        [
            metrics["cohen_kappa"]
            for metrics in analysis["inter_judge_agreement"].values()
        ]
    )

    if avg_kappa > 0.8:
        analysis["recommendations"].append(
            "EXCELLENT: High inter-judge agreement (Kappa > 0.8). Judges are reliable."
        )
    elif avg_kappa > 0.6:
        analysis["recommendations"].append(
            "GOOD: Moderate inter-judge agreement (Kappa > 0.6). Acceptable for research."
        )
    else:
        analysis["recommendations"].append(
            "WARNING: Low inter-judge agreement (Kappa < 0.6). Consider using ensemble or stronger judge."
        )

    # Check if there's a clear best judge (highest agreement with others)
    avg_agreement_per_judge = defaultdict(list)
    for pair_key, metrics in analysis["inter_judge_agreement"].items():
        for judge in judge_names:
            if judge in pair_key:
                avg_agreement_per_judge[judge].append(metrics["cohen_kappa"])

    best_judge = max(avg_agreement_per_judge.items(), key=lambda x: np.mean(x[1]))
    analysis["most_reliable_judge"] = {
        "judge": best_judge[0],
        "avg_kappa_with_others": np.mean(best_judge[1]),
    }

    return analysis


def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Print formatted analysis report."""
    print("\n" + "=" * 80)
    print("LLM-AS-JUDGE META-ANALYSIS REPORT")
    print("=" * 80)

    # Individual judge performance
    print("\n1. INDIVIDUAL JUDGE PERFORMANCE")
    print("-" * 80)
    for judge, stats in analysis["judges"].items():
        print(f"\n{judge.upper()}:")
        print(
            f"  Model: {stats['config']['judge_provider']}/{stats['config']['judge_model']}"
        )
        print(f"  Overall Accuracy: {stats['overall_accuracy']:.4f}")
        print(f"  Total Questions: {stats['total_questions']}")

        print(f"\n  Hardest Tasks:")
        for task in stats["task_difficulty_analysis"]["hardest_tasks"]:
            print(
                f"    - {task['task']}: {task['accuracy']:.4f} ({task['count']} questions)"
            )

        print(f"\n  Easiest Tasks:")
        for task in stats["task_difficulty_analysis"]["easiest_tasks"]:
            print(
                f"    - {task['task']}: {task['accuracy']:.4f} ({task['count']} questions)"
            )

    # Inter-judge agreement
    print("\n\n2. INTER-JUDGE AGREEMENT (Reliability)")
    print("-" * 80)
    for pair, metrics in analysis["inter_judge_agreement"].items():
        print(f"\n{pair}:")
        print(f"  Agreement Rate: {metrics['agreement_rate']:.4f}")
        print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"  Pearson r: {metrics['pearson_r']:.4f}")
        print(f"  Samples: {metrics['n_samples']}")

    kappa_interpretation = """
    Kappa Interpretation (Landis & Koch, 1977):
      < 0.00: Poor
      0.00 - 0.20: Slight
      0.21 - 0.40: Fair
      0.41 - 0.60: Moderate
      0.61 - 0.80: Substantial
      0.81 - 1.00: Almost Perfect
    """
    print(kappa_interpretation)

    # Strictness ranking
    print("\n\n3. JUDGE STRICTNESS RANKING")
    print("-" * 80)
    print("(Lower accuracy = stricter/harder grading)")
    for rank, (judge, accuracy) in enumerate(analysis["strictness_ranking"].items(), 1):
        print(f"  {rank}. {judge}: {accuracy:.4f}")

    # Most reliable judge
    print("\n\n4. MOST RELIABLE JUDGE")
    print("-" * 80)
    print(f"  Judge: {analysis['most_reliable_judge']['judge']}")
    print(
        f"  Avg Kappa with others: {analysis['most_reliable_judge']['avg_kappa_with_others']:.4f}"
    )

    # Task consensus
    print("\n\n5. TASK DIFFICULTY CONSENSUS")
    print("-" * 80)
    sorted_tasks = sorted(
        analysis["task_difficulty_consensus"].items(),
        key=lambda x: x[1]["mean_accuracy"],
    )

    print("\nHardest tasks (all judges agree):")
    for task, stats in sorted_tasks[:5]:
        print(f"  {task}: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")

    print("\nEasiest tasks (all judges agree):")
    for task, stats in sorted_tasks[-5:]:
        print(f"  {task}: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")

    print("\nTasks with highest judge disagreement:")
    high_variance = sorted(
        analysis["task_difficulty_consensus"].items(),
        key=lambda x: x[1]["std_accuracy"],
        reverse=True,
    )[:5]
    for task, stats in high_variance:
        print(f"  {task}: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"    Coefficient of Variation: {stats['coefficient_of_variation']:.4f}")

    # Recommendations
    print("\n\n6. RECOMMENDATIONS")
    print("-" * 80)
    for i, rec in enumerate(analysis["recommendations"], 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 80)


def export_to_csv(analysis: Dict[str, Any], output_dir: str) -> None:
    """Export analysis results to CSV files for further analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Export inter-judge agreement
    agreement_data = []
    for pair, metrics in analysis["inter_judge_agreement"].items():
        agreement_data.append({"judge_pair": pair, **metrics})
    pd.DataFrame(agreement_data).to_csv(
        output_dir / "inter_judge_agreement.csv", index=False
    )

    # Export task consensus
    task_data = []
    for task, stats in analysis["task_difficulty_consensus"].items():
        task_data.append(
            {
                "task": task,
                "mean_accuracy": stats["mean_accuracy"],
                "std_accuracy": stats["std_accuracy"],
                "cv": stats["coefficient_of_variation"],
            }
        )
    pd.DataFrame(task_data).to_csv(
        output_dir / "task_difficulty_consensus.csv", index=False
    )

    print(f"\nCSV files exported to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Meta-analysis of LLM-as-judge evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/analyze_judges.py \\
    --results results/eval_claude.json results/eval_gpt52.json results/eval_qwen.json \\
    --names claude gpt52 qwen \\
    --output-dir analysis/
        """,
    )

    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Paths to evaluation result JSON files",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Names for each judge (in same order as --results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--export-csv", action="store_true", help="Export results to CSV files"
    )

    args = parser.parse_args()

    if len(args.results) != len(args.names):
        parser.error("Number of result files must match number of names")

    # Run analysis
    result_files = list(zip(args.names, args.results))
    analysis = analyze_all_judges(result_files)

    # Print report
    print_analysis_report(analysis)

    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(analysis, args.output_dir)

    # Save full analysis to JSON
    output_path = Path(args.output_dir) / "meta_analysis.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to: {output_path}")
