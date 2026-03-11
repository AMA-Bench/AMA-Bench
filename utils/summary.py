"""Summary and reporting utilities for evaluation results."""

from typing import List
from collections import defaultdict

from utils.qa_result import QAResult


def print_summary(
    results: List[QAResult],
    provider: str = "",
    model: str = "",
    llm_as_judge: str = "none",
):
    """
    Print evaluation summary with detailed metrics.

    Args:
        results: List of QAResult objects
        provider: Model provider name
        model: Model name
        llm_as_judge: LLM judge mode
    """
    if not results:
        print("\n❌ No results to summarize")
        return

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")

    # Overall metrics
    avg_em = sum(r.exact_match for r in results) / len(results)
    avg_f1 = sum(r.f1_score or 0 for r in results) / len(results)
    total_time = sum(r.evaluation_time for r in results)

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Total QA pairs: {len(results)}")
    print(f"Average Exact Match: {avg_em:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    # LLM judge metrics
    if llm_as_judge != "none":
        llm_scores = [r.llm_judge_score for r in results if r.llm_judge_score is not None]
        if llm_scores:
            avg_llm = sum(llm_scores) / len(llm_scores)
            accuracy = sum(1 for s in llm_scores if s >= 0.5) / len(llm_scores)
            print(f"Average LLM Judge Score: {avg_llm:.4f}")
            print(f"LLM Judge Accuracy: {accuracy:.4f} ({sum(1 for s in llm_scores if s >= 0.5)}/{len(llm_scores)})")

    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per QA: {total_time/len(results):.2f}s")

    # Per task type
    _print_task_type_metrics(results, llm_as_judge)

    # Per QA type
    _print_qa_type_metrics(results, llm_as_judge)

    print(f"{'='*70}\n")


def _print_task_type_metrics(results: List[QAResult], llm_as_judge: str):
    """Print metrics grouped by task type."""
    task_metrics = defaultdict(lambda: {"em": [], "f1": [], "llm_judge": []})

    for r in results:
        task_metrics[r.task_type]["em"].append(r.exact_match)
        if r.f1_score is not None:
            task_metrics[r.task_type]["f1"].append(r.f1_score)
        if r.llm_judge_score is not None:
            task_metrics[r.task_type]["llm_judge"].append(r.llm_judge_score)

    print(f"\n{'='*70}")
    print("Per Task Type:")
    print(f"{'='*70}")

    for task_type, metrics in sorted(task_metrics.items()):
        avg_em = sum(metrics["em"]) / len(metrics["em"])
        avg_f1 = sum(metrics["f1"]) / len(metrics["f1"]) if metrics["f1"] else 0.0

        metrics_str = f"  {task_type}: EM={avg_em:.4f}, F1={avg_f1:.4f}"

        if llm_as_judge != "none" and metrics["llm_judge"]:
            avg_llm = sum(metrics["llm_judge"]) / len(metrics["llm_judge"])
            llm_accuracy = sum(1 for s in metrics["llm_judge"] if s >= 0.5) / len(metrics["llm_judge"])
            metrics_str += f", LLM Judge={avg_llm:.4f}, Accuracy={llm_accuracy:.4f}"

        metrics_str += f" (n={len(metrics['em'])})"
        print(metrics_str)


def _print_qa_type_metrics(results: List[QAResult], llm_as_judge: str):
    """Print metrics grouped by QA type (A, B, C, D)."""
    qa_type_metrics = defaultdict(lambda: {"em": [], "f1": [], "llm_judge": []})

    for r in results:
        if r.qa_type:
            qa_type_metrics[r.qa_type]["em"].append(r.exact_match)
            if r.f1_score is not None:
                qa_type_metrics[r.qa_type]["f1"].append(r.f1_score)
            if r.llm_judge_score is not None:
                qa_type_metrics[r.qa_type]["llm_judge"].append(r.llm_judge_score)

    if not qa_type_metrics:
        return

    print(f"\n{'='*70}")
    print("Per QA Type:")
    print(f"{'='*70}")

    for qa_type in sorted(qa_type_metrics.keys()):
        metrics = qa_type_metrics[qa_type]
        avg_em = sum(metrics["em"]) / len(metrics["em"])
        avg_f1 = sum(metrics["f1"]) / len(metrics["f1"]) if metrics["f1"] else 0.0

        metrics_str = f"  Type {qa_type}: EM={avg_em:.4f}, F1={avg_f1:.4f}"

        if llm_as_judge != "none" and metrics["llm_judge"]:
            avg_llm = sum(metrics["llm_judge"]) / len(metrics["llm_judge"])
            llm_accuracy = sum(1 for s in metrics["llm_judge"] if s >= 0.5) / len(metrics["llm_judge"])
            metrics_str += f", LLM Judge={avg_llm:.4f}, Accuracy={llm_accuracy:.4f}"

        metrics_str += f" (n={len(metrics['em'])})"
        print(metrics_str)


def print_compact_summary(results: List[QAResult], llm_as_judge: str = "none"):
    """
    Print compact summary with metrics formatted as comma-separated values.

    Useful for quick comparison across different models.

    Args:
        results: List of QAResult objects
        llm_as_judge: LLM judge mode
    """
    if not results:
        print("No results to summarize")
        return

    qa_type_metrics = defaultdict(lambda: {"em": [], "f1": [], "llm_judge": []})

    for r in results:
        if r.qa_type:
            qa_type_metrics[r.qa_type]["em"].append(r.exact_match)
            if r.f1_score is not None:
                qa_type_metrics[r.qa_type]["f1"].append(r.f1_score)
            if r.llm_judge_score is not None:
                qa_type_metrics[r.qa_type]["llm_judge"].append(r.llm_judge_score)

    print("\nCompact Format:")
    print("="*70)

    # F1 scores
    f1_scores = []
    for qa_type in ['A', 'B', 'C', 'D']:
        if qa_type in qa_type_metrics and qa_type_metrics[qa_type]["f1"]:
            avg_f1 = sum(qa_type_metrics[qa_type]["f1"]) / len(qa_type_metrics[qa_type]["f1"])
            f1_scores.append(f"{avg_f1:.4f}")
        else:
            f1_scores.append("N/A")

    # Calculate average
    all_f1 = [r.f1_score for r in results if r.f1_score is not None and r.qa_type]
    if all_f1:
        avg_f1 = sum(all_f1) / len(all_f1)
        f1_scores.append(f"{avg_f1:.4f}")
    else:
        f1_scores.append("N/A")

    print(f"F1 score(A,B,C,D,ave): {','.join(f1_scores)}")

    # LLM judge scores
    if llm_as_judge != "none":
        llm_scores = []
        for qa_type in ['A', 'B', 'C', 'D']:
            if qa_type in qa_type_metrics and qa_type_metrics[qa_type]["llm_judge"]:
                avg_llm = sum(qa_type_metrics[qa_type]["llm_judge"]) / len(qa_type_metrics[qa_type]["llm_judge"])
                llm_scores.append(f"{avg_llm:.4f}")
            else:
                llm_scores.append("N/A")

        # Calculate average
        all_llm = [r.llm_judge_score for r in results if r.llm_judge_score is not None and r.qa_type]
        if all_llm:
            avg_llm = sum(all_llm) / len(all_llm)
            llm_scores.append(f"{avg_llm:.4f}")
        else:
            llm_scores.append("N/A")

        print(f"LLM-as-judge score(A,B,C,D,ave): {','.join(llm_scores)}")

    print("="*70)
