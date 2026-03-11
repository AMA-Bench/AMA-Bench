#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Metrics for QA Assessment.

This module provides various evaluation metrics:
- Exact Match (EM)
- F1 Score (token-level)
- Numeric Accuracy
- LLM-as-Judge

Usage:
    from evaluation_metrics import compute_exact_match, compute_f1_score
"""

import re
from typing import Optional, List, Set
from collections import Counter


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Strip whitespace
    """
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return normalize_text(text).split()


def compute_exact_match(predicted: str, golden: str) -> float:
    """
    Compute exact match score.

    Returns:
        1.0 if exact match (after normalization), 0.0 otherwise
    """
    pred_normalized = normalize_text(predicted)
    gold_normalized = normalize_text(golden)

    return 1.0 if pred_normalized == gold_normalized else 0.0


def compute_f1_score(predicted: str, golden: str) -> float:
    """
    Compute token-level F1 score.

    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = tokenize(predicted)
    gold_tokens = tokenize(golden)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    # Count common tokens
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = pred_counter & gold_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_numeric_accuracy(predicted: str, golden: str, tolerance: float = 1e-6) -> float:
    """
    Compute accuracy for numeric answers.

    Extracts numbers from both strings and compares with tolerance.

    Returns:
        1.0 if numbers match within tolerance, 0.0 otherwise, or falls back to exact match if no numbers found
    """
    # Extract numbers from strings
    pred_numbers = re.findall(r'-?\d+\.?\d*', predicted)
    gold_numbers = re.findall(r'-?\d+\.?\d*', golden)

    if not pred_numbers or not gold_numbers:
        # Fall back to exact match if no numbers found
        return compute_exact_match(predicted, golden)

    # Compare first number found
    try:
        pred_num = float(pred_numbers[0])
        gold_num = float(gold_numbers[0])
        return 1.0 if abs(pred_num - gold_num) <= tolerance else 0.0
    except ValueError:
        return compute_exact_match(predicted, golden)


def compute_llm_as_judge(
    question: str,
    golden_answer: str,
    predicted_answer: str,
    judge_client,
    scale: int = 5,
    task_description: str = "",
    task_type: str = "",
    episode_id: str = "",
) -> float:
    """
    Use LLM to judge answer quality with binary (yes/no) judgment.

    Args:
        question: The question asked
        golden_answer: Reference answer
        predicted_answer: Model's predicted answer
        judge_client: ModelClient instance for LLM judge
        scale: Not used (kept for backward compatibility)
        task_description: Description of the task context (optional)
        task_type: Type of the task (optional)
        episode_id: Episode ID for reference (optional)

    Returns:
        Binary score: 1.0 if correct, 0.0 if incorrect
    """
    # Build context information
    context_parts = []
    if task_type:
        context_parts.append(f"Task Type: {task_type}")
    if episode_id:
        context_parts.append(f"Episode ID: {episode_id}")
    if task_description:
        context_parts.append(f"Task Context: {task_description}")
    
    context_str = "\n".join(context_parts) if context_parts else ""
    
    judge_prompt = f"""You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

{context_str}

Question: {question}

Reference Answer: {golden_answer}

Predicted Answer: {predicted_answer}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:<think></think>"""

    try:
        # Increase max_tokens to handle potential thinking tags and ensure complete response
        response = judge_client.query(judge_prompt, temperature=0.0, max_tokens=2048)
        
        # Remove thinking tags if present (some models use <think>...</think>)
        response_cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response_cleaned = response_cleaned.strip()
        
        # Extract the last occurrence of yes or no (case-insensitive)
        # This handles cases where model explains before answering
        response_lower = response_cleaned.lower()
        
        # Find all matches of "yes" or "no" as complete words
        yes_matches = list(re.finditer(r'\byes\b', response_lower))
        no_matches = list(re.finditer(r'\bno\b', response_lower))
        
        # Determine which comes last
        last_yes_pos = yes_matches[-1].start() if yes_matches else -1
        last_no_pos = no_matches[-1].start() if no_matches else -1
        
        if last_yes_pos > last_no_pos:
            return 1.0
        elif last_no_pos > last_yes_pos:
            return 0.0
        else:
            # Fallback to F1 if no valid response
            print(f"Warning: Could not parse LLM judge response: '{response}'. Falling back to F1 score.")
            return compute_f1_score(predicted_answer, golden_answer)
    except Exception as e:
        print(f"Error: LLM judge failed: {e}")
        raise


def compute_multi_choice_accuracy(predicted: str, golden: str) -> float:
    """
    Compute accuracy for multiple choice answers.

    Handles various formats:
    - "1", "option 1", "answer: 1", etc.

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Extract first digit/number from both
    pred_match = re.search(r'\d+', predicted)
    gold_match = re.search(r'\d+', golden)

    if pred_match and gold_match:
        pred_num = pred_match.group()
        gold_num = gold_match.group()
        return 1.0 if pred_num == gold_num else 0.0

    # Fall back to exact match
    return compute_exact_match(predicted, golden)


def compute_contains_score(predicted: str, golden: str) -> float:
    """
    Check if golden answer is contained in prediction (or vice versa).

    Useful for answers that may have additional context.

    Returns:
        1.0 if one contains the other, 0.0 otherwise
    """
    pred_normalized = normalize_text(predicted)
    gold_normalized = normalize_text(golden)

    if gold_normalized in pred_normalized or pred_normalized in gold_normalized:
        return 1.0
    return 0.0


def compute_set_overlap(predicted: str, golden: str) -> float:
    """
    Compute set overlap for list-like answers.

    Example: "apple, banana" vs "banana, apple" -> 1.0

    Returns:
        Jaccard similarity between token sets
    """
    pred_tokens = set(tokenize(predicted))
    gold_tokens = set(tokenize(golden))

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    intersection = pred_tokens & gold_tokens
    union = pred_tokens | gold_tokens

    return len(intersection) / len(union)


def compute_all_metrics(predicted: str, golden: str, judge_client=None) -> dict:
    """
    Compute all available metrics for a prediction.

    Args:
        predicted: Predicted answer
        golden: Golden/reference answer
        judge_client: Optional ModelClient for LLM-as-judge

    Returns:
        Dictionary with all metric scores
    """
    metrics = {
        "exact_match": compute_exact_match(predicted, golden),
        "f1_score": compute_f1_score(predicted, golden),
        "numeric_accuracy": compute_numeric_accuracy(predicted, golden),
        "multi_choice_accuracy": compute_multi_choice_accuracy(predicted, golden),
        "contains_score": compute_contains_score(predicted, golden),
        "set_overlap": compute_set_overlap(predicted, golden),
    }

    if judge_client:
        # This would need the question too, so skip for now in this function
        pass

    return metrics


# Export main functions
__all__ = [
    "normalize_text",
    "tokenize",
    "compute_exact_match",
    "compute_f1_score",
    "compute_numeric_accuracy",
    "compute_llm_as_judge",
    "compute_multi_choice_accuracy",
    "compute_contains_score",
    "compute_set_overlap",
    "compute_all_metrics",
]
