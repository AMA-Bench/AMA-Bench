"""I/O utilities for saving and loading evaluation results."""

import json
from pathlib import Path
from typing import List
from dataclasses import asdict

from utils.qa_result import QAResult


def save_results(results: List[QAResult], output_path: str):
    """
    Save evaluation results to JSON file.

    Args:
        results: List of QAResult objects
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_dict = [asdict(r) for r in results]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ Results saved to {output_path}")


def load_results(input_path: str) -> List[QAResult]:
    """
    Load evaluation results from JSON file.

    Args:
        input_path: Path to input JSON file

    Returns:
        List of QAResult objects
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        results_dict = json.load(f)

    results = []
    for r in results_dict:
        results.append(QAResult(**r))

    return results
