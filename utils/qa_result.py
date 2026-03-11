"""QA Result data structure for AMA-Bench evaluation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QAResult:
    """Result of a single QA evaluation."""
    file_name: str
    episode_id: int
    task_type: str
    qa_index: int
    question: str
    golden_answer: str
    predicted_answer: str
    exact_match: float
    f1_score: Optional[float]
    llm_judge_score: Optional[float]
    evaluation_time: float
    qa_type: Optional[str] = None
