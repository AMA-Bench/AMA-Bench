"""
Long Context Method - Uses the full trajectory as context without compression or retrieval.

This method does not call any embedding service and is suitable for offline or
network-restricted runs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, override

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils.embedding import EmbeddingEngine

from .base import *


@dataclass
class LongContextConfig(BaseConfig):
    """Configuration for LongContextMethod"""

    max_model_length: int = 16384  # Model's maximum context length
    max_response_tokens: int = 4096  # Reserved for response
    model_name: str = (
        None  # Tokenizer model name (optional, for token counting and truncation)
    )
    safety_buffer: int = 300  # Fixed overhead for prompt template/formatting tokens


@dataclass
class LongContextMemory(BaseMemory):
    """Memory object for long context method - simply stores the full text"""

    full_text: str


class LongContextMethod(BaseMethod):
    """
    Long context method.

    Uses the entire trajectory as context without any compression or retrieval.
    Suitable for models with large context windows.
    Does not require embeddings or network access for memory construction/retrieval.
    """

    def __init__(self, config_path: os.PathLike = None):
        """
        Initialize long context method.

        Args:
            config_path: Path to configuration file (optional)
                        Config can specify: max_model_length, max_response_tokens
        """
        super().__init__(config_path=config_path)

        self.config = self._parse_config()
        self.tokenizer = self._load_tokenizer(self.config.model_name)

    @override
    def _parse_config(self) -> LongContextConfig:
        config_dict = self._load_config(self.config_path)
        vllm_launch_dict: Dict[str, Any] = config_dict.get("vllm_launch", {})

        max_model_length = config_dict.get("max_model_length") or vllm_launch_dict.get(
            "max_model_len"
        )
        max_response_tokens = config_dict.get(
            "max_response_tokens"
        ) or vllm_launch_dict.get("max_response_len")
        model_path = config_dict.get("model")

        return LongContextConfig(max_model_length, max_response_tokens, model_path)

    def _load_tokenizer(self, model_name: str = None) -> PreTrainedTokenizerBase | None:
        if model_name is None:
            return None
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception:
            return None

    def _encode_prompt_tokens(self, prompt: str):
        """Return token id list for prompt, or None if tokenizer unavailable."""
        if self.tokenizer is None:
            return None
        try:
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        except Exception:
            return None

    def truncate_prompt(
        self, prompt: str, target_length: Optional[int] = None
    ) -> Tuple[str, Optional[int]]:
        """Truncate prompt to fit within the context window.

        Keeps the first 70% and last 30% of the allowed budget so that
        recent context (end of trajectory) is always preserved.

        Returns:
            (truncated_prompt, actual_token_count_or_None)
        """
        if target_length is not None:
            max_allowed = target_length
        elif self.config.max_model_length is not None:
            max_allowed = (
                self.config.max_model_length
                - self.config.max_response_tokens
                - self.config.safety_buffer
            )
        else:
            return prompt, None

        max_allowed = max(100, max_allowed)

        token_ids = self._encode_prompt_tokens(prompt)
        if token_ids is None:
            return prompt, None

        token_count = len(token_ids)
        if token_count <= max_allowed:
            return prompt, token_count

        head_tokens = int(max_allowed * 0.7)
        tail_tokens = max_allowed - head_tokens

        truncated_ids = token_ids[:head_tokens]
        if tail_tokens > 0:
            truncated_ids = truncated_ids + token_ids[-tail_tokens:]

        try:
            truncated_prompt = self.tokenizer.decode(
                truncated_ids, skip_special_tokens=True
            )
        except Exception:
            truncated_prompt = self.tokenizer.decode(truncated_ids)

        truncated_len = len(self._encode_prompt_tokens(truncated_prompt) or []) or None
        # print(f"  Prompt tokens {token_count} exceed limit {max_allowed}; truncated to {truncated_len} tokens.")
        return truncated_prompt, truncated_len

    @override
    def memory_construction(self, traj_text: str, task: str = "") -> LongContextMemory:
        full_text = traj_text
        if task:
            full_text = (
                f"## Task Description\n{task}\n\n"
                f"## Agent Trajectory\n"
                f"The following is a step-by-step trajectory of the agent's actions and observations:\n\n"
                f"{traj_text}"
            )
        return LongContextMemory(full_text)

    @override
    def memory_retrieve(
        self,
        memory: LongContextMemory,
        questions: Union[str, List[str]],
        mcq_mode: bool = False,
    ) -> str:
        """Retrieve context from memory, accounting for question overhead.

        When *questions* is a plain string (single-question / backward-compat mode)
        the method returns the truncated context only, and the caller is responsible
        for assembling the final prompt.

        When *questions* is a list the method builds and returns the **complete**
        prompt in ``create_long_context_prompt`` format: context → questions →
        instructions → answer slots.  The trajectory section is truncated so that
        the whole prompt fits within the model's context window.

        Args:
            memory: LongContextMemory object produced by memory_construction.
            questions: Either a single question string or a list of question strings.
            mcq_mode: Whether to use MCQ answer format (default False / open-ended).

        Returns:
            Truncated context string (str input) or complete formatted prompt (list input).
        """
        if not isinstance(memory, LongContextMemory):
            raise ValueError("Memory must be a LongContextMemory object")

        if isinstance(questions, str):
            # Single question: return truncated context only (caller builds the prompt)
            question_ids = self._encode_prompt_tokens(questions)
            question_overhead = len(question_ids) if question_ids else 0
            target = (
                self.config.max_model_length
                - self.config.max_response_tokens
                - self.config.safety_buffer
                - question_overhead
            )
            truncated, _ = self.truncate_prompt(
                memory.full_text, target_length=max(100, target)
            )
            return truncated

        # --- List mode: build the complete create_long_context_prompt-style prompt ---
        questions_block = "\n".join(
            f"Question {i}: {q}\n" for i, q in enumerate(questions, 1)
        )

        if mcq_mode:
            section_intro = (
                "Please answer the following multiple-choice questions based on "
                "the task description and agent trajectory above."
            )
            instructions = (
                "For each question, select all correct options and respond using "
                "the format (A), (B), (C), or (D). "
                "If multiple options are correct, combine them like (A)(B)."
            )
            answer_slots = "\n".join(
                f"Answer[{i}]: [(A)/(B)/(C)/(D) or combination such as (A)(B)]"
                for i in range(1, len(questions) + 1)
            )
        else:
            section_intro = (
                "Please answer the following questions based on the task description "
                "and agent trajectory above. For each question, provide a direct and "
                "concise answer."
            )
            instructions = "Please provide answers in the following format:"
            answer_slots = "\n".join(
                f"Answer[{i}]: [your answer here]" for i in range(1, len(questions) + 1)
            )

        suffix = (
            f"\n\n## Questions\n{section_intro}\n\n"
            f"{questions_block}\n"
            f"## Instructions\n{instructions}\n\n"
            f"{answer_slots}"
        )

        # Deduct suffix tokens from available budget so the full prompt fits
        suffix_ids = self._encode_prompt_tokens(suffix)
        suffix_overhead = len(suffix_ids) if suffix_ids else 0
        target = (
            self.config.max_model_length
            - self.config.max_response_tokens
            - self.config.safety_buffer
            - suffix_overhead
        )
        truncated_context, _ = self.truncate_prompt(
            memory.full_text, target_length=max(100, target)
        )
        return truncated_context + suffix
