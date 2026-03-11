"""
Long Context Method - Uses the full trajectory as context without compression or retrieval
"""

from typing import Any
import json
import yaml
from pathlib import Path

from src.method.base_method import BaseMethod


class LongContextMemory:
    """Memory object for long context method - simply stores the full text"""

    def __init__(self, full_text: str):
        self.full_text = full_text


class LongContextMethod(BaseMethod):
    """
    Long context method.

    Uses the entire trajectory as context without any compression or retrieval.
    Suitable for models with large context windows.
    """

    def __init__(self, config_path: str = None, embedding_engine: Any = None):
        """
        Initialize long context method.

        Args:
            config_path: Path to configuration file (optional)
                        Config can specify: max_model_length, max_response_tokens, chars_per_token
            embedding_engine: Optional embedding engine (not used by LongContext, for compatibility)
        """
        # Default values
        self.max_model_length = 16384  # Model's maximum context length
        self.max_response_tokens = 4096  # Reserved for response
        self.chars_per_token = 4  # Character-to-token ratio (rough estimate)

        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            self.max_model_length = config.get('max_model_length', self.max_model_length)
            self.max_response_tokens = config.get('max_response_tokens', self.max_response_tokens)
            self.chars_per_token = config.get('chars_per_token', self.chars_per_token)

        # Calculate max input tokens
        self.max_input_tokens = self.max_model_length - self.max_response_tokens
        self.embedding_engine = embedding_engine  # Not used, for compatibility

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from JSON or YAML file or parse JSON string.

        Args:
            config_path: Path to config file or JSON string

        Returns:
            Configuration dictionary
        """

        # Try to load as file
        config_file = Path(config_path)
        

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def memory_construction(self, traj_text: str, task: str = "") -> LongContextMemory:
        """
        Store the full trajectory text.

        Args:
            traj_text: String-formatted trajectory text
            task: Task description (will be prepended to trajectory)

        Returns:
            LongContextMemory object containing the full text
        """
        # Combine task and trajectory
        full_text = traj_text
        if task:
            full_text = f"# Task\n{task}\n\n# Agent Trajectory\n{traj_text}"

        return LongContextMemory(full_text)

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within token limits by keeping first 50% and last 50%.

        Args:
            text: Input text

        Returns:
            Truncated text
        """
        # Estimate token count
        estimated_tokens = len(text) / self.chars_per_token

        if estimated_tokens <= self.max_input_tokens:
            return text

        # Calculate target character count
        target_chars = int(self.max_input_tokens * self.chars_per_token)

        # Keep first 50% and last 50%
        half_chars = target_chars // 2

        first_half = text[:half_chars]
        last_half = text[-half_chars:]

        # Add separator
        truncated = first_half + "\n\n... [middle section truncated] ...\n\n" + last_half

        return truncated

    def memory_retrieve(self, memory: LongContextMemory, question: str) -> str:
        """
        Return the full trajectory text (no retrieval needed).

        Args:
            memory: LongContextMemory object
            question: Question (not used in this method)

        Returns:
            Full trajectory text (truncated if necessary)
        """
        if not isinstance(memory, LongContextMemory):
            raise ValueError("Memory must be a LongContextMemory object")

        return self._truncate_text(memory.full_text)
