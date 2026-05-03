"""
Base class for memory construction and retrieval methods.

All methods in src/method should inherit from this base class and implement:
1. memory_construction: Build memory from string-formatted trajectory
2. memory_retrieve: Retrieve relevant memory information for a question
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import _MISSING_TYPE, dataclass, fields
from pathlib import Path
from typing import Any, Dict

import yaml

from model_client import ModelClient
from utils.embedding import EmbeddingEngine


@dataclass
class BaseConfig:
    """Base configuration dataclass for memory methods. Can be extended by specific methods."""

    # ^ Reference: https://stackoverflow.com/a/69944614
    def __post_init__(self):
        # Loop through the fields
        for field in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if (
                not isinstance(field.default, _MISSING_TYPE)
                and getattr(self, field.name) is None
            ):
                setattr(self, field.name, field.default)


@dataclass
class BaseMemory:
    """
    Base class for memory objects.
    This class can be extended to define specific memory structures for different methods.
    """

    pass


class BaseMethod(ABC):
    """
    Base class for all memory methods.

    Each method must implement two core functions:
    1. memory_construction: Convert trajectory text to memory object
    2. memory_retrieve: Retrieve relevant information from memory for answering questions
    """

    # * see get_method() in method_register.py
    def __init__(
        self,
        config_path: os.PathLike = None,
        client: ModelClient = None,
        embedding_engine: EmbeddingEngine = None,
    ):
        self.config_path = config_path
        self.client = client
        self.embedding_engine = embedding_engine

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file (JSON or YAML)

        Returns:
            Configuration dictionary
        """
        if config_path is None:
            return {}
        path = Path(config_path)
        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f) or {}
            elif path.suffix == ".json":
                return json.load(f)
        return {}

    @staticmethod
    def _pick(*values):
        for v in values:
            if v is not None:
                return v
        return None

    @abstractmethod
    def _parse_config(self) -> BaseConfig:
        """
        Parse configuration and set method parameters.
        This function can be overridden by specific methods to handle method-specific config parameters.
        """
        pass

    @abstractmethod
    def memory_construction(self, traj_text: str, task: str = "") -> BaseMemory:
        """
        Build memory from string-formatted trajectory.

        Args:
            traj_text: String-formatted trajectory text
            task: Task description (optional)

        Returns:
            Memory object (format depends on specific method implementation)
        """
        pass

    @abstractmethod
    def memory_retrieve(self, memory: BaseMemory, question: str) -> str:
        """
        Retrieve relevant memory information for a question.

        Args:
            memory: Memory object built by memory_construction
            question: Question to answer

        Returns:
            Retrieved memory information as string
        """
        pass
