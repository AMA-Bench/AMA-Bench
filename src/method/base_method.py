""" 
Base class for memory construction and retrieval methods.

All methods in src/method should inherit from this base class and implement:
1. memory_construction: Build memory from string-formatted trajectory
2. memory_retrieve: Retrieve relevant memory information for a question
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import json
import yaml


class BaseMethod(ABC):
    """
    Base class for all memory methods.

    Each method must implement two core functions:
    1. memory_construction: Convert trajectory text to memory object
    2. memory_retrieve: Retrieve relevant information from memory for answering questions
    """
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Configuration dictionary
        """
        path = Path(config_path)
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif path.suffix == '.json':
                return json.load(f)
        return {}

    @abstractmethod
    def memory_construction(self, traj_text: str, task: str = "") -> Any:
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
    def memory_retrieve(self, memory: Any, question: str) -> str:
        """
        Retrieve relevant memory information for a question.

        Args:
            memory: Memory object built by memory_construction
            question: Question to answer

        Returns:
            Retrieved memory information as string
        """
        pass
