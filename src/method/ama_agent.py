"""
AMA-Agent Method - Adaptive Memory Agent with structured state memory and retrieval

This module provides two key functions:
1. memory_construction: Build state memory from trajectory
2. memory_retrieve: Retrieve relevant context for answering questions
"""
from typing import Any, Dict, Optional

from src.method.base_method import BaseMethod
from src.method.ama_agent_core.construct import construct_state_memory
from src.method.ama_agent_core.retrieve import memory_retrieve as _do_retrieve


class AMAAgentMemory:
    """Memory object for AMA-Agent method"""

    def __init__(self, memory_data: Dict[str, Any]):
        """
        Initialize memory from constructed memory data.

        Args:
            memory_data: Dictionary containing state_mem, causal_graph,
                         text_mem, embed_mem, trajectory
        """
        self.state_mem = memory_data.get('state_mem')
        self.causal_graph = memory_data.get('causal_graph')  # None if causal=False
        self.text_mem = memory_data.get('text_mem')
        self.embed_mem = memory_data.get('embed_mem')  # None if embed_engine not provided
        self.trajectory = memory_data.get('trajectory')

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for retrieval."""
        return {
            'state_mem': self.state_mem,
            'causal_graph': self.causal_graph,
            'text_mem': self.text_mem,
            'embed_mem': self.embed_mem,
            'trajectory': self.trajectory
        }


class AMAAgentMethod(BaseMethod):
    """
    AMA-Agent method using structured state memory construction and retrieval.

    This method:
    1. Constructs hierarchical state memory from trajectory
    2. Performs multi-stage retrieval with top-k chunks and LLM sufficiency judgment
    3. Integrates memory-augmented QA
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        client: Optional[Any] = None,
        embedding_engine: Optional[Any] = None
    ):
        config = self._load_config(config_path)
        self.temperature = config.get('temperature', 0)
        self.max_tokens = config.get('max_tokens', 8192)
        self.chunk_size = config.get('chunk_size', 8192)
        self.top_k = config.get('top_k', 5)
        self.causal = config.get('causal', False)

        self.client = client
        self.embedding_engine = embedding_engine

    def _call_llm(self, prompt: str) -> tuple:
        """Synchronous LLM call using the provided client."""
        response = self.client.query(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        return None, response 

    def memory_construction(self, traj_text: str, task: str = "") -> AMAAgentMemory:
        """
        Build structured state memory from trajectory text.

        1. Parses trajectory text into different turns
        2. Compresses trajectory into state memory using LLM
        3. Optionally extracts causal graph (if causal=True in config)
        4. Optionally builds embeddings (if embedding_engine is provided)

        Args:
            traj_text: String-formatted trajectory text
            task: Task description

        Returns:
            AMAAgentMemory object containing state memory and optional causal graph
        """
        memory_data = construct_state_memory(
            trajectory_text=traj_text,
            task=task,
            call_llm_func=self._call_llm,
            chunk_size=self.chunk_size,
            embed_engine=self.embedding_engine,
            causal=self.causal
        )
        return AMAAgentMemory(memory_data)

    def memory_retrieve(self, memory: AMAAgentMemory, question: str) -> str:
        """
        Retrieve relevant context from memory to answer a question.

        1. Retrieves top_k=5 chunks
        2. LLM judges if the retrieved context is sufficient

        Args:
            memory: AMAAgentMemory object built by memory_construction
            question: Question to answer

        Returns:
            Retrieved context string
        """
        return _do_retrieve(
            memory=memory.to_dict(),
            question=question,
            call_llm_func=self._call_llm,
            top_k=self.top_k
        )
