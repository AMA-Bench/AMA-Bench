"""
AMA-Agent Method - Adaptive Memory Agent with structured state memory and retrieval

This module provides two key functions:
1. memory_construction: Build state memory from trajectory
2. memory_retrieve: Retrieve relevant context for answering questions
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, override

from .ama_agent_core.construct import construct_state_memory
from .ama_agent_core.retrieve import memory_retrieve as _do_retrieve
from .base import *


@dataclass
class AMAAgentConfig(BaseConfig):
    """Configuration for AMA-Agent method"""

    temperature: float = 0
    chunk_size: int = 2048
    session_size: int = 16384
    top_k: int = 5
    causal: bool = False


@dataclass
class AMAAgentMemory(BaseMemory):
    """Memory object for AMA-Agent method"""

    state_mem: str
    text_mem: Dict[str, Any]
    trajectory: str
    causal_graph: List[Dict[str, Any]] = None  # None if causal=False
    embed_mem: Dict[str, Any] = None  # None if embedding_engine not provided

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AMAAgentMemory":
        return cls(**d)


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
        config_path: os.PathLike = None,
        client: ModelClient = None,
        embedding_engine: EmbeddingEngine = None,
    ):
        super().__init__(
            config_path=config_path, client=client, embedding_engine=embedding_engine
        )

        self.config = self._parse_config()

        # max_tokens and max_model_length come from the LLM config, not method config
        llm_cfg = (
            client.config if (client is not None and hasattr(client, "config")) else {}
        )
        vllm_launch = llm_cfg.get("vllm_launch", {})
        self.max_tokens = llm_cfg.get("max_tokens") or vllm_launch.get(
            "max_response_len", 8192
        )
        self.max_model_length = llm_cfg.get("max_model_len") or vllm_launch.get(
            "max_model_len", 131072
        )

    def _call_llm(self, prompt: str) -> tuple:
        """Synchronous LLM call using the provided client."""
        response = self.client.query(
            prompt, temperature=self.config.temperature, max_tokens=self.max_tokens
        )
        return None, response

    @override
    def _parse_config(self) -> AMAAgentConfig:
        config_dict = self._load_config(self.config_path)
        return AMAAgentConfig(
            temperature=config_dict.get("temperature"),
            chunk_size=config_dict.get("chunk_size"),
            session_size=config_dict.get("session_size"),
            top_k=config_dict.get("top_k"),
            causal=config_dict.get("causal"),
        )

    @override
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
            chunk_size=self.config.chunk_size,
            session_size=self.config.session_size,
            embed_engine=self.embedding_engine,
            causal=self.config.causal,
        )
        return AMAAgentMemory.from_dict(memory_data)

    @override
    def memory_retrieve(self, memory: AMAAgentMemory, question: str) -> str:
        """
        Retrieve relevant context from memory to answer a question.

        Multi-stage pipeline (delegated to retrieve.memory_retrieve):
        1. Check if state memory alone is sufficient
        2. K=5 similarity-based node retrieval via embedding engine (port from config)
        3. LLM self-evaluation; if insufficient, invoke graph-node search or keyword search
        4. Synthesize all gathered evidence

        Args:
            memory: AMAAgentMemory object built by memory_construction
            question: Question to answer

        Returns:
            Retrieved context string
        """
        return _do_retrieve(
            memory=asdict(memory),
            question=question,
            call_llm_func=self._call_llm,
            top_k=self.config.top_k,
            embed_engine=self.embedding_engine,
            max_context_length=self.max_model_length - self.max_tokens,
        )
