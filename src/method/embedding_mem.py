"""
Embedding-based Memory Method - Uses semantic embeddings for memory construction and retrieval
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, override

import numpy as np
from gguf import Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from utils.embedding import EmbeddingEngine

from .base import *


@dataclass
class EmbeddingConfig(BaseConfig):
    """Configuration for embedding-based method"""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    use_faiss: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.use_faiss &= FAISS_AVAILABLE


@dataclass
class EmbeddingMemory(BaseMemory):
    """Memory object for embedding-based method"""

    documents: List[str]
    embeddings: np.ndarray
    index: faiss.IndexFlatIP | None = None  # FAISS index if available


class EmbeddingMethod(BaseMethod):
    """
    Embedding-based memory method.

    Uses semantic embeddings to retrieve relevant trajectory segments.
    Supports both FAISS (if available) and simple cosine similarity.
    """

    def __init__(
        self, config_path: os.PathLike = None, embedding_engine: EmbeddingEngine = None
    ):
        """
        Initialize embedding method.

        Args:
            embedding_model: Name of the embedding model
            top_k: Number of top documents to retrieve
            use_faiss: Whether to use FAISS index (requires faiss-cpu package)
            config_path: Path to configuration file (optional)
            embedding_engine: Optional external embedding engine (from utils.embedding)
        """
        super().__init__(config_path=config_path, embedding_engine=embedding_engine)
        self.config = self._parse_config()

        self.tokenizer, self.model = self._load_embedding_model(
            self.config.embedding_model
        )

    @override
    def _parse_config(self):
        config_dict = self._load_config(self.config_path)
        return EmbeddingConfig(
            embedding_model=config_dict.get("embedding_model"),
            top_k=config_dict.get("top_k"),
            use_faiss=config_dict.get("use_faiss"),
        )

    def _load_embedding_model(
        self, model_name: str
    ) -> Tuple[Optional[PreTrainedTokenizerBase], Optional[PreTrainedModel]]:
        if model_name is None:
            return None, None

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for EmbeddingMethod. "
                "Install with: pip install transformers torch"
            )

        try:
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                model_name
            )
            model: PreTrainedModel = AutoModel.from_pretrained(model_name)
            model.eval()
            return tokenizer, model
        except Exception:
            return None, None

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        # Use external embedding engine if available
        if self.embedding_engine is not None:
            return self.embedding_engine.encode(texts)

        # Otherwise use local model
        embeddings = []

        for text in texts:
            # Tokenize
            inputs: Dict[str, torch.Tensor] = self.tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )

            # Get embeddings
            with torch.no_grad():
                outputs: BaseModelOutput = self.model(**inputs)

            # Use mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask

            embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings)

    @override
    def memory_construction(self, traj_text: str, task: str = "") -> EmbeddingMemory:
        """
        Build embedding index from trajectory text.

        Args:
            traj_text: String-formatted trajectory text
            task: Task description (will be prepended to trajectory)

        Returns:
            EmbeddingMemory object containing documents and embeddings
        """
        # Add task description if provided
        full_text = traj_text
        if task:
            full_text = f"Task: {task}\n\n{traj_text}"

        # Split trajectory into documents (one per turn)
        documents = []
        lines = full_text.split("\n")

        current_turn = []
        for line in lines:
            if line.strip().startswith("Turn ") or line.strip().startswith("Step "):
                if current_turn:
                    documents.append("\n".join(current_turn))
                    current_turn = []
            current_turn.append(line)

        # Add the last turn
        if current_turn:
            documents.append("\n".join(current_turn))

        # If no turns found, split by chunks
        if not documents:
            # Split into chunks of ~500 characters
            chunk_size = 500
            documents = [
                full_text[i : i + chunk_size]
                for i in range(0, len(full_text), chunk_size)
            ]

        # Encode documents to embeddings
        embeddings = self._encode_text(documents)

        # Build FAISS index if enabled
        index = None
        if self.config.use_faiss:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(
                dimension
            )  # Inner product (cosine similarity after normalization)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings)

        return EmbeddingMemory(documents, embeddings, index)

    @override
    def memory_retrieve(self, memory: EmbeddingMemory, question: str) -> str:
        """
        Retrieve relevant documents using semantic similarity.

        Args:
            memory: EmbeddingMemory object
            question: Question to retrieve information for

        Returns:
            Retrieved context as string (top-k documents concatenated)
        """
        if not isinstance(memory, EmbeddingMemory):
            raise ValueError("Memory must be an EmbeddingMemory object")

        # Encode question
        question_embedding = self._encode_text([question])

        # Retrieve top-k documents
        if memory.index is not None:
            # Use FAISS index
            faiss.normalize_L2(question_embedding)
            scores, indices = memory.index.search(question_embedding, self.config.top_k)
            top_indices = indices[0].tolist()
        else:
            # Use simple cosine similarity
            # Normalize embeddings
            question_norm = question_embedding / (
                np.linalg.norm(question_embedding) + 1e-9
            )
            doc_norms = memory.embeddings / (
                np.linalg.norm(memory.embeddings, axis=1, keepdims=True) + 1e-9
            )

            # Compute cosine similarities
            similarities = np.dot(doc_norms, question_norm.T).flatten()

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][: self.config.top_k].tolist()

        # Get top documents
        retrieved_docs = [
            memory.documents[i] for i in top_indices if i < len(memory.documents)
        ]

        # Concatenate retrieved documents
        retrieved_context = "\n\n".join(retrieved_docs)

        return retrieved_context
