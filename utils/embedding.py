from typing import List, Optional, Dict, Any
import numpy as np
import os


class EmbeddingEngine:
    """
    Embedding Engine for generating text embeddings.

    Supports multiple embedding models including:
    - HuggingFace models (e.g., sentence-transformers)
    - Local embedding models via vLLM or API
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        """
        Initialize embedding engine.

        Args:
            model_name: Name or path of embedding model (e.g., "qwen3-embedding-4B")
            base_url: Base URL for API-based embedding models
            api_key: API key for authentication
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length for embedding
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

        # Initialize based on model type
        if model_name:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model based on configuration."""
        # Check if using API-based embedding
        if self.base_url:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                self.use_api = True
            except ImportError:
                raise ImportError(
                    "openai package is required for API-based embeddings. "
                    "Install with: pip install openai"
                )
        else:
            # Use local HuggingFace model
            try:
                import torch
                from transformers import AutoTokenizer, AutoModel

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()
                self.use_api = False

                # Move to GPU if available
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(self.device)

            except ImportError:
                raise ImportError(
                    "transformers and torch are required for local embeddings. "
                    "Install with: pip install transformers torch"
                )

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if not texts:
            return np.array([])

        if self.use_api:
            return self._encode_with_api(texts)
        else:
            return self._encode_with_local_model(texts)

    def _encode_with_api(self, texts: List[str]) -> np.ndarray:
        """Encode texts using API-based embedding model."""
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _encode_with_local_model(self, texts: List[str]) -> np.ndarray:
        """Encode texts using local HuggingFace model."""
        import torch

        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if embeddings else np.array([])

    def __call__(self, text: str) -> np.ndarray:
        """
        Encode a single text string.

        Args:
            text: Text string to encode

        Returns:
            Embedding vector as numpy array
        """
        return self.encode([text])[0]


