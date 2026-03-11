# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
BM25 Method - Uses BM25 retrieval for memory construction and retrieval
"""

from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from src.method.base_method import BaseMethod


class BM25Memory:
    """Memory object for BM25 method"""

    def __init__(self, documents: List[str], bm25_index: BM25Okapi, corpus_tokens: List[List[str]]):
        self.documents = documents
        self.bm25_index = bm25_index
        self.corpus_tokens = corpus_tokens


class BM25Method(BaseMethod):
    """
    BM25-based memory method.

    Uses BM25 (Best Matching 25) ranking function to retrieve relevant trajectory segments.
    """

    def __init__(self, top_k: int = 5, config_path: str = None, embedding_engine: Any = None):
        """
        Initialize BM25 method.

        Args:
            top_k: Number of top documents to retrieve for each question
            config_path: Path to configuration file (optional)
            embedding_engine: Optional embedding engine (not used by BM25, for compatibility)
        """

        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            top_k = config.get('top_k', top_k)

        self.top_k = top_k
        self.embedding_engine = embedding_engine  # Not used, for compatibility

    def memory_construction(self, traj_text: str, task: str = "") -> BM25Memory:
        """
        Build BM25 index from trajectory text.

        Args:
            traj_text: String-formatted trajectory text
            task: Task description (optional, not used in BM25)

        Returns:
            BM25Memory object containing the index and documents
        """
        # Split trajectory into documents (one per turn)
        # Each turn is separated by double newline
        documents = []
        lines = traj_text.split('\n')

        current_turn = []
        for line in lines:
            if line.strip().startswith('Turn ') or line.strip().startswith('Step '):
                if current_turn:
                    documents.append('\n'.join(current_turn))
                    current_turn = []
            current_turn.append(line)

        # Add the last turn
        if current_turn:
            documents.append('\n'.join(current_turn))

        # If no turns found, treat entire text as one document
        if not documents:
            documents = [traj_text]

        # Tokenize documents for BM25 (simple whitespace tokenization)
        corpus_tokens = [doc.lower().split() for doc in documents]

        # Build BM25 index
        bm25_index = BM25Okapi(corpus_tokens)

        return BM25Memory(documents, bm25_index, corpus_tokens)

    def memory_retrieve(self, memory: BM25Memory, question: str) -> str:
        """
        Retrieve relevant documents using BM25.

        Args:
            memory: BM25Memory object
            question: Question to retrieve information for

        Returns:
            Retrieved context as string (top-k documents concatenated)
        """
        if not isinstance(memory, BM25Memory):
            raise ValueError("Memory must be a BM25Memory object")

        # Tokenize question
        query_tokens = question.lower().split()

        # Retrieve top-k documents using BM25
        scores = memory.bm25_index.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]

        # Get top documents
        retrieved_docs = [memory.documents[i] for i in top_indices]

        # Concatenate retrieved documents
        retrieved_context = "\n\n".join(retrieved_docs)

        return retrieved_context
