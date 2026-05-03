import os
import subprocess
import time
from typing import List, Optional

import numpy as np


class EmbeddingEngine:
    """
    Embedding Engine for generating text embeddings.

    Supports multiple embedding models including:
    - HuggingFace models (e.g., sentence-transformers)
    - Local embedding models via vLLM or API

    When base_url is provided and auto_launch=True, automatically starts a local
    vLLM embedding server if the endpoint is not already reachable.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        batch_size: int = 8,
        max_length: int = 512,
        # auto-launch parameters
        auto_launch: bool = False,
        host: str = "127.0.0.1",
        port: int = 8003,
        runner: str = "pooling",
        cuda_visible_devices: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        startup_timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_length = max_length
        self.auto_launch = auto_launch
        self.host = host
        self.port = port
        self.runner = runner
        self.cuda_visible_devices = cuda_visible_devices
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.startup_timeout = startup_timeout
        self.model = None
        self.tokenizer = None
        self._server_proc = None

        if model_name:
            self._initialize_model()

    # ──────────────────────────────────────────────────────────────────────
    # Server management
    # ──────────────────────────────────────────────────────────────────────

    def _health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def _is_server_ready(self) -> bool:
        """Return True if the vLLM server is already up and responding."""
        try:
            import urllib.request

            with urllib.request.urlopen(self._health_url(), timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _launch_server(self) -> None:
        """
        Start the vLLM embedding server as a background process and block
        until it is ready (or startup_timeout seconds have elapsed).
        """
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--runner",
            self.runner,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]

        env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)

        log_path = f"embedding_server_{self.port}.log"
        print(
            f"Launching vLLM embedding server: {self.model_name} "
            f"on {self.host}:{self.port} (GPU {self.cuda_visible_devices})"
        )
        print(f"Server log → {log_path}")

        with open(log_path, "w") as log_f:
            self._server_proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )

        # Wait for the server to become healthy
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._is_server_ready():
                print(f"Embedding server ready at http://{self.host}:{self.port}")
                return
            if self._server_proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM embedding server exited unexpectedly (code "
                    f"{self._server_proc.returncode}). Check {log_path}."
                )
            time.sleep(3)

        self._server_proc.terminate()
        raise TimeoutError(
            f"vLLM embedding server did not become ready within "
            f"{self.startup_timeout}s. Check {log_path}."
        )

    def _maybe_launch_server(self) -> None:
        """Launch the embedding server if auto_launch is True and it is not already running."""
        if not self.auto_launch:
            return
        if self._is_server_ready():
            print(f"Embedding server already running at http://{self.host}:{self.port}")
            return
        self._launch_server()

    # ──────────────────────────────────────────────────────────────────────
    # Model initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _initialize_model(self):
        """Initialize the embedding model based on configuration."""
        if self.base_url:
            # Optionally auto-launch the vLLM embedding server
            self._maybe_launch_server()
            try:
                from openai import OpenAI

                self.client = OpenAI(
                    base_url=self.base_url, api_key=self.api_key, timeout=120.0
                )
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
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()
                self.use_api = False

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(self.device)

            except ImportError:
                raise ImportError(
                    "transformers and torch are required for local embeddings. "
                    "Install with: pip install transformers torch"
                )

    # ──────────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────────

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        if self.use_api:
            return self._encode_with_api(texts)
        else:
            return self._encode_with_local_model(texts)

    def _encode_with_api(self, texts: List[str]) -> np.ndarray:
        """Encode texts using API-based embedding model (with retry)."""
        try:
            from openai import BadRequestError
        except ImportError:
            BadRequestError = None

        embeddings = []
        max_retries = 3
        # Use 2 chars/token estimate (conservative for multilingual/code content)
        # to stay safely below the model's token context limit.
        # Qwen3 tokenizer encodes typical content at ~1.6 chars/token, so
        # max_length * 2 gives ~max_length / 1.6 * 2 ≈ 1.25× headroom.
        _char_limit = self.max_length * 2

        for i in range(0, len(texts), self.batch_size):
            batch = [t[:_char_limit] for t in texts[i : i + self.batch_size]]
            last_exc = None
            current_char_limit = _char_limit
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=batch, model=self.model_name
                    )
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    # 400 means token limit exceeded: truncate harder and retry immediately
                    if BadRequestError is not None and isinstance(e, BadRequestError):
                        current_char_limit = current_char_limit // 2
                        batch = [
                            t[:current_char_limit]
                            for t in texts[i : i + self.batch_size]
                        ]
                    elif attempt < max_retries - 1:
                        time.sleep(2**attempt)
            if last_exc is not None:
                raise last_exc
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _encode_with_local_model(self, texts: List[str]) -> np.ndarray:
        """Encode texts using local HuggingFace model."""
        import torch

        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if embeddings else np.array([])

    def __call__(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def shutdown(self) -> None:
        """Terminate the embedding server if it was launched by this instance."""
        if self._server_proc is not None and self._server_proc.poll() is None:
            print(f"Shutting down embedding server (PID {self._server_proc.pid})")
            self._server_proc.terminate()
            self._server_proc = None
