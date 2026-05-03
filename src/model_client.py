import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ModelClient:
    """Unified client for different model providers."""

    def __init__(self, config_path: str, server_type: str = "api", **kwargs):
        """
        Initialize model client.

        Args:
            config_path: Path to YAML config file containing provider, model, api_key, etc.
            server_type: Server type ("api" or "vllm")
            **kwargs: Additional provider-specific arguments (e.g., host, port for vllm)
        """
        # Load config file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        # Extract provider and model from config
        self.provider = config.get("provider", "").lower()
        self.model = config.get("model", "")
        self.config = config
        vllm_launch = config.get("vllm_launch", {})
        self._max_model_len = config.get("max_model_len") or vllm_launch.get(
            "max_model_len", 131072
        )

        # For vllm server type, override provider
        if server_type == "vllm":
            self.provider = "custom"
            # Use host/port from kwargs or config (support both 'host' and 'vllm_host' naming)
            host = config.get("vllm_host") or config.get("host", "localhost")
            port = config.get("vllm_port") or config.get("port", 8000)
            config["base_url"] = f"http://{host}:{port}/v1"

        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize provider-specific client."""
        if self.provider == "custom":
            from openai import OpenAI

            base_url = self.config.get("base_url")
            api_key = self.config.get("api_key", "EMPTY")
            return OpenAI(base_url=base_url, api_key=api_key, timeout=180.0)

        elif self.provider == "openai":
            from openai import OpenAI

            # Use api_key from config if provided, otherwise will use OPENAI_API_KEY env var
            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                return OpenAI(api_key=api_key, timeout=180.0)
            else:
                # Let OpenAI SDK handle the API key (will use OPENAI_API_KEY env var)
                return OpenAI(timeout=180.0)

        elif self.provider == "deepseek":
            from openai import OpenAI

            api_key = self.config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DeepSeek API key not found in config or DEEPSEEK_API_KEY environment variable"
                )
            return OpenAI(
                api_key=api_key, base_url="https://api.deepseek.com", timeout=180.0
            )

        elif self.provider == "gemini":
            import google.generativeai as genai

            api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key not found in config or GOOGLE_API_KEY environment variable"
                )
            genai.configure(api_key=api_key)
            return genai

        elif self.provider in ["anthropic", "claude"]:
            from anthropic import Anthropic

            api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                return Anthropic(api_key=api_key)
            else:
                # Let Anthropic SDK handle the API key (will use ANTHROPIC_API_KEY env var)
                return Anthropic()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def query(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
        system: Optional[str] = None,
    ) -> str:
        """Query model with prompt with retry logic for rate limits."""
        import re as _re

        _truncated = False
        attempt = 0
        while attempt < max_retries:
            try:
                if self.provider in ["custom", "deepseek"]:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content.strip()

                elif self.provider == "openai":
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=max_tokens,
                        )
                        return response.choices[0].message.content.strip()
                    except Exception as e:
                        if "max_completion_tokens" in str(
                            e
                        ) or "unsupported_parameter" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            return response.choices[0].message.content.strip()
                        else:
                            raise

                elif self.provider == "gemini":
                    model = self.client.GenerativeModel(self.model)
                    response = model.generate_content(
                        prompt,
                        generation_config=self.client.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        ),
                    )
                    return response.text.strip()

                elif self.provider in ["anthropic", "claude"]:
                    request_params = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if system:
                        request_params["system"] = system
                    response = self.client.messages.create(**request_params)
                    if not response.content:
                        if (
                            hasattr(response, "stop_reason")
                            and response.stop_reason == "refusal"
                        ):
                            raise ValueError(
                                f"Claude refused to respond. This may be due to content policy. Stop reason: {response.stop_reason}"
                            )
                        raise ValueError(
                            f"Empty response content from Claude API. Response: {response}"
                        )
                    if not hasattr(response.content[0], "text"):
                        raise ValueError(
                            f"Response content has no text attribute. Content type: {type(response.content[0])}, Content: {response.content[0]}"
                        )
                    return response.content[0].text.strip()

                else:
                    raise ValueError(
                        f"Query not implemented for provider: {self.provider}"
                    )

            except Exception as e:
                error_str = str(e)
                # Don't retry on refusals or permanent errors
                if "refused" in error_str.lower() or "refusal" in error_str.lower():
                    raise
                # Handle context-length 400 errors: reduce max_tokens without consuming an attempt
                is_context_length_error = (
                    "400" in error_str or "BadRequestError" in error_str
                ) and (
                    "context length" in error_str.lower()
                    or "context_length" in error_str.lower()
                    or "maximum context" in error_str.lower()
                    or "input_tokens" in error_str.lower()
                )
                if is_context_length_error:
                    if _truncated:
                        raise
                    _truncated = True
                    m_limit = _re.search(r"maximum context length is (\d+)", error_str)
                    m_input = _re.search(
                        r"prompt contains at least (\d+) input tokens", error_str
                    )
                    if m_limit and m_input:
                        model_limit = int(m_limit.group(1))
                        input_tokens = int(m_input.group(1))
                    else:
                        model_limit = self._max_model_len
                        input_tokens = len(prompt) // 4
                    target_input_tokens = max(256, model_limit - max_tokens)
                    scale = target_input_tokens / max(input_tokens, 1)
                    new_char_len = max(
                        200, min(int(len(prompt) * scale), len(prompt) - 1)
                    )
                    head_len = int(new_char_len * 0.5)
                    old_len = len(prompt)
                    prompt = (
                        prompt[:head_len]
                        + "\n...[truncated]...\n"
                        + prompt[-(new_char_len - head_len) :]
                    )
                    print(
                        f"Context length exceeded, truncating prompt {old_len} -> ~{len(prompt)} chars, retrying..."
                    )
                    continue
                # Don't retry on other 400/client errors (permanent)
                if "400" in error_str or "BadRequestError" in error_str:
                    raise
                # Check if it's a rate limit error that should be retried
                attempt += 1
                if attempt >= max_retries:
                    raise
                if "rate" in error_str.lower() or "429" in error_str:
                    wait_time = 2 ** (attempt - 1)
                    print(
                        f"Rate limit hit, retrying in {wait_time}s... (attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    wait_time = min(2 ** (attempt - 1), 10)
                    print(f"Error: {error_str}")
                    print(
                        f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)

        raise RuntimeError(f"Failed after {max_retries} retries")
