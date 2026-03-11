import json
import argparse
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
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
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        # Extract provider and model from config
        self.provider = config.get('provider', '').lower()
        self.model = config.get('model', '')
        self.config = config

        # For vllm server type, override provider
        if server_type == "vllm":
            self.provider = "vllm"
            # Use host/port from kwargs or config (support both 'host' and 'vllm_host' naming)
            kwargs.setdefault('host', config.get('vllm_host') or config.get('host', 'localhost'))
            kwargs.setdefault('port', config.get('vllm_port') or config.get('port', 8000))

        self.client = self._initialize_client(**kwargs)

    def _initialize_client(self, **kwargs):
        """Initialize provider-specific client."""
        if self.provider == "vllm":
            from openai import OpenAI
            base_url = f"http://{kwargs.get('host', 'localhost')}:{kwargs.get('port', 8000)}/v1"
            return OpenAI(base_url=base_url, api_key="EMPTY")

        elif self.provider == "openai":
            from openai import OpenAI
            api_key = self.config.get("api_key", "")
            return OpenAI(api_key=api_key)

        elif self.provider == "deepseek":
            from openai import OpenAI
            api_key = self.config.get("api_key", "")
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = self.config.get("api_key", "")
            genai.configure(api_key=api_key)
            return genai

        elif self.provider in ["anthropic", "claude"]:
            from anthropic import Anthropic
            api_key = self.config.get("api_key", "")
            return Anthropic(api_key=api_key)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def query(self, prompt: str, temperature: float = 0.0, max_tokens: int = 4096, max_retries: int = 5) -> str:
        """Query model with prompt with retry logic for rate limits."""
        for attempt in range(max_retries):
            try:
                if self.provider in ["vllm", "deepseek"]:
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
                        if "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
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
                        )
                    )
                    return response.text.strip()

                elif self.provider in ["anthropic", "claude"]:
                    response = self.client.messages.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.content[0].text.strip()

                else:
                    raise ValueError(f"Query not implemented for provider: {self.provider}")
                    
            except Exception as e:
                error_str = str(e)
                print(f"Error: {error_str}")
        
        raise RuntimeError(f"Failed after {max_retries} retries")

