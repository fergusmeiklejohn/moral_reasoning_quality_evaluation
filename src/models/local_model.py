"""
Local model provider implementation.

Supports models running via vLLM, Ollama, or custom inference endpoints.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import time
import requests

from .base import BaseLLMProvider
from ..data.schemas import ModelResponse


class LocalVLLMProvider(BaseLLMProvider):
    """
    Provider for models running via vLLM inference server.

    Assumes vLLM is running with OpenAI-compatible API at specified endpoint.
    """

    def _initialize(self) -> None:
        """Initialize connection to vLLM server."""
        self.endpoint = self.config.get("endpoint", "http://localhost:8000/v1/completions")
        self.base_url = self.config.get("base_url", "http://localhost:8000")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
        seed: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response via vLLM server."""
        start_time = time.time()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if seed is not None:
            payload["seed"] = seed

        payload.update(kwargs)

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            response_time = time.time() - start_time

            data = response.json()
            raw_text = data["choices"][0]["text"]
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            tokens_used = usage.get("total_tokens")
            finish_reason = data["choices"][0].get("finish_reason", "unknown")

            parsed_choice, reasoning = self._parse_response(raw_text)

            return ModelResponse(
                raw_text=raw_text,
                parsed_choice=parsed_choice,
                reasoning=reasoning,
                timestamp=datetime.utcnow(),
                response_time_seconds=response_time,
                tokens_used=tokens_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason
            )

        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"vLLM server error: {str(e)}"

            return ModelResponse(
                raw_text=error_msg,
                parsed_choice="ERROR",
                reasoning=error_msg,
                timestamp=datetime.utcnow(),
                response_time_seconds=response_time,
                tokens_used=0,
                finish_reason="error"
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM model information."""
        return {
            "provider": "vllm",
            "model_name": self.model_name,
            "endpoint": self.endpoint,
            "supports_seed": True,
            "supports_temperature": True
        }


class OllamaProvider(BaseLLMProvider):
    """Provider for models running via Ollama."""

    def _initialize(self) -> None:
        """Initialize connection to Ollama."""
        self.endpoint = self.config.get("endpoint", "http://localhost:11434/api/generate")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
        seed: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response via Ollama."""
        start_time = time.time()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }

        if seed is not None:
            payload["options"]["seed"] = seed

        # Add any additional options
        if kwargs:
            payload["options"].update(kwargs)

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            response_time = time.time() - start_time

            data = response.json()
            raw_text = data["response"]
            output_tokens = data.get("eval_count")
            input_tokens = data.get("prompt_eval_count")
            if output_tokens is not None or input_tokens is not None:
                tokens_used = (output_tokens or 0) + (input_tokens or 0)
            else:
                tokens_used = None

            parsed_choice, reasoning = self._parse_response(raw_text)

            return ModelResponse(
                raw_text=raw_text,
                parsed_choice=parsed_choice,
                reasoning=reasoning,
                timestamp=datetime.utcnow(),
                response_time_seconds=response_time,
                tokens_used=tokens_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason="stop"
            )

        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Ollama error: {str(e)}"

            return ModelResponse(
                raw_text=error_msg,
                parsed_choice="ERROR",
                reasoning=error_msg,
                timestamp=datetime.utcnow(),
                response_time_seconds=response_time,
                tokens_used=0,
                finish_reason="error"
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        return {
            "provider": "ollama",
            "model_name": self.model_name,
            "endpoint": self.endpoint,
            "supports_seed": True,
            "supports_temperature": True
        }
