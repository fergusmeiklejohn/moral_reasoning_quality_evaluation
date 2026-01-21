"""
Grok model provider implementation via xAI API.

Supports Grok models using the OpenAI-compatible xAI API.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import time

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import BaseLLMProvider
from ..data.schemas import ModelResponse


class GrokProvider(BaseLLMProvider):
    """Provider for Grok models via xAI API."""

    XAI_BASE_URL = "https://api.x.ai/v1"

    def _initialize(self) -> None:
        """Initialize xAI client (OpenAI-compatible)."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library required for Grok. Install with: pip install openai"
            )

        base_url = self.config.get("base_url", self.XAI_BASE_URL)
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
        seed: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a response using xAI API.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            **kwargs: Additional parameters

        Returns:
            ModelResponse object
        """
        start_time = time.time()

        # Build API call parameters
        api_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # Add seed if provided
        if seed is not None:
            api_params["seed"] = seed

        # Add any additional parameters
        api_params.update(kwargs)

        # Make API call
        try:
            completion = self.client.chat.completions.create(**api_params)
            response_time = time.time() - start_time

            # Extract response
            raw_text = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens if completion.usage else None
            input_tokens = (
                completion.usage.prompt_tokens if completion.usage else None
            )
            output_tokens = (
                completion.usage.completion_tokens if completion.usage else None
            )
            finish_reason = completion.choices[0].finish_reason

            # Parse choice and reasoning
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
            error_msg = f"Grok API error: {str(e)}"

            return ModelResponse(
                raw_text=error_msg,
                parsed_choice="ERROR",
                reasoning=error_msg,
                timestamp=datetime.utcnow(),
                response_time_seconds=response_time,
                tokens_used=0,
                finish_reason="error"
            )

    def generate_conversation(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a response in a multi-turn conversation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            ModelResponse object
        """
        start_time = time.time()

        # Convert messages to OpenAI format (already compatible)
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        # Build API call parameters
        api_params = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # Add any additional parameters
        api_params.update(kwargs)

        # Make API call
        try:
            completion = self.client.chat.completions.create(**api_params)
            response_time = time.time() - start_time

            # Extract response
            raw_text = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens if completion.usage else None
            input_tokens = (
                completion.usage.prompt_tokens if completion.usage else None
            )
            output_tokens = (
                completion.usage.completion_tokens if completion.usage else None
            )
            finish_reason = completion.choices[0].finish_reason

            # Parse choice and reasoning
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
            error_msg = f"Grok API error: {str(e)}"

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
        """Get Grok model information."""
        return {
            "provider": "grok",
            "model_name": self.model_name,
            "supports_seed": True,
            "supports_temperature": True,
            "supports_conversation": True,
            "api_base": self.XAI_BASE_URL
        }
