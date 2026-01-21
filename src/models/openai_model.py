"""
OpenAI model provider implementation.

Supports GPT-4, GPT-4o, GPT-3.5-turbo, and other OpenAI models.
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


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models."""

    def _initialize(self) -> None:
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        self.client = OpenAI(api_key=self.api_key)

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
        Generate a response using OpenAI API.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            ModelResponse object
        """
        start_time = time.time()

        # Build API call parameters
        # Use max_completion_tokens for newer models (gpt-4o, gpt-5.x, o1, etc.)
        api_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_completion_tokens": max_tokens,
        }

        # Add seed if provided (supported in newer models)
        if seed is not None:
            api_params["seed"] = seed

        # Add any additional parameters
        api_params.update(kwargs)

        # Make API call
        try:
            completion = self.client.chat.completions.create(**api_params)
            response_time = time.time() - start_time

            # Extract response - handle None content (can happen with refusals or reasoning models)
            message = completion.choices[0].message
            raw_text = message.content

            # Check for refusal or empty content (newer models may refuse via refusal field)
            # Also check for empty string - reasoning models may use all tokens for thinking
            if raw_text is None or raw_text == "":
                refusal = getattr(message, 'refusal', None)
                if refusal:
                    raw_text = f"[Model refused: {refusal}]"
                else:
                    # Content might be empty - check finish reason and token details
                    finish = completion.choices[0].finish_reason
                    # Check if reasoning tokens exhausted the budget (GPT-5.x, o1, etc.)
                    reasoning_tokens = 0
                    if hasattr(completion.usage, 'completion_tokens_details'):
                        details = completion.usage.completion_tokens_details
                        reasoning_tokens = getattr(details, 'reasoning_tokens', 0) or 0

                    if finish == "length" and reasoning_tokens > 0:
                        raw_text = f"[Reasoning tokens ({reasoning_tokens}) exhausted budget - increase max_tokens]"
                    elif finish == "content_filter":
                        raw_text = "[Response filtered by content policy]"
                    elif finish == "length":
                        raw_text = "[Response truncated - hit max tokens]"
                    else:
                        raw_text = f"[No content returned - finish_reason: {finish}]"
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
            error_msg = f"OpenAI API error: {str(e)}"

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
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles are 'user', 'assistant', or 'system'.
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            ModelResponse object containing the response
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
        # Use max_completion_tokens for newer models (gpt-4o, gpt-5.x, o1, etc.)
        api_params = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_completion_tokens": max_tokens,
        }

        # Add any additional parameters
        api_params.update(kwargs)

        # Make API call
        try:
            completion = self.client.chat.completions.create(**api_params)
            response_time = time.time() - start_time

            # Extract response - handle None content (can happen with refusals or reasoning models)
            message = completion.choices[0].message
            raw_text = message.content

            # Check for refusal or empty content (newer models may refuse via refusal field)
            # Also check for empty string - reasoning models may use all tokens for thinking
            if raw_text is None or raw_text == "":
                refusal = getattr(message, 'refusal', None)
                if refusal:
                    raw_text = f"[Model refused: {refusal}]"
                else:
                    # Content might be empty - check finish reason and token details
                    finish = completion.choices[0].finish_reason
                    # Check if reasoning tokens exhausted the budget (GPT-5.x, o1, etc.)
                    reasoning_tokens = 0
                    if hasattr(completion.usage, 'completion_tokens_details'):
                        details = completion.usage.completion_tokens_details
                        reasoning_tokens = getattr(details, 'reasoning_tokens', 0) or 0

                    if finish == "length" and reasoning_tokens > 0:
                        raw_text = f"[Reasoning tokens ({reasoning_tokens}) exhausted budget - increase max_tokens]"
                    elif finish == "content_filter":
                        raw_text = "[Response filtered by content policy]"
                    elif finish == "length":
                        raw_text = "[Response truncated - hit max tokens]"
                    else:
                        raw_text = f"[No content returned - finish_reason: {finish}]"
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
            error_msg = f"OpenAI API error: {str(e)}"

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
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "supports_seed": True,
            "supports_temperature": True,
            "supports_conversation": True,
            "api_base": self.client.base_url if hasattr(self.client, 'base_url') else None
        }
