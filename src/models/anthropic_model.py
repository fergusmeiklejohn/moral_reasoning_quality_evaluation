"""
Anthropic model provider implementation.

Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, and other Anthropic models.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import time

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import BaseLLMProvider
from ..data.schemas import ModelResponse


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models."""

    def _initialize(self) -> None:
        """Initialize Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

        self.client = Anthropic(api_key=self.api_key)

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
        Generate a response using Anthropic API.

        Note: Anthropic does not currently support random seeds for reproducibility.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed (not supported by Anthropic, ignored)
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            ModelResponse object
        """
        start_time = time.time()

        cache_control = kwargs.pop("cache_control", None)
        # Build API call parameters
        user_content = [{"type": "text", "text": prompt}]
        if cache_control:
            user_content[0]["cache_control"] = cache_control

        api_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_tokens,
        }

        # Anthropic doesn't allow both temperature and top_p
        # Only set one - prefer temperature, use top_p only if explicitly non-default
        if top_p != 1.0:
            api_params["top_p"] = top_p
        else:
            api_params["temperature"] = temperature

        # Add any additional parameters
        api_params.update(kwargs)

        # Note: Anthropic doesn't support seed parameter
        if seed is not None:
            print(f"Warning: Random seed {seed} specified but not supported by Anthropic")

        # Make API call
        try:
            message = self.client.messages.create(**api_params)
            response_time = time.time() - start_time

            # Extract response
            raw_text = message.content[0].text
            input_tokens = getattr(message.usage, "input_tokens", None)
            output_tokens = getattr(message.usage, "output_tokens", None)
            tokens_used = None
            if input_tokens is not None or output_tokens is not None:
                tokens_used = (input_tokens or 0) + (output_tokens or 0)
            finish_reason = message.stop_reason

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
            error_msg = f"Anthropic API error: {str(e)}"

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
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            ModelResponse object containing the response
        """
        start_time = time.time()

        # Separate system message from conversation messages
        system_content = None
        api_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Anthropic handles system prompt separately
                system_content = content
            else:
                # Convert content to Anthropic format
                api_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })

        # Build API call parameters
        api_params = {
            "model": self.model_name,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }

        # Anthropic doesn't allow both temperature and top_p
        # Only set one - prefer temperature, use top_p only if explicitly non-default
        if top_p != 1.0:
            api_params["top_p"] = top_p
        else:
            api_params["temperature"] = temperature

        # Add system prompt if present
        if system_content:
            api_params["system"] = system_content

        # Add any additional parameters
        api_params.update(kwargs)

        # Make API call
        try:
            message = self.client.messages.create(**api_params)
            response_time = time.time() - start_time

            # Extract response
            raw_text = message.content[0].text
            input_tokens = getattr(message.usage, "input_tokens", None)
            output_tokens = getattr(message.usage, "output_tokens", None)
            tokens_used = None
            if input_tokens is not None or output_tokens is not None:
                tokens_used = (input_tokens or 0) + (output_tokens or 0)
            finish_reason = message.stop_reason

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
            error_msg = f"Anthropic API error: {str(e)}"

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
        """Get Anthropic model information."""
        return {
            "provider": "anthropic",
            "model_name": self.model_name,
            "supports_seed": False,  # Anthropic doesn't support seeds
            "supports_temperature": True,
            "supports_conversation": True,
            "note": "Claude models do not support random seeds for reproducibility"
        }
