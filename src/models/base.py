"""
Base model interface for LLM abstraction.

This module defines the abstract interface that all LLM providers must implement,
allowing seamless swapping of models in experiments.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

from ..data.schemas import ModelResponse, Choice


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLM provider.

        Args:
            model_name: Name/identifier of the model
            api_key: API key for authentication (if needed)
            **kwargs: Additional provider-specific configuration
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the provider (setup client, validate credentials, etc.)."""
        pass

    @abstractmethod
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
        Generate a response from the model.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility (if supported)
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse object containing the parsed response

        Raises:
            Exception: If generation fails after retries
        """
        pass

    def _parse_response(self, raw_text: str) -> tuple[Choice, str]:
        """
        Parse the raw response to extract choice and reasoning.

        Args:
            raw_text: Raw text response from the model

        Returns:
            Tuple of (parsed_choice, reasoning_text)
        """
        lines = raw_text.strip().split('\n')

        # Try to find the choice in the first few lines
        choice = Choice.ERROR
        reasoning_lines = []
        choice_found = False

        for line in lines:
            line_upper = line.strip().upper()

            if not choice_found:
                if "CHOICE A" in line_upper or line_upper == "A":
                    choice = Choice.A
                    choice_found = True
                    continue
                elif "CHOICE B" in line_upper or line_upper == "B":
                    choice = Choice.B
                    choice_found = True
                    continue
                elif any(word in line_upper for word in ["REFUSE", "CANNOT", "DECLINE", "UNABLE"]):
                    choice = Choice.REFUSE
                    choice_found = True
                    continue

            # After finding choice, collect reasoning
            if choice_found and line.strip():
                reasoning_lines.append(line.strip())

        reasoning = " ".join(reasoning_lines) if reasoning_lines else raw_text

        return choice, reasoning

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model metadata (name, version, provider, etc.)
        """
        pass

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

        This is used for V2 experiments where we need to maintain
        conversation context across multiple turns (initial + probes).

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles are 'user', 'assistant', or 'system'.
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse object containing the response

        Note:
            Default implementation converts to single prompt. Subclasses
            should override for proper multi-turn support.
        """
        # Default implementation: concatenate messages into a single prompt
        # Subclasses should override this for proper multi-turn support
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[System]: {content}")
            elif role == "assistant":
                prompt_parts.append(f"[Assistant]: {content}")
            else:
                prompt_parts.append(f"[User]: {content}")

        combined_prompt = "\n\n".join(prompt_parts)
        combined_prompt += "\n\n[Assistant]:"

        return self.generate(
            prompt=combined_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )

    def validate_connection(self) -> bool:
        """
        Validate that the provider is properly configured and can connect.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Simple test prompt
            test_prompt = "Respond with: OK"
            response = self.generate(
                test_prompt,
                temperature=0.0,
                max_tokens=10
            )
            # Consider the connection valid if the call succeeded without an error
            # finish_reason. Some providers may not return a parsable Choice for
            # this generic prompt, so we treat any non-error completion as success.
            return response.finish_reason != "error"
        except Exception as e:
            print(f"Connection validation failed: {e}")
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"


class MockLLMProvider(BaseLLMProvider):
    """Mock provider for testing purposes."""

    def _initialize(self) -> None:
        """No initialization needed for mock."""
        pass

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
        seed: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a mock response."""
        import random
        if seed is not None:
            random.seed(seed)

        start_time = time.time()

        # Simulate random choice
        choice = random.choice([Choice.A, Choice.B])
        reasoning = "This is a mock response for testing purposes."

        response_time = time.time() - start_time

        return ModelResponse(
            raw_text=f"{choice.value}\n{reasoning}",
            parsed_choice=choice,
            reasoning=reasoning,
            timestamp=datetime.utcnow(),
            response_time_seconds=response_time,
            tokens_used=50,
            input_tokens=20,
            output_tokens=30,
            finish_reason="stop"
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model info."""
        return {
            "provider": "mock",
            "model_name": self.model_name,
            "version": "1.0.0",
            "supports_seed": True,
            "supports_temperature": True
        }
