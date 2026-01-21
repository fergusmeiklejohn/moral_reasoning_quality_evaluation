"""
Google model provider implementation.

Supports Gemini 1.5 Pro, Gemini 1.5 Flash, and other Google models.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import time

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from .base import BaseLLMProvider
from ..data.schemas import ModelResponse


class GoogleProvider(BaseLLMProvider):
    """Provider for Google Gemini models."""

    def _initialize(self) -> None:
        """Initialize Google Generative AI client."""
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "Google Generative AI library not installed. "
                "Install with: pip install google-generativeai"
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

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
        Generate a response using Google Gemini API.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed (not supported by Google, ignored)
            **kwargs: Additional Google-specific parameters

        Returns:
            ModelResponse object
        """
        start_time = time.time()

        # Build generation config
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }

        # Add any additional parameters
        generation_config.update(kwargs)

        # Note: Google doesn't support seed parameter
        if seed is not None:
            print(f"Warning: Random seed {seed} specified but not supported by Google")

        # Make API call
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            response_time = time.time() - start_time

            # Extract response safely - response.text throws if blocked/empty
            raw_text = self._extract_text_safely(response)
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"

            # Token accounting (best effort from available metadata)
            usage = getattr(response, "usage_metadata", None)
            tokens_used = getattr(usage, "total_token_count", None) if usage else None
            input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
            # Candidates token count is a reasonable proxy for output tokens
            output_tokens = getattr(usage, "candidates_token_count", None) if usage else None

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
            error_msg = f"Google API error: {str(e)}"

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
            **kwargs: Additional Google-specific parameters

        Returns:
            ModelResponse object containing the response
        """
        start_time = time.time()

        # Build generation config
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        generation_config.update(kwargs)

        # Convert messages to Gemini format
        # Gemini uses 'user' and 'model' roles, and handles system via system_instruction
        history = []
        system_instruction = None

        for msg in messages[:-1]:  # All but the last message go into history
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})
            else:
                history.append({"role": "user", "parts": [content]})

        # Get the last message as the current input
        last_msg = messages[-1] if messages else {"role": "user", "content": ""}
        current_content = last_msg.get("content", "")

        # Make API call
        try:
            # Create a chat session with history
            if system_instruction:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
            else:
                model = self.model

            chat = model.start_chat(history=history)
            response = chat.send_message(
                current_content,
                generation_config=generation_config
            )
            response_time = time.time() - start_time

            # Extract response safely - response.text throws if blocked/empty
            raw_text = self._extract_text_safely(response)
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"

            # Token accounting
            usage = getattr(response, "usage_metadata", None)
            tokens_used = getattr(usage, "total_token_count", None) if usage else None
            input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
            output_tokens = getattr(usage, "candidates_token_count", None) if usage else None

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
            error_msg = f"Google API error: {str(e)}"

            return ModelResponse(
                raw_text=error_msg,
                parsed_choice="ERROR",
                reasoning=error_msg,
                timestamp=datetime.utcnow(),
                response_time_seconds=response_time,
                tokens_used=0,
                finish_reason="error"
            )

    def _extract_text_safely(self, response) -> str:
        """
        Safely extract text from a Google Gemini response.

        The response.text accessor throws ValueError if the response is blocked
        or has no valid parts. This method handles those cases gracefully.

        Args:
            response: The GenerateContentResponse object

        Returns:
            The extracted text, or an error message if extraction fails
        """
        try:
            # First try the quick accessor
            return response.text
        except ValueError:
            # Response may be blocked or empty - check candidates manually
            if not response.candidates:
                # Check for prompt feedback (content filtering)
                if hasattr(response, 'prompt_feedback'):
                    block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                    if block_reason:
                        return f"[Response blocked: {block_reason}]"
                return "[No response candidates returned]"

            # Try to extract from first candidate's parts
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts:
                    # Concatenate all text parts
                    texts = []
                    for part in parts:
                        if hasattr(part, 'text'):
                            texts.append(part.text)
                    if texts:
                        return "".join(texts)

            # Check finish reason for more context
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason:
                return f"[Response incomplete: {finish_reason.name}]"

            return "[Unable to extract response text]"

    def get_model_info(self) -> Dict[str, Any]:
        """Get Google model information."""
        return {
            "provider": "google",
            "model_name": self.model_name,
            "supports_seed": False,
            "supports_temperature": True,
            "supports_conversation": True,
            "note": "Google Gemini models do not support random seeds for reproducibility"
        }
