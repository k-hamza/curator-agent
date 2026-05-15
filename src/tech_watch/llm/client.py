"""
OpenAI-compatible LLM client.

Wraps the OpenAI-compatible API endpoint (Ollama or any other provider)
with two methods:
- complete() : plain text response
- complete_json() : structured JSON response with automatic retry

Designed to be provider-agnostic — works with Ollama, OpenRouter,
Together AI, or any OpenAI-compatible endpoint.

Usage:
    from tech_watch.llm.client import LLMClient
    from tech_watch.config.settings import Settings

    client = LLMClient.from_settings(settings)
    text = await client.complete("Summarise this article: ...")
    data = await client.complete_json("Extract topics as JSON: ...", schema=MyModel)
"""

import json
from typing import Any, Type, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError

from tech_watch.config.settings import Settings


T = TypeVar("T", bound=BaseModel)

# Default system prompt used when none is provided
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant specialised in technology news analysis. "
    "Be concise and accurate."
)


class LLMError(Exception):
    """Raised when the LLM API returns an error or an unparseable response."""
    pass


class LLMClient:
    """
    Async client for OpenAI-compatible LLM APIs.

    All requests use the /v1/chat/completions endpoint.
    Supports plain text and structured JSON responses.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
    ) -> None:
        """
        Args:
            base_url: Base URL of the OpenAI-compatible API
                      (e.g. 'http://localhost:11434/v1' for Ollama).
            model:    Model name to use for completions.
            timeout:  HTTP timeout in seconds. LLMs can be slow — 120s default.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout, connect=10.0)
        self._endpoint = f"{self._base_url}/chat/completions"

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        """
        Convenience constructor — build a client from application settings.

        Args:
            settings: Validated application settings.

        Returns:
            Configured LLMClient instance.
        """
        return cls(
            base_url=settings.agent.llm_base_url,
            model=settings.agent.model,
        )

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.3,
    ) -> str:
        """
        Send a prompt and return the LLM response as plain text.

        Args:
            prompt:        User message sent to the LLM.
            system_prompt: System message that sets the LLM behaviour.
            temperature:   Sampling temperature (0.0 = deterministic).
                           Lower values produce more consistent outputs,
                           which is preferable for filtering and summarisation.

        Returns:
            The LLM response text, stripped of leading/trailing whitespace.

        Raises:
            LLMError: If the API call fails or returns an unexpected response.
        """
        payload = self._build_payload(prompt, system_prompt, temperature)
        response_data = await self._post(payload)
        return self._extract_text(response_data)

    async def complete_json(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_retries: int = 2,
    ) -> T:
        """
        Send a prompt and return the response parsed into a Pydantic model.

        Uses JSON mode when available. Retries on parse failure by asking
        the LLM to correct its output.

        Args:
            prompt:       User message. Should explicitly request JSON output.
            schema:       Pydantic model class to parse the response into.
            system_prompt: System message.
            temperature:  Lower than complete() default — JSON needs consistency.
            max_retries:  Number of correction attempts on parse failure.

        Returns:
            An instance of the schema model populated with the LLM response.

        Raises:
            LLMError: If parsing fails after all retries.
        """
        # Append JSON instruction to system prompt
        json_system_prompt = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON only. "
            f"No markdown, no code blocks, no explanation — raw JSON only."
        )

        last_error: Exception | None = None
        last_text: str = ""

        for attempt in range(max_retries + 1):
            if attempt == 0:
                current_prompt = prompt
            else:
                # Ask the LLM to fix its previous invalid output
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response was not valid JSON:\n{last_text}\n"
                    f"Error: {last_error}\n"
                    f"Please respond with valid JSON only."
                )
                logger.debug(
                    f"JSON retry {attempt}/{max_retries} after parse error"
                )

            payload = self._build_payload(
                current_prompt, json_system_prompt, temperature
            )
            response_data = await self._post(payload)
            last_text = self._extract_text(response_data)

            try:
                parsed = self._parse_json(last_text, schema)
                return parsed
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    f"JSON parse attempt {attempt + 1}/{max_retries + 1} failed: "
                    f"{type(e).__name__}: {e}"
                )

        raise LLMError(
            f"Failed to parse LLM response as {schema.__name__} "
            f"after {max_retries + 1} attempts. "
            f"Last response: {last_text[:200]}"
        )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
    ) -> dict[str, Any]:
        """Build the request payload for /v1/chat/completions."""
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST to the completions endpoint and return the parsed JSON response.

        Raises:
            LLMError: On any HTTP or network error.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            raise LLMError(
                f"LLM request timed out after {self._timeout.read}s "
                f"(model: {self._model})"
            )
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"LLM API returned HTTP {e.response.status_code}: "
                f"{e.response.text[:200]}"
            )
        except httpx.RequestError as e:
            raise LLMError(
                f"Network error reaching LLM API at {self._base_url}: {e}"
            )

    def _extract_text(self, response_data: dict[str, Any]) -> str:
        """
        Extract the assistant message text from a chat completions response.

        Raises:
            LLMError: If the response structure is unexpected.
        """
        try:
            return response_data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(
                f"Unexpected LLM response structure: {e}\n"
                f"Response: {str(response_data)[:300]}"
            )

    def _parse_json(self, text: str, schema: Type[T]) -> T:
        """
        Parse a JSON string into a Pydantic model.

        Handles common LLM formatting issues:
        - Strips <think>...</think> blocks (qwen3 and other thinking models)
        - Strips markdown code fences (```json ... ```)
        - Strips leading/trailing whitespace
        """
        import re

        cleaned = text.strip()

        # Remove <think>...</think> blocks produced by thinking models (e.g. qwen3)
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            cleaned = "\n".join(lines[1:-1]).strip()

        data = json.loads(cleaned)
        return schema.model_validate(data)
