"""
Anthropic LLM Provider — Claude model family.

Features:
    - Claude 3.5 Sonnet / Claude 3 Opus / Claude 3 Haiku support
    - Proper system message handling (Anthropic uses a dedicated field)
    - Streaming support
    - Automatic retry with exponential backoff
    - Rate limiting and cost tracking

Reference:
    - Anthropic API: https://docs.anthropic.com/en/docs/
    - Claude models: https://docs.anthropic.com/en/docs/about-claude/models
"""

import time
import logging
from typing import Any, Dict, Iterator, List, Optional

from .base import (
    RetryConfig,
    RateLimitConfig,
    UsageStats,
    TokenBucketRateLimiter,
    estimate_cost,
    estimate_tokens,
    retry_on_error,
)

logger = logging.getLogger(__name__)

# Default max tokens for Claude models (required by the API)
_DEFAULT_MAX_TOKENS = 4096


class AnthropicLLM:
    """
    Anthropic Claude LLM provider.

    Claude's API differs from OpenAI in a few important ways:

    1. ``system`` is a top-level parameter, not a message role.
    2. ``max_tokens`` is **required** (not optional).
    3. The response format wraps content in a ``content`` list.

    Args:
        model_name: Claude model name (default: ``"claude-3-5-sonnet-20241022"``).
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
        max_tokens: Default max completion tokens (default: 4096).
        temperature: Default temperature (default: 0.7).
        timeout: Request timeout in seconds (default: 120).
        max_retries: Retry attempts for transient errors (default: 3).
        rate_limit: ``RateLimitConfig`` instance or dict.
        **kwargs: Additional configuration.

    Example::

        from llm_assessment.providers.anthropic_provider import AnthropicLLM

        llm = AnthropicLLM(api_key="sk-ant-...")
        print(llm.generate("Explain quantum computing in one sentence."))
    """

    def __init__(self, model_name="claude-3-5-sonnet-20241022", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # Backward-compat
        self.call_count = 0
        self.total_tokens = 0

        # Usage tracking
        self._usage = UsageStats()

        # Default params
        self._max_tokens = kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS)
        self._temperature = kwargs.pop("temperature", 0.7)
        self._timeout = kwargs.pop("timeout", 120)

        # Retry
        self._retry_config = RetryConfig(
            max_retries=kwargs.pop("max_retries", 3)
        )

        # Rate limiter
        rl_cfg = kwargs.pop("rate_limit", None)
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # ---- Create Anthropic client (lazy import) ----
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package.\n"
                "Install it with: pip install anthropic>=0.18.0"
            )

        api_key = kwargs.pop("api_key", None)
        base_url = kwargs.pop("base_url", None)

        client_kwargs = {"timeout": self._timeout, "max_retries": 0}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = anthropic.Anthropic(**client_kwargs)

        # BaseLLM init
        try:
            from ..core.llm_wrapper import BaseLLM
            if isinstance(self, BaseLLM):
                BaseLLM.__init__(self, model_name, **kwargs)
        except (ImportError, TypeError):
            pass

    # ------------------------------------------------------------------ #
    #  Core generation
    # ------------------------------------------------------------------ #

    @retry_on_error
    def generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        """Generate a response from a single prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self._create_message(messages, **kwargs)

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """
        Multi-turn chat completion.

        System messages in ``messages`` are extracted and passed via
        Anthropic's ``system`` parameter.  The remaining messages must
        alternate between ``user`` and ``assistant`` roles.
        """
        return self._create_message(messages, **kwargs)

    # ------------------------------------------------------------------ #
    #  Streaming
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """Stream a response from a single prompt."""
        messages = [{"role": "user", "content": prompt}]
        for chunk in self._stream_message(messages, **kwargs):
            yield chunk

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """Stream a multi-turn chat completion."""
        for chunk in self._stream_message(messages, **kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    #  Batch
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=4, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.generate, p, **kwargs): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("batch_generate error at index %d: %s", idx, e)
                    results[idx] = "[Error: {}]".format(e)
        return results

    # ------------------------------------------------------------------ #
    #  Health check
    # ------------------------------------------------------------------ #

    def health_check(self):
        # type: () -> Dict[str, Any]
        start = time.time()
        try:
            resp = self.generate("Say 'ok'.", max_tokens=10)
            latency = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "response_preview": resp[:50],
            }
        except Exception as e:
            latency = (time.time() - start) * 1000
            return {
                "status": "unhealthy",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "error": str(e),
            }

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def get_stats(self):
        # type: () -> Dict[str, Any]
        stats = self._usage.to_dict()
        stats["model_name"] = self.model_name
        stats["call_count"] = self._usage.total_calls
        stats["total_tokens"] = self._usage.total_tokens
        return stats

    def __repr__(self):
        return "AnthropicLLM(model='{}')".format(self.model_name)

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_system_message(messages):
        # type: (List[Dict[str, str]]) -> tuple
        """
        Separate system messages from conversation messages.

        Returns:
            (system_text, filtered_messages)
        """
        system_parts = []
        filtered = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_parts.append(msg.get("content", ""))
            else:
                filtered.append(msg)

        system_text = "\n\n".join(system_parts) if system_parts else None

        # Anthropic requires messages to start with a "user" message
        # and alternate roles.  If the first message is "assistant",
        # prepend a placeholder user message.
        if filtered and filtered[0].get("role") == "assistant":
            filtered.insert(0, {"role": "user", "content": "(continue)"})

        return system_text, filtered

    def _create_message(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """Non-streaming message creation."""
        self._rate_limiter.acquire(estimate_tokens(str(messages)))

        system_text, filtered_messages = self._extract_system_message(messages)
        request_kwargs = self._build_request_kwargs(kwargs)

        if system_text:
            request_kwargs["system"] = system_text

        start = time.time()
        response = self.client.messages.create(
            model=self.model_name,
            messages=filtered_messages,
            **request_kwargs,
        )
        latency_ms = (time.time() - start) * 1000

        # Extract usage
        usage = response.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        # Extract text content
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return content

    def _stream_message(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """Streaming message creation."""
        self._rate_limiter.acquire(estimate_tokens(str(messages)))

        system_text, filtered_messages = self._extract_system_message(messages)
        request_kwargs = self._build_request_kwargs(kwargs)

        if system_text:
            request_kwargs["system"] = system_text

        start = time.time()
        collected = []
        prompt_tokens = 0
        completion_tokens = 0

        with self.client.messages.stream(
            model=self.model_name,
            messages=filtered_messages,
            **request_kwargs,
        ) as stream:
            for text in stream.text_stream:
                collected.append(text)
                yield text

            # Get final usage from the stream's final message
            final_message = stream.get_final_message()
            if final_message and final_message.usage:
                prompt_tokens = final_message.usage.input_tokens
                completion_tokens = final_message.usage.output_tokens

        latency_ms = (time.time() - start) * 1000

        if not prompt_tokens:
            prompt_tokens = estimate_tokens(str(messages))
            completion_tokens = estimate_tokens("".join(collected))

        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

    def _build_request_kwargs(self, overrides):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        kwargs = {}

        kwargs["max_tokens"] = overrides.pop("max_tokens", self._max_tokens)

        temperature = overrides.pop("temperature", self._temperature)
        kwargs["temperature"] = temperature

        # Pass through safe keys
        safe_keys = {"top_p", "top_k", "stop_sequences", "metadata"}
        for k in safe_keys:
            if k in overrides:
                kwargs[k] = overrides.pop(k)

        return kwargs

    def _track_usage(self, prompt_tokens, completion_tokens, latency_ms):
        # type: (int, int, float) -> None
        self._usage.total_calls += 1
        self._usage.total_prompt_tokens += prompt_tokens
        self._usage.total_completion_tokens += completion_tokens
        self._usage.total_tokens += prompt_tokens + completion_tokens
        self._usage.total_latency_ms += latency_ms
        self._usage.total_cost_usd += estimate_cost(
            self.model_name, prompt_tokens, completion_tokens
        )
        self.call_count = self._usage.total_calls
        self.total_tokens = self._usage.total_tokens
