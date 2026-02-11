"""
OpenAI LLM Provider — Supports OpenAI API and Azure OpenAI Service.

Features:
    - GPT-4o / GPT-4 / GPT-3.5 / o1 / o3-mini model families
    - Azure OpenAI Service with deployment-based routing
    - Streaming support
    - Automatic retry with exponential backoff
    - Rate limiting (token bucket)
    - Detailed usage tracking and cost estimation

Reference:
    - OpenAI API docs: https://platform.openai.com/docs/api-reference
    - Azure OpenAI: https://learn.microsoft.com/en-us/azure/ai-services/openai/
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


class OpenAILLM:
    """
    OpenAI LLM provider (ChatCompletion API).

    Supports all chat-based OpenAI models including GPT-4o, o1, o3-mini.

    Args:
        model_name: Model identifier (default: ``"gpt-4o-mini"``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Override the API base URL (useful for proxies).
        organization: OpenAI organization ID.
        timeout: Request timeout in seconds (default: 120).
        max_retries: Maximum retry attempts for transient errors (default: 3).
        temperature: Default sampling temperature (default: 0.7).
        max_tokens: Default max tokens for generation.
        rate_limit: ``RateLimitConfig`` instance or dict.
        **kwargs: Additional configuration passed to the client.

    Example::

        from llm_assessment.providers.openai_provider import OpenAILLM

        llm = OpenAILLM(model_name="gpt-4o-mini", api_key="sk-...")
        print(llm.generate("Hello!"))
    """

    def __init__(self, model_name="gpt-4o-mini", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # Backward-compat
        self.call_count = 0
        self.total_tokens = 0

        # Usage tracking
        self._usage = UsageStats()

        # Default generation params
        self._temperature = kwargs.pop("temperature", 0.7)
        self._max_tokens = kwargs.pop("max_tokens", None)
        self._timeout = kwargs.pop("timeout", 120)

        # Retry config
        max_retries_sdk = kwargs.pop("max_retries", 3)
        self._retry_config = RetryConfig(max_retries=max_retries_sdk)

        # Rate limiter
        rl_cfg = kwargs.pop("rate_limit", None)
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # ---- Create OpenAI client (lazy import) ----
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the 'openai' package.\n"
                "Install it with: pip install openai>=1.0.0"
            )

        api_key = kwargs.pop("api_key", None)
        base_url = kwargs.pop("base_url", None)
        organization = kwargs.pop("organization", None)

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        client_kwargs["timeout"] = self._timeout
        # Let the SDK handle its own retries as a last-resort fallback
        client_kwargs["max_retries"] = 0

        self.client = openai.OpenAI(**client_kwargs)

        # Initialize BaseLLM if used as subclass
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
        return self._chat_completion(messages, **kwargs)

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """Multi-turn chat completion."""
        return self._chat_completion(messages, **kwargs)

    # ------------------------------------------------------------------ #
    #  Streaming
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """Stream a response from a single prompt, yielding text chunks."""
        messages = [{"role": "user", "content": prompt}]
        for chunk in self._stream_chat_completion(messages, **kwargs):
            yield chunk

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """Stream a multi-turn chat completion."""
        for chunk in self._stream_chat_completion(messages, **kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    #  Batch
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=4, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """Generate responses for multiple prompts using thread pool."""
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
        """Verify the OpenAI API is reachable."""
        start = time.time()
        try:
            resp = self.generate("Say 'ok'.", max_tokens=5)
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
        # Backward compat
        stats["call_count"] = self._usage.total_calls
        stats["total_tokens"] = self._usage.total_tokens
        return stats

    def __repr__(self):
        return "OpenAILLM(model='{}')".format(self.model_name)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _chat_completion(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """Perform a non-streaming ChatCompletion request."""
        self._rate_limiter.acquire(estimate_tokens(str(messages)))

        request_kwargs = self._build_request_kwargs(kwargs)
        request_kwargs["stream"] = False

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **request_kwargs,
        )
        latency_ms = (time.time() - start) * 1000

        # Extract usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        content = response.choices[0].message.content or ""
        return content

    def _stream_chat_completion(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """Perform a streaming ChatCompletion request."""
        self._rate_limiter.acquire(estimate_tokens(str(messages)))

        request_kwargs = self._build_request_kwargs(kwargs)
        request_kwargs["stream"] = True
        request_kwargs["stream_options"] = {"include_usage": True}

        start = time.time()
        collected_content = []
        prompt_tokens = 0
        completion_tokens = 0

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **request_kwargs,
        )

        for chunk in stream:
            # Usage info comes in the last chunk
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    collected_content.append(delta.content)
                    yield delta.content

        latency_ms = (time.time() - start) * 1000

        # Fall back to estimation if usage not provided
        if not prompt_tokens:
            full_text = "".join(collected_content)
            prompt_tokens = estimate_tokens(str(messages))
            completion_tokens = estimate_tokens(full_text)

        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

    def _build_request_kwargs(self, overrides):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        """Merge default params with per-call overrides."""
        kwargs = {}

        temperature = overrides.pop("temperature", self._temperature)
        max_tokens = overrides.pop("max_tokens", self._max_tokens)

        # o1 / o3 models don't support temperature
        model_lower = self.model_name.lower()
        if not any(model_lower.startswith(p) for p in ("o1", "o3")):
            kwargs["temperature"] = temperature

        if max_tokens is not None:
            # o1/o3 models use 'max_completion_tokens' instead of 'max_tokens'
            if any(model_lower.startswith(p) for p in ("o1", "o3")):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        # Pass through remaining kwargs (top_p, stop, etc.)
        # Filter out our internal keys
        internal_keys = {
            "api_key", "base_url", "organization", "timeout",
            "rate_limit", "retry", "max_retries",
        }
        for k, v in overrides.items():
            if k not in internal_keys:
                kwargs[k] = v

        return kwargs

    def _track_usage(self, prompt_tokens, completion_tokens, latency_ms):
        # type: (int, int, float) -> None
        """Record usage statistics."""
        self._usage.total_calls += 1
        self._usage.total_prompt_tokens += prompt_tokens
        self._usage.total_completion_tokens += completion_tokens
        self._usage.total_tokens += prompt_tokens + completion_tokens
        self._usage.total_latency_ms += latency_ms
        self._usage.total_cost_usd += estimate_cost(
            self.model_name, prompt_tokens, completion_tokens
        )
        # Backward compat
        self.call_count = self._usage.total_calls
        self.total_tokens = self._usage.total_tokens


class AzureOpenAILLM:
    """
    Azure OpenAI Service provider.

    Uses the Azure-specific endpoint and API version format.

    Args:
        model_name: The Azure *deployment name* (not the model name).
        api_key: Azure OpenAI API key. Falls back to ``AZURE_OPENAI_API_KEY``.
        azure_endpoint: Azure resource endpoint
            (e.g. ``"https://my-resource.openai.azure.com"``).
        api_version: Azure API version (default: ``"2024-10-21"``).
        **kwargs: Same options as ``OpenAILLM``.

    Example::

        from llm_assessment.providers.openai_provider import AzureOpenAILLM

        llm = AzureOpenAILLM(
            model_name="my-gpt4o-deployment",
            azure_endpoint="https://my-resource.openai.azure.com",
            api_key="...",
        )
    """

    def __init__(self, model_name="gpt-4o", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # Backward-compat
        self.call_count = 0
        self.total_tokens = 0

        # Usage tracking
        self._usage = UsageStats()

        self._temperature = kwargs.pop("temperature", 0.7)
        self._max_tokens = kwargs.pop("max_tokens", None)
        self._timeout = kwargs.pop("timeout", 120)

        # Retry / rate limit
        self._retry_config = RetryConfig(
            max_retries=kwargs.pop("max_retries", 3)
        )
        rl_cfg = kwargs.pop("rate_limit", None)
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # ---- Azure OpenAI client ----
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Azure OpenAI provider requires the 'openai' package.\n"
                "Install it with: pip install openai>=1.0.0"
            )

        api_key = kwargs.pop("api_key", None)
        azure_endpoint = kwargs.pop("azure_endpoint", None)
        api_version = kwargs.pop("api_version", "2024-10-21")

        if not azure_endpoint:
            import os
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not api_key:
            import os
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")

        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=self._timeout,
            max_retries=0,
        )

        try:
            from ..core.llm_wrapper import BaseLLM
            if isinstance(self, BaseLLM):
                BaseLLM.__init__(self, model_name, **kwargs)
        except (ImportError, TypeError):
            pass

    # The rest of the methods delegate to the same pattern as OpenAILLM.
    # We share the implementation via composition rather than inheritance
    # to keep each class self‑contained for clarity.

    @retry_on_error
    def generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        messages = [{"role": "user", "content": prompt}]
        return self._chat_completion(messages, **kwargs)

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        return self._chat_completion(messages, **kwargs)

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        messages = [{"role": "user", "content": prompt}]
        for chunk in self._stream_chat_completion(messages, **kwargs):
            yield chunk

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        for chunk in self._stream_chat_completion(messages, **kwargs):
            yield chunk

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
                    results[idx] = "[Error: {}]".format(e)
        return results

    def health_check(self):
        # type: () -> Dict[str, Any]
        start = time.time()
        try:
            resp = self.generate("Say 'ok'.", max_tokens=5)
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

    def get_stats(self):
        # type: () -> Dict[str, Any]
        stats = self._usage.to_dict()
        stats["model_name"] = self.model_name
        stats["call_count"] = self._usage.total_calls
        stats["total_tokens"] = self._usage.total_tokens
        return stats

    def __repr__(self):
        return "AzureOpenAILLM(model='{}')".format(self.model_name)

    # ---- Internals (same logic as OpenAILLM) ----

    def _chat_completion(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        self._rate_limiter.acquire(estimate_tokens(str(messages)))
        request_kwargs = self._build_request_kwargs(kwargs)
        request_kwargs["stream"] = False

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **request_kwargs,
        )
        latency_ms = (time.time() - start) * 1000

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return response.choices[0].message.content or ""

    def _stream_chat_completion(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        self._rate_limiter.acquire(estimate_tokens(str(messages)))
        request_kwargs = self._build_request_kwargs(kwargs)
        request_kwargs["stream"] = True

        start = time.time()
        collected = []

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **request_kwargs,
        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    collected.append(delta.content)
                    yield delta.content

        latency_ms = (time.time() - start) * 1000
        full_text = "".join(collected)
        self._track_usage(
            estimate_tokens(str(messages)),
            estimate_tokens(full_text),
            latency_ms,
        )

    def _build_request_kwargs(self, overrides):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        kwargs = {}
        temperature = overrides.pop("temperature", self._temperature)
        max_tokens = overrides.pop("max_tokens", self._max_tokens)

        model_lower = self.model_name.lower()
        if not any(model_lower.startswith(p) for p in ("o1", "o3")):
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            if any(model_lower.startswith(p) for p in ("o1", "o3")):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        internal_keys = {
            "api_key", "base_url", "organization", "timeout",
            "azure_endpoint", "api_version",
            "rate_limit", "retry", "max_retries",
        }
        for k, v in overrides.items():
            if k not in internal_keys:
                kwargs[k] = v
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
