"""
vLLM Provider — High-performance local inference via vLLM's OpenAI-compatible server.

vLLM exposes an OpenAI-compatible REST API, so this provider uses the
``openai`` SDK pointed at the vLLM endpoint.  This gives us streaming,
batching, and tool support for free.

Features:
    - OpenAI-compatible API (drop-in replacement)
    - High-throughput batched inference
    - Streaming support
    - Multiple model serving via vLLM

Prerequisites:
    1. Install vLLM: ``pip install vllm``
    2. Start the server::

        python -m vllm.entrypoints.openai.api_server \\
            --model meta-llama/Llama-3.2-3B-Instruct \\
            --port 8000

Reference:
    - vLLM docs: https://docs.vllm.ai/en/latest/
    - OpenAI compat: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
"""

import time
import logging
from typing import Any, Dict, Iterator, List, Optional

from .base import (
    RetryConfig,
    RateLimitConfig,
    UsageStats,
    TokenBucketRateLimiter,
    estimate_tokens,
    retry_on_error,
)

logger = logging.getLogger(__name__)


class VllmLLM:
    """
    vLLM provider via its OpenAI-compatible server.

    Args:
        model_name: Model name served by vLLM.
        base_url: vLLM server URL (default: ``"http://localhost:8000/v1"``).
        api_key: API key (vLLM uses ``"EMPTY"`` by default).
        temperature: Default temperature (default: 0.7).
        max_tokens: Default max tokens (default: 1024).
        timeout: Request timeout (default: 300s — large models can be slow).
        **kwargs: Additional configuration.

    Example::

        from llm_assessment.providers.vllm_provider import VllmLLM

        llm = VllmLLM(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            base_url="http://localhost:8000/v1",
        )
        print(llm.generate("What is AI?"))
    """

    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        self.call_count = 0
        self.total_tokens = 0
        self._usage = UsageStats()

        self._temperature = kwargs.pop("temperature", 0.7)
        self._max_tokens = kwargs.pop("max_tokens", 1024)
        self._timeout = kwargs.pop("timeout", 300)

        # Retry
        self._retry_config = RetryConfig(
            max_retries=kwargs.pop("max_retries", 2)
        )

        # Rate limiter (usually not needed for local vLLM)
        rl_cfg = kwargs.pop("rate_limit", RateLimitConfig(enabled=False))
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # ---- OpenAI client pointed at vLLM ----
        try:
            import openai
        except ImportError:
            raise ImportError(
                "vLLM provider uses the OpenAI SDK for API compatibility.\n"
                "Install it with: pip install openai>=1.0.0"
            )

        base_url = kwargs.pop("base_url", "http://localhost:8000/v1")
        api_key = kwargs.pop("api_key", "EMPTY")

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self._timeout,
            max_retries=0,
        )

        # BaseLLM
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
        messages = [{"role": "user", "content": prompt}]
        return self._chat_completion(messages, **kwargs)

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        return self._chat_completion(messages, **kwargs)

    # ------------------------------------------------------------------ #
    #  Streaming
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        messages = [{"role": "user", "content": prompt}]
        for chunk in self._stream_chat_completion(messages, **kwargs):
            yield chunk

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        for chunk in self._stream_chat_completion(messages, **kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    #  Batch
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=4, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """
        Batch generation.

        vLLM handles concurrent requests efficiently via continuous
        batching, so using multiple workers is beneficial.
        """
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

    # ------------------------------------------------------------------ #
    #  Health check
    # ------------------------------------------------------------------ #

    def health_check(self):
        # type: () -> Dict[str, Any]
        """Check vLLM server connectivity."""
        start = time.time()
        try:
            # List available models
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            latency = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "available_models": model_ids,
            }
        except Exception as e:
            latency = (time.time() - start) * 1000
            return {
                "status": "unhealthy",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "error": str(e),
                "hint": (
                    "Is vLLM running? Start with:\n"
                    "  python -m vllm.entrypoints.openai.api_server "
                    "--model {} --port 8000".format(self.model_name)
                ),
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
        stats["total_cost_usd"] = 0.0
        return stats

    def __repr__(self):
        return "VllmLLM(model='{}')".format(self.model_name)

    # ------------------------------------------------------------------ #
    #  Internals (reuse OpenAI-compatible API)
    # ------------------------------------------------------------------ #

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
        self._track_usage(
            estimate_tokens(str(messages)),
            estimate_tokens("".join(collected)),
            latency_ms,
        )

    def _build_request_kwargs(self, overrides):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        kwargs = {}
        kwargs["temperature"] = overrides.pop("temperature", self._temperature)

        max_tokens = overrides.pop("max_tokens", self._max_tokens)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for key in ("top_p", "stop", "seed", "frequency_penalty", "presence_penalty"):
            if key in overrides:
                kwargs[key] = overrides.pop(key)

        return kwargs

    def _track_usage(self, prompt_tokens, completion_tokens, latency_ms):
        # type: (int, int, float) -> None
        self._usage.total_calls += 1
        self._usage.total_prompt_tokens += prompt_tokens
        self._usage.total_completion_tokens += completion_tokens
        self._usage.total_tokens += prompt_tokens + completion_tokens
        self._usage.total_latency_ms += latency_ms
        self.call_count = self._usage.total_calls
        self.total_tokens = self._usage.total_tokens
