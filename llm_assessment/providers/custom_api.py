"""
Custom API LLM Provider — Connect to any OpenAI-compatible or custom REST API.

Designed for enterprise / self-hosted LLM endpoints that expose a REST API
but don't conform to any specific SDK.

Features:
    - Flexible request/response mapping for arbitrary API formats
    - OpenAI-compatible mode for quick setup
    - Custom header injection (auth tokens, API keys, etc.)
    - Configurable JSON paths for extracting response content
    - Streaming support (SSE-based)
    - Retry, rate limiting, usage tracking

Example API formats supported:
    - OpenAI-compatible (``/v1/chat/completions``)
    - Ollama-style (``/api/generate``)
    - Custom enterprise APIs with arbitrary JSON structures

Reference:
    Inspired by LiteLLM's custom API handling.
"""

import json
import time
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional

from .base import (
    RetryConfig,
    RateLimitConfig,
    UsageStats,
    TokenBucketRateLimiter,
    estimate_tokens,
    retry_on_error,
)

logger = logging.getLogger(__name__)


def _extract_json_path(data, path, default=""):
    """
    Extract a value from nested JSON using dot-separated path.

    Example::
        _extract_json_path({"a": {"b": "c"}}, "a.b")  # "c"
        _extract_json_path({"choices": [{"text": "hi"}]}, "choices.0.text")  # "hi"
    """
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(key)]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return current


class CustomAPILLM:
    """
    Custom REST API LLM provider.

    Args:
        model_name: Model identifier sent in the request body.
        base_url: API base URL (required).
        api_key: API key sent in the ``Authorization`` header.
        headers: Additional HTTP headers (dict).
        chat_endpoint: Path for chat completions (default: ``"/v1/chat/completions"``).
        generate_endpoint: Path for text generation (default: same as chat).
        request_template: Dict template for request body.
            Use ``{prompt}``, ``{messages}``, ``{model}``, ``{max_tokens}``,
            ``{temperature}`` as placeholders.
        response_content_path: JSON path to extract response text
            (default: ``"choices.0.message.content"``).
        api_style: Shortcut for common API styles.
            ``"openai"`` (default), ``"ollama"``, or ``"custom"``.
        temperature: Default temperature (default: 0.7).
        max_tokens: Default max tokens (default: 1024).
        timeout: Request timeout in seconds (default: 120).
        **kwargs: Additional configuration.

    Example — OpenAI-compatible API::

        llm = CustomAPILLM(
            model_name="my-model",
            base_url="https://my-api.example.com",
            api_key="sk-...",
            api_style="openai",
        )

    Example — Fully custom API::

        llm = CustomAPILLM(
            model_name="custom-v1",
            base_url="https://api.internal.corp",
            headers={"X-API-Key": "secret"},
            chat_endpoint="/inference",
            request_template={
                "model_id": "{model}",
                "input": {"messages": "{messages}"},
                "params": {"max_length": "{max_tokens}", "temp": "{temperature}"},
            },
            response_content_path="output.text",
        )
    """

    def __init__(self, model_name="custom-model", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        self.call_count = 0
        self.total_tokens = 0
        self._usage = UsageStats()

        # API config
        self.base_url = kwargs.pop("base_url", "http://localhost:8000")
        self.base_url = self.base_url.rstrip("/")

        api_key = kwargs.pop("api_key", None)
        extra_headers = kwargs.pop("headers", {})

        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = "Bearer {}".format(api_key)
        self._headers.update(extra_headers)

        # Endpoint paths
        api_style = kwargs.pop("api_style", "openai")

        if api_style == "ollama":
            default_chat = "/api/chat"
            default_gen = "/api/generate"
            default_content_path = "message.content"
        else:
            # OpenAI-compatible (default)
            default_chat = "/v1/chat/completions"
            default_gen = "/v1/chat/completions"
            default_content_path = "choices.0.message.content"

        self._chat_endpoint = kwargs.pop("chat_endpoint", default_chat)
        self._generate_endpoint = kwargs.pop("generate_endpoint", default_gen)
        self._response_content_path = kwargs.pop(
            "response_content_path", default_content_path
        )
        self._api_style = api_style

        # Custom request template
        self._request_template = kwargs.pop("request_template", None)

        # Generation defaults
        self._temperature = kwargs.pop("temperature", 0.7)
        self._max_tokens = kwargs.pop("max_tokens", 1024)
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

        # Verify requests is available
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError(
                "CustomAPILLM requires the 'requests' package.\n"
                "Install it with: pip install requests>=2.28.0"
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
        """Generate text from a prompt."""
        if self._api_style == "ollama":
            return self._ollama_generate(prompt, **kwargs)
        # Default: use chat-style with a single user message
        messages = [{"role": "user", "content": prompt}]
        return self._chat_request(messages, **kwargs)

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """Multi-turn chat completion."""
        return self._chat_request(messages, **kwargs)

    # ------------------------------------------------------------------ #
    #  Streaming
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        messages = [{"role": "user", "content": prompt}]
        for chunk in self._stream_chat_request(messages, **kwargs):
            yield chunk

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        for chunk in self._stream_chat_request(messages, **kwargs):
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
                    results[idx] = "[Error: {}]".format(e)
        return results

    # ------------------------------------------------------------------ #
    #  Health check
    # ------------------------------------------------------------------ #

    def health_check(self):
        # type: () -> Dict[str, Any]
        import requests

        start = time.time()
        try:
            # Try a simple GET to the base URL
            resp = requests.get(
                self.base_url,
                headers=self._headers,
                timeout=10,
            )
            latency = (time.time() - start) * 1000
            return {
                "status": "healthy" if resp.status_code < 500 else "degraded",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "http_status": resp.status_code,
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
        return "CustomAPILLM(model='{}', url='{}')".format(
            self.model_name, self.base_url
        )

    # ------------------------------------------------------------------ #
    #  Internal — request builders
    # ------------------------------------------------------------------ #

    def _chat_request(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """Send a non-streaming chat request."""
        import requests

        self._rate_limiter.acquire(estimate_tokens(str(messages)))

        temperature = kwargs.pop("temperature", self._temperature)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        if self._request_template:
            payload = self._render_template(
                self._request_template,
                messages=messages,
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # OpenAI-compatible format
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

        url = "{}{}".format(self.base_url, self._chat_endpoint)

        start = time.time()
        resp = requests.post(
            url,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        latency_ms = (time.time() - start) * 1000

        data = resp.json()
        content = _extract_json_path(data, self._response_content_path, "")
        if not isinstance(content, str):
            content = str(content)

        # Try to get usage from response
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0) or estimate_tokens(str(messages))
        completion_tokens = usage.get("completion_tokens", 0) or estimate_tokens(content)
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    def _ollama_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        """Ollama-style /api/generate request."""
        import requests

        self._rate_limiter.acquire(estimate_tokens(prompt))

        temperature = kwargs.pop("temperature", self._temperature)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        url = "{}{}".format(self.base_url, self._generate_endpoint)

        start = time.time()
        resp = requests.post(
            url,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        latency_ms = (time.time() - start) * 1000

        data = resp.json()
        content = data.get("response", "")

        prompt_tokens = data.get("prompt_eval_count", 0) or estimate_tokens(prompt)
        completion_tokens = data.get("eval_count", 0) or estimate_tokens(content)
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    def _stream_chat_request(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """Send a streaming SSE-based request."""
        import requests

        self._rate_limiter.acquire(estimate_tokens(str(messages)))

        temperature = kwargs.pop("temperature", self._temperature)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = "{}{}".format(self.base_url, self._chat_endpoint)

        start = time.time()
        collected = []

        resp = requests.post(
            url,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
            stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            # Handle SSE format: "data: {...}"
            if line_str.startswith("data: "):
                line_str = line_str[6:]
            if line_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(line_str)
                # Try OpenAI SSE format
                text = _extract_json_path(
                    data, "choices.0.delta.content", ""
                )
                if not text:
                    # Try Ollama streaming format
                    text = data.get("message", {}).get("content", "")
                if not text:
                    text = data.get("response", "")
                if text:
                    collected.append(text)
                    yield text
            except (json.JSONDecodeError, KeyError):
                continue

        latency_ms = (time.time() - start) * 1000
        self._track_usage(
            estimate_tokens(str(messages)),
            estimate_tokens("".join(collected)),
            latency_ms,
        )

    @staticmethod
    def _render_template(template, **values):
        """
        Recursively render a request template dict by replacing placeholders.

        Supports ``{prompt}``, ``{messages}``, ``{model}``, ``{max_tokens}``,
        ``{temperature}`` in string values.
        """
        if isinstance(template, dict):
            return {
                k: CustomAPILLM._render_template(v, **values)
                for k, v in template.items()
            }
        if isinstance(template, list):
            return [CustomAPILLM._render_template(v, **values) for v in template]
        if isinstance(template, str):
            for key, val in values.items():
                placeholder = "{" + key + "}"
                if template == placeholder:
                    return val  # Return the actual type, not stringified
                if placeholder in template:
                    template = template.replace(placeholder, str(val))
            return template
        return template

    def _track_usage(self, prompt_tokens, completion_tokens, latency_ms):
        # type: (int, int, float) -> None
        self._usage.total_calls += 1
        self._usage.total_prompt_tokens += prompt_tokens
        self._usage.total_completion_tokens += completion_tokens
        self._usage.total_tokens += prompt_tokens + completion_tokens
        self._usage.total_latency_ms += latency_ms
        self.call_count = self._usage.total_calls
        self.total_tokens = self._usage.total_tokens
