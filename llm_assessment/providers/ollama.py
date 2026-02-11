"""
Ollama LLM Provider — Local model inference via Ollama REST API.

Zero-cost local model inference supporting Llama 3, Mistral, Qwen, Gemma,
and any model available through Ollama's model library.

Features:
    - No API key required (runs entirely locally)
    - Streaming support
    - Multi-model management (pull, list, check availability)
    - Configurable generation parameters (temperature, top_p, etc.)
    - Automatic model pulling on first use (optional)

Reference:
    - Ollama: https://ollama.com
    - REST API: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
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


class OllamaLLM:
    """
    Ollama local LLM provider.

    Communicates with a locally running Ollama server via its REST API.

    Args:
        model_name: Model tag (default: ``"llama3.2"``).
            Use ``ollama list`` to see available models.
        base_url: Ollama server URL (default: ``"http://localhost:11434"``).
        timeout: Request timeout in seconds (default: 300 — local inference
            can be slow on CPU).
        temperature: Default temperature (default: 0.7).
        num_ctx: Context window size (default: model's built-in).
        auto_pull: Automatically pull a model if it's not found locally
            (default: ``False``).
        **kwargs: Additional configuration.

    Example::

        from llm_assessment.providers.ollama import OllamaLLM

        llm = OllamaLLM(model_name="llama3.2")
        print(llm.generate("What is machine learning?"))

        # Use a different model
        llm = OllamaLLM(model_name="qwen2:7b")

        # Check available models
        print(llm.list_models())
    """

    def __init__(self, model_name="llama3.2", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # Backward-compat
        self.call_count = 0
        self.total_tokens = 0

        # Usage tracking (all local — zero cost)
        self._usage = UsageStats()

        self.base_url = kwargs.pop("base_url", "http://localhost:11434")
        self._timeout = kwargs.pop("timeout", 300)
        self._temperature = kwargs.pop("temperature", 0.7)
        self._num_ctx = kwargs.pop("num_ctx", None)
        self._auto_pull = kwargs.pop("auto_pull", False)

        # Retry (for transient connection errors)
        self._retry_config = RetryConfig(
            max_retries=kwargs.pop("max_retries", 2),
            base_delay=0.5,
        )

        # Rate limiter (usually not needed locally, but available)
        rl_cfg = kwargs.pop("rate_limit", RateLimitConfig(enabled=False))
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # Lazy import
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError(
                "Ollama provider requires the 'requests' package.\n"
                "Install it with: pip install requests>=2.28.0"
            )

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
        """Generate a response using Ollama's /api/generate endpoint."""
        import requests

        self._ensure_model_available()
        self._rate_limiter.acquire()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(kwargs),
        }

        # Optional system prompt
        system = kwargs.get("system")
        if system:
            payload["system"] = system

        start = time.time()
        resp = requests.post(
            "{}/api/generate".format(self.base_url),
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        latency_ms = (time.time() - start) * 1000

        data = resp.json()
        content = data.get("response", "")

        # Ollama returns eval_count / prompt_eval_count when available
        prompt_tokens = data.get("prompt_eval_count", 0) or estimate_tokens(prompt)
        completion_tokens = data.get("eval_count", 0) or estimate_tokens(content)
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """Multi-turn chat using Ollama's /api/chat endpoint."""
        import requests

        self._ensure_model_available()
        self._rate_limiter.acquire()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": self._build_options(kwargs),
        }

        start = time.time()
        resp = requests.post(
            "{}/api/chat".format(self.base_url),
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        latency_ms = (time.time() - start) * 1000

        data = resp.json()
        content = data.get("message", {}).get("content", "")

        prompt_tokens = data.get("prompt_eval_count", 0) or estimate_tokens(str(messages))
        completion_tokens = data.get("eval_count", 0) or estimate_tokens(content)
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    # ------------------------------------------------------------------ #
    #  Streaming
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """Stream generation via Ollama's streaming API."""
        import requests

        self._ensure_model_available()
        self._rate_limiter.acquire()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": self._build_options(kwargs),
        }

        system = kwargs.get("system")
        if system:
            payload["system"] = system

        start = time.time()
        collected = []

        resp = requests.post(
            "{}/api/generate".format(self.base_url),
            json=payload,
            timeout=self._timeout,
            stream=True,
        )
        resp.raise_for_status()

        prompt_tokens = 0
        completion_tokens = 0

        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            text = data.get("response", "")
            if text:
                collected.append(text)
                yield text

            # Final chunk contains usage stats
            if data.get("done"):
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)

        latency_ms = (time.time() - start) * 1000
        if not prompt_tokens:
            prompt_tokens = estimate_tokens(prompt)
            completion_tokens = estimate_tokens("".join(collected))
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """Stream chat via Ollama's streaming API."""
        import requests

        self._ensure_model_available()
        self._rate_limiter.acquire()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": self._build_options(kwargs),
        }

        start = time.time()
        collected = []

        resp = requests.post(
            "{}/api/chat".format(self.base_url),
            json=payload,
            timeout=self._timeout,
            stream=True,
        )
        resp.raise_for_status()

        prompt_tokens = 0
        completion_tokens = 0

        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            msg = data.get("message", {})
            text = msg.get("content", "")
            if text:
                collected.append(text)
                yield text

            if data.get("done"):
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)

        latency_ms = (time.time() - start) * 1000
        if not prompt_tokens:
            prompt_tokens = estimate_tokens(str(messages))
            completion_tokens = estimate_tokens("".join(collected))
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

    # ------------------------------------------------------------------ #
    #  Batch
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=2, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """
        Batch generation.

        Note: Ollama processes sequentially on most hardware, so
        ``max_workers`` should be kept low (1-2) to avoid contention.
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
                    logger.error("batch_generate error at index %d: %s", idx, e)
                    results[idx] = "[Error: {}]".format(e)
        return results

    # ------------------------------------------------------------------ #
    #  Health check & model management
    # ------------------------------------------------------------------ #

    def health_check(self):
        # type: () -> Dict[str, Any]
        """Check Ollama server connectivity and model availability."""
        import requests

        start = time.time()
        try:
            # Check server
            resp = requests.get(
                "{}/api/tags".format(self.base_url),
                timeout=5,
            )
            resp.raise_for_status()

            models = [m["name"] for m in resp.json().get("models", [])]
            model_available = any(
                self.model_name in m for m in models
            )

            latency = (time.time() - start) * 1000
            return {
                "status": "healthy" if model_available else "degraded",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "model_available": model_available,
                "available_models": models[:10],  # First 10
            }
        except Exception as e:
            latency = (time.time() - start) * 1000
            return {
                "status": "unhealthy",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "error": str(e),
                "hint": "Is Ollama running? Start it with: ollama serve",
            }

    def list_models(self):
        # type: () -> List[str]
        """List all locally available Ollama models."""
        import requests

        resp = requests.get(
            "{}/api/tags".format(self.base_url),
            timeout=10,
        )
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def pull_model(self, model_name=None):
        # type: (Optional[str]) -> None
        """Pull (download) a model from the Ollama library."""
        import requests

        model = model_name or self.model_name
        logger.info("Pulling model '%s' — this may take a while...", model)

        resp = requests.post(
            "{}/api/pull".format(self.base_url),
            json={"name": model, "stream": False},
            timeout=3600,  # Models can be large
        )
        resp.raise_for_status()
        logger.info("Model '%s' pulled successfully.", model)

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def get_stats(self):
        # type: () -> Dict[str, Any]
        stats = self._usage.to_dict()
        stats["model_name"] = self.model_name
        stats["call_count"] = self._usage.total_calls
        stats["total_tokens"] = self._usage.total_tokens
        stats["total_cost_usd"] = 0.0  # Always free
        return stats

    def __repr__(self):
        return "OllamaLLM(model='{}', url='{}')".format(
            self.model_name, self.base_url
        )

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _ensure_model_available(self):
        # type: () -> None
        """Pull the model automatically if configured and not present."""
        if not self._auto_pull:
            return

        import requests
        try:
            resp = requests.post(
                "{}/api/show".format(self.base_url),
                json={"name": self.model_name},
                timeout=10,
            )
            if resp.status_code == 404:
                self.pull_model()
        except Exception:
            pass  # Will fail later with a clearer error

    def _build_options(self, kwargs):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        """Build Ollama-specific generation options."""
        options = {}

        temperature = kwargs.pop("temperature", self._temperature)
        options["temperature"] = temperature

        if self._num_ctx:
            options["num_ctx"] = self._num_ctx

        # Map common params to Ollama option names
        param_map = {
            "top_p": "top_p",
            "top_k": "top_k",
            "max_tokens": "num_predict",
            "stop": "stop",
            "seed": "seed",
            "repeat_penalty": "repeat_penalty",
        }
        for our_key, ollama_key in param_map.items():
            if our_key in kwargs:
                options[ollama_key] = kwargs.pop(our_key)

        return options

    def _track_usage(self, prompt_tokens, completion_tokens, latency_ms):
        # type: (int, int, float) -> None
        self._usage.total_calls += 1
        self._usage.total_prompt_tokens += prompt_tokens
        self._usage.total_completion_tokens += completion_tokens
        self._usage.total_tokens += prompt_tokens + completion_tokens
        self._usage.total_latency_ms += latency_ms
        # Cost is always 0 for local models
        self.call_count = self._usage.total_calls
        self.total_tokens = self._usage.total_tokens
