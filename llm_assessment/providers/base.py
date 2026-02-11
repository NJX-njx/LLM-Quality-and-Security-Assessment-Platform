"""
Shared utilities for LLM providers.

Includes retry logic, rate limiting, usage tracking, and cost estimation.
All providers share these infrastructure components.
"""

import time
import random
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================
# Configuration Dataclasses
# ============================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass
class RateLimitConfig:
    """Configuration for token-bucket rate limiting."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    enabled: bool = True


@dataclass
class UsageStats:
    """Aggregated usage statistics for a provider instance."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    errors: int = 0
    retries: int = 0

    @property
    def avg_latency_ms(self):
        # type: () -> float
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    def to_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "errors": self.errors,
            "retries": self.retries,
        }


@dataclass
class LLMResponse:
    """Unified response object returned by provider internals."""
    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0
    raw_response: Any = None


# ============================================================
# Rate Limiter — Token Bucket Algorithm
# ============================================================

class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter.

    Controls both request rate and token throughput to stay within
    API provider limits.
    """

    def __init__(self, config=None):
        # type: (Optional[RateLimitConfig]) -> None
        if config is None:
            config = RateLimitConfig()
        self.config = config
        self._lock = threading.Lock()
        self._request_tokens = float(config.requests_per_minute)
        self._token_tokens = float(config.tokens_per_minute)
        self._last_refill = time.monotonic()
        self._max_request_tokens = float(config.requests_per_minute)
        self._max_token_tokens = float(config.tokens_per_minute)

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        request_refill = elapsed * (self._max_request_tokens / 60.0)
        token_refill = elapsed * (self._max_token_tokens / 60.0)
        self._request_tokens = min(
            self._max_request_tokens, self._request_tokens + request_refill
        )
        self._token_tokens = min(
            self._max_token_tokens, self._token_tokens + token_refill
        )
        self._last_refill = now

    def acquire(self, estimated_tokens=1):
        # type: (int) -> None
        """Block until the request is allowed under rate limits."""
        if not self.config.enabled:
            return

        while True:
            with self._lock:
                self._refill()
                if (self._request_tokens >= 1.0
                        and self._token_tokens >= estimated_tokens):
                    self._request_tokens -= 1.0
                    self._token_tokens -= estimated_tokens
                    return
            # Back off briefly before re-checking
            time.sleep(0.05)


# ============================================================
# Retry Decorator — Exponential Backoff with Jitter
# ============================================================

def retry_on_error(func=None, config=None):
    """
    Decorator that retries LLM API calls on transient failures.

    Supports:
    - Exponential backoff with optional jitter
    - HTTP status-code–based retry decisions
    - Keyword-based heuristic retry (rate limit, overloaded, etc.)

    Usage::

        @retry_on_error
        def my_api_call(...): ...

        @retry_on_error(config=RetryConfig(max_retries=5))
        def my_api_call(...): ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(config.max_retries + 1):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Decide if the error is worth retrying
                    is_retryable = False

                    # Check HTTP status code (works with openai, anthropic SDKs)
                    status_code = getattr(e, "status_code", None)
                    if status_code is None:
                        status_code = getattr(e, "http_status", None)
                    if status_code and status_code in config.retryable_status_codes:
                        is_retryable = True

                    # Heuristic keyword match
                    error_str = str(e).lower()
                    retryable_keywords = [
                        "rate limit", "rate_limit", "overloaded",
                        "429", "503", "too many requests",
                        "connection", "timeout", "server error",
                    ]
                    if any(kw in error_str for kw in retryable_keywords):
                        is_retryable = True

                    if not is_retryable or attempt == config.max_retries:
                        raise

                    # Compute delay
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay,
                    )
                    if config.jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        "Retry %d/%d for %s after error: %s — waiting %.1fs",
                        attempt + 1,
                        config.max_retries,
                        f.__name__,
                        str(e)[:120],
                        delay,
                    )

                    # Track retry in usage stats if possible
                    if args and hasattr(args[0], "_usage"):
                        args[0]._usage.retries += 1

                    time.sleep(delay)

            raise last_exception  # pragma: no cover

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ============================================================
# Cost Estimation — Per-Model Pricing Tables
# ============================================================

# Pricing: (input_cost_per_1M_tokens, output_cost_per_1M_tokens) in USD
# Prices as of early 2025.  Add new models as needed.
MODEL_PRICING = {
    # ---- OpenAI ----
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-preview": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-4-32k": (60.00, 120.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-3.5-turbo-16k": (3.00, 4.00),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1-preview": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # ---- Anthropic ----
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-3.5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    # ---- DeepSeek ----
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # ---- Local / Free ----
    "mock-model": (0.0, 0.0),
}


def estimate_cost(model_name, prompt_tokens, completion_tokens):
    # type: (str, int, int) -> float
    """Estimate the USD cost for an API call based on token counts."""
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        # Try prefix matching (e.g. "gpt-4o-2024-..." matches "gpt-4o")
        for key in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
            if model_name.startswith(key):
                pricing = MODEL_PRICING[key]
                break
    if pricing is None:
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * pricing[0]
    output_cost = (completion_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


# ============================================================
# Token Estimation (heuristic, no tokenizer dependency)
# ============================================================

def estimate_tokens(text):
    # type: (str) -> int
    """
    Rough token count heuristic.

    For accurate counts, use ``tiktoken`` (OpenAI) or provider-specific
    tokenizers.  This approximation is used only when actual token counts
    are unavailable (e.g., Ollama without usage info).
    """
    if not text:
        return 0
    # CJK characters ≈ 1.5 tokens each; ASCII ≈ 0.25 tokens per char
    cjk = sum(
        1 for c in text
        if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f"
    )
    ascii_chars = len(text) - cjk
    return max(1, int(ascii_chars / 4 + cjk / 1.5))
