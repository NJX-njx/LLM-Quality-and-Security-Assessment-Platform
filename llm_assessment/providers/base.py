"""
LLM 提供者共享工具模块

包含重试逻辑、速率限制、用量跟踪和成本估算等基础设施组件。
所有提供者共享这些基础组件。
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
# 配置数据类
# ============================================================

@dataclass
class RetryConfig:
    """重试行为配置，支持指数退避"""
    max_retries: int = 3                # 最大重试次数
    base_delay: float = 1.0             # 基础延迟（秒）
    max_delay: float = 60.0             # 最大延迟（秒）
    exponential_base: float = 2.0       # 指数基数
    jitter: bool = True                 # 是否添加随机抖动
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)  # 可重试的 HTTP 状态码


@dataclass
class RateLimitConfig:
    """令牌桶速率限制配置"""
    requests_per_minute: int = 60       # 每分钟请求数限制
    tokens_per_minute: int = 90000      # 每分钟 token 数限制
    enabled: bool = True                # 是否启用速率限制


@dataclass
class UsageStats:
    """提供者实例的累计用量统计"""
    total_calls: int = 0                # 总调用次数
    total_prompt_tokens: int = 0        # 总提示词 token 数
    total_completion_tokens: int = 0    # 总完成 token 数
    total_tokens: int = 0              # 总 token 数
    total_cost_usd: float = 0.0        # 总费用（美元）
    total_latency_ms: float = 0.0      # 总延迟（毫秒）
    errors: int = 0                    # 错误次数
    retries: int = 0                   # 重试次数

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
    """提供者内部返回的统一响应对象"""
    content: str                # 响应内容
    model: str = ""             # 模型名称
    prompt_tokens: int = 0      # 提示词 token 数
    completion_tokens: int = 0  # 完成 token 数
    total_tokens: int = 0       # 总 token 数
    finish_reason: str = ""     # 完成原因
    latency_ms: float = 0.0    # 延迟（毫秒）
    raw_response: Any = None    # 原始响应


# ============================================================
# 速率限制器 — 令牌桶算法
# ============================================================

class TokenBucketRateLimiter:
    """
    线程安全的令牌桶速率限制器。

    同时控制请求速率和 token 吞吐量，以保持在
    API 提供者的限制内。
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
        """根据经过的时间补充令牌"""
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
        """阻塞直到请求在速率限制内被允许"""
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
            # 短暂等待后重新检查
            time.sleep(0.05)


# ============================================================
# 重试装饰器 — 指数退避 + 随机抖动
# ============================================================

def retry_on_error(func=None, config=None):
    """
    对 LLM API 调用进行瞬态故障重试的装饰器。

    支持：
    - 指数退避 + 可选随机抖动
    - 基于 HTTP 状态码的重试决策
    - 基于关键词的启发式重试（速率限制、过载等）
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

                    # 判断是否值得重试
                    is_retryable = False

                    # 检查 HTTP 状态码（适用于 openai、anthropic SDK）
                    status_code = getattr(e, "status_code", None)
                    if status_code is None:
                        status_code = getattr(e, "http_status", None)
                    if status_code and status_code in config.retryable_status_codes:
                        is_retryable = True

                    # 启发式关键词匹配
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

                    # 计算延迟
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

                    # 如果可能，在用量统计中记录重试
                    if args and hasattr(args[0], "_usage"):
                        args[0]._usage.retries += 1

                    time.sleep(delay)

            raise last_exception  # pragma: no cover

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ============================================================
# 成本估算 — 按模型定价表
# ============================================================

# 定价：(每百万 token 输入成本, 每百万 token 输出成本) 单位美元
# 价格截至 2025 年初，如有新模型可随时添加
MODEL_PRICING = {
    # ---- OpenAI 系列 ----
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
    # ---- Anthropic 系列 ----
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-3.5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    # ---- DeepSeek 系列 ----
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # ---- 本地/免费模型 ----
    "mock-model": (0.0, 0.0),
}


def estimate_cost(model_name, prompt_tokens, completion_tokens):
    # type: (str, int, int) -> float
    """根据 token 数量估算 API 调用的美元成本"""
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        # 尝试前缀匹配（如 "gpt-4o-2024-..." 匹配 "gpt-4o"）
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
# Token 估算（启发式，无需分词器依赖）
# ============================================================

def estimate_tokens(text):
    # type: (str) -> int
    """
    粗略的 token 计数启发算法。

    精确计数请使用 tiktoken（OpenAI）或提供者特定的分词器。
    此近似值仅在实际 token 计数不可用时使用。
    """
    if not text:
        return 0
    # 中文字符 ≈ 1.5 tokens；ASCII 字符 ≈ 0.25 tokens
    cjk = sum(
        1 for c in text
        if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f"
    )
    ascii_chars = len(text) - cjk
    return max(1, int(ascii_chars / 4 + cjk / 1.5))
