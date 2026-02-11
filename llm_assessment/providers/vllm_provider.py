"""
vLLM 提供者 — 通过 vLLM 的 OpenAI 兼容服务器实现高性能本地推理。

vLLM 暴露了一个 OpenAI 兼容的 REST API，因此本提供者使用
``openai`` SDK 指向 vLLM 端点。这使我们免费获得流式输出、
批处理和工具支持。

功能特性：
    - OpenAI 兼容 API（可直接替换）
    - 高吞吐量批处理推理
    - 流式输出支持
    - 多模型服务

前置条件：
    1. 安装 vLLM: ``pip install vllm``
    2. 启动服务器

参考：
    - vLLM 文档: https://docs.vllm.ai/en/latest/
    - OpenAI 兼容: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
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
    vLLM 提供者，通过其 OpenAI 兼容服务器。

    参数：
        model_name: vLLM 服务的模型名称。
        base_url: vLLM 服务器 URL（默认: ``"http://localhost:8000/v1"``）。
        api_key: API 密钥（vLLM 默认使用 ``"EMPTY"``）。
        temperature: 默认温度（默认: 0.7）。
        max_tokens: 默认最大 token 数（默认: 1024）。
        timeout: 请求超时（默认: 300秒 — 大模型可能较慢）。
        **kwargs: 其他配置。
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

        # 重试配置
        self._retry_config = RetryConfig(
            max_retries=kwargs.pop("max_retries", 2)
        )

        # 速率限制器（本地 vLLM 通常不需要）
        rl_cfg = kwargs.pop("rate_limit", RateLimitConfig(enabled=False))
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # ---- 指向 vLLM 的 OpenAI 客户端 ----
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
    #  核心生成方法
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
    #  流式输出
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
    #  批量生成
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=4, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """
        批量生成。

        vLLM 通过连续批处理高效处理并发请求，
        因此使用多线程是有益的。
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
    #  健康检查
    # ------------------------------------------------------------------ #

    def health_check(self):
        # type: () -> Dict[str, Any]
        """检查 vLLM 服务器连接性"""
        start = time.time()
        try:
            # 列出可用模型
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
    #  统计信息
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
    #  内部实现（复用 OpenAI 兼容 API）
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
