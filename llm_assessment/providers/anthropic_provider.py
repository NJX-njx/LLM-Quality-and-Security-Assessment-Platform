"""
Anthropic LLM 提供者 — Claude 模型系列。

功能特性：
    - 支持 Claude 3.5 Sonnet / Claude 3 Opus / Claude 3 Haiku
    - 正确处理系统消息（Anthropic 使用专用字段）
    - 流式输出支持
    - 自动重试（指数退避）
    - 速率限制和成本跟踪

参考：
    - Anthropic API: https://docs.anthropic.com/en/docs/
    - Claude 模型: https://docs.anthropic.com/en/docs/about-claude/models
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

# Claude 模型的默认最大 token 数（API 必须参数）
_DEFAULT_MAX_TOKENS = 4096


class AnthropicLLM:
    """
    Anthropic Claude LLM 提供者。

    Claude 的 API 与 OpenAI 有几个重要区别：

    1. ``system`` 是顶层参数，而非消息角色。
    2. ``max_tokens`` 是**必填的**（非可选）。
    3. 响应格式将内容包装在 ``content`` 列表中。

    参数：
        model_name: Claude 模型名称（默认: ``"claude-3-5-sonnet-20241022"``）。
        api_key: Anthropic API 密钥。回退到 ``ANTHROPIC_API_KEY`` 环境变量。
        max_tokens: 默认最大完成 token 数（默认: 4096）。
        temperature: 默认温度（默认: 0.7）。
        timeout: 请求超时（秒，默认: 120）。
        max_retries: 瞬态错误重试次数（默认: 3）。
        rate_limit: ``RateLimitConfig`` 实例或字典。
        **kwargs: 其他配置。
    """

    def __init__(self, model_name="claude-3-5-sonnet-20241022", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # 向后兼容字段
        self.call_count = 0
        self.total_tokens = 0

        # 用量跟踪
        self._usage = UsageStats()

        # 默认参数
        self._max_tokens = kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS)
        self._temperature = kwargs.pop("temperature", 0.7)
        self._timeout = kwargs.pop("timeout", 120)

        # 重试配置
        self._retry_config = RetryConfig(
            max_retries=kwargs.pop("max_retries", 3)
        )

        # 速率限制器
        rl_cfg = kwargs.pop("rate_limit", None)
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # ---- 创建 Anthropic 客户端（延迟导入） ----
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

        # BaseLLM 初始化
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
        """从单个提示词生成响应"""
        messages = [{"role": "user", "content": prompt}]
        return self._create_message(messages, **kwargs)

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """
        多轮对话完成。

        ``messages`` 中的系统消息会被提取并通过
        Anthropic 的 ``system`` 参数传递。其余消息必须
        在 ``user`` 和 ``assistant`` 角色之间交替。
        """
        return self._create_message(messages, **kwargs)

    # ------------------------------------------------------------------ #
    #  流式输出
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """从单个提示词流式生成响应"""
        messages = [{"role": "user", "content": prompt}]
        for chunk in self._stream_message(messages, **kwargs):
            yield chunk

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """流式多轮对话完成"""
        for chunk in self._stream_message(messages, **kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    #  批量生成
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
    #  健康检查
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
    #  统计信息
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
    #  内部实现
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_system_message(messages):
        # type: (List[Dict[str, str]]) -> tuple
        """
        将系统消息与对话消息分离。

        返回：
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

        # Anthropic 要求消息以 "user" 角色开始
        # 并且角色交替。如果第一条消息是 "assistant"，
        # 则前置一个占位符 user 消息。
        if filtered and filtered[0].get("role") == "assistant":
            filtered.insert(0, {"role": "user", "content": "(continue)"})

        return system_text, filtered

    def _create_message(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """非流式消息创建"""
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

        # 提取用量信息
        usage = response.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        # 提取文本内容
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return content

    def _stream_message(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """流式消息创建"""
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

            # 从流的最终消息中获取用量信息
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

        # 透传安全的 key
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
