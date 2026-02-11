"""
OpenAI LLM 提供者 — 支持 OpenAI API 和 Azure OpenAI 服务。

功能特性：
    - 支持 GPT-4o / GPT-4 / GPT-3.5 / o1 / o3-mini 模型系列
    - Azure OpenAI 服务，基于部署的路由
    - 流式输出支持
    - 自动重试（指数退避）
    - 速率限制（令牌桶）
    - 详细的用量跟踪和成本估算

参考：
    - OpenAI API 文档: https://platform.openai.com/docs/api-reference
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
    OpenAI LLM 提供者（ChatCompletion API）。

    支持所有基于对话的 OpenAI 模型，包括 GPT-4o、o1、o3-mini。

    参数：
        model_name: 模型标识符（默认: ``"gpt-4o-mini"``）。
        api_key: OpenAI API 密钥。会回退到 ``OPENAI_API_KEY`` 环境变量。
        base_url: 覆盖 API 基础 URL（适用于代理）。
        organization: OpenAI 组织 ID。
        timeout: 请求超时时间（秒，默认: 120）。
        max_retries: 瞬态错误的最大重试次数（默认: 3）。
        temperature: 默认采样温度（默认: 0.7）。
        max_tokens: 生成的最大 token 数。
        rate_limit: ``RateLimitConfig`` 实例或字典。
        **kwargs: 传递给客户端的其他配置。
    """

    def __init__(self, model_name="gpt-4o-mini", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # 向后兼容字段
        self.call_count = 0
        self.total_tokens = 0

        # 用量跟踪
        self._usage = UsageStats()

        # 默认生成参数
        self._temperature = kwargs.pop("temperature", 0.7)
        self._max_tokens = kwargs.pop("max_tokens", None)
        self._timeout = kwargs.pop("timeout", 120)

        # 重试配置
        max_retries_sdk = kwargs.pop("max_retries", 3)
        self._retry_config = RetryConfig(max_retries=max_retries_sdk)

        # 速率限制器
        rl_cfg = kwargs.pop("rate_limit", None)
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # ---- 创建 OpenAI 客户端（延迟导入） ----
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
        # 让 SDK 的自带重试作为最后的备用方案
        client_kwargs["max_retries"] = 0

        self.client = openai.OpenAI(**client_kwargs)

        # 如果作为子类使用则初始化 BaseLLM
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
        return self._chat_completion(messages, **kwargs)

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """多轮对话完成"""
        return self._chat_completion(messages, **kwargs)

    # ------------------------------------------------------------------ #
    #  流式输出
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """从单个提示词流式生成响应，逐块产出文本"""
        messages = [{"role": "user", "content": prompt}]
        for chunk in self._stream_chat_completion(messages, **kwargs):
            yield chunk

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """流式多轮对话完成"""
        for chunk in self._stream_chat_completion(messages, **kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    #  批量生成
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=4, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """使用线程池为多个提示词批量生成响应"""
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
        """验证 OpenAI API 是否可达"""
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
    #  统计信息
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
    #  内部辅助方法
    # ------------------------------------------------------------------ #

    def _chat_completion(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """执行非流式 ChatCompletion 请求"""
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

        # 提取用量信息
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        content = response.choices[0].message.content or ""
        return content

    def _stream_chat_completion(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """执行流式 ChatCompletion 请求"""
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
            # 用量信息在最后一个 chunk 中
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    collected_content.append(delta.content)
                    yield delta.content

        latency_ms = (time.time() - start) * 1000

        # 如果未提供用量信息，回退到估算
        if not prompt_tokens:
            full_text = "".join(collected_content)
            prompt_tokens = estimate_tokens(str(messages))
            completion_tokens = estimate_tokens(full_text)

        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

    def _build_request_kwargs(self, overrides):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        """将默认参数与每次调用的覆盖参数合并"""
        kwargs = {}

        temperature = overrides.pop("temperature", self._temperature)
        max_tokens = overrides.pop("max_tokens", self._max_tokens)

        # o1 / o3 模型不支持 temperature 参数
        model_lower = self.model_name.lower()
        if not any(model_lower.startswith(p) for p in ("o1", "o3")):
            kwargs["temperature"] = temperature

        if max_tokens is not None:
            # o1/o3 模型使用 'max_completion_tokens' 而非 'max_tokens'
            if any(model_lower.startswith(p) for p in ("o1", "o3")):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        # 透传其余 kwargs（top_p, stop 等）
        # 过滤掉内部使用的 key
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
        """记录用量统计"""
        self._usage.total_calls += 1
        self._usage.total_prompt_tokens += prompt_tokens
        self._usage.total_completion_tokens += completion_tokens
        self._usage.total_tokens += prompt_tokens + completion_tokens
        self._usage.total_latency_ms += latency_ms
        self._usage.total_cost_usd += estimate_cost(
            self.model_name, prompt_tokens, completion_tokens
        )
        # 向后兼容
        self.call_count = self._usage.total_calls
        self.total_tokens = self._usage.total_tokens


class AzureOpenAILLM:
    """
    Azure OpenAI 服务提供者。

    使用 Azure 特定的端点和 API 版本格式。

    参数：
        model_name: Azure 部署名称（非模型名）。
        api_key: Azure OpenAI API 密钥。回退到 ``AZURE_OPENAI_API_KEY``。
        azure_endpoint: Azure 资源端点
            （如 ``"https://my-resource.openai.azure.com"``）。
        api_version: Azure API 版本（默认: ``"2024-10-21"``）。
        **kwargs: 与 ``OpenAILLM`` 相同的选项。
    """

    def __init__(self, model_name="gpt-4o", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # 向后兼容字段
        self.call_count = 0
        self.total_tokens = 0

        # 用量跟踪
        self._usage = UsageStats()

        self._temperature = kwargs.pop("temperature", 0.7)
        self._max_tokens = kwargs.pop("max_tokens", None)
        self._timeout = kwargs.pop("timeout", 120)

        # 重试 / 速率限制
        self._retry_config = RetryConfig(
            max_retries=kwargs.pop("max_retries", 3)
        )
        rl_cfg = kwargs.pop("rate_limit", None)
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # ---- Azure OpenAI 客户端 ----
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

    # 其余方法与 OpenAILLM 采用相同模式。
    # 通过组合而非继承共享实现，保持每个类的独立性和清晰性。

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

    # ---- 内部实现（与 OpenAILLM 相同的逻辑） ----

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
