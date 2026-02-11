"""
自定义 API LLM 提供者 — 连接任何 OpenAI 兼容或自定义 REST API。

专为企业/自托管 LLM 端点设计，这些端点暴露 REST API
但不符合任何特定的 SDK 规范。

功能特性：
    - 灵活的请求/响应映射，支持任意 API 格式
    - OpenAI 兼容模式，可快速设置
    - 自定义请求头注入（认证令牌、API 密钥等）
    - 可配置的 JSON 路径用于提取响应内容
    - 流式输出支持（基于 SSE）
    - 重试、速率限制、用量跟踪

支持的 API 格式示例：
    - OpenAI 兼容（``/v1/chat/completions``）
    - Ollama 风格（``/api/generate``）
    - 任意 JSON 结构的自定义企业 API
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
    使用点分隔路径从嵌套 JSON 中提取值。

    示例::
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
    自定义 REST API LLM 提供者。

    参数：
        model_name: 请求体中发送的模型标识符。
        base_url: API 基础 URL（必填）。
        api_key: 在 ``Authorization`` 头中发送的 API 密钥。
        headers: 额外的 HTTP 头（字典）。
        chat_endpoint: 对话完成的路径（默认: ``"/v1/chat/completions"``）。
        generate_endpoint: 文本生成的路径。
        request_template: 请求体的字典模板。
            使用 ``{prompt}``、``{messages}``、``{model}``、``{max_tokens}``、
            ``{temperature}`` 作为占位符。
        response_content_path: 提取响应文本的 JSON 路径。
        api_style: 常见 API 风格的快捷方式。
            ``"openai"``（默认）、``"ollama"`` 或 ``"custom"``。
        temperature: 默认温度（默认: 0.7）。
        max_tokens: 默认最大 token 数（默认: 1024）。
        timeout: 请求超时（秒，默认: 120）。
        **kwargs: 其他配置。
    """
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

        # API 配置
        self.base_url = kwargs.pop("base_url", "http://localhost:8000")
        self.base_url = self.base_url.rstrip("/")

        api_key = kwargs.pop("api_key", None)
        extra_headers = kwargs.pop("headers", {})

        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = "Bearer {}".format(api_key)
        self._headers.update(extra_headers)

        # 端点路径
        api_style = kwargs.pop("api_style", "openai")

        if api_style == "ollama":
            default_chat = "/api/chat"
            default_gen = "/api/generate"
            default_content_path = "message.content"
        else:
            # OpenAI 兼容（默认）
            default_chat = "/v1/chat/completions"
            default_gen = "/v1/chat/completions"
            default_content_path = "choices.0.message.content"

        self._chat_endpoint = kwargs.pop("chat_endpoint", default_chat)
        self._generate_endpoint = kwargs.pop("generate_endpoint", default_gen)
        self._response_content_path = kwargs.pop(
            "response_content_path", default_content_path
        )
        self._api_style = api_style

        # 自定义请求模板
        self._request_template = kwargs.pop("request_template", None)

        # 生成默认值
        self._temperature = kwargs.pop("temperature", 0.7)
        self._max_tokens = kwargs.pop("max_tokens", 1024)
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

        # 验证 requests 是否可用
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
    #  核心生成方法
    # ------------------------------------------------------------------ #

    @retry_on_error
    def generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        """从提示词生成文本"""
        if self._api_style == "ollama":
            return self._ollama_generate(prompt, **kwargs)
        # 默认：使用对话风格的单个用户消息
        messages = [{"role": "user", "content": prompt}]
        return self._chat_request(messages, **kwargs)

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """多轮对话完成"""
        return self._chat_request(messages, **kwargs)

    # ------------------------------------------------------------------ #
    #  流式输出
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
                    results[idx] = "[Error: {}]".format(e)
        return results

    # ------------------------------------------------------------------ #
    #  健康检查
    # ------------------------------------------------------------------ #

    def health_check(self):
        # type: () -> Dict[str, Any]
        import requests

        start = time.time()
        try:
            # 尝试对基础 URL 发送简单 GET 请求
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
        return "CustomAPILLM(model='{}', url='{}')".format(
            self.model_name, self.base_url
        )

    # ------------------------------------------------------------------ #
    #  内部 — 请求构建器
    # ------------------------------------------------------------------ #

    def _chat_request(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """发送非流式对话请求"""
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
            # OpenAI 兼容格式
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

        # 尝试从响应中获取用量信息
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0) or estimate_tokens(str(messages))
        completion_tokens = usage.get("completion_tokens", 0) or estimate_tokens(content)
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    def _ollama_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        """Ollama 风格的 /api/generate 请求"""
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
        """发送基于 SSE 的流式请求"""
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
            # 处理 SSE 格式: "data: {...}"
            if line_str.startswith("data: "):
                line_str = line_str[6:]
            if line_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(line_str)
                # 尝试 OpenAI SSE 格式
                text = _extract_json_path(
                    data, "choices.0.delta.content", ""
                )
                if not text:
                    # 尝试 Ollama 流式格式
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
        递归渲染请求模板字典，替换占位符。

        支持字符串值中的 ``{prompt}``、``{messages}``、``{model}``、
        ``{max_tokens}``、``{temperature}`` 占位符。
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
