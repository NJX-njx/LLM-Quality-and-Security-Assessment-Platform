"""
Ollama LLM 提供者 — 通过 Ollama REST API 进行本地模型推理。

零成本本地模型推理，支持 Llama 3、Mistral、Qwen、Gemma、
以及 Ollama 模型库中的任何模型。

功能特性：
    - 无需 API 密钥（完全本地运行）
    - 流式输出支持
    - 多模型管理（拉取、列表、检查可用性）
    - 可配置生成参数（temperature、top_p 等）
    - 首次使用时自动拉取模型（可选）

参考：
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
    Ollama 本地 LLM 提供者。

    通过 REST API 与本地运行的 Ollama 服务器通信。

    参数：
        model_name: 模型标签（默认: ``"llama3.2"``）。
            使用 ``ollama list`` 查看可用模型。
        base_url: Ollama 服务器 URL（默认: ``"http://localhost:11434"``）。
        timeout: 请求超时（秒，默认: 300 — CPU 推理可能较慢）。
        temperature: 默认温度（默认: 0.7）。
        num_ctx: 上下文窗口大小（默认: 模型内置值）。
        auto_pull: 未找到时自动拉取模型（默认: ``False``）。
        **kwargs: 其他配置。
    """

    def __init__(self, model_name="llama3.2", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # 向后兼容字段
        self.call_count = 0
        self.total_tokens = 0

        # 用量跟踪（本地运行 — 零成本）
        self._usage = UsageStats()

        self.base_url = kwargs.pop("base_url", "http://localhost:11434")
        self._timeout = kwargs.pop("timeout", 300)
        self._temperature = kwargs.pop("temperature", 0.7)
        self._num_ctx = kwargs.pop("num_ctx", None)
        self._auto_pull = kwargs.pop("auto_pull", False)

        # 重试（用于瞬态连接错误）
        self._retry_config = RetryConfig(
            max_retries=kwargs.pop("max_retries", 2),
            base_delay=0.5,
        )

        # 速率限制器（本地通常不需要，但可用）
        rl_cfg = kwargs.pop("rate_limit", RateLimitConfig(enabled=False))
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # 延迟导入
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError(
                "Ollama provider requires the 'requests' package.\n"
                "Install it with: pip install requests>=2.28.0"
            )

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
        """使用 Ollama 的 /api/generate 端点生成响应"""
        import requests

        self._ensure_model_available()
        self._rate_limiter.acquire()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(kwargs),
        }

        # 可选系统提示词
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

        # Ollama 在可用时返回 eval_count / prompt_eval_count
        prompt_tokens = data.get("prompt_eval_count", 0) or estimate_tokens(prompt)
        completion_tokens = data.get("eval_count", 0) or estimate_tokens(content)
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """使用 Ollama 的 /api/chat 端点进行多轮对话"""
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
    #  流式输出
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """通过 Ollama 流式 API 生成响应"""
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

            # 最后一个 chunk 包含用量统计
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
        """通过 Ollama 流式 API 进行对话"""
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
    #  批量生成
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=2, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """
        批量生成。

        注意：Ollama 在大多数硬件上顺序处理，``max_workers``
        应保持较低（1-2）以避免竞争。
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
    #  健康检查和模型管理
    # ------------------------------------------------------------------ #

    def health_check(self):
        # type: () -> Dict[str, Any]
        """检查 Ollama 服务器连接性和模型可用性"""
        import requests

        start = time.time()
        try:
            # 检查服务器
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
                "available_models": models[:10],  # 前 10 个
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
        """列出所有本地可用的 Ollama 模型"""
        import requests

        resp = requests.get(
            "{}/api/tags".format(self.base_url),
            timeout=10,
        )
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def pull_model(self, model_name=None):
        # type: (Optional[str]) -> None
        """从 Ollama 模型库拉取（下载）模型"""
        import requests

        model = model_name or self.model_name
        logger.info("Pulling model '%s' — this may take a while...", model)

        resp = requests.post(
            "{}/api/pull".format(self.base_url),
            json={"name": model, "stream": False},
            timeout=3600,  # 模型可能很大
        )
        resp.raise_for_status()
        logger.info("Model '%s' pulled successfully.", model)

    # ------------------------------------------------------------------ #
    #  统计信息
    # ------------------------------------------------------------------ #

    def get_stats(self):
        # type: () -> Dict[str, Any]
        stats = self._usage.to_dict()
        stats["model_name"] = self.model_name
        stats["call_count"] = self._usage.total_calls
        stats["total_tokens"] = self._usage.total_tokens
        stats["total_cost_usd"] = 0.0  # 本地始终免费
        return stats

    def __repr__(self):
        return "OllamaLLM(model='{}', url='{}')".format(
            self.model_name, self.base_url
        )

    # ------------------------------------------------------------------ #
    #  内部实现
    # ------------------------------------------------------------------ #

    def _ensure_model_available(self):
        # type: () -> None
        """如果已配置且模型不存在，则自动拉取"""
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
            pass  # 将在后续操作中以更清晰的错误失败

    def _build_options(self, kwargs):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        """构建 Ollama 特定的生成选项"""
        options = {}

        temperature = kwargs.pop("temperature", self._temperature)
        options["temperature"] = temperature

        if self._num_ctx:
            options["num_ctx"] = self._num_ctx

        # 将通用参数映射到 Ollama 选项名称
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
        # 本地模型成本始终为 0
        self.call_count = self._usage.total_calls
        self.total_tokens = self._usage.total_tokens
