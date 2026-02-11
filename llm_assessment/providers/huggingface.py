"""
HuggingFace LLM 提供者 — 推理 API 和本地 transformers。

两种运行模式：
    1. **HuggingFaceLLM** — 无服务器推理 API（本地无需 GPU）
    2. **HuggingFaceLocalLLM** — 通过 ``transformers`` + ``torch`` 本地推理

功能特性：
    - 支持 HuggingFace Hub 上的任何文本生成模型
    - 流式输出支持（API 模式）
    - 可配置生成参数
    - 推理 API 成本跟踪（PRO 端点）

参考：
    - 推理 API: https://huggingface.co/docs/api-inference/
    - huggingface_hub: https://huggingface.co/docs/huggingface_hub/
    - transformers: https://huggingface.co/docs/transformers/
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


class HuggingFaceLLM:
    """
    HuggingFace 推理 API 提供者（无服务器模式）。

    使用 ``huggingface_hub.InferenceClient`` 调用托管在
    HuggingFace 基础设施上的模型。支持免费层模型和
    专用 / PRO 推理端点。

    参数：
        model_name: HuggingFace 模型 ID 或推理端点 URL。
        api_key: HuggingFace API token。回退到 ``HF_TOKEN`` 环境变量。
        timeout: 请求超时（秒，默认: 120）。
        temperature: 默认温度（默认: 0.7）。
        max_tokens: 默认最大新 token 数（默认: 1024）。
        rate_limit: ``RateLimitConfig`` 或字典。
        **kwargs: 传递给推理客户端的其他参数。
    """

    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # 向后兼容字段
        self.call_count = 0
        self.total_tokens = 0

        self._usage = UsageStats()

        self._temperature = kwargs.pop("temperature", 0.7)
        self._max_tokens = kwargs.pop("max_tokens", 1024)
        self._timeout = kwargs.pop("timeout", 120)

        # 重试配置
        self._retry_config = RetryConfig(
            max_retries=kwargs.pop("max_retries", 3)
        )

        # Rate limiter
        rl_cfg = kwargs.pop("rate_limit", None)
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # ---- 创建 InferenceClient ----
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "HuggingFace provider requires 'huggingface_hub'.\n"
                "Install it with: pip install huggingface_hub>=0.20.0"
            )

        api_key = kwargs.pop("api_key", None)
        if not api_key:
            import os
            api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        self.client = InferenceClient(
            model=model_name,
            token=api_key,
            timeout=self._timeout,
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
        """使用文本生成 API 生成响应"""
        self._rate_limiter.acquire(estimate_tokens(prompt))

        gen_kwargs = self._build_gen_kwargs(kwargs)

        start = time.time()
        response = self.client.text_generation(
            prompt,
            **gen_kwargs,
        )
        latency_ms = (time.time() - start) * 1000

        # text_generation 直接返回生成的文本
        content = response if isinstance(response, str) else str(response)

        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(content)
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """使用 chat_completion API 进行对话完成"""
        self._rate_limiter.acquire(estimate_tokens(str(messages)))

        gen_kwargs = self._build_gen_kwargs(kwargs)

        start = time.time()
        response = self.client.chat_completion(
            messages=messages,
            **gen_kwargs,
        )
        latency_ms = (time.time() - start) * 1000

        # 从 ChatCompletionOutput 提取内容
        content = ""
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content or ""
        elif isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else estimate_tokens(str(messages))
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else estimate_tokens(content)
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    # ------------------------------------------------------------------ #
    #  流式输出
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """流式文本生成"""
        self._rate_limiter.acquire(estimate_tokens(prompt))
        gen_kwargs = self._build_gen_kwargs(kwargs)

        start = time.time()
        collected = []

        stream = self.client.text_generation(
            prompt,
            stream=True,
            **gen_kwargs,
        )

        for token in stream:
            text = token if isinstance(token, str) else str(token)
            collected.append(text)
            yield text

        latency_ms = (time.time() - start) * 1000
        self._track_usage(
            estimate_tokens(prompt),
            estimate_tokens("".join(collected)),
            latency_ms,
        )

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """流式对话完成"""
        self._rate_limiter.acquire(estimate_tokens(str(messages)))
        gen_kwargs = self._build_gen_kwargs(kwargs)

        start = time.time()
        collected = []

        stream = self.client.chat_completion(
            messages=messages,
            stream=True,
            **gen_kwargs,
        )

        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", "") or ""
                if text:
                    collected.append(text)
                    yield text

        latency_ms = (time.time() - start) * 1000
        self._track_usage(
            estimate_tokens(str(messages)),
            estimate_tokens("".join(collected)),
            latency_ms,
        )

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
        start = time.time()
        try:
            status = self.client.get_model_status(self.model_name)
            latency = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "model_state": str(getattr(status, "state", "unknown")),
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
        return "HuggingFaceLLM(model='{}')".format(self.model_name)

    # ------------------------------------------------------------------ #
    #  内部实现
    # ------------------------------------------------------------------ #

    def _build_gen_kwargs(self, overrides):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        kwargs = {}
        kwargs["temperature"] = overrides.pop("temperature", self._temperature)
        kwargs["max_new_tokens"] = overrides.pop("max_tokens", self._max_tokens)

        for key in ("top_p", "top_k", "repetition_penalty", "stop", "seed"):
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


# ====================================================================== #
#  本地 Transformers 提供者
# ====================================================================== #

class HuggingFaceLocalLLM:
    """
    本地 HuggingFace transformers 提供者。

    使用 ``transformers.pipeline`` 进行本地模型推理。
    需要安装 ``torch`` + ``transformers``。

    参数：
        model_name: HuggingFace 模型 ID（默认: ``"microsoft/Phi-3-mini-4k-instruct"``）。
        device: 加载模型的设备（``"auto"``、``"cpu"``、``"cuda"``、``"mps"``）。
        torch_dtype: 模型权重的数据类型（``"auto"``、``"float16"``、``"bfloat16"``）。
        temperature: 默认温度（默认: 0.7）。
        max_tokens: 默认最大新 token 数（默认: 512）。
        load_in_4bit: 启用 4-bit 量化（通过 bitsandbytes）。
        load_in_8bit: 启用 8-bit 量化。
        **kwargs: 传递给 ``pipeline()`` 或 ``from_pretrained()`` 的额外参数。
    """

    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        self.call_count = 0
        self.total_tokens = 0
        self._usage = UsageStats()

        self._temperature = kwargs.pop("temperature", 0.7)
        self._max_tokens = kwargs.pop("max_tokens", 512)
        self._device = kwargs.pop("device", "auto")
        self._torch_dtype = kwargs.pop("torch_dtype", "auto")
        self._load_in_4bit = kwargs.pop("load_in_4bit", False)
        self._load_in_8bit = kwargs.pop("load_in_8bit", False)

        # 速率限制器（本地默认禁用）
        rl_cfg = kwargs.pop("rate_limit", RateLimitConfig(enabled=False))
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # 延迟模型加载 — 推迟到首次调用
        self._pipeline = None
        self._tokenizer = None
        self._model_kwargs = kwargs

        # BaseLLM
        try:
            from ..core.llm_wrapper import BaseLLM
            if isinstance(self, BaseLLM):
                BaseLLM.__init__(self, model_name, **kwargs)
        except (ImportError, TypeError):
            pass

    def _load_model(self):
        """首次使用时延迟加载模型和分词器"""
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        except ImportError:
            raise ImportError(
                "HuggingFaceLocalLLM requires 'torch' and 'transformers'.\n"
                "Install with: pip install torch transformers"
            )

        logger.info("Loading model '%s' — this may take a while...", self.model_name)

        # 解析 torch 数据类型
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._torch_dtype, "auto")

        model_kwargs = {}
        if self._load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif self._load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self._device,
            **model_kwargs,
        )

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
        )

        logger.info("Model '%s' loaded successfully.", self.model_name)

    # ------------------------------------------------------------------ #
    #  Core generation
    # ------------------------------------------------------------------ #

    def generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        self._load_model()
        self._rate_limiter.acquire()

        temperature = kwargs.pop("temperature", self._temperature)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        start = time.time()
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            return_full_text=False,
            **kwargs,
        )
        latency_ms = (time.time() - start) * 1000

        content = outputs[0]["generated_text"] if outputs else ""

        prompt_tokens = len(self._tokenizer.encode(prompt))
        completion_tokens = len(self._tokenizer.encode(content))
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        self._load_model()
        self._rate_limiter.acquire()

        temperature = kwargs.pop("temperature", self._temperature)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        # 如果有对话模板则应用
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # 回退：拼接消息
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append("{}: {}".format(role, content))
            prompt = "\n".join(parts) + "\nassistant: "

        start = time.time()
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            return_full_text=False,
            **kwargs,
        )
        latency_ms = (time.time() - start) * 1000

        content = outputs[0]["generated_text"] if outputs else ""

        prompt_tokens = len(self._tokenizer.encode(prompt))
        completion_tokens = len(self._tokenizer.encode(content))
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    # ------------------------------------------------------------------ #
    # 流式输出（本地 transformers TextIteratorStreamer）
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """
        使用 TextIteratorStreamer 进行流式生成。

        如果 TextIteratorStreamer 不可用，则回退到非流式模式。
        """
        self._load_model()

        try:
            from transformers import TextIteratorStreamer
            import threading
        except ImportError:
            yield self.generate(prompt, **kwargs)
            return

        temperature = kwargs.pop("temperature", self._temperature)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._pipeline.model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else None,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }

        thread = threading.Thread(
            target=self._pipeline.model.generate, kwargs=generation_kwargs
        )

        start = time.time()
        thread.start()

        collected = []
        for text in streamer:
            collected.append(text)
            yield text

        thread.join()
        latency_ms = (time.time() - start) * 1000

        prompt_tokens = len(self._tokenizer.encode(prompt))
        completion_tokens = len(self._tokenizer.encode("".join(collected)))
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """流式对话 — 转换为提示词并流式输出"""
        self._load_model()

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            parts = []
            for msg in messages:
                parts.append("{}: {}".format(msg.get("role", "user"), msg.get("content", "")))
            prompt = "\n".join(parts) + "\nassistant: "

        for chunk in self.stream_generate(prompt, **kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    #  批量生成 (HuggingFaceLocalLLM)
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=1, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """顺序批量 — GPU 竞争会使并行化适得其反"""
        return [self.generate(p, **kwargs) for p in prompts]

    # ------------------------------------------------------------------ #
    #  Health check
    # ------------------------------------------------------------------ #

    def health_check(self):
        # type: () -> Dict[str, Any]
        start = time.time()
        try:
            self._load_model()
            latency = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "device": str(self._pipeline.device) if self._pipeline else "unknown",
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
        stats["total_cost_usd"] = 0.0  # 本地推理 = 免费
        return stats

    def __repr__(self):
        return "HuggingFaceLocalLLM(model='{}', device='{}')".format(
            self.model_name, self._device
        )

    # ------------------------------------------------------------------ #
    #  内部实现 (HuggingFaceLocalLLM)
    # ------------------------------------------------------------------ #

    def _track_usage(self, prompt_tokens, completion_tokens, latency_ms):
        # type: (int, int, float) -> None
        self._usage.total_calls += 1
        self._usage.total_prompt_tokens += prompt_tokens
        self._usage.total_completion_tokens += completion_tokens
        self._usage.total_tokens += prompt_tokens + completion_tokens
        self._usage.total_latency_ms += latency_ms
        self.call_count = self._usage.total_calls
        self.total_tokens = self._usage.total_tokens
