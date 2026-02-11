"""
HuggingFace LLM Providers — Inference API and local transformers.

Two operating modes:
    1. **HuggingFaceLLM** — Serverless Inference API (no GPU required locally)
    2. **HuggingFaceLocalLLM** — Local inference via ``transformers`` + ``torch``

Features:
    - Supports any text-generation model on the HuggingFace Hub
    - Streaming support (API mode)
    - Configurable generation parameters
    - Cost tracking for Inference API (PRO endpoints)

Reference:
    - Inference API: https://huggingface.co/docs/api-inference/
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
    HuggingFace Inference API provider (serverless).

    Uses ``huggingface_hub.InferenceClient`` to call models hosted on
    HuggingFace's infrastructure.  Supports free-tier models and
    dedicated / PRO inference endpoints.

    Args:
        model_name: HuggingFace model ID or Inference Endpoint URL
            (default: ``"meta-llama/Llama-3.2-3B-Instruct"``).
        api_key: HuggingFace API token.  Falls back to ``HF_TOKEN`` env var.
        timeout: Request timeout in seconds (default: 120).
        temperature: Default temperature (default: 0.7).
        max_tokens: Default max new tokens (default: 1024).
        rate_limit: ``RateLimitConfig`` or dict.
        **kwargs: Additional params passed to the inference client.

    Example::

        from llm_assessment.providers.huggingface import HuggingFaceLLM

        llm = HuggingFaceLLM(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            api_key="hf_...",
        )
        print(llm.generate("Explain gravity."))
    """

    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # Backward-compat
        self.call_count = 0
        self.total_tokens = 0

        self._usage = UsageStats()

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

        # ---- Create InferenceClient ----
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
    #  Core generation
    # ------------------------------------------------------------------ #

    @retry_on_error
    def generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        """Generate using the text-generation API."""
        self._rate_limiter.acquire(estimate_tokens(prompt))

        gen_kwargs = self._build_gen_kwargs(kwargs)

        start = time.time()
        response = self.client.text_generation(
            prompt,
            **gen_kwargs,
        )
        latency_ms = (time.time() - start) * 1000

        # text_generation returns the generated text directly
        content = response if isinstance(response, str) else str(response)

        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(content)
        self._track_usage(prompt_tokens, completion_tokens, latency_ms)

        return content

    @retry_on_error
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """Chat completion using chat_completion API."""
        self._rate_limiter.acquire(estimate_tokens(str(messages)))

        gen_kwargs = self._build_gen_kwargs(kwargs)

        start = time.time()
        response = self.client.chat_completion(
            messages=messages,
            **gen_kwargs,
        )
        latency_ms = (time.time() - start) * 1000

        # Extract content from ChatCompletionOutput
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
    #  Streaming
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """Stream text generation."""
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
        """Stream chat completion."""
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
    #  Internals
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
#  Local Transformers Provider
# ====================================================================== #

class HuggingFaceLocalLLM:
    """
    Local HuggingFace transformers provider.

    Uses ``transformers.pipeline`` for local model inference.
    Requires ``torch`` + ``transformers`` installed.

    Args:
        model_name: HuggingFace model ID (default: ``"microsoft/Phi-3-mini-4k-instruct"``).
        device: Device to load the model on (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
        torch_dtype: Dtype for model weights (``"auto"``, ``"float16"``, ``"bfloat16"``).
        temperature: Default temperature (default: 0.7).
        max_tokens: Default max new tokens (default: 512).
        load_in_4bit: Enable 4-bit quantization via bitsandbytes.
        load_in_8bit: Enable 8-bit quantization.
        **kwargs: Extra args passed to ``pipeline()`` or ``from_pretrained()``.

    Example::

        from llm_assessment.providers.huggingface import HuggingFaceLocalLLM

        llm = HuggingFaceLocalLLM(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            device="auto",
            torch_dtype="float16",
        )
        print(llm.generate("What is deep learning?"))
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

        # Rate limiter (disabled by default for local)
        rl_cfg = kwargs.pop("rate_limit", RateLimitConfig(enabled=False))
        if isinstance(rl_cfg, dict):
            rl_cfg = RateLimitConfig(**rl_cfg)
        self._rate_limiter = TokenBucketRateLimiter(rl_cfg)

        # Lazy model loading — deferred until first call
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
        """Lazy-load the model and tokenizer on first use."""
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

        # Resolve torch dtype
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

        # Apply chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: concatenate messages
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
    # Streaming (local transformers TextIteratorStreamer)
    # ------------------------------------------------------------------ #

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """
        Stream generation using TextIteratorStreamer.

        Falls back to non-streaming if TextIteratorStreamer is not available.
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
        """Stream chat — converts to prompt and streams."""
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
    #  Batch
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=1, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """Sequential batch — GPU contention makes parallelism harmful."""
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
        stats["total_cost_usd"] = 0.0  # Local inference = free
        return stats

    def __repr__(self):
        return "HuggingFaceLocalLLM(model='{}', device='{}')".format(
            self.model_name, self._device
        )

    # ------------------------------------------------------------------ #
    #  Internal
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
