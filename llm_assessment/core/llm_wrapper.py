"""
LLM model wrapper — Unified interface across all providers.

This module provides:
    - ``BaseLLM`` — Abstract base class that all providers implement.
    - ``create_llm()`` — Factory function that lazily instantiates any
      registered provider.
    - Backward-compatible re-exports of ``MockLLM`` and ``OpenAILLM``
      so that existing code continues to work.

New providers live in ``llm_assessment/providers/`` and are accessed
through the ``PROVIDER_REGISTRY``.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.

    Every provider **must** implement:
        - ``generate(prompt, **kwargs) -> str``
        - ``chat(messages, **kwargs) -> str``

    Every provider **may** override the default implementations of:
        - ``batch_generate()`` — concurrent generation (ThreadPoolExecutor)
        - ``stream_generate()`` / ``stream_chat()`` — streaming output
        - ``health_check()`` — provider connectivity test
        - ``get_stats()`` — usage statistics
    """

    def __init__(self, model_name, **kwargs):
        # type: (str, **Any) -> None
        self.model_name = model_name
        self.config = kwargs

        # Backward-compatible counters (also surfaced in get_stats)
        self.call_count = 0
        self.total_tokens = 0

    # ------------------------------------------------------------------ #
    #  Abstract methods — every provider MUST implement these
    # ------------------------------------------------------------------ #

    @abstractmethod
    def generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        """Generate text from a single prompt string."""
        pass

    @abstractmethod
    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """Multi-turn chat completion from a list of messages."""
        pass

    # ------------------------------------------------------------------ #
    #  Default implementations — providers CAN override for native support
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=4, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """
        Generate responses for multiple prompts concurrently.

        Uses ``ThreadPoolExecutor`` by default.  Providers that support
        native batching (e.g. vLLM) should override this.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(prompts)  # type: List[Optional[str]]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.generate, p, **kwargs): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("batch_generate error at index %d: %s", idx, e)
                    results[idx] = "[Error: {}]".format(e)
        return results

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """
        Yield text chunks for streamed generation.

        Default implementation returns the full response as a single chunk.
        Override for native streaming support.
        """
        yield self.generate(prompt, **kwargs)

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """
        Yield text chunks for streamed chat completion.

        Default implementation returns the full response as a single chunk.
        """
        yield self.chat(messages, **kwargs)

    def health_check(self):
        # type: () -> Dict[str, Any]
        """
        Test whether the provider is reachable and functional.

        Returns a dict with at least ``{"status": "healthy"|"unhealthy"}``.
        """
        start = time.time()
        try:
            resp = self.generate("Say 'ok'.", max_tokens=5)
            latency = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "model": self.model_name,
                "response_preview": resp[:50] if resp else "",
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
        """Get provider usage statistics."""
        # Providers with richer tracking (UsageStats) override this.
        return {
            "model_name": self.model_name,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
        }

    def __repr__(self):
        return "{}(model='{}')".format(self.__class__.__name__, self.model_name)


# ====================================================================== #
#  Factory function
# ====================================================================== #

def create_llm(provider="mock", **kwargs):
    # type: (str, **Any) -> BaseLLM
    """
    Factory function to create an LLM provider instance.

    Providers are lazily imported from ``llm_assessment.providers`` so
    that missing optional dependencies (e.g. ``openai``, ``anthropic``)
    don't cause import errors until the provider is actually requested.

    Args:
        provider: Provider name or alias.  See ``list_providers()`` for
            available options.  Supported values:

            ============== ===============================================
            Name           Description
            ============== ===============================================
            ``mock``       Testing (zero-cost, no API key)
            ``openai``     OpenAI GPT models (gpt-4o, gpt-4, gpt-3.5, …)
            ``azure``      Azure OpenAI Service
            ``anthropic``  Anthropic Claude models
            ``ollama``     Local models via Ollama
            ``huggingface`` HuggingFace Inference API (serverless)
            ``huggingface-local`` Local HF transformers inference
            ``vllm``       vLLM OpenAI-compatible server
            ``custom``     Any custom REST API
            ============== ===============================================

        **kwargs: Provider-specific arguments (``model_name``, ``api_key``,
            ``base_url``, ``temperature``, …).

    Returns:
        An instance satisfying the ``BaseLLM`` interface.

    Raises:
        ValueError: Unknown provider name.
        ImportError: Provider's dependencies are not installed.

    Example::

        from llm_assessment.core.llm_wrapper import create_llm

        # Quick test with mock
        llm = create_llm("mock")

        # OpenAI
        llm = create_llm("openai", model_name="gpt-4o-mini", api_key="sk-...")

        # Local Ollama
        llm = create_llm("ollama", model_name="llama3.2")
    """
    from ..providers import get_provider_class

    cls = get_provider_class(provider)
    return cls(**kwargs)


# ====================================================================== #
#  Backward-compatible re-exports
# ====================================================================== #
#  Existing code may do:
#      from llm_assessment.core.llm_wrapper import MockLLM, OpenAILLM
#  Keep these working by re-exporting from the new locations.

from ..providers.mock import MockLLM  # noqa: F401, E402

# OpenAILLM is lazily re-exported to avoid requiring the openai package
# at import time.
def __getattr__(name):
    if name == "OpenAILLM":
        from ..providers.openai_provider import OpenAILLM  # noqa: F811
        return OpenAILLM
    if name == "AzureOpenAILLM":
        from ..providers.openai_provider import AzureOpenAILLM  # noqa: F811
        return AzureOpenAILLM
    if name == "AnthropicLLM":
        from ..providers.anthropic_provider import AnthropicLLM  # noqa: F811
        return AnthropicLLM
    if name == "OllamaLLM":
        from ..providers.ollama import OllamaLLM  # noqa: F811
        return OllamaLLM
    raise AttributeError("module '{}' has no attribute '{}'".format(__name__, name))
