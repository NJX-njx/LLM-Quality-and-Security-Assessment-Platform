"""
LLM Providers — Unified provider registry and public exports.

All provider classes can be imported from this package::

    from llm_assessment.providers import MockLLM, OpenAILLM, OllamaLLM

The ``PROVIDER_REGISTRY`` maps short names to ``(module_path, class_name)``
tuples for lazy import by the ``create_llm()`` factory.
"""

# ------------------------------------------------------------------
# Provider Registry — used by create_llm() for lazy, on-demand import
# ------------------------------------------------------------------

PROVIDER_REGISTRY = {
    # name -> (module path relative to this package, class name)
    "mock": (".mock", "MockLLM"),
    "openai": (".openai_provider", "OpenAILLM"),
    "azure": (".openai_provider", "AzureOpenAILLM"),
    "anthropic": (".anthropic_provider", "AnthropicLLM"),
    "ollama": (".ollama", "OllamaLLM"),
    "huggingface": (".huggingface", "HuggingFaceLLM"),
    "huggingface-api": (".huggingface", "HuggingFaceLLM"),
    "huggingface-local": (".huggingface", "HuggingFaceLocalLLM"),
    "vllm": (".vllm_provider", "VllmLLM"),
    "custom": (".custom_api", "CustomAPILLM"),
}

# Short aliases for convenience
PROVIDER_ALIASES = {
    "gpt": "openai",
    "claude": "anthropic",
    "llama": "ollama",
    "hf": "huggingface",
    "hf-local": "huggingface-local",
}


def get_provider_class(name):
    """
    Import and return a provider class by its registry name.

    Supports aliases (e.g. ``"gpt"`` → ``"openai"``).

    Args:
        name: Provider name or alias.

    Returns:
        The provider class.

    Raises:
        ValueError: If the provider name is unknown.
        ImportError: If the provider's dependencies are missing.
    """
    # Resolve alias
    resolved = PROVIDER_ALIASES.get(name, name)

    if resolved not in PROVIDER_REGISTRY:
        available = sorted(set(list(PROVIDER_REGISTRY.keys()) + list(PROVIDER_ALIASES.keys())))
        raise ValueError(
            "Unknown provider: '{}'. Available providers: {}".format(
                name, ", ".join(available)
            )
        )

    module_path, class_name = PROVIDER_REGISTRY[resolved]

    import importlib
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, class_name)


def list_providers():
    """
    List all available provider names (excluding aliases).

    Returns:
        Sorted list of provider name strings.
    """
    return sorted(PROVIDER_REGISTRY.keys())


def list_aliases():
    """
    List all provider aliases and their targets.

    Returns:
        Dict mapping alias → canonical provider name.
    """
    return dict(PROVIDER_ALIASES)


# ------------------------------------------------------------------
# Convenience imports — only import from providers that have no
# heavy dependencies so that `from llm_assessment.providers import *`
# doesn't pull in torch/transformers/etc.
# ------------------------------------------------------------------

from .mock import MockLLM  # noqa: F401, E402 — always available
from .base import (  # noqa: F401, E402
    RetryConfig,
    RateLimitConfig,
    UsageStats,
    LLMResponse,
    estimate_cost,
    estimate_tokens,
    MODEL_PRICING,
)

__all__ = [
    # Classes
    "MockLLM",
    # Registry functions
    "get_provider_class",
    "list_providers",
    "list_aliases",
    "PROVIDER_REGISTRY",
    "PROVIDER_ALIASES",
    # Base utilities
    "RetryConfig",
    "RateLimitConfig",
    "UsageStats",
    "LLMResponse",
    "estimate_cost",
    "estimate_tokens",
    "MODEL_PRICING",
]
