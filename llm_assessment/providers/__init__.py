"""
LLM 提供者模块 - 统一的提供者注册表和公共导出。

所有提供者类可从此包导入：

    from llm_assessment.providers import MockLLM, OpenAILLM, OllamaLLM

PROVIDER_REGISTRY 将短名称映射到 (module_path, class_name) 元组，
用于 create_llm() 工厂函数的懒加载导入。
"""

# ------------------------------------------------------------------
# 提供者注册表 — create_llm() 用于按需懒加载导入
# ------------------------------------------------------------------

PROVIDER_REGISTRY = {
    # 名称 -> (相对于此包的模块路径, 类名)
    "mock": (".mock", "MockLLM"),                    # 模拟提供者（测试用）
    "openai": (".openai_provider", "OpenAILLM"),      # OpenAI GPT 系列
    "azure": (".openai_provider", "AzureOpenAILLM"),   # Azure OpenAI 服务
    "anthropic": (".anthropic_provider", "AnthropicLLM"),  # Anthropic Claude 系列
    "ollama": (".ollama", "OllamaLLM"),               # Ollama 本地模型
    "huggingface": (".huggingface", "HuggingFaceLLM"), # HuggingFace 推理 API
    "huggingface-api": (".huggingface", "HuggingFaceLLM"),
    "huggingface-local": (".huggingface", "HuggingFaceLocalLLM"),  # HuggingFace 本地推理
    "vllm": (".vllm_provider", "VllmLLM"),            # vLLM 高性能推理
    "custom": (".custom_api", "CustomAPILLM"),         # 自定义 REST API
}

# 短别名映射，方便使用
PROVIDER_ALIASES = {
    "gpt": "openai",           # GPT -> OpenAI
    "claude": "anthropic",     # Claude -> Anthropic
    "llama": "ollama",         # Llama -> Ollama
    "hf": "huggingface",       # HF -> HuggingFace
    "hf-local": "huggingface-local",
}


def get_provider_class(name):
    """
    根据注册名称导入并返回提供者类。

    支持别名（如 "gpt" → "openai"）。

    Args:
        name: 提供者名称或别名。

    Returns:
        提供者类。

    Raises:
        ValueError: 未知的提供者名称。
        ImportError: 提供者的依赖未安装。
    """
    # 解析别名
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
    """列出所有可用的提供者名称（不包含别名）"""
    return sorted(PROVIDER_REGISTRY.keys())


def list_aliases():
    """列出所有提供者别名及其目标"""
    return dict(PROVIDER_ALIASES)


# ------------------------------------------------------------------
# 便捷导入 — 只导入无重依赖的提供者，
# 避免 `from llm_assessment.providers import *` 拉入 torch/transformers 等
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
