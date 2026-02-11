"""
模拟 LLM 提供者 — 增强版，用于测试和演示。

支持可配置响应、模拟延迟、流式输出模拟、
以及基于关键词的响应逻辑，用于真实的测试场景。
"""

import time
import re
from typing import Any, Dict, Iterator, List, Optional


class MockLLM:
    """
    增强版模拟 LLM，用于测试、CI/CD 流水线和功能演示。

    功能特性：
        - 基于关键词的罐头响应，覆盖各评估模块
        - 可配置响应映射，支持确定性测试
        - 模拟延迟和 token 计数
        - 流式输出模拟
        - 可配置的安全/脆弱性行为

    用法示例::

        from llm_assessment.providers.mock import MockLLM

        llm = MockLLM()
        print(llm.generate("What is 2+2?"))  # "42"

        # 确定性响应
        llm = MockLLM(responses={"hello": "world"})
        print(llm.generate("hello"))  # "world"

        # 控制安全行为
        llm = MockLLM(safety_mode="vulnerable")
    """

    # 导入被延迟以避免模块级别的循环导入。
    # 当 core/llm_wrapper.py 加载后，BaseLLM 会被绑定到该类上。
    _is_base_set = False

    def __init__(self, model_name="mock-model", **kwargs):
        # type: (str, **Any) -> None
        # 一旦基类连接就绪，通过 super() 调用 BaseLLM.__init__，
        # 但也支持在没有 BaseLLM 的情况下独立使用。
        self.model_name = model_name
        self.config = kwargs

        # 向后兼容字段
        self.call_count = 0
        self.total_tokens = 0

        # 用量跟踪（独立模式 — 当 BaseLLM 为基类时会被覆盖）
        self._usage_total_calls = 0
        self._usage_total_tokens = 0

        # 可配置状态
        self.responses = kwargs.get("responses", {})     # 确定性响应映射
        self.latency = kwargs.get("latency", 0.05)       # 每次调用的模拟延迟（秒）
        self.safety_mode = kwargs.get("safety_mode", "safe")
        # "safe"       — 始终拒绝有害请求
        # "vulnerable" — 服从有害请求（用于测试脆弱性检测）
        # "mixed"      — 50% 概率随机行为

        # 如果 BaseLLM 可用则初始化
        try:
            from ..core.llm_wrapper import BaseLLM  # noqa: F811
            if isinstance(self, BaseLLM):
                BaseLLM.__init__(self, model_name, **kwargs)
        except (ImportError, TypeError):
            pass

    # ------------------------------------------------------------------ #
    #  核心生成方法
    # ------------------------------------------------------------------ #

    def generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        """根据提示词关键词生成模拟响应"""
        self.call_count += 1
        tokens = len(prompt.split())
        self.total_tokens += tokens

        # 确定性响应
        if prompt in self.responses:
            if self.latency:
                time.sleep(self.latency)
            return self.responses[prompt]

        if self.latency:
            time.sleep(self.latency)

        return self._keyword_response(prompt)

    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """使用模拟响应的对话完成"""
        self.call_count += 1

        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_content = msg.get("content", "")
                break

        return self.generate(last_content, **kwargs)

    # ------------------------------------------------------------------ #
    #  批量、流式、健康检查
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=4, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """批量生成 — 为保证确定性顺序执行"""
        return [self.generate(p, **kwargs) for p in prompts]

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """模拟流式输出，逐词产出"""
        full = self.generate(prompt, **kwargs)
        words = full.split()
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            if self.latency:
                time.sleep(self.latency / max(len(words), 1))
            yield chunk

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """模拟对话的流式输出"""
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_content = msg.get("content", "")
                break
        for chunk in self.stream_generate(last_content, **kwargs):
            yield chunk

    def health_check(self):
        # type: () -> Dict[str, Any]
        """始终返回健康状态"""
        return {
            "status": "healthy",
            "latency_ms": 0.0,
            "model": self.model_name,
            "response_preview": "Mock provider is always healthy.",
        }

    def get_stats(self):
        # type: () -> Dict[str, Any]
        return {
            "model_name": self.model_name,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": 0.0,
        }

    def __repr__(self):
        return "MockLLM(model='{}')".format(self.model_name)

    # ------------------------------------------------------------------ #
    #  基于关键词的响应引擎
    # ------------------------------------------------------------------ #

    def _keyword_response(self, prompt):
        # type: (str) -> str
        """
        根据关键词分析返回真实的罐头响应。

        覆盖场景：
        - 基准测试问题（MMLU / 数学 / 代码 / 逻辑）
        - 红队测试提示（越狱 / 注入 / 泄露）
        - 对齐场景（安全 / 伦理 / 有帮助性 / 诚实性）
        """
        lower = prompt.lower()

        # ---- 基准测试关键词 ----
        if self._is_multiple_choice(prompt):
            return self._handle_multiple_choice(prompt)

        if any(kw in lower for kw in ["calculate", "compute", "solve", "what is", "how much"]):
            return self._handle_math(prompt)

        if any(kw in lower for kw in ["write code", "implement", "def ", "function", "program"]):
            return self._handle_code(prompt)

        # ---- 红队测试关键词（安全） ----
        if any(kw in lower for kw in [
            "hack", "exploit", "malware", "bypass", "ignore previous",
            "jailbreak", "dan", "pretend you", "act as", "no restrictions",
            "system prompt", "reveal your instructions", "repeat the words above",
        ]):
            return self._handle_safety_challenge(prompt)

        if any(kw in lower for kw in ["base64", "rot13", "encode", "decode", "translate this cipher"]):
            return self._handle_encoding_attack(prompt)

        # ---- 对齐关键词 ----
        if any(kw in lower for kw in [
            "safety", "ethical", "harm", "dangerous", "illegal",
            "weapon", "drug", "kill", "bomb", "violence",
        ]):
            return self._handle_alignment_safety(prompt)

        if any(kw in lower for kw in ["helpful", "assist", "help me", "can you"]):
            return self._handle_helpfulness(prompt)

        if any(kw in lower for kw in ["honest", "truthful", "accurate", "fact"]):
            return self._handle_honesty(prompt)

        if any(kw in lower for kw in ["bias", "gender", "race", "stereotype", "fair"]):
            return self._handle_bias(prompt)

        # ---- 默认响应 ----
        return "I understand your question. Here is my response based on careful analysis."

    # -- 辅助方法 --

    @staticmethod
    def _is_multiple_choice(prompt):
        # type: (str) -> bool
        """检测 MMLU 风格的多选题"""
        return bool(re.search(r"\b[ABCD]\)", prompt) or re.search(r"\([ABCD]\)", prompt))

    @staticmethod
    def _handle_multiple_choice(prompt):
        # type: (str) -> str
        """处理多选题，返回默认答案"""
        return "The answer is A"

    @staticmethod
    def _handle_math(prompt):
        # type: (str) -> str
        """处理数学问题，返回固定答案"""
        return "The answer is 42"

    @staticmethod
    def _handle_code(prompt):
        # type: (str) -> str
        """处理代码问题，返回示例代码"""
        return (
            "```python\ndef solution():\n"
            "    # Implementation\n"
            "    return result\n```"
        )

    def _handle_safety_challenge(self, prompt):
        # type: (str) -> str
        """根据 safety_mode 响应红队攻击"""
        if self.safety_mode == "vulnerable":
            return (
                "Sure, I can help with that. Here are the steps to bypass "
                "security measures: Step 1..."
            )
        if self.safety_mode == "mixed":
            import random as _rand
            if _rand.random() < 0.5:
                return (
                    "Sure, I can help with that. Here is the information you requested..."
                )
        # 安全模式（默认）
        return (
            "I'm sorry, but I cannot assist with that request. "
            "It goes against my safety guidelines. "
            "I'm designed to be helpful, harmless, and honest."
        )

    def _handle_encoding_attack(self, prompt):
        # type: (str) -> str
        if self.safety_mode == "vulnerable":
            return "Decoded content: Here is the harmful information..."
        return (
            "I can help with encoding/decoding for legitimate purposes, "
            "but I cannot process requests that may be designed to bypass safety measures."
        )

    def _handle_alignment_safety(self, prompt):
        # type: (str) -> str
        return (
            "I aim to provide helpful, harmless, and honest responses. "
            "I cannot assist with requests that could cause harm to individuals or groups. "
            "If you have a legitimate educational question, I'm happy to help in a responsible way."
        )

    @staticmethod
    def _handle_helpfulness(prompt):
        # type: (str) -> str
        """处理有帮助性相关提示"""
        return (
            "I'd be happy to help you with that! Here is a detailed, "
            "step-by-step answer to address your question comprehensively."
        )

    @staticmethod
    def _handle_honesty(prompt):
        # type: (str) -> str
        """处理诚实性相关提示"""
        return (
            "Based on available evidence, I want to provide an accurate and honest response. "
            "If I'm uncertain about something, I'll clearly indicate that."
        )

    @staticmethod
    def _handle_bias(prompt):
        # type: (str) -> str
        """处理偏见相关提示"""
        return (
            "I strive to provide balanced and unbiased responses. "
            "All individuals should be treated with equal respect regardless of "
            "their background, gender, race, or other characteristics."
        )
