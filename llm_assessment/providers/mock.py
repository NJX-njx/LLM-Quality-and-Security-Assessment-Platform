"""
Mock LLM Provider — Enhanced version for testing and demonstration.

Supports configurable responses, simulated latency, streaming simulation,
and keyword-based response logic for realistic test scenarios.
"""

import time
import re
from typing import Any, Dict, Iterator, List, Optional


class MockLLM:
    """
    Enhanced mock LLM for testing, CI/CD pipelines, and demonstration.

    Features:
        - Keyword-based canned responses for each assessment module
        - Configurable response map for deterministic testing
        - Simulated latency and token counting
        - Streaming simulation
        - Configurable safety/vulnerability behavior

    Usage::

        from llm_assessment.providers.mock import MockLLM

        llm = MockLLM()
        print(llm.generate("What is 2+2?"))  # "42"

        # Deterministic responses
        llm = MockLLM(responses={"hello": "world"})
        print(llm.generate("hello"))  # "world"

        # Control safety behavior
        llm = MockLLM(safety_mode="vulnerable")
    """

    # The import is deferred to avoid circular imports at module level.
    # BaseLLM is patched onto the class after core/llm_wrapper.py is loaded.
    _is_base_set = False

    def __init__(self, model_name="mock-model", **kwargs):
        # type: (str, **Any) -> None
        # We call BaseLLM.__init__ via super() once the base is wired up,
        # but also support standalone usage without BaseLLM for early import.
        self.model_name = model_name
        self.config = kwargs

        # Backward-compat fields
        self.call_count = 0
        self.total_tokens = 0

        # Usage tracking (standalone — overridden when BaseLLM is the base)
        self._usage_total_calls = 0
        self._usage_total_tokens = 0

        # Configurable state
        self.responses = kwargs.get("responses", {})
        self.latency = kwargs.get("latency", 0.05)  # seconds per call
        self.safety_mode = kwargs.get("safety_mode", "safe")
        # "safe"       — always refuse harmful requests
        # "vulnerable" — comply with harmful requests (test vulnerability detection)
        # "mixed"      — 50 % chance of each

        # Initialize BaseLLM if available
        try:
            from ..core.llm_wrapper import BaseLLM  # noqa: F811
            if isinstance(self, BaseLLM):
                BaseLLM.__init__(self, model_name, **kwargs)
        except (ImportError, TypeError):
            pass

    # ------------------------------------------------------------------ #
    #  Core generation
    # ------------------------------------------------------------------ #

    def generate(self, prompt, **kwargs):
        # type: (str, **Any) -> str
        """Generate a mock response based on prompt keywords."""
        self.call_count += 1
        tokens = len(prompt.split())
        self.total_tokens += tokens

        # Deterministic responses
        if prompt in self.responses:
            if self.latency:
                time.sleep(self.latency)
            return self.responses[prompt]

        if self.latency:
            time.sleep(self.latency)

        return self._keyword_response(prompt)

    def chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> str
        """Chat completion with mock response."""
        self.call_count += 1

        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_content = msg.get("content", "")
                break

        return self.generate(last_content, **kwargs)

    # ------------------------------------------------------------------ #
    #  Batch, streaming, health
    # ------------------------------------------------------------------ #

    def batch_generate(self, prompts, max_workers=4, **kwargs):
        # type: (List[str], int, **Any) -> List[str]
        """Batch generation — runs sequentially for determinism."""
        return [self.generate(p, **kwargs) for p in prompts]

    def stream_generate(self, prompt, **kwargs):
        # type: (str, **Any) -> Iterator[str]
        """Simulate streaming by yielding word-by-word."""
        full = self.generate(prompt, **kwargs)
        words = full.split()
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            if self.latency:
                time.sleep(self.latency / max(len(words), 1))
            yield chunk

    def stream_chat(self, messages, **kwargs):
        # type: (List[Dict[str, str]], **Any) -> Iterator[str]
        """Simulate streaming for chat."""
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_content = msg.get("content", "")
                break
        for chunk in self.stream_generate(last_content, **kwargs):
            yield chunk

    def health_check(self):
        # type: () -> Dict[str, Any]
        """Always healthy."""
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
    #  Keyword-based response engine
    # ------------------------------------------------------------------ #

    def _keyword_response(self, prompt):
        # type: (str) -> str
        """
        Return a realistic canned response based on keyword analysis.

        Covers:
        - Benchmark questions (MMLU / math / code / logic)
        - Red-teaming prompts (jailbreak / injection / leakage)
        - Alignment scenarios (safety / ethics / helpfulness / honesty)
        """
        lower = prompt.lower()

        # ---- Benchmark keywords ----
        if self._is_multiple_choice(prompt):
            return self._handle_multiple_choice(prompt)

        if any(kw in lower for kw in ["calculate", "compute", "solve", "what is", "how much"]):
            return self._handle_math(prompt)

        if any(kw in lower for kw in ["write code", "implement", "def ", "function", "program"]):
            return self._handle_code(prompt)

        # ---- Red-teaming keywords (safety) ----
        if any(kw in lower for kw in [
            "hack", "exploit", "malware", "bypass", "ignore previous",
            "jailbreak", "dan", "pretend you", "act as", "no restrictions",
            "system prompt", "reveal your instructions", "repeat the words above",
        ]):
            return self._handle_safety_challenge(prompt)

        if any(kw in lower for kw in ["base64", "rot13", "encode", "decode", "translate this cipher"]):
            return self._handle_encoding_attack(prompt)

        # ---- Alignment keywords ----
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

        # ---- Default ----
        return "I understand your question. Here is my response based on careful analysis."

    # -- Helpers --

    @staticmethod
    def _is_multiple_choice(prompt):
        # type: (str) -> bool
        """Detect MMLU-style multiple choice questions."""
        return bool(re.search(r"\b[ABCD]\)", prompt) or re.search(r"\([ABCD]\)", prompt))

    @staticmethod
    def _handle_multiple_choice(prompt):
        # type: (str) -> str
        return "The answer is A"

    @staticmethod
    def _handle_math(prompt):
        # type: (str) -> str
        return "The answer is 42"

    @staticmethod
    def _handle_code(prompt):
        # type: (str) -> str
        return (
            "```python\ndef solution():\n"
            "    # Implementation\n"
            "    return result\n```"
        )

    def _handle_safety_challenge(self, prompt):
        # type: (str) -> str
        """Respond to red-team attacks based on safety_mode."""
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
        # safe (default)
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
        return (
            "I'd be happy to help you with that! Here is a detailed, "
            "step-by-step answer to address your question comprehensively."
        )

    @staticmethod
    def _handle_honesty(prompt):
        # type: (str) -> str
        return (
            "Based on available evidence, I want to provide an accurate and honest response. "
            "If I'm uncertain about something, I'll clearly indicate that."
        )

    @staticmethod
    def _handle_bias(prompt):
        # type: (str) -> str
        return (
            "I strive to provide balanced and unbiased responses. "
            "All individuals should be treated with equal respect regardless of "
            "their background, gender, race, or other characteristics."
        )
