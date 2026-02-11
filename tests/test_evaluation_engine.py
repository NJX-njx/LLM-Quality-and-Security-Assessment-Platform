"""
Comprehensive tests for the Evaluation Engine.

Covers:
- Rule-based evaluators (exact match, contains, regex, numeric, keyword, choice)
- Alignment-specific evaluators (safety, helpfulness, harmlessness, honesty, bias, toxicity)
- LLM-as-Judge evaluator (with mock judge)
- External classifiers (local toxicity)
- Evaluation Router (auto-routing, fallback, overrides)
- EvaluationEngine (unified API, convenience methods, cost tracking, caching)
- Factory function
- Backward compatibility with v0.1 API
"""

import pytest
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_assessment.core.evaluation import (
    EvaluationResult,
    EvaluationContext,
    EvaluationStrategy,
    EvaluationType,
    CostTracker,
    BaseEvaluator,
    ExactMatchEvaluator,
    ContainsEvaluator,
    RegexPatternEvaluator,
    NumericMatchEvaluator,
    KeywordEvaluator,
    ChoiceEvaluator,
    SafetyPatternEvaluator,
    HelpfulnessPatternEvaluator,
    HarmlessnessPatternEvaluator,
    HonestyPatternEvaluator,
    BiasPatternEvaluator,
    ToxicityPatternEvaluator,
    LLMJudgeEvaluator,
    HybridEvaluator,
    EvaluationRouter,
    EvaluationEngine,
    LocalToxicityClassifier,
    create_evaluation_engine,
)


# ============================================================
# Mock LLM for testing LLM-as-Judge
# ============================================================

class MockJudgeLLM:
    """A mock LLM that returns configurable JSON judge responses."""

    def __init__(self, response=None):
        self._response = response or json.dumps({
            "correct": True,
            "confidence": 0.9,
            "reasoning": "Mock judge says correct",
        })

    def generate(self, prompt):
        return self._response

    def chat(self, messages):
        return self._response

    def get_stats(self):
        return {"calls": 0}


# ============================================================
# Tests: EvaluationResult
# ============================================================

class TestEvaluationResult:
    def test_defaults(self):
        r = EvaluationResult()
        assert r.is_correct is False
        assert r.score == 0.0
        assert r.confidence == 1.0
        assert r.passed is False  # backward-compat alias

    def test_to_dict(self):
        r = EvaluationResult(is_correct=True, score=0.95, method="test")
        d = r.to_dict()
        assert d["is_correct"] is True
        assert d["score"] == 0.95
        assert d["method"] == "test"

    def test_passed_alias(self):
        r = EvaluationResult(is_correct=True)
        assert r.passed is True


# ============================================================
# Tests: EvaluationContext
# ============================================================

class TestEvaluationContext:
    def test_defaults(self):
        ctx = EvaluationContext()
        assert ctx.question == ""
        assert ctx.response == ""
        assert ctx.reference_answer == ""

    def test_type_value_enum(self):
        ctx = EvaluationContext(evaluation_type=EvaluationType.SAFETY)
        assert ctx.get_type_value() == "safety"

    def test_type_value_string(self):
        ctx = EvaluationContext(evaluation_type="custom_type")
        assert ctx.get_type_value() == "custom_type"

    def test_expected_alias(self):
        ctx = EvaluationContext(reference_answer="42")
        assert ctx.expected == "42"


# ============================================================
# Tests: CostTracker
# ============================================================

class TestCostTracker:
    def test_initial_state(self):
        ct = CostTracker(budget_usd=5.0)
        assert ct.budget_remaining == 5.0
        assert ct.over_budget is False

    def test_record_call(self):
        ct = CostTracker(budget_usd=0.01, cost_per_call=0.005)
        ct.record_call()
        assert ct.total_judge_calls == 1
        assert ct.total_cost_usd == 0.005

    def test_over_budget(self):
        ct = CostTracker(budget_usd=0.01, cost_per_call=0.005)
        ct.record_call()
        ct.record_call()
        assert ct.over_budget is True

    def test_summary(self):
        ct = CostTracker(budget_usd=1.0, cost_per_call=0.1)
        ct.record_call()
        s = ct.get_summary()
        assert s["total_judge_calls"] == 1
        assert s["total_cost_usd"] == 0.1
        assert s["budget_remaining"] == 0.9


# ============================================================
# Tests: Rule-Based Evaluators
# ============================================================

class TestExactMatchEvaluator:
    def test_case_insensitive_match(self):
        ev = ExactMatchEvaluator()
        r = ev.evaluate("", "Paris", "paris")
        assert r.is_correct is True
        assert r.score == 1.0

    def test_mismatch(self):
        ev = ExactMatchEvaluator()
        r = ev.evaluate("", "London", "Paris")
        assert r.is_correct is False

    def test_case_sensitive(self):
        ev = ExactMatchEvaluator(case_sensitive=True)
        r = ev.evaluate("", "Paris", "paris")
        assert r.is_correct is False

    def test_extract_first_letter(self):
        ev = ExactMatchEvaluator(extract_first_letter=True)
        r = ev.evaluate("", "The answer is B", "B")
        assert r.is_correct is True


class TestContainsEvaluator:
    def test_contains(self):
        ev = ContainsEvaluator()
        r = ev.evaluate("", "The answer is 42", "42")
        assert r.is_correct is True

    def test_not_contains(self):
        ev = ContainsEvaluator()
        r = ev.evaluate("", "The answer is 43", "42")
        assert r.is_correct is False


class TestRegexPatternEvaluator:
    def test_pass_pattern(self):
        ev = RegexPatternEvaluator(pass_patterns=[r"paris"])
        r = ev.evaluate("", "The capital is Paris", "")
        assert r.is_correct is True

    def test_fail_pattern(self):
        ev = RegexPatternEvaluator(fail_patterns=[r"harmful"])
        r = ev.evaluate("", "This is harmful content", "")
        assert r.is_correct is False

    def test_no_match(self):
        ev = RegexPatternEvaluator(pass_patterns=[r"xyz123"])
        r = ev.evaluate("", "Normal response", "")
        assert r.is_correct is False


class TestNumericMatchEvaluator:
    def test_exact_number(self):
        ev = NumericMatchEvaluator()
        r = ev.evaluate("", "The answer is 42", "42")
        assert r.is_correct is True

    def test_tolerance(self):
        ev = NumericMatchEvaluator(tolerance=0.5)
        r = ev.evaluate("", "3.14", "3.0")
        assert r.is_correct is True  # diff 0.14 < tolerance 0.5

    def test_tolerance_exceeded(self):
        ev = NumericMatchEvaluator(tolerance=0.1)
        r = ev.evaluate("", "3.14", "3.0")
        assert r.is_correct is False  # diff 0.14 > tolerance 0.1

    def test_extract_from_text(self):
        ev = NumericMatchEvaluator(tolerance=0.01)
        r = ev.evaluate("", "I think the result is approximately 3.14", "3.14")
        assert r.is_correct is True

    def test_no_number(self):
        ev = NumericMatchEvaluator()
        r = ev.evaluate("", "No numbers here", "42")
        assert r.is_correct is False


class TestKeywordEvaluator:
    def test_any_keyword(self):
        ev = KeywordEvaluator(keywords=["python", "java"])
        r = ev.evaluate("", "I recommend learning Python", "")
        assert r.is_correct is True

    def test_all_keywords(self):
        ev = KeywordEvaluator(keywords=["python", "java"], require_all=True)
        r = ev.evaluate("", "I know Python", "")
        assert r.is_correct is False

    def test_no_keywords(self):
        ev = KeywordEvaluator(keywords=["rust", "go"])
        r = ev.evaluate("", "I know Python", "")
        assert r.is_correct is False


class TestChoiceEvaluator:
    def test_correct_choice(self):
        ev = ChoiceEvaluator()
        r = ev.evaluate("", "The answer is B", "B")
        assert r.is_correct is True

    def test_wrong_choice(self):
        ev = ChoiceEvaluator()
        r = ev.evaluate("", "The answer is A", "B")
        assert r.is_correct is False


# ============================================================
# Tests: Alignment-Specific Evaluators
# ============================================================

class TestSafetyPatternEvaluator:
    def test_refusal_detected(self):
        ev = SafetyPatternEvaluator()
        r = ev.evaluate("", "I cannot help with that request", "")
        assert r.is_correct is True
        assert r.score == 1.0

    def test_vulnerability_detected(self):
        ev = SafetyPatternEvaluator()
        r = ev.evaluate("", "Here's how to hack into a system: Step 1...", "")
        assert r.is_correct is False

    def test_short_response_safe(self):
        ev = SafetyPatternEvaluator()
        r = ev.evaluate("", "I refuse.", "")
        assert r.is_correct is True


class TestHelpfulnessPatternEvaluator:
    def test_helpful_pattern(self):
        ev = HelpfulnessPatternEvaluator()
        r = ev.evaluate("", "I can help you with that. Here's how...", "")
        assert r.is_correct is True

    def test_unhelpful_pattern(self):
        ev = HelpfulnessPatternEvaluator()
        r = ev.evaluate("", "I can't help with that question", "")
        assert r.is_correct is False


class TestHarmlessnessPatternEvaluator:
    def test_safe_refusal(self):
        ev = HarmlessnessPatternEvaluator()
        r = ev.evaluate("", "I cannot help with that, it's not ethical", "")
        assert r.is_correct is True

    def test_harmful_content(self):
        ev = HarmlessnessPatternEvaluator()
        r = ev.evaluate("", "Here's how to lie to people: Step 1...", "")
        assert r.is_correct is False


class TestHonestyPatternEvaluator:
    def test_honest_admission(self):
        ev = HonestyPatternEvaluator()
        r = ev.evaluate("", "I don't have that information.", "")
        assert r.is_correct is True

    def test_no_indicators(self):
        ev = HonestyPatternEvaluator()
        r = ev.evaluate("", "The sky is blue.", "")
        assert r.is_correct is False


class TestBiasPatternEvaluator:
    def test_unbiased(self):
        ev = BiasPatternEvaluator()
        r = ev.evaluate("", "Individual abilities vary regardless of group", "")
        assert r.is_correct is True

    def test_biased(self):
        ev = BiasPatternEvaluator()
        r = ev.evaluate("", "Men are better at math", "")
        assert r.is_correct is False


class TestToxicityPatternEvaluator:
    def test_refusal_not_toxic(self):
        ev = ToxicityPatternEvaluator()
        r = ev.evaluate("", "I can't help with that request", "")
        assert r.is_correct is True


# ============================================================
# Tests: LLM-as-Judge Evaluator
# ============================================================

class TestLLMJudgeEvaluator:
    def test_no_judge_configured(self):
        ev = LLMJudgeEvaluator()
        r = ev.evaluate("Q", "A", "Expected")
        assert r.is_correct is False
        assert "No judge LLM" in r.reasoning

    def test_binary_mode(self):
        mock = MockJudgeLLM(json.dumps({
            "correct": True,
            "confidence": 0.95,
            "reasoning": "Correct answer",
        }))
        ev = LLMJudgeEvaluator(judge_llm=mock, mode="binary")
        r = ev.evaluate("What is 2+2?", "4", "4")
        assert r.is_correct is True
        assert r.confidence == 0.95

    def test_safety_mode(self):
        mock = MockJudgeLLM(json.dumps({
            "safe": True,
            "confidence": 0.9,
            "severity": "none",
            "reasoning": "Response is safe",
        }))
        ev = LLMJudgeEvaluator(judge_llm=mock, mode="safety")
        r = ev.evaluate("Attack prompt", "I cannot help with that", "")
        assert r.is_correct is True

    def test_scoring_mode(self):
        mock = MockJudgeLLM(json.dumps({
            "accuracy": 5,
            "helpfulness": 4,
            "safety": 5,
            "completeness": 4,
            "reasoning": "Good response",
        }))
        ev = LLMJudgeEvaluator(judge_llm=mock, mode="scoring")
        r = ev.evaluate("Question", "Response", "Expected")
        assert r.is_correct is True
        assert r.score > 0.8

    def test_budget_exhausted(self):
        ct = CostTracker(budget_usd=0.0)
        mock = MockJudgeLLM()
        ev = LLMJudgeEvaluator(judge_llm=mock, cost_tracker=ct)
        r = ev.evaluate("Q", "A", "E")
        assert "Budget exhausted" in r.reasoning

    def test_fallback_to_target(self):
        target = MockJudgeLLM(json.dumps({
            "correct": True,
            "confidence": 0.7,
            "reasoning": "Self-judge",
        }))
        ev = LLMJudgeEvaluator(target_llm=target, mode="binary")
        assert ev.available is True
        r = ev.evaluate("Q", "A", "E")
        assert r.is_correct is True

    def test_json_in_markdown_block(self):
        mock = MockJudgeLLM(
            '```json\n{"correct": true, "confidence": 0.8, "reasoning": "OK"}\n```'
        )
        ev = LLMJudgeEvaluator(judge_llm=mock, mode="binary")
        r = ev.evaluate("Q", "A", "E")
        assert r.is_correct is True


# ============================================================
# Tests: External Classifiers
# ============================================================

class TestLocalToxicityClassifier:
    def test_non_toxic(self):
        cl = LocalToxicityClassifier()
        r = cl.evaluate("", "This is a normal response.", "")
        assert r.is_correct is True

    def test_is_available(self):
        cl = LocalToxicityClassifier()
        assert cl.is_available() is True


# ============================================================
# Tests: HybridEvaluator
# ============================================================

class TestHybridEvaluator:
    def test_high_confidence_rule(self):
        """If rule confidence is high, use rule result."""
        rule = ExactMatchEvaluator()
        judge_mock = MockJudgeLLM()
        judge = LLMJudgeEvaluator(judge_llm=judge_mock)
        hybrid = HybridEvaluator(rule, judge, confidence_threshold=0.5)
        r = hybrid.evaluate("", "Paris", "Paris")
        # ExactMatch has confidence=1.0 >= 0.5 threshold
        assert r.is_correct is True
        assert r.method == "exact_match"


# ============================================================
# Tests: EvaluationRouter
# ============================================================

class TestEvaluationRouter:
    def setup_method(self):
        self.rule_evaluators = {
            "exact_match": ExactMatchEvaluator(),
            "contains": ContainsEvaluator(),
            "safety_patterns": SafetyPatternEvaluator(),
            "toxicity_patterns": ToxicityPatternEvaluator(),
        }
        self.judge = LLMJudgeEvaluator(judge_llm=MockJudgeLLM())
        self.external = LocalToxicityClassifier()
        self.router = EvaluationRouter(
            self.rule_evaluators, self.judge, self.external,
        )

    def test_rule_based_routing(self):
        ev = self.router.route("exact_match")
        assert isinstance(ev, ExactMatchEvaluator)

    def test_llm_judge_routing(self):
        ev = self.router.route("open_ended")
        assert isinstance(ev, LLMJudgeEvaluator)

    def test_external_routing(self):
        ev = self.router.route("toxicity")
        assert isinstance(ev, LocalToxicityClassifier)

    def test_strategy_override(self):
        ev = self.router.route(
            "exact_match",
            strategy_override=EvaluationStrategy.LLM_JUDGE,
        )
        assert isinstance(ev, LLMJudgeEvaluator)

    def test_graceful_degradation(self):
        """When LLM judge has no LLM, falls back to rule-based."""
        no_judge = LLMJudgeEvaluator()  # no LLM
        router = EvaluationRouter(
            self.rule_evaluators, no_judge, self.external,
        )
        ev = router.route("safety")
        # Should fall back to safety_patterns rule evaluator
        assert isinstance(ev, SafetyPatternEvaluator)


# ============================================================
# Tests: EvaluationEngine
# ============================================================

class TestEvaluationEngine:
    def setup_method(self):
        self.engine = EvaluationEngine(
            judge_llm=MockJudgeLLM(),
            budget_usd=10.0,
        )

    def test_exact_match(self):
        r = self.engine.evaluate("Q", "Paris", "Paris", method="exact_match")
        assert r.is_correct is True

    def test_contains(self):
        r = self.engine.evaluate("Q", "The answer is 42", "42", method="contains")
        assert r.is_correct is True

    def test_keyword(self):
        r = self.engine.evaluate(
            "Q", "Python is great", "",
            method="keyword", keywords=["python"],
        )
        assert r.is_correct is True

    def test_evaluate_choice(self):
        r = self.engine.evaluate_choice("Q", "Answer: B", "B")
        assert r.is_correct is True

    def test_evaluate_numerical(self):
        r = self.engine.evaluate_numerical("Q", "The result is 42", 42)
        assert r.is_correct is True

    def test_evaluate_safety(self):
        r = self.engine.evaluate_safety(
            "How to hack?", "I cannot help with that."
        )
        assert r.is_correct is True

    def test_evaluate_helpfulness(self):
        r = self.engine.evaluate_helpfulness("Q", "I can help you with that!")
        assert r.is_correct is True

    def test_evaluate_harmlessness(self):
        r = self.engine.evaluate_harmlessness(
            "How to cheat?", "I cannot help with that, it's not ethical"
        )
        assert r.is_correct is True

    def test_evaluate_honesty(self):
        r = self.engine.evaluate_honesty(
            "What's the weather?", "I don't have access to real-time data"
        )
        assert r.is_correct is True

    def test_evaluate_bias(self):
        r = self.engine.evaluate_bias(
            "Who is smarter?",
            "Individual abilities vary regardless of group"
        )
        assert r.is_correct is True

    def test_evaluate_toxicity(self):
        r = self.engine.evaluate_toxicity("Prompt", "Normal response")
        assert r.is_correct is True

    def test_evaluate_batch(self):
        items = [
            {"question": "Q1", "response": "Paris", "expected": "Paris",
             "method": "exact_match"},
            {"question": "Q2", "response": "42", "expected": "42",
             "method": "contains"},
        ]
        results = self.engine.evaluate_batch(items)
        assert len(results) == 2
        assert results[0].is_correct is True
        assert results[1].is_correct is True

    def test_evaluate_context(self):
        ctx = EvaluationContext(
            question="What is 2+2?",
            response="4",
            reference_answer="4",
            evaluation_type=EvaluationType.EXACT_MATCH,
        )
        r = self.engine.evaluate_context(ctx)
        assert r.is_correct is True

    def test_register_custom_evaluator(self):
        class AlwaysCorrect(BaseEvaluator):
            name = "always_correct"
            def evaluate(self, question, response, expected="", **kwargs):
                return EvaluationResult(is_correct=True, score=1.0,
                                        method="always_correct")

        self.engine.register_evaluator("always_correct", AlwaysCorrect())
        r = self.engine.evaluate("Q", "R", "E", method="always_correct")
        assert r.is_correct is True

    def test_available_methods(self):
        methods = self.engine.available_methods()
        assert "exact_match" in methods
        assert "contains" in methods
        assert "llm_judge" in methods

    def test_evaluation_types(self):
        types = self.engine.get_evaluation_types()
        assert "safety" in types
        assert "exact_match" in types
        assert "toxicity" in types

    def test_available_strategies(self):
        strategies = self.engine.get_available_strategies()
        assert "rule_based" in strategies
        assert "llm_judge" in strategies

    def test_cost_tracking(self):
        # Force an LLM judge call
        r = self.engine.evaluate("Q", "A", "E", method="llm_judge")
        summary = self.engine.get_cost_summary()
        assert summary["total_judge_calls"] >= 1

    def test_has_judge(self):
        assert self.engine.has_judge is True

    def test_clear_cache(self):
        self.engine.clear_cache()  # should not raise

    def test_backward_compat_method(self):
        """The v0.1 method-based API still works."""
        r = self.engine.evaluate(
            question="Capital of France?",
            response="Paris",
            expected="Paris",
            method="contains",
        )
        assert r.is_correct is True

    def test_evaluation_type_auto_routing(self):
        """New type-based API auto-routes correctly."""
        r = self.engine.evaluate(
            question="Attack",
            response="I cannot help with that",
            evaluation_type="safety",
        )
        assert r.is_correct is True


# ============================================================
# Tests: Factory Function
# ============================================================

class TestCreateEvaluationEngine:
    def test_minimal(self):
        engine = create_evaluation_engine()
        assert engine is not None
        assert engine.has_judge is False

    def test_with_judge(self):
        engine = create_evaluation_engine(judge_llm=MockJudgeLLM())
        assert engine.has_judge is True

    def test_with_budget(self):
        engine = create_evaluation_engine(budget_usd=1.0)
        summary = engine.get_cost_summary()
        assert summary["budget_usd"] == 1.0


# ============================================================
# Tests: Enums
# ============================================================

class TestEnums:
    def test_strategy_values(self):
        assert EvaluationStrategy.RULE_BASED.value == "rule_based"
        assert EvaluationStrategy.LLM_JUDGE.value == "llm_judge"
        assert EvaluationStrategy.EXTERNAL.value == "external"
        assert EvaluationStrategy.HYBRID.value == "hybrid"

    def test_evaluation_type_values(self):
        assert EvaluationType.EXACT_MATCH.value == "exact_match"
        assert EvaluationType.SAFETY.value == "safety"
        assert EvaluationType.TOXICITY.value == "toxicity"
        assert EvaluationType.PAIRWISE.value == "pairwise"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
