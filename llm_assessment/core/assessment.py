"""
Enhanced Assessment Orchestrator for the LLM Assessment Platform.

This is the central coordination layer that:

* Discovers and registers benchmark / red-team / alignment modules.
* Filters modules based on :class:`AssessmentConfig`.
* Executes tests with progress tracking, error resilience, and
  customisable callbacks.
* Aggregates results into a weighted health score.
* Emits lifecycle events via both **Callback** and **EventBus** APIs.

Backward-compatible: the v0.1 calling convention is preserved::

    platform = AssessmentPlatform(llm)
    results  = platform.run_all(max_benchmark_questions=5)
    platform.save_results("results.json")

Enhanced usage with config + callbacks::

    from llm_assessment.core.config import AssessmentConfig

    config = AssessmentConfig.from_yaml("configs/default.yaml")
    platform = AssessmentPlatform(llm, config=config)
    platform.add_callback(ProgressCallback())
    platform.on("test_completed", lambda d: print(d["test_name"]))
    results = platform.run_all()
"""

import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# Event System
# ============================================================

class AssessmentEvent:
    """String constants for assessment lifecycle events.

    Subscribe via :meth:`EventBus.subscribe` or
    :meth:`AssessmentPlatform.on`.
    """
    ASSESSMENT_STARTED = "assessment_started"
    ASSESSMENT_COMPLETED = "assessment_completed"
    MODULE_STARTED = "module_started"
    MODULE_COMPLETED = "module_completed"
    TEST_STARTED = "test_started"
    TEST_COMPLETED = "test_completed"
    ERROR_OCCURRED = "error_occurred"
    PROGRESS_UPDATED = "progress_updated"


class EventBus:
    """Simple publish-subscribe event bus for assessment lifecycle.

    Usage::

        bus = EventBus()
        bus.subscribe("test_completed", lambda d: print(d))
        bus.emit("test_completed", test_name="MMLU", score=95.0)
    """

    def __init__(self):
        # type: () -> None
        self._handlers = {}  # type: Dict[str, List[Callable]]

    def subscribe(self, event_type, handler):
        # type: (str, Callable) -> None
        """Register *handler* for *event_type*."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type, handler):
        # type: (str, Callable) -> None
        """Remove a previously registered handler."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event_type, **data):
        # type: (str, **Any) -> None
        """Emit an event, calling all registered handlers."""
        data["event_type"] = event_type
        data["timestamp"] = datetime.now().isoformat()
        for handler in self._handlers.get(event_type, []):
            try:
                handler(data)
            except Exception:
                logger.warning(
                    "Event handler error for %s: %s",
                    event_type, traceback.format_exc(),
                )

    def clear(self):
        # type: () -> None
        """Remove all handlers."""
        self._handlers.clear()


# ============================================================
# Callback System
# ============================================================

class CallbackHandler:
    """Base class for assessment lifecycle callbacks.

    Subclass and override only the hooks you need.  All methods are
    no-ops by default.
    """

    def on_assessment_start(self, metadata):
        # type: (Dict[str, Any]) -> None
        """Called when a full assessment begins."""
        pass

    def on_module_start(self, module_type, module_count):
        # type: (str, int) -> None
        """Called before a module (benchmark / red_team / alignment) begins."""
        pass

    def on_test_start(self, test_name, module_type):
        # type: (str, str) -> None
        """Called before an individual test starts."""
        pass

    def on_test_complete(self, test_name, result, duration_ms):
        # type: (str, Dict[str, Any], float) -> None
        """Called after an individual test finishes."""
        pass

    def on_module_complete(self, module_type, results, duration_ms):
        # type: (str, List[Dict[str, Any]], float) -> None
        """Called after a module finishes."""
        pass

    def on_assessment_complete(self, results):
        # type: (Dict[str, Any]) -> None
        """Called when the full assessment is done."""
        pass

    def on_error(self, error, context):
        # type: (Exception, Dict[str, Any]) -> None
        """Called when an error occurs (if ``continue_on_error`` is True)."""
        pass

    def on_progress(self, current, total, message):
        # type: (int, int, str) -> None
        """Called to report progress updates."""
        pass


class LoggingCallback(CallbackHandler):
    """Built-in callback that logs lifecycle events to stdout.

    Used by default when ``execution.show_progress`` is True.
    """

    def on_assessment_start(self, metadata):
        print("\n" + "=" * 70)
        print("LLM QUALITY & SECURITY ASSESSMENT PLATFORM")
        print("=" * 70)
        print("Model: {}".format(metadata.get("model_name", "unknown")))
        print("Session: {}".format(metadata.get("session_id", "N/A")))
        print("Start time: {}".format(metadata.get("start_time", "N/A")))
        if metadata.get("assessment_name"):
            print("Assessment: {}".format(metadata["assessment_name"]))
        print("=" * 70 + "\n")

    def on_module_start(self, module_type, module_count):
        labels = {
            "benchmark": "[1/3] Running Capability Benchmarks",
            "red_team": "[2/3] Running Security Red Team Tests",
            "alignment": "[3/3] Running Alignment Verification",
        }
        label = labels.get(module_type, module_type)
        print("\n{} ({} tests)...".format(label, module_count))
        print("-" * 70)

    def on_test_start(self, test_name, module_type):
        print("\nRunning {}...".format(test_name))

    def on_test_complete(self, test_name, result, duration_ms):
        if "score" in result:
            print("  Score: {:.1f}% ({}/{}) [{:.0f}ms]".format(
                result["score"],
                result.get("correct_answers", 0),
                result.get("total_questions", 0),
                duration_ms,
            ))
        elif "security_score" in result:
            total = result.get("total_tests", 0)
            vulns = result.get("vulnerabilities_found", 0)
            print("  Security Score: {:.1f}% ({}/{} passed) [{:.0f}ms]".format(
                result["security_score"],
                total - vulns,
                total,
                duration_ms,
            ))
        elif "alignment_score" in result:
            print("  Alignment Score: {:.1f}% ({}/{} passed) [{:.0f}ms]".format(
                result["alignment_score"],
                result.get("passed_tests", 0),
                result.get("total_tests", 0),
                duration_ms,
            ))

    def on_module_complete(self, module_type, results, duration_ms):
        print("\n  Module '{}' completed: {} tests in {:.1f}s".format(
            module_type, len(results), duration_ms / 1000,
        ))

    def on_assessment_complete(self, results):
        summary = results.get("summary", {})
        print("\n" + "=" * 70)
        print("ASSESSMENT COMPLETE")
        print("=" * 70)
        print("Overall Health Score: {:.1f}/100 ({})".format(
            summary.get("overall_health_score", 0),
            summary.get("health_rating", "N/A"),
        ))
        print("  Capability:  {:.1f}".format(summary.get("benchmark_average", 0)))
        print("  Security:    {:.1f}".format(summary.get("security_average", 0)))
        print("  Alignment:   {:.1f}".format(summary.get("alignment_average", 0)))
        print("  Vulnerabilities: {}".format(
            summary.get("total_vulnerabilities", 0),
        ))
        print("  Duration: {:.1f}s".format(
            summary.get("duration_seconds", 0),
        ))
        print("=" * 70)

    def on_error(self, error, context):
        print("  [WARNING] Error in {}: {}".format(
            context.get("test_name", "unknown"), error,
        ))


class ProgressCallback(CallbackHandler):
    """Built-in callback that shows a *tqdm* progress bar per module."""

    def __init__(self):
        self._pbar = None

    def on_module_start(self, module_type, module_count):
        try:
            from tqdm import tqdm
            self._pbar = tqdm(
                total=module_count,
                desc=module_type,
                unit="test",
                leave=True,
            )
        except ImportError:
            self._pbar = None

    def on_test_complete(self, test_name, result, duration_ms):
        if self._pbar is not None:
            self._pbar.update(1)

    def on_module_complete(self, module_type, results, duration_ms):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


# ============================================================
# Module Registry
# ============================================================

class ModuleRegistry:
    """Registry for assessment modules with auto-discovery.

    Wraps the existing ``get_available_*()`` pattern and adds:

    * Explicit registration of custom module instances.
    * Filtering by name and category.
    * Lazy auto-discovery on first access.

    Usage::

        registry = ModuleRegistry()
        registry.auto_discover()
        benchmarks = registry.get_benchmarks(names=["MMLU-general"])
    """

    def __init__(self):
        # type: () -> None
        self._benchmarks = []          # type: List[Any]
        self._red_team_tests = []      # type: List[Any]
        self._alignment_tests = []     # type: List[Any]
        self._custom_benchmarks = []   # type: List[Any]
        self._custom_red_team = []     # type: List[Any]
        self._custom_alignment = []    # type: List[Any]
        self._discovered = False

    # ----------------------------------------------------------
    # Auto-discovery
    # ----------------------------------------------------------

    def auto_discover(self):
        # type: () -> None
        """Discover modules from built-in ``get_available_*()`` functions."""
        if self._discovered:
            return

        try:
            from ..benchmark.benchmarks import get_available_benchmarks
            self._benchmarks = list(get_available_benchmarks())
        except Exception as exc:
            logger.warning("Could not discover benchmarks: %s", exc)

        try:
            from ..red_teaming.tests import get_available_red_team_tests
            self._red_team_tests = list(get_available_red_team_tests())
        except Exception as exc:
            logger.warning("Could not discover red-team tests: %s", exc)

        try:
            from ..alignment.tests import get_available_alignment_tests
            self._alignment_tests = list(get_available_alignment_tests())
        except Exception as exc:
            logger.warning("Could not discover alignment tests: %s", exc)

        self._discovered = True
        logger.info(
            "Discovered %d benchmarks, %d red-team tests, %d alignment tests",
            len(self._benchmarks),
            len(self._red_team_tests),
            len(self._alignment_tests),
        )

    # ----------------------------------------------------------
    # Registration
    # ----------------------------------------------------------

    def register_benchmark(self, benchmark):
        # type: (Any) -> None
        """Register a custom benchmark instance."""
        self._custom_benchmarks.append(benchmark)

    def register_red_team_test(self, test):
        # type: (Any) -> None
        """Register a custom red-team test instance."""
        self._custom_red_team.append(test)

    def register_alignment_test(self, test):
        # type: (Any) -> None
        """Register a custom alignment test instance."""
        self._custom_alignment.append(test)

    # ----------------------------------------------------------
    # Retrieval (with filtering)
    # ----------------------------------------------------------

    def get_benchmarks(self, names=None, categories=None):
        # type: (Optional[List[str]], Optional[List[str]]) -> List[Any]
        """Return benchmarks, optionally filtered by *names* / *categories*."""
        self.auto_discover()
        all_items = self._benchmarks + self._custom_benchmarks
        return self._filter(all_items, names, categories)

    def get_red_team_tests(self, names=None, categories=None):
        # type: (Optional[List[str]], Optional[List[str]]) -> List[Any]
        """Return red-team tests, optionally filtered."""
        self.auto_discover()
        all_items = self._red_team_tests + self._custom_red_team
        return self._filter(all_items, names, categories)

    def get_alignment_tests(self, names=None, categories=None):
        # type: (Optional[List[str]], Optional[List[str]]) -> List[Any]
        """Return alignment tests, optionally filtered."""
        self.auto_discover()
        all_items = self._alignment_tests + self._custom_alignment
        return self._filter(all_items, names, categories)

    def list_modules(self):
        # type: () -> Dict[str, List[Dict[str, str]]]
        """Return a summary dict of all registered modules."""
        self.auto_discover()

        def _summarise(items):
            return [
                {"name": getattr(m, "name", str(m)),
                 "category": getattr(m, "category", "unknown")}
                for m in items
            ]

        return {
            "benchmarks": _summarise(
                self._benchmarks + self._custom_benchmarks,
            ),
            "red_team_tests": _summarise(
                self._red_team_tests + self._custom_red_team,
            ),
            "alignment_tests": _summarise(
                self._alignment_tests + self._custom_alignment,
            ),
        }

    # ----------------------------------------------------------
    # Internal filter
    # ----------------------------------------------------------

    @staticmethod
    def _filter(items, names, categories):
        # type: (List[Any], Optional[List[str]], Optional[List[str]]) -> List[Any]
        result = []
        for item in items:
            item_name = getattr(item, "name", "")
            item_cat = getattr(item, "category", "")
            if names is not None and item_name not in names:
                continue
            if categories is not None and item_cat not in categories:
                continue
            result.append(item)
        return result


# ============================================================
# Assessment Session
# ============================================================

@dataclass
class AssessmentSession:
    """Immutable metadata for a single assessment run.

    Created automatically when :meth:`AssessmentPlatform.run_all` begins.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = ""
    assessment_name: str = ""
    start_time: str = ""
    end_time: str = ""
    tags: List[str] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "assessment_name": self.assessment_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tags": self.tags,
            "errors": self.errors,
        }


# ============================================================
# Health Score Calculator
# ============================================================

class HealthScoreCalculator:
    """Calculates the overall health score from module results.

    Encapsulates the scoring logic so it can be unit-tested and
    customised independently of the orchestrator.
    """

    def __init__(self, scoring_config=None):
        # type: (Optional[Any]) -> None
        from .config import ScoringConfig
        self._config = scoring_config or ScoringConfig()

    def calculate(self, benchmark_results, red_team_results, alignment_results):
        # type: (List[Dict], List[Dict], List[Dict]) -> Dict[str, Any]
        """Compute the health score summary.

        Returns a dict compatible with the v0.1 ``summary`` format.
        """
        benchmark_avg = self._avg(benchmark_results, "score")
        security_avg = self._avg(red_team_results, "security_score")
        alignment_avg = self._avg(alignment_results, "alignment_score")

        overall = (
            benchmark_avg * self._config.benchmark_weight
            + security_avg * self._config.security_weight
            + alignment_avg * self._config.alignment_weight
        )

        total_vulns = sum(
            r.get("vulnerabilities_found", 0) for r in red_team_results
        )

        return {
            "overall_health_score": round(overall, 2),
            "health_rating": self._config.get_rating(overall),
            "benchmark_average": round(benchmark_avg, 2),
            "security_average": round(security_avg, 2),
            "alignment_average": round(alignment_avg, 2),
            "total_vulnerabilities": total_vulns,
            "weights": {
                "benchmark": self._config.benchmark_weight,
                "security": self._config.security_weight,
                "alignment": self._config.alignment_weight,
            },
        }

    @staticmethod
    def _avg(results, key):
        # type: (List[Dict], str) -> float
        if not results:
            return 0.0
        total = sum(r.get(key, 0) for r in results)
        return total / len(results)


# ============================================================
# Assessment Platform (The Orchestrator)
# ============================================================

class AssessmentPlatform:
    """
    Unified LLM Quality and Security Assessment Platform.

    This is the main entry point.  It coordinates:

    * **Benchmark** testing (capability evaluation)
    * **Red-teaming** (security testing)
    * **Alignment** checking (value alignment)

    New in v2: config-driven execution, event system, module registry,
    error resilience, customisable scoring.

    Args:
        llm: The LLM instance to assess (must implement ``generate()``).
        config: Either an :class:`AssessmentConfig`, a plain ``dict``
            (v0.1 compat), or ``None`` for defaults.
        callbacks: Optional list of :class:`CallbackHandler` instances.

    Backward Compatibility:
        The v0.1 API is fully supported::

            platform = AssessmentPlatform(llm)
            results = platform.run_all(max_benchmark_questions=5)
    """

    def __init__(self, llm, config=None, callbacks=None):
        # type: (Any, Any, Optional[List[CallbackHandler]]) -> None
        self.llm = llm

        # ---- Config ----
        self._config = self._normalise_config(config)

        # ---- Subsystems ----
        self._registry = ModuleRegistry()
        self._event_bus = EventBus()
        self._callbacks = list(callbacks or [])
        self._scorer = HealthScoreCalculator(self._config.scoring)

        # ---- Evaluation engine (lazy init) ----
        self._evaluation_engine = None  # type: Optional[Any]

        # ---- State ----
        self._session = None  # type: Optional[AssessmentSession]
        self._results = {
            "benchmark": [],       # type: List[Dict[str, Any]]
            "red_teaming": [],     # type: List[Dict[str, Any]]
            "alignment": [],       # type: List[Dict[str, Any]]
        }

        # ---- Add built-in callbacks ----
        if self._config.execution.show_progress:
            self._callbacks.append(LoggingCallback())

    # ----------------------------------------------------------
    # Configuration helpers
    # ----------------------------------------------------------

    @staticmethod
    def _normalise_config(config):
        # type: (Any) -> Any
        """Accept dict, AssessmentConfig, or None -> always AssessmentConfig."""
        from .config import AssessmentConfig

        if config is None:
            return AssessmentConfig()
        if isinstance(config, AssessmentConfig):
            return config
        if isinstance(config, dict):
            return AssessmentConfig.from_dict(config)
        return AssessmentConfig()

    @property
    def config(self):
        # type: () -> Any
        """The current :class:`AssessmentConfig`."""
        return self._config

    def load_config(self, filepath):
        # type: (str) -> None
        """Load configuration from a YAML file and apply it."""
        from .config import AssessmentConfig
        self._config = AssessmentConfig.from_yaml(filepath)
        self._scorer = HealthScoreCalculator(self._config.scoring)

    def update_config(self, **overrides):
        # type: (**Any) -> None
        """Merge overrides into the current config."""
        self._config = self._config.merge(overrides)
        self._scorer = HealthScoreCalculator(self._config.scoring)

    # ----------------------------------------------------------
    # Module registry
    # ----------------------------------------------------------

    @property
    def registry(self):
        # type: () -> ModuleRegistry
        """Access the module registry."""
        return self._registry

    def register_benchmark(self, benchmark):
        # type: (Any) -> None
        """Register a custom benchmark."""
        self._registry.register_benchmark(benchmark)

    def register_red_team_test(self, test):
        # type: (Any) -> None
        """Register a custom red-team test."""
        self._registry.register_red_team_test(test)

    def register_alignment_test(self, test):
        # type: (Any) -> None
        """Register a custom alignment test."""
        self._registry.register_alignment_test(test)

    def list_modules(self):
        # type: () -> Dict[str, Any]
        """List all discovered and registered modules."""
        return self._registry.list_modules()

    # ----------------------------------------------------------
    # Evaluation engine
    # ----------------------------------------------------------

    @property
    def evaluation_engine(self):
        # type: () -> Any
        """Lazy-initialised :class:`EvaluationEngine`."""
        if self._evaluation_engine is None:
            from .evaluation import create_evaluation_engine
            eval_cfg = self._config.evaluation
            judge_llm = None
            perspective_key = None
            budget = 10.0
            if eval_cfg is not None:
                judge_provider = getattr(eval_cfg, "judge_provider", None)
                judge_model = getattr(eval_cfg, "judge_model", None)
                judge_api_key = getattr(eval_cfg, "judge_api_key", None)
                if judge_provider and judge_model:
                    try:
                        from .llm_wrapper import create_llm
                        judge_llm = create_llm(
                            provider=judge_provider,
                            model=judge_model,
                            api_key=judge_api_key,
                        )
                    except Exception:
                        pass
                perspective_key = getattr(
                    eval_cfg, "perspective_api_key", None)
                budget = getattr(eval_cfg, "judge_budget_usd", 10.0)
            self._evaluation_engine = create_evaluation_engine(
                judge_llm=judge_llm,
                target_llm=self.llm,
                perspective_api_key=perspective_key,
                budget_usd=budget,
                config=eval_cfg,
            )
        return self._evaluation_engine

    # ----------------------------------------------------------
    # Event system
    # ----------------------------------------------------------

    @property
    def event_bus(self):
        # type: () -> EventBus
        return self._event_bus

    def on(self, event_type, handler):
        # type: (str, Callable) -> None
        """Subscribe a function to a lifecycle event.

        Shortcut for ``platform.event_bus.subscribe(event_type, handler)``.
        """
        self._event_bus.subscribe(event_type, handler)

    def add_callback(self, handler):
        # type: (CallbackHandler) -> None
        """Add a :class:`CallbackHandler`."""
        self._callbacks.append(handler)

    def remove_callback(self, handler):
        # type: (CallbackHandler) -> None
        """Remove a previously added callback."""
        if handler in self._callbacks:
            self._callbacks.remove(handler)

    # ----------------------------------------------------------
    # Core execution
    # ----------------------------------------------------------

    def run_all(self, max_benchmark_questions=None):
        # type: (Optional[int]) -> Dict[str, Any]
        """Run all enabled assessment modules (one-click health check).

        This is the primary entry point.  Backward compatible with v0.1.

        Args:
            max_benchmark_questions: Override ``benchmark.max_questions``
                from the config (v0.1 compat parameter).

        Returns:
            Complete assessment results dict.
        """
        # Apply v0.1 compat override
        if max_benchmark_questions is not None:
            self._config.benchmark.max_questions = max_benchmark_questions

        # Validate config
        try:
            self._config.validate()
        except ValueError as exc:
            logger.warning("Config validation warning: %s", exc)

        # Start session
        self._start_session()

        overall_start = time.monotonic()

        # Run each enabled module
        if self._config.benchmark.enabled:
            self.run_benchmarks()

        if self._config.red_team.enabled:
            self.run_red_teaming()

        if self._config.alignment.enabled:
            self.run_alignment()

        # Finalise
        overall_duration_ms = (time.monotonic() - overall_start) * 1000
        self._session.end_time = datetime.now().isoformat()

        results = self.get_results()
        results["summary"]["duration_seconds"] = round(
            overall_duration_ms / 1000, 2,
        )

        self._emit_event(
            AssessmentEvent.ASSESSMENT_COMPLETED,
            results=results,
        )
        self._notify_callbacks("on_assessment_complete", results)

        return results

    def run_benchmarks(self, max_questions=None, names=None, categories=None):
        # type: (Optional[int], Optional[List[str]], Optional[List[str]]) -> List[Dict[str, Any]]
        """Run benchmark tests (optionally filtered).

        Args:
            max_questions: Override per-benchmark question limit.
            names: Only run benchmarks with these names.
            categories: Only run benchmarks in these categories.

        Returns:
            List of benchmark result dicts.
        """
        if self._session is None:
            self._start_session()

        cfg = self._config.benchmark
        effective_max = max_questions if max_questions is not None else cfg.max_questions
        filter_names = names if names is not None else cfg.benchmarks
        filter_cats = categories if categories is not None else cfg.categories

        benchmarks = self._registry.get_benchmarks(
            names=filter_names, categories=filter_cats,
        )

        self._results["benchmark"] = []
        return self._run_module(
            module_type="benchmark",
            tests=benchmarks,
            result_key="benchmark",
            run_kwargs={"max_questions": effective_max},
        )

    def run_red_teaming(self, names=None, categories=None):
        # type: (Optional[List[str]], Optional[List[str]]) -> List[Dict[str, Any]]
        """Run red-team tests (optionally filtered)."""
        if self._session is None:
            self._start_session()

        cfg = self._config.red_team
        filter_names = names if names is not None else cfg.tests
        filter_cats = categories if categories is not None else cfg.categories

        tests = self._registry.get_red_team_tests(
            names=filter_names, categories=filter_cats,
        )

        self._results["red_teaming"] = []
        return self._run_module(
            module_type="red_team",
            tests=tests,
            result_key="red_teaming",
        )

    def run_alignment(self, names=None, categories=None):
        # type: (Optional[List[str]], Optional[List[str]]) -> List[Dict[str, Any]]
        """Run alignment tests (optionally filtered)."""
        if self._session is None:
            self._start_session()

        cfg = self._config.alignment
        filter_names = names if names is not None else cfg.tests
        filter_cats = categories if categories is not None else cfg.categories

        tests = self._registry.get_alignment_tests(
            names=filter_names, categories=filter_cats,
        )

        self._results["alignment"] = []
        return self._run_module(
            module_type="alignment",
            tests=tests,
            result_key="alignment",
        )

    # ----------------------------------------------------------
    # Result accessors
    # ----------------------------------------------------------

    def get_results(self):
        # type: () -> Dict[str, Any]
        """Get all assessment results (backward compatible format)."""
        metadata = self._session.to_dict() if self._session else {
            "model_name": getattr(self.llm, "model_name", "unknown"),
            "start_time": None,
            "end_time": None,
        }
        # v0.1 compat — ensure 'model_name' is always present
        if "model_name" not in metadata:
            metadata["model_name"] = getattr(self.llm, "model_name", "unknown")

        summary = self._generate_summary()

        return {
            "metadata": metadata,
            "benchmark_results": list(self._results["benchmark"]),
            "red_teaming_results": list(self._results["red_teaming"]),
            "alignment_results": list(self._results["alignment"]),
            "summary": summary,
        }

    def _generate_summary(self):
        # type: () -> Dict[str, Any]
        """Generate summary statistics via :class:`HealthScoreCalculator`."""
        summary = self._scorer.calculate(
            self._results["benchmark"],
            self._results["red_teaming"],
            self._results["alignment"],
        )

        # Attach model stats
        if hasattr(self.llm, "get_stats"):
            summary["model_stats"] = self.llm.get_stats()
        else:
            summary["model_stats"] = {}

        # Attach error summary
        if self._session and self._session.errors:
            summary["errors"] = list(self._session.errors)

        return summary

    def save_results(self, filepath="assessment_results.json"):
        # type: (str) -> None
        """Save results to a JSON file."""
        results = self.get_results()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\nResults saved to: {}".format(filepath))

    def generate_report(self, filepath=None, format="html"):
        # type: (Optional[str], str) -> str
        """Generate a report from current results.

        Args:
            filepath: If given, save the report to this file.
            format: ``"html"`` or ``"text"``.

        Returns:
            The report content as a string.
        """
        from .report import ReportGenerator

        results = self.get_results()
        report_gen = ReportGenerator(results)

        if format == "html":
            content = report_gen.generate_html_report()
        else:
            content = report_gen.generate_text_report()

        if filepath:
            report_gen.save_report(filepath, format=format)

        return content

    # ----------------------------------------------------------
    # Internal — session lifecycle
    # ----------------------------------------------------------

    def _start_session(self):
        # type: () -> None
        """Create a new session and emit the start event."""
        self._session = AssessmentSession(
            model_name=getattr(self.llm, "model_name", "unknown"),
            assessment_name=self._config.assessment_name or "",
            start_time=datetime.now().isoformat(),
            tags=list(self._config.tags),
            config_snapshot=self._config.to_dict(),
        )

        self._results = {
            "benchmark": [],
            "red_teaming": [],
            "alignment": [],
        }

        metadata = self._session.to_dict()
        self._emit_event(AssessmentEvent.ASSESSMENT_STARTED, **metadata)
        self._notify_callbacks("on_assessment_start", metadata)

    # ----------------------------------------------------------
    # Internal — module execution
    # ----------------------------------------------------------

    def _run_module(self, module_type, tests, result_key, run_kwargs=None):
        # type: (str, List[Any], str, Optional[Dict]) -> List[Dict[str, Any]]
        """Execute all tests in a module with error handling and events.

        This is the core execution loop shared across benchmark,
        red_team, and alignment modules.
        """
        run_kwargs = run_kwargs or {}
        module_results = []

        self._emit_event(
            AssessmentEvent.MODULE_STARTED,
            module_type=module_type,
            module_count=len(tests),
        )
        self._notify_callbacks("on_module_start", module_type, len(tests))

        module_start = time.monotonic()

        for idx, test in enumerate(tests):
            test_name = getattr(test, "name", str(test))
            test_category = getattr(test, "category", "unknown")

            self._emit_event(
                AssessmentEvent.TEST_STARTED,
                test_name=test_name,
                module_type=module_type,
                index=idx,
            )
            self._notify_callbacks("on_test_start", test_name, module_type)

            test_start = time.monotonic()

            try:
                if self._config.execution.dry_run:
                    result_dict = self._dry_run_result(
                        test_name, test_category, module_type,
                    )
                else:
                    result_dict = self._execute_single_test(
                        test, module_type, run_kwargs,
                    )
            except Exception as exc:
                duration_ms = (time.monotonic() - test_start) * 1000
                error_info = {
                    "test_name": test_name,
                    "module_type": module_type,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }

                if self._session:
                    self._session.errors.append(error_info)

                self._emit_event(
                    AssessmentEvent.ERROR_OCCURRED,
                    error=str(exc),
                    context=error_info,
                )
                self._notify_callbacks("on_error", exc, error_info)

                if self._config.execution.continue_on_error:
                    logger.warning(
                        "Test '%s' failed, continuing: %s", test_name, exc,
                    )
                    result_dict = self._error_result(
                        test_name, test_category, module_type, exc,
                    )
                else:
                    raise
            else:
                duration_ms = (time.monotonic() - test_start) * 1000

            result_dict["duration_ms"] = round(duration_ms, 1)
            module_results.append(result_dict)

            self._emit_event(
                AssessmentEvent.TEST_COMPLETED,
                test_name=test_name,
                result=result_dict,
                duration_ms=duration_ms,
            )
            self._notify_callbacks(
                "on_test_complete", test_name, result_dict, duration_ms,
            )

            # Progress
            self._emit_event(
                AssessmentEvent.PROGRESS_UPDATED,
                current=idx + 1,
                total=len(tests),
                message="{}: {}/{}".format(module_type, idx + 1, len(tests)),
            )
            self._notify_callbacks(
                "on_progress", idx + 1, len(tests),
                "{}: {}/{}".format(module_type, idx + 1, len(tests)),
            )

        module_duration_ms = (time.monotonic() - module_start) * 1000

        # Store results
        self._results[result_key] = module_results

        self._emit_event(
            AssessmentEvent.MODULE_COMPLETED,
            module_type=module_type,
            results_count=len(module_results),
            duration_ms=module_duration_ms,
        )
        self._notify_callbacks(
            "on_module_complete", module_type, module_results, module_duration_ms,
        )

        return module_results

    def _execute_single_test(self, test, module_type, run_kwargs):
        # type: (Any, str, Dict) -> Dict[str, Any]
        """Execute a single test and normalise its result to a dict."""
        if module_type == "benchmark":
            result = test.run(
                self.llm,
                max_questions=run_kwargs.get("max_questions"),
            )
            return {
                "name": result.name,
                "category": result.category,
                "score": result.score,
                "total_questions": result.total_questions,
                "correct_answers": result.correct_answers,
                "details": result.details,
            }
        elif module_type == "red_team":
            result = test.run(self.llm)
            return {
                "name": result.name,
                "category": result.category,
                "security_score": result.security_score,
                "vulnerabilities_found": result.vulnerabilities_found,
                "total_tests": result.total_tests,
                "details": result.details,
            }
        elif module_type == "alignment":
            result = test.run(self.llm)
            return {
                "name": result.name,
                "category": result.category,
                "alignment_score": result.alignment_score,
                "passed_tests": result.passed_tests,
                "total_tests": result.total_tests,
                "details": result.details,
            }
        else:
            raise ValueError("Unknown module type: {}".format(module_type))

    @staticmethod
    def _dry_run_result(test_name, category, module_type):
        # type: (str, str, str) -> Dict[str, Any]
        """Produce a dummy result when ``dry_run`` is True."""
        base = {"name": test_name, "category": category, "details": {"dry_run": True}}
        if module_type == "benchmark":
            base.update(score=0.0, total_questions=0, correct_answers=0)
        elif module_type == "red_team":
            base.update(security_score=100.0, vulnerabilities_found=0, total_tests=0)
        elif module_type == "alignment":
            base.update(alignment_score=0.0, passed_tests=0, total_tests=0)
        return base

    @staticmethod
    def _error_result(test_name, category, module_type, exc):
        # type: (str, str, str, Exception) -> Dict[str, Any]
        """Produce an error placeholder result."""
        base = {
            "name": test_name,
            "category": category,
            "details": {"error": str(exc)},
        }
        if module_type == "benchmark":
            base.update(score=0.0, total_questions=0, correct_answers=0)
        elif module_type == "red_team":
            base.update(security_score=0.0, vulnerabilities_found=0, total_tests=0)
        elif module_type == "alignment":
            base.update(alignment_score=0.0, passed_tests=0, total_tests=0)
        return base

    # ----------------------------------------------------------
    # Internal — event helpers
    # ----------------------------------------------------------

    def _emit_event(self, event_type, **data):
        # type: (str, **Any) -> None
        """Emit an event through the EventBus."""
        self._event_bus.emit(event_type, **data)

    def _notify_callbacks(self, method_name, *args):
        # type: (str, *Any) -> None
        """Invoke *method_name* on all registered callbacks."""
        for cb in self._callbacks:
            fn = getattr(cb, method_name, None)
            if fn is not None:
                try:
                    fn(*args)
                except Exception:
                    logger.warning(
                        "Callback %s.%s raised: %s",
                        type(cb).__name__, method_name,
                        traceback.format_exc(),
                    )

    # ----------------------------------------------------------
    # Async placeholder
    # ----------------------------------------------------------

    def run_all_async(self, max_benchmark_questions=None):
        """Placeholder for future async execution.

        Currently delegates to the sync :meth:`run_all`.  A future
        version will use ``asyncio`` for concurrent module execution.
        """
        logger.info("Async execution not yet implemented; running synchronously.")
        return self.run_all(max_benchmark_questions=max_benchmark_questions)
