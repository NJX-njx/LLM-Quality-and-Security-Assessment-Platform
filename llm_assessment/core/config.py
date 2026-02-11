"""
Configuration management for the LLM Assessment Platform.

Provides typed configuration via dataclasses with YAML serialization,
preset profiles, and hierarchical config merging.

Usage::

    from llm_assessment.core.config import AssessmentConfig, get_preset_config

    # Default config
    config = AssessmentConfig()

    # Load from YAML
    config = AssessmentConfig.from_yaml("configs/default.yaml")

    # Use a preset
    config = get_preset_config("quick_test")

    # Merge overrides
    config = config.merge({"benchmark": {"max_questions": 10}})
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Any, Optional, List, Tuple
import copy
import os
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Module Configuration Dataclasses
# ============================================================

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark evaluation module.

    Attributes:
        enabled: Whether to run benchmarks at all.
        max_questions: Cap the number of questions per benchmark (None = no cap).
        benchmarks: Whitelist of benchmark names to run (None = all available).
        categories: Whitelist of categories to include (None = all).
        timeout_per_question: Max seconds to wait per question.
        shuffle_questions: Randomize question order.
    """
    enabled: bool = True
    max_questions: Optional[int] = None
    benchmarks: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    timeout_per_question: float = 30.0
    shuffle_questions: bool = False

    def should_run(self, name, category):
        # type: (str, str) -> bool
        """Check whether a specific benchmark should execute."""
        if not self.enabled:
            return False
        if self.benchmarks is not None and name not in self.benchmarks:
            return False
        if self.categories is not None and category not in self.categories:
            return False
        return True


@dataclass
class RedTeamConfig:
    """Configuration for the security red teaming module.

    Attributes:
        enabled: Whether to run red‑team tests.
        tests: Whitelist of test names (None = all).
        categories: Whitelist of categories (None = all).
        severity_threshold: Minimum severity to include ("low" | "medium" | "high" | "critical").
        auto_attack_enabled: Enable LLM‑driven automatic attack generation.
        max_attacks_per_test: Cap attacks per test (None = no cap).
    """
    enabled: bool = True
    tests: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    severity_threshold: str = "low"
    auto_attack_enabled: bool = False
    max_attacks_per_test: Optional[int] = None

    _SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def should_run(self, name, category):
        # type: (str, str) -> bool
        """Check whether a specific red‑team test should execute."""
        if not self.enabled:
            return False
        if self.tests is not None and name not in self.tests:
            return False
        if self.categories is not None and category not in self.categories:
            return False
        return True

    def meets_severity(self, severity):
        # type: (str) -> bool
        """Return True if *severity* meets or exceeds the threshold."""
        threshold_idx = self._SEVERITY_ORDER.get(self.severity_threshold, 0)
        severity_idx = self._SEVERITY_ORDER.get(severity, 0)
        return severity_idx >= threshold_idx


@dataclass
class AlignmentConfig:
    """Configuration for the alignment verification module.

    Attributes:
        enabled: Whether to run alignment tests.
        tests: Whitelist of test names (None = all).
        categories: Whitelist of categories (None = all).
    """
    enabled: bool = True
    tests: Optional[List[str]] = None
    categories: Optional[List[str]] = None

    def should_run(self, name, category):
        # type: (str, str) -> bool
        """Check whether a specific alignment test should execute."""
        if not self.enabled:
            return False
        if self.tests is not None and name not in self.tests:
            return False
        if self.categories is not None and category not in self.categories:
            return False
        return True


# ============================================================
# Engine Configuration Dataclasses
# ============================================================

@dataclass
class EvaluationConfig:
    """Configuration for the evaluation engine.

    Attributes:
        default_method: Default evaluation strategy
            ("rule", "llm_judge", or "hybrid").
        judge_model: Model name for LLM‑as‑Judge.
        judge_provider: Provider for the judge model.
        judge_api_key: API key for the judge provider.
        fallback_to_rule: Fall back to rule‑based evaluation on judge failure.
        cache_evaluations: Cache evaluation results to avoid re‑computation.
        confidence_threshold: Minimum confidence for LLM judge verdicts.
    """
    default_method: str = "rule"
    judge_model: Optional[str] = None
    judge_provider: Optional[str] = None
    judge_api_key: Optional[str] = None
    fallback_to_rule: bool = True
    cache_evaluations: bool = True
    confidence_threshold: float = 0.7


@dataclass
class ScoringConfig:
    """Configuration for health score calculation.

    The three module weights **must** sum to 1.0.

    Attributes:
        benchmark_weight: Weight for capability benchmarks (default 0.4).
        security_weight: Weight for security red‑teaming (default 0.3).
        alignment_weight: Weight for alignment verification (default 0.3).
        excellent_threshold: Score >= this → "Excellent".
        good_threshold: Score >= this → "Good".
        fair_threshold: Score >= this → "Fair".
        category_weights: Optional per‑category weight overrides.
    """
    benchmark_weight: float = 0.4
    security_weight: float = 0.3
    alignment_weight: float = 0.3
    excellent_threshold: float = 90.0
    good_threshold: float = 75.0
    fair_threshold: float = 60.0
    category_weights: Dict[str, float] = field(default_factory=dict)

    def validate_weights(self):
        # type: () -> None
        """Raise ``ValueError`` if module weights do not sum to 1.0."""
        total = self.benchmark_weight + self.security_weight + self.alignment_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                "Scoring weights must sum to 1.0, got {:.3f} "
                "(benchmark={}, security={}, alignment={})".format(
                    total,
                    self.benchmark_weight,
                    self.security_weight,
                    self.alignment_weight,
                )
            )

    def get_rating(self, score):
        # type: (float) -> str
        """Map a numeric 0‑100 score to a human‑readable rating."""
        if score >= self.excellent_threshold:
            return "Excellent"
        elif score >= self.good_threshold:
            return "Good"
        elif score >= self.fair_threshold:
            return "Fair"
        else:
            return "Needs Improvement"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    formats: List[str] = field(default_factory=lambda: ["html"])
    output_dir: str = "."
    include_details: bool = True
    include_recommendations: bool = True
    max_detail_examples: int = 10
    template_dir: Optional[str] = None


@dataclass
class ExecutionConfig:
    """Configuration for execution behaviour.

    Attributes:
        continue_on_error: If True, a failing test won't abort the run.
        show_progress: Display progress bars (requires *tqdm*).
        verbose: Print extra diagnostic information.
        dry_run: Discover modules but skip actual LLM calls.
        max_concurrent: Reserved for future async execution.
        retry_failed: Re‑run failed tests once at the end.
        log_level: Python logging level name.
    """
    continue_on_error: bool = True
    show_progress: bool = True
    verbose: bool = False
    dry_run: bool = False
    max_concurrent: int = 1
    retry_failed: bool = False
    log_level: str = "INFO"


# ============================================================
# Top‑Level Assessment Configuration
# ============================================================

@dataclass
class AssessmentConfig:
    """
    Top‑level configuration for the Assessment Platform.

    Aggregates all module and engine configs into a single hierarchical
    structure that can be loaded from YAML, constructed programmatically,
    or obtained from named presets.
    """
    # Module configs
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    red_team: RedTeamConfig = field(default_factory=RedTeamConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)

    # Engine configs
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Provider config (convenience — CLI passes these to create_llm)
    provider: str = "mock"
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    provider_options: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    assessment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # ----------------------------------------------------------
    # Serialization helpers
    # ----------------------------------------------------------

    def to_dict(self):
        # type: () -> Dict[str, Any]
        """Serialize to a plain dictionary (JSON‑compatible)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        # type: (Dict[str, Any]) -> AssessmentConfig
        """Deserialize from a dictionary with nested dataclass parsing."""
        if not isinstance(data, dict):
            return cls()

        config = cls()
        _mapping = {
            "benchmark": BenchmarkConfig,
            "red_team": RedTeamConfig,
            "alignment": AlignmentConfig,
            "evaluation": EvaluationConfig,
            "scoring": ScoringConfig,
            "report": ReportConfig,
            "execution": ExecutionConfig,
        }
        for key, dc_cls in _mapping.items():
            if key in data and isinstance(data[key], dict):
                setattr(config, key, _parse_dataclass(dc_cls, data[key]))

        for key in ("provider", "model_name", "api_key", "assessment_name"):
            if key in data:
                setattr(config, key, data[key])
        if "provider_options" in data and isinstance(data["provider_options"], dict):
            config.provider_options = dict(data["provider_options"])
        if "tags" in data and isinstance(data["tags"], list):
            config.tags = list(data["tags"])
        return config

    # ----------------------------------------------------------
    # YAML I/O
    # ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, filepath):
        # type: (str) -> AssessmentConfig
        """Load configuration from a YAML file.

        Requires ``pyyaml``.  Install via ``pip install pyyaml`` or
        ``pip install llm-assessment-platform[config]``.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required for YAML configuration. "
                "Install with: pip install pyyaml"
            )
        with open(filepath, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def save_yaml(self, filepath):
        # type: (str) -> None
        """Save configuration to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required. Install with: pip install pyyaml"
            )
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            yaml.dump(
                self.to_dict(),
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    # ----------------------------------------------------------
    # Merging
    # ----------------------------------------------------------

    def merge(self, overrides):
        # type: (Dict[str, Any]) -> AssessmentConfig
        """Return a **new** config with *overrides* deep‑merged."""
        base = self.to_dict()
        _deep_merge(base, overrides)
        return AssessmentConfig.from_dict(base)

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    def validate(self):
        # type: () -> None
        """Run basic validation checks.  Raises ``ValueError`` on failure."""
        self.scoring.validate_weights()
        if self.execution.max_concurrent < 1:
            raise ValueError("execution.max_concurrent must be >= 1")
        valid_methods = ("rule", "llm_judge", "hybrid")
        if self.evaluation.default_method not in valid_methods:
            raise ValueError(
                "evaluation.default_method must be one of {}".format(valid_methods)
            )


# ============================================================
# Preset Configurations
# ============================================================

_PRESETS = {
    "default": lambda: AssessmentConfig(),

    "quick_test": lambda: AssessmentConfig(
        benchmark=BenchmarkConfig(max_questions=3),
        red_team=RedTeamConfig(max_attacks_per_test=3),
        execution=ExecutionConfig(verbose=False),
        report=ReportConfig(include_details=False),
        assessment_name="quick-test",
    ),

    "full_assessment": lambda: AssessmentConfig(
        benchmark=BenchmarkConfig(max_questions=None),
        red_team=RedTeamConfig(auto_attack_enabled=True),
        evaluation=EvaluationConfig(default_method="hybrid"),
        report=ReportConfig(
            formats=["html", "json"],
            include_details=True,
        ),
        execution=ExecutionConfig(verbose=True),
        assessment_name="full-assessment",
    ),

    "security_focused": lambda: AssessmentConfig(
        benchmark=BenchmarkConfig(enabled=False),
        red_team=RedTeamConfig(auto_attack_enabled=True),
        alignment=AlignmentConfig(
            tests=["Harmlessness", "Bias & Fairness"],
        ),
        scoring=ScoringConfig(
            benchmark_weight=0.0,
            security_weight=0.5,
            alignment_weight=0.5,
        ),
        report=ReportConfig(formats=["html"]),
        assessment_name="security-focused",
    ),

    "benchmark_only": lambda: AssessmentConfig(
        red_team=RedTeamConfig(enabled=False),
        alignment=AlignmentConfig(enabled=False),
        scoring=ScoringConfig(
            benchmark_weight=1.0,
            security_weight=0.0,
            alignment_weight=0.0,
        ),
        assessment_name="benchmark-only",
    ),

    "alignment_only": lambda: AssessmentConfig(
        benchmark=BenchmarkConfig(enabled=False),
        red_team=RedTeamConfig(enabled=False),
        scoring=ScoringConfig(
            benchmark_weight=0.0,
            security_weight=0.0,
            alignment_weight=1.0,
        ),
        assessment_name="alignment-only",
    ),
}


def get_preset_config(name):
    # type: (str) -> AssessmentConfig
    """Return a **copy** of a named preset configuration.

    Available presets: ``default``, ``quick_test``, ``full_assessment``,
    ``security_focused``, ``benchmark_only``, ``alignment_only``.
    """
    factory = _PRESETS.get(name)
    if factory is None:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(
            "Unknown preset '{}'. Available: {}".format(name, available)
        )
    return factory()


def list_presets():
    # type: () -> List[str]
    """Return the names of all available preset configurations."""
    return sorted(_PRESETS.keys())


# ============================================================
# Internal Helpers
# ============================================================

def _parse_dataclass(dc_class, data):
    """Parse a dict into a dataclass, silently ignoring unknown keys."""
    if isinstance(data, dc_class):
        return data
    if not isinstance(data, dict):
        return dc_class()
    known = {f.name for f in fields(dc_class)}
    filtered = {k: v for k, v in data.items() if k in known}
    return dc_class(**filtered)


def _deep_merge(base, overrides):
    # type: (dict, dict) -> None
    """Recursively merge *overrides* into *base* (mutates *base*)."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
