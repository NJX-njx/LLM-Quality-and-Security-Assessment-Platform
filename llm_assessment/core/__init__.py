"""核心模块初始化 - 包含评估编排、LLM 封装、配置管理、数据集加载、提示词模板和报告生成"""

# --- Assessment Orchestrator ---
from .assessment import (
    AssessmentPlatform,
    AssessmentEvent,
    AssessmentSession,
    EventBus,
    CallbackHandler,
    LoggingCallback,
    ProgressCallback,
    ModuleRegistry,
    HealthScoreCalculator,
)

# --- Configuration ---
from .config import (
    AssessmentConfig,
    BenchmarkConfig,
    RedTeamConfig,
    AlignmentConfig,
    EvaluationConfig,
    ScoringConfig,
    ReportConfig,
    ExecutionConfig,
    get_preset_config,
    list_presets,
)

# --- Evaluation Engine ---
from .evaluation import (
    EvaluationEngine,
    EvaluationResult,
    EvaluationContext,
    EvaluationStrategy,
    EvaluationType,
    CostTracker,
    BaseEvaluator,
    LLMJudgeEvaluator,
    HybridEvaluator,
    EvaluationRouter,
    BaseExternalClassifier,
    PerspectiveAPIClassifier,
    LocalToxicityClassifier,
    create_evaluation_engine,
)

# --- Dataset Loading ---
from .datasets import (
    StandardQuestion,
    DatasetRegistry,
    BaseDatasetLoader,
)

# --- Prompt Templates ---
from .prompts import PromptTemplate, PromptTemplateRegistry, get_default_registry

__all__ = [
    # Orchestrator
    "AssessmentPlatform",
    "AssessmentEvent",
    "AssessmentSession",
    "EventBus",
    "CallbackHandler",
    "LoggingCallback",
    "ProgressCallback",
    "ModuleRegistry",
    "HealthScoreCalculator",
    # Config
    "AssessmentConfig",
    "BenchmarkConfig",
    "RedTeamConfig",
    "AlignmentConfig",
    "EvaluationConfig",
    "ScoringConfig",
    "ReportConfig",
    "ExecutionConfig",
    "get_preset_config",
    "list_presets",
    # Evaluation
    "EvaluationEngine",
    "EvaluationResult",
    "EvaluationContext",
    "EvaluationStrategy",
    "EvaluationType",
    "CostTracker",
    "BaseEvaluator",
    "LLMJudgeEvaluator",
    "HybridEvaluator",
    "EvaluationRouter",
    "BaseExternalClassifier",
    "PerspectiveAPIClassifier",
    "LocalToxicityClassifier",
    "create_evaluation_engine",
    # Datasets
    "StandardQuestion",
    "DatasetRegistry",
    "BaseDatasetLoader",
    # Prompts
    "PromptTemplate",
    "PromptTemplateRegistry",
    "get_default_registry",
]
