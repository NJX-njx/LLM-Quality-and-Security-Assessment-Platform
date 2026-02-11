"""
LLM 质量与安全评估平台

统一的大语言模型评估平台，涵盖能力基准测试、安全红队测试和对齐验证三大维度。
"""

__version__ = "0.2.0"  # 版本号
__author__ = "LLM Assessment Team"  # 作者

# 导入核心组件供外部使用
from .core.assessment import (  # 评估平台主类 + 事件系统
    AssessmentPlatform,
    AssessmentEvent,
    EventBus,
    CallbackHandler,
    LoggingCallback,
    ProgressCallback,
    ModuleRegistry,
)
from .core.config import (  # 配置管理
    AssessmentConfig,
    get_preset_config,
    list_presets,
)
from .core.report import ReportGenerator  # 报告生成器
from .core.llm_wrapper import create_llm, BaseLLM  # LLM 工厂函数和基类
from .core.evaluation import (  # 评估引擎
    EvaluationEngine,
    EvaluationResult,
    EvaluationContext,
    EvaluationType,
    EvaluationStrategy,
    create_evaluation_engine,
)

__all__ = [
    # Orchestrator
    "AssessmentPlatform",
    "AssessmentEvent",
    "EventBus",
    "CallbackHandler",
    "LoggingCallback",
    "ProgressCallback",
    "ModuleRegistry",
    # Config
    "AssessmentConfig",
    "get_preset_config",
    "list_presets",
    # Report
    "ReportGenerator",
    # LLM
    "create_llm",
    "BaseLLM",
    # Evaluation
    "EvaluationEngine",
    "EvaluationResult",
    "EvaluationContext",
    "EvaluationType",
    "EvaluationStrategy",
    "create_evaluation_engine",
]
