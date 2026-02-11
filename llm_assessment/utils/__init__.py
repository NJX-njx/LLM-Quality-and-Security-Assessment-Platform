"""工具模块初始化 - 包含评估指标、文本匹配、统计计算等通用工具函数"""

from .metrics import (
    exact_match,
    contains_match,
    regex_match,
    keyword_match,
    numerical_match,
    extract_choice_letter,
    extract_number,
    accuracy,
    weighted_average,
    normalize_score,
    severity_to_score,
)
