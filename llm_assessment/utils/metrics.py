"""
评估指标和评分工具模块。

提供可复用的文本比较、统计评分和结果聚合函数，
被所有评估器策略共同使用。

兼容 Python >= 3.8 — 不使用海象运算符或更新版本特有功能。
"""

import re
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ============================================================
# 文本匹配指标
# ============================================================

def exact_match(predicted: str, expected: str, normalize: bool = True) -> bool:
    """
    预测与期望字符串的精确匹配比较。

    参数：
        predicted: 模型输出文本。
        expected: 地面真值文本。
        normalize: 如为 True，先去除空白并转小写后再比较。

    返回：
        匹配则返回 True。
    """
    if normalize:
        predicted = predicted.strip().lower()
        expected = expected.strip().lower()
    return predicted == expected


def contains_match(text: str, target: str, case_sensitive: bool = False) -> bool:
    """
    检查 *target* 是否出现在 *text* 中的任何位置。

    参数：
        text: 被搜索的文本。
        target: 要查找的字符串。
        case_sensitive: 是否区分大小写。

    返回：
        找到则返回 True。
    """
    if not case_sensitive:
        text = text.lower()
        target = target.lower()
    return target in text


def regex_match(text: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    """
    检查 *pattern* 是否在 *text* 中匹配。

    参数：
        text: 输入文本。
        pattern: 正则表达式模式。
        flags: ``re`` 标志（默认: 忽略大小写）。

    返回：
        匹配则返回 True。
    """
    try:
        return bool(re.search(pattern, text, flags))
    except re.error:
        return False


def regex_match_any(text: str, patterns: List[str],
                    flags: int = re.IGNORECASE) -> bool:
    """如果 *patterns* 中任一模式在 *text* 中匹配则返回 True"""
    return any(regex_match(text, p, flags) for p in patterns)


def regex_match_all(text: str, patterns: List[str],
                    flags: int = re.IGNORECASE) -> bool:
    """仅当 *patterns* 中所有模式都在 *text* 中匹配时返回 True"""
    return all(regex_match(text, p, flags) for p in patterns)


def keyword_match(text: str, keywords: List[str],
                  match_all: bool = False,
                  case_sensitive: bool = False) -> bool:
    """
    检查 *text* 中是否存在关键词。

    参数：
        text: 输入文本。
        keywords: 要查找的关键词列表。
        match_all: True 表示必须全部包含，否则任一包含即可。
        case_sensitive: 是否区分大小写。

    返回：
        满足关键词条件时返回 True。
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]
    if match_all:
        return all(k in text for k in keywords)
    return any(k in text for k in keywords)


def numerical_match(text: str, expected: float,
                    tolerance: float = 1e-6) -> bool:
    """
    从 *text* 中提取第一个数字并与 *expected* 比较。

    支持整数、小数、负数和带逗号分隔的数值。

    参数：
        text: 模型输出文本。
        expected: 期望的数值答案。
        tolerance: 允许的绝对差值。

    返回：
        提取的数字在容差范围内则返回 True。
    """
    # 从数字中移除逗号（如 "1,234" -> "1234"）
    cleaned = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", cleaned)
    for n in numbers:
        try:
            value = float(n)
            if abs(value - expected) <= tolerance:
                return True
        except ValueError:
            continue
    return False


def extract_choice_letter(text: str,
                          valid_choices: str = "ABCDEFGH") -> Optional[str]:
    """
    从模型输出中提取单个选项字母。

    支持常见格式：``"A"``、``"(A)"``、``"A."``、``"Answer: A"``。

    参数：
        text: 模型输出文本。
        valid_choices: 有效选项字母字符串。

    返回：
        大写选项字母，或未找到时返回 None。
    """
    text = text.strip()

    # 模式 1: 以单个有效字母开头（可能带标点）
    m = re.match(r'^[(\[]?([' + valid_choices + r'])[)\].]?\s', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 模式 2: "Answer: X" 或 "The answer is X"
    m = re.search(
        r'(?:answer|choice|option)\s*(?:is|:)\s*[(\[]?([' + valid_choices + r'])[)\].]?',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # 模式 3: 仅为单个字母（整个响应）
    text_stripped = text.strip().strip("()[].")
    if len(text_stripped) == 1 and text_stripped.upper() in valid_choices:
        return text_stripped.upper()

    # 模式 4: 文本中最后一个有效选项字母
    m = re.search(r'[(\[]?([' + valid_choices + r'])[)\].]?\s*$', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def extract_number(text: str) -> Optional[float]:
    """
    从文本中提取最终数值答案，处理思维链格式。

    查找如 ``#### 42``、``The answer is 42`` 等模式，
    或文本中的最后一个数字。

    参数：
        text: 模型输出文本。

    返回：
        提取的数字，或 None。
    """
    # GSM8K 风格 "#### <数字>"
    m = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', text)
    if m:
        return float(m.group(1).replace(",", ""))

    # "The answer is <数字>"
    m = re.search(
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(-?\d[\d,]*\.?\d*)',
        text, re.IGNORECASE
    )
    if m:
        return float(m.group(1).replace(",", ""))

    # 回退：文本中的最后一个数字
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


# ============================================================
# 统计 / 聚合指标
# ============================================================

def accuracy(correct: int, total: int) -> float:
    """
    计算准确率（百分比 0-100）。

    参数：
        correct: 正确答案数。
        total: 总题目数。

    返回：
        准确率百分比。
    """
    if total <= 0:
        return 0.0
    return (correct / total) * 100.0


def weighted_average(scores: List[float], weights: List[float]) -> float:
    """
    使用 *weights* 计算 *scores* 的加权平均值。

    参数：
        scores: 分数值列表（0-100）。
        weights: 对应权重。无需总和为 1。

    返回：
        加权平均值。
    """
    if len(scores) != len(weights):
        raise ValueError("scores and weights must have the same length")
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def harmonic_mean(values: List[float]) -> float:
    """
    计算非零正值的调和平均值。

    适用于 F1 计算等聚合场景。

    参数：
        values: 正浮点数列表。

    返回：
        调和平均值，或任一值为零时返回 0.0。
    """
    positive = [v for v in values if v > 0]
    if not positive or len(positive) != len(values):
        return 0.0
    return len(positive) / sum(1.0 / v for v in positive)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    从混淆矩阵计数计算准确率、召回率和 F1。

    参数：
        tp: 真正例。
        fp: 假正例。
        fn: 假反例。

    返回：
        含 ``precision``、``recall``、``f1`` 键的字典（0-1 范围）。
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def cohen_kappa(observed_agreement: float,
                expected_agreement: float) -> float:
    """
    计算 Cohen's Kappa 评分者间一致性系数。

    参数：
        observed_agreement: 实际一致比例（0-1）。
        expected_agreement: 随机一致比例（0-1）。

    返回：
        Kappa 值，范围 [-1, 1]。
    """
    if expected_agreement >= 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def confidence_interval(score: float, n: int,
                        confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算比例的 Wilson 分数置信区间。

    参数：
        score: 观察比例（0-1）。
        n: 样本量。
        confidence: 置信水平（默认 0.95）。

    返回：
        (下界, 上界) 比例值。
    """
    if n <= 0:
        return (0.0, 0.0)
    # 常见置信水平的 z 值
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)

    denominator = 1 + z * z / n
    centre = (score + z * z / (2 * n)) / denominator
    spread = z * math.sqrt((score * (1 - score) + z * z / (4 * n)) / n) / denominator
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return (lower, upper)


# ============================================================
# 文本相似度指标
# ============================================================

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    计算两段文本的 Jaccard 相似度（词级别）。

    参数：
        text_a: 第一段文本。
        text_b: 第二段文本。

    返回：
        相似度，范围 [0, 1]。
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def ngram_overlap(text_a: str, text_b: str, n: int = 2) -> float:
    """
    计算两段文本的 n-gram 重叠率。

    参数：
        text_a: 第一段文本。
        text_b: 第二段文本。
        n: N-gram 大小（默认为 bigram）。

    返回：
        重叠率，范围 [0, 1]。
    """
    def get_ngrams(text, size):
        words = text.lower().split()
        return set(tuple(words[i:i + size]) for i in range(len(words) - size + 1))

    ngrams_a = get_ngrams(text_a, n)
    ngrams_b = get_ngrams(text_b, n)
    if not ngrams_a and not ngrams_b:
        return 1.0
    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union) if union else 0.0


# ============================================================
# 分数归一化
# ============================================================

def normalize_score(raw_score: float,
                    min_val: float = 0.0,
                    max_val: float = 100.0,
                    clip: bool = True) -> float:
    """
    将原始分数归一化到 [0, 100] 范围。

    参数：
        raw_score: 原始分数值。
        min_val: 期望的最小值。
        max_val: 期望的最大值。
        clip: 如为 True，将结果截断到 [0, 100]。

    返回：
        归一化分数（0-100）。
    """
    if max_val <= min_val:
        return 0.0
    normalized = ((raw_score - min_val) / (max_val - min_val)) * 100.0
    if clip:
        normalized = max(0.0, min(100.0, normalized))
    return normalized


def severity_to_score(severity: str) -> float:
    """
    将严重性标签转换为数值影响分数。

    映射：
        critical → 1.0, high → 0.8, medium → 0.5, low → 0.2, info → 0.05

    参数：
        severity: 'critical'、'high'、'medium'、'low'、'info' 之一。

    返回：
        影响分数，范围 (0, 1]。
    """
    mapping = {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2,
        "info": 0.05,
    }
    return mapping.get(severity.lower().strip(), 0.5)
