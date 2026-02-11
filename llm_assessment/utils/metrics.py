"""
Metrics and scoring utilities for evaluation.

Provides reusable text comparison, statistical scoring, and result
aggregation functions used across all evaluator strategies.

Compatible with Python >= 3.8 — no walrus operator or newer-only features.
"""

import re
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ============================================================
# Text Matching Metrics
# ============================================================

def exact_match(predicted: str, expected: str, normalize: bool = True) -> bool:
    """
    Exact match comparison between predicted and expected strings.

    Args:
        predicted: Model output text.
        expected: Ground truth text.
        normalize: If True, strip whitespace and lower-case before comparing.

    Returns:
        True if the texts match.
    """
    if normalize:
        predicted = predicted.strip().lower()
        expected = expected.strip().lower()
    return predicted == expected


def contains_match(text: str, target: str, case_sensitive: bool = False) -> bool:
    """
    Check if *target* appears anywhere in *text*.

    Args:
        text: The haystack.
        target: The needle.
        case_sensitive: Whether comparison is case-sensitive.

    Returns:
        True if ``target`` is found inside ``text``.
    """
    if not case_sensitive:
        text = text.lower()
        target = target.lower()
    return target in text


def regex_match(text: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    """
    Check if *pattern* matches anywhere in *text*.

    Args:
        text: Input text to search.
        pattern: Regular expression pattern.
        flags: ``re`` flags (default: IGNORECASE).

    Returns:
        True if a match is found.
    """
    try:
        return bool(re.search(pattern, text, flags))
    except re.error:
        return False


def regex_match_any(text: str, patterns: List[str],
                    flags: int = re.IGNORECASE) -> bool:
    """Return True if *any* of the ``patterns`` match inside ``text``."""
    return any(regex_match(text, p, flags) for p in patterns)


def regex_match_all(text: str, patterns: List[str],
                    flags: int = re.IGNORECASE) -> bool:
    """Return True only if *all* of the ``patterns`` match inside ``text``."""
    return all(regex_match(text, p, flags) for p in patterns)


def keyword_match(text: str, keywords: List[str],
                  match_all: bool = False,
                  case_sensitive: bool = False) -> bool:
    """
    Check for the presence of keywords in *text*.

    Args:
        text: Input text.
        keywords: List of keywords to look for.
        match_all: If True, *all* keywords must be present; otherwise *any*.
        case_sensitive: Whether comparison is case-sensitive.

    Returns:
        True when the keyword condition is satisfied.
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
    Extract the first number from *text* and compare to *expected*.

    Handles integers, decimals, negative numbers, and comma-separated values.

    Args:
        text: Model output text.
        expected: Expected numerical answer.
        tolerance: Allowed absolute difference.

    Returns:
        True if extracted number is within ``tolerance`` of ``expected``.
    """
    # Remove commas from numbers (e.g. "1,234" -> "1234")
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
    Extract a single choice letter from model output.

    Handles common formats: ``"A"``, ``"(A)"``, ``"A."``, ``"Answer: A"``.

    Args:
        text: Model output text.
        valid_choices: String of valid choice letters.

    Returns:
        Uppercase choice letter, or None if not found.
    """
    text = text.strip()

    # Pattern 1: Starts with a single valid letter (possibly with punctuation)
    m = re.match(r'^[(\[]?([' + valid_choices + r'])[)\].]?\s', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 2: "Answer: X" or "The answer is X"
    m = re.search(
        r'(?:answer|choice|option)\s*(?:is|:)\s*[(\[]?([' + valid_choices + r'])[)\].]?',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # Pattern 3: Just a single letter on its own (entire response)
    text_stripped = text.strip().strip("()[].")
    if len(text_stripped) == 1 and text_stripped.upper() in valid_choices:
        return text_stripped.upper()

    # Pattern 4: Last letter in text that is a valid choice
    m = re.search(r'[(\[]?([' + valid_choices + r'])[)\].]?\s*$', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def extract_number(text: str) -> Optional[float]:
    """
    Extract the final numerical answer from text, handling chain-of-thought.

    Looks for patterns like ``#### 42``, ``The answer is 42``, or just the
    last number in the text.

    Args:
        text: Model output text.

    Returns:
        Extracted number or None.
    """
    # GSM8K-style "#### <number>"
    m = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', text)
    if m:
        return float(m.group(1).replace(",", ""))

    # "The answer is <number>"
    m = re.search(
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(-?\d[\d,]*\.?\d*)',
        text, re.IGNORECASE
    )
    if m:
        return float(m.group(1).replace(",", ""))

    # Fallback: last number in text
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


# ============================================================
# Statistical / Aggregate Metrics
# ============================================================

def accuracy(correct: int, total: int) -> float:
    """
    Compute accuracy as a percentage (0-100).

    Args:
        correct: Number of correct answers.
        total: Total number of questions.

    Returns:
        Accuracy percentage.
    """
    if total <= 0:
        return 0.0
    return (correct / total) * 100.0


def weighted_average(scores: List[float], weights: List[float]) -> float:
    """
    Compute weighted average of *scores* using *weights*.

    Args:
        scores: List of score values (0-100).
        weights: Corresponding weights. Need not sum to 1.

    Returns:
        Weighted average.

    Raises:
        ValueError: If lists have different lengths.
    """
    if len(scores) != len(weights):
        raise ValueError("scores and weights must have the same length")
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def harmonic_mean(values: List[float]) -> float:
    """
    Compute the harmonic mean of non-zero positive values.

    Useful for F1-like aggregation.

    Args:
        values: List of positive floats.

    Returns:
        Harmonic mean, or 0.0 if any value is zero.
    """
    positive = [v for v in values if v > 0]
    if not positive or len(positive) != len(values):
        return 0.0
    return len(positive) / sum(1.0 / v for v in positive)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 from confusion-matrix counts.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.

    Returns:
        Dict with ``precision``, ``recall``, ``f1`` keys (0-1 range).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def cohen_kappa(observed_agreement: float,
                expected_agreement: float) -> float:
    """
    Compute Cohen's Kappa inter-rater agreement coefficient.

    Args:
        observed_agreement: Proportion of actual agreement (0-1).
        expected_agreement: Proportion of agreement by chance (0-1).

    Returns:
        Kappa value in [-1, 1].
    """
    if expected_agreement >= 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def confidence_interval(score: float, n: int,
                        confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    Args:
        score: Observed proportion (0-1).
        n: Sample size.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower_bound, upper_bound) as proportions.
    """
    if n <= 0:
        return (0.0, 0.0)
    # z-value for common confidence levels
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)

    denominator = 1 + z * z / n
    centre = (score + z * z / (2 * n)) / denominator
    spread = z * math.sqrt((score * (1 - score) + z * z / (4 * n)) / n) / denominator
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return (lower, upper)


# ============================================================
# Text Similarity Metrics
# ============================================================

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between two texts (word-level).

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Similarity in [0, 1].
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
    Compute n-gram overlap ratio between two texts.

    Args:
        text_a: First text.
        text_b: Second text.
        n: N-gram size (default bigram).

    Returns:
        Overlap ratio in [0, 1].
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
# Score Normalization
# ============================================================

def normalize_score(raw_score: float,
                    min_val: float = 0.0,
                    max_val: float = 100.0,
                    clip: bool = True) -> float:
    """
    Normalize a raw score into the [0, 100] range.

    Args:
        raw_score: The raw score value.
        min_val: Minimum expected value.
        max_val: Maximum expected value.
        clip: If True, clip the result to [0, 100].

    Returns:
        Normalized score (0-100).
    """
    if max_val <= min_val:
        return 0.0
    normalized = ((raw_score - min_val) / (max_val - min_val)) * 100.0
    if clip:
        normalized = max(0.0, min(100.0, normalized))
    return normalized


def severity_to_score(severity: str) -> float:
    """
    Convert a severity label to a numeric impact score.

    Mapping:
        critical → 1.0, high → 0.8, medium → 0.5, low → 0.2, info → 0.05

    Args:
        severity: One of 'critical', 'high', 'medium', 'low', 'info'.

    Returns:
        Impact score in (0, 1].
    """
    mapping = {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2,
        "info": 0.05,
    }
    return mapping.get(severity.lower().strip(), 0.5)
