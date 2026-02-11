"""
Unified Data & Dataset Layer for LLM Assessment Platform.

Provides a standard abstraction for loading evaluation data from multiple
sources: built-in JSONL files, HuggingFace ``datasets``, and user-supplied
local files.  Three canonical data classes unify all modules:

- :class:`StandardQuestion` — benchmark questions
- :class:`AttackPrompt` — red-teaming attack prompts
- :class:`AlignmentScenario` — alignment/safety test scenarios

High-level entry point :class:`DatasetLoader` orchestrates all data access
with automatic HuggingFace → built-in fallback.

Usage::

    from llm_assessment.core.datasets import DatasetLoader

    loader = DatasetLoader()
    questions = loader.load_benchmark("mmlu", max_samples=50)
    attacks   = loader.load_attacks("jailbreak")
    scenarios = loader.load_alignment("hhh")

Design references:
- EleutherAI lm-evaluation-harness (task abstraction)
- HuggingFace datasets (streaming & caching)
- OpenAI evals (registry + config pattern)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
import json
import hashlib
import os
import logging
import random

logger = logging.getLogger(__name__)


# ============================================================
# Standardised Question Format
# ============================================================

@dataclass
class StandardQuestion:
    """Universal question format shared across all benchmarks.

    This dataclass normalises questions regardless of their source
    (HuggingFace, JSONL, hardcoded).  Individual benchmarks convert
    their native data into ``StandardQuestion`` instances before
    evaluation.

    Attributes:
        question_id: Unique identifier within the dataset.
        question: The question text.
        choices: Optional list of answer choices (e.g. for MCQ).
        correct_answer: The expected correct answer (letter, number,
            or free text).
        category: Top-level category (knowledge, reasoning, coding, etc.).
        subcategory: Fine-grained sub-topic.
        subject: Optional subject/topic label.
        difficulty: Optional difficulty level (easy/medium/hard/expert).
        metadata: Arbitrary extra data.
        evaluation_type: How to evaluate the answer:
            ``"exact_match"`` | ``"contains"`` | ``"regex"`` |
            ``"code_exec"`` | ``"llm_judge"`` | ``"numeric_match"`` |
            ``"keyword"``.
        source: Origin dataset name.
        explanation: Optional explanation of the answer.
        context: Optional context passage for the question.
    """
    question_id: str = ""
    question: str = ""
    choices: Optional[List[str]] = None
    correct_answer: str = ""
    category: str = "general"
    subcategory: str = ""
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_type: str = "exact_match"
    source: str = ""
    explanation: str = ""
    context: str = ""

    def to_dict(self):
        # type: () -> Dict[str, Any]
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        # type: (Dict[str, Any]) -> StandardQuestion
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        # Ensure correct_answer is a string
        if "correct_answer" in filtered:
            filtered["correct_answer"] = str(filtered["correct_answer"])
        return cls(**filtered)

    def format_prompt(self, include_choices=True, instruction=None):
        # type: (bool, Optional[str]) -> str
        """Format this question into a prompt string for the LLM.

        Args:
            include_choices: Append labelled choices for MCQs.
            instruction: Override the default answer instruction.

        Returns:
            The formatted prompt.
        """
        parts = [self.question]

        if include_choices and self.choices:
            parts.append("")
            parts.append("Choices:")
            for idx, choice in enumerate(self.choices):
                parts.append("{letter}. {text}".format(
                    letter=chr(65 + idx), text=choice,
                ))

        if instruction:
            parts.append("")
            parts.append(instruction)
        elif self.choices:
            parts.append("")
            parts.append("Answer with just the letter (A, B, C, or D):")
        else:
            parts.append("")
            parts.append("Provide a concise answer:")

        return "\n".join(parts)


# ============================================================
# Attack Prompt Format
# ============================================================

@dataclass
class AttackPrompt:
    """Canonical attack prompt format for the red-teaming module.

    Each attack prompt carries metadata about its category, severity,
    and expected safe/vulnerable behaviour patterns.

    For multi-turn attacks, ``turns`` is a list of message dicts
    (``[{"role": "user", "content": "..."}]``) instead of a single
    ``prompt`` string.
    """
    id: str = ""
    prompt: str = ""
    category: str = "jailbreak"        # jailbreak / injection / leakage / encoding / multi_turn
    subcategory: str = ""
    severity: str = "medium"           # low / medium / high / critical
    attack_type: str = ""
    description: str = ""
    expected_safe_behavior: str = ""
    tags: List[str] = field(default_factory=list)
    reference: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    turns: Optional[List[Dict[str, str]]] = None  # multi-turn messages

    def to_dict(self):
        # type: () -> Dict[str, Any]
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # type: (Dict[str, Any]) -> AttackPrompt
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ============================================================
# Alignment Scenario Format
# ============================================================

@dataclass
class AlignmentScenario:
    """Canonical scenario format for alignment / safety tests.

    Used by HHH tests, refusal calibration, bias detection,
    and hallucination evaluation.
    """
    id: str = ""
    prompt: str = ""
    category: str = "alignment"        # helpfulness / harmlessness / honesty / bias / refusal / hallucination
    subcategory: str = ""
    expected_behavior: str = ""
    should_refuse: Optional[bool] = None  # for refusal calibration
    evaluation_type: str = "regex"     # regex / llm_judge / keyword
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        # type: () -> Dict[str, Any]
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # type: (Dict[str, Any]) -> AlignmentScenario
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ============================================================
# JSONL I/O Utilities
# ============================================================

def load_jsonl(filepath):
    # type: (str) -> List[Dict[str, Any]]
    """Load data from a JSONL file (one JSON object per line).

    Blank lines and lines starting with ``//`` are silently skipped.
    """
    items = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "JSONL parse error at %s:%d — %s", filepath, lineno, exc
                )
    return items


def save_jsonl(filepath, items):
    # type: (str, List[Dict[str, Any]]) -> None
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def _file_hash(filepath):
    # type: (str) -> str
    """Return MD5 hex digest of a file for cache-invalidation."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Base Dataset Loader
# ============================================================

class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    Subclasses must implement :meth:`load` which returns a list of
    :class:`StandardQuestion`.
    """

    @abstractmethod
    def load(self, max_items=None, **kwargs):
        # type: (Optional[int], **Any) -> List[StandardQuestion]
        """Load questions from the data source."""
        pass

    def _apply_limit(self, questions, max_items, shuffle=False, seed=42):
        # type: (List[StandardQuestion], Optional[int], bool, int) -> List[StandardQuestion]
        """Optionally shuffle and truncate a question list."""
        if shuffle:
            questions = list(questions)
            rng = random.Random(seed)
            rng.shuffle(questions)
        if max_items is not None and max_items > 0:
            questions = questions[:max_items]
        return questions


# ============================================================
# Built-in (Hardcoded) Dataset Loader
# ============================================================

class BuiltinDatasetLoader(BaseDatasetLoader):
    """Load questions from a Python list of dicts (hardcoded data).

    This is backward-compatible with the existing benchmark
    ``load_questions()`` pattern.
    """

    def __init__(self, data, source_name="builtin"):
        # type: (List[Dict[str, Any]], str) -> None
        self._data = data
        self._source_name = source_name

    def load(self, max_items=None, shuffle=False, seed=42, **kwargs):
        # type: (Optional[int], bool, int, **Any) -> List[StandardQuestion]
        questions = []
        for idx, item in enumerate(self._data):
            q = StandardQuestion.from_dict(item)
            if not q.question_id:
                q.question_id = "{}-{}".format(self._source_name, idx)
            if not q.source:
                q.source = self._source_name
            questions.append(q)
        return self._apply_limit(questions, max_items, shuffle=shuffle, seed=seed)


# ============================================================
# JSONL Dataset Loader
# ============================================================

class JsonlDatasetLoader(BaseDatasetLoader):
    """Load questions from a JSONL (JSON Lines) file.

    Each line must be a valid JSON object conforming to the
    :class:`StandardQuestion` schema.
    """

    def __init__(self, filepath, source_name=None, transform_fn=None):
        # type: (str, Optional[str], Optional[Callable]) -> None
        self._filepath = filepath
        self._source_name = source_name or os.path.basename(filepath)
        self._transform_fn = transform_fn

    def load(self, max_items=None, shuffle=False, seed=42, **kwargs):
        # type: (Optional[int], bool, int, **Any) -> List[StandardQuestion]
        questions = []
        raw = load_jsonl(self._filepath)
        for idx, data in enumerate(raw):
            if self._transform_fn:
                data = self._transform_fn(data)
            q = StandardQuestion.from_dict(data)
            if not q.question_id:
                q.question_id = "{}-{}".format(self._source_name, idx)
            if not q.source:
                q.source = self._source_name
            questions.append(q)
        return self._apply_limit(questions, max_items, shuffle=shuffle, seed=seed)


# ============================================================
# Package Data Loader (built-in JSONL files)
# ============================================================

class PackageDataLoader:
    """Load data from the built-in JSONL files shipped with the package.

    Resolves paths relative to ``llm_assessment/data/`` automatically
    and caches results in memory.
    """

    def __init__(self):
        from ..data import DATA_DIR
        self._data_dir = DATA_DIR
        self._cache = {}  # filepath -> (hash, data)

    def _resolve(self, subdir, filename):
        # type: (str, str) -> str
        path = os.path.join(self._data_dir, subdir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Built-in data file not found: %s (looked in %s)"
                % (filename, path)
            )
        return path

    def load_raw(self, subdir, filename, max_items=None, shuffle=False, seed=42):
        # type: (str, str, Optional[int], bool, int) -> List[Dict[str, Any]]
        """Load raw dicts from ``llm_assessment/data/<subdir>/<filename>``.

        Results are cached; cache is invalidated when the file changes.
        """
        path = self._resolve(subdir, filename)
        fhash = _file_hash(path)
        if path in self._cache and self._cache[path][0] == fhash:
            data = list(self._cache[path][1])
        else:
            data = load_jsonl(path)
            self._cache[path] = (fhash, data)
            data = list(data)  # copy for safety
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(data)
        if max_items is not None and max_items > 0:
            data = data[:max_items]
        return data

    # Typed loaders ---------------------------------------------------------

    def load_questions(self, filename, **kwargs):
        # type: (str, **Any) -> List[StandardQuestion]
        """Load benchmark questions from ``data/benchmarks/<filename>``."""
        raw = self.load_raw("benchmarks", filename, **kwargs)
        return [StandardQuestion.from_dict(item) for item in raw]

    def load_attacks(self, filename, **kwargs):
        # type: (str, **Any) -> List[AttackPrompt]
        """Load attack prompts from ``data/attacks/<filename>``."""
        raw = self.load_raw("attacks", filename, **kwargs)
        return [AttackPrompt.from_dict(item) for item in raw]

    def load_scenarios(self, filename, **kwargs):
        # type: (str, **Any) -> List[AlignmentScenario]
        """Load alignment scenarios from ``data/alignment/<filename>``."""
        raw = self.load_raw("alignment", filename, **kwargs)
        return [AlignmentScenario.from_dict(item) for item in raw]

    def load_hallucination_prompts(self, filename="factual_prompts.jsonl", **kwargs):
        # type: (str, **Any) -> List[AlignmentScenario]
        """Load hallucination prompts from ``data/hallucination/``."""
        raw = self.load_raw("hallucination", filename, **kwargs)
        return [AlignmentScenario.from_dict(item) for item in raw]

    def list_files(self, subdir):
        # type: (str) -> List[str]
        """List available JSONL files in a sub-directory."""
        dirpath = os.path.join(self._data_dir, subdir)
        if not os.path.isdir(dirpath):
            return []
        return sorted(
            f for f in os.listdir(dirpath)
            if f.endswith(".jsonl") and not f.startswith("_")
        )


# ============================================================
# HuggingFace Dataset Loader
# ============================================================

class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """Load questions from a HuggingFace ``datasets`` dataset.

    Requires the ``datasets`` package (imported lazily).
    """

    def __init__(
        self,
        dataset_id,            # type: str
        split="test",          # type: str
        config_name=None,      # type: Optional[str]
        transform_fn=None,     # type: Optional[Callable]
        source_name=None,      # type: Optional[str]
    ):
        self._dataset_id = dataset_id
        self._split = split
        self._config_name = config_name
        self._transform_fn = transform_fn
        self._source_name = source_name or dataset_id

    def load(self, max_items=None, shuffle=False, seed=42, **kwargs):
        # type: (Optional[int], bool, int, **Any) -> List[StandardQuestion]
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for HuggingFace data. "
                "Install with: pip install datasets"
            )

        logger.info(
            "Loading HuggingFace dataset: %s (split=%s, config=%s)",
            self._dataset_id, self._split, self._config_name,
        )
        ds = load_dataset(
            self._dataset_id,
            self._config_name,
            split=self._split,
            **kwargs,
        )

        questions = []
        for idx, row in enumerate(ds):
            raw = dict(row)
            if self._transform_fn:
                raw = self._transform_fn(raw)
            q = StandardQuestion.from_dict(raw)
            if not q.question_id:
                q.question_id = "{}-{}".format(self._source_name, idx)
            if not q.source:
                q.source = self._source_name
            questions.append(q)

        return self._apply_limit(questions, max_items, shuffle=shuffle, seed=seed)


# ============================================================
# Dataset Registry (lower-level)
# ============================================================

class DatasetRegistry:
    """Registry for named dataset loaders.

    Benchmarks register their loaders here so that the orchestrator
    (or CLI) can discover and load datasets by name.

    Usage::

        DatasetRegistry.register("mmlu_sample", BuiltinDatasetLoader(data))
        questions = DatasetRegistry.load("mmlu_sample", max_items=5)
    """

    _loaders = {}  # type: Dict[str, BaseDatasetLoader]

    @classmethod
    def register(cls, name, loader):
        # type: (str, BaseDatasetLoader) -> None
        if name in cls._loaders:
            logger.warning("Overwriting existing dataset loader: %s", name)
        cls._loaders[name] = loader

    @classmethod
    def get(cls, name):
        # type: (str) -> Optional[BaseDatasetLoader]
        return cls._loaders.get(name)

    @classmethod
    def load(cls, name, max_items=None, **kwargs):
        # type: (str, Optional[int], **Any) -> List[StandardQuestion]
        loader = cls._loaders.get(name)
        if loader is None:
            raise KeyError(
                "No dataset loader registered for '{}'. "
                "Available: {}".format(name, list(cls._loaders.keys()))
            )
        return loader.load(max_items=max_items, **kwargs)

    @classmethod
    def list_datasets(cls):
        # type: () -> List[str]
        return sorted(cls._loaders.keys())

    @classmethod
    def clear(cls):
        # type: () -> None
        cls._loaders.clear()


# ============================================================
# Transform Functions for Common HuggingFace Datasets
# ============================================================

def mmlu_transform(row):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Transform a HuggingFace MMLU row into StandardQuestion format."""
    answer_idx = row.get("answer", 0)
    answer_letter = chr(65 + answer_idx) if isinstance(answer_idx, int) else str(answer_idx)
    return {
        "question": row.get("question", ""),
        "choices": row.get("choices", []),
        "correct_answer": answer_letter,
        "subject": row.get("subject", "general"),
        "category": "knowledge",
        "evaluation_type": "exact_match",
    }


def gsm8k_transform(row):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Transform a HuggingFace GSM8K row into StandardQuestion format."""
    raw_answer = row.get("answer", "")
    final_answer = raw_answer
    if "####" in raw_answer:
        final_answer = raw_answer.split("####")[-1].strip()
    return {
        "question": row.get("question", ""),
        "correct_answer": final_answer,
        "category": "reasoning",
        "subcategory": "math",
        "evaluation_type": "numeric_match",
        "explanation": raw_answer,
    }


def truthfulqa_transform(row):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Transform a HuggingFace TruthfulQA row into StandardQuestion format."""
    return {
        "question": row.get("question", ""),
        "correct_answer": row.get("best_answer", ""),
        "category": "factuality",
        "evaluation_type": "llm_judge",
        "metadata": {
            "correct_answers": row.get("correct_answers", []),
            "incorrect_answers": row.get("incorrect_answers", []),
            "category": row.get("category", ""),
        },
    }


def humaneval_transform(row):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Transform a HuggingFace HumanEval row into StandardQuestion format."""
    return {
        "question_id": row.get("task_id", ""),
        "question": row.get("prompt", ""),
        "correct_answer": row.get("canonical_solution", ""),
        "category": "coding",
        "subcategory": "python",
        "evaluation_type": "code_exec",
        "metadata": {
            "test": row.get("test", ""),
            "entry_point": row.get("entry_point", ""),
        },
    }


def cmmlu_transform(row):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Transform a HuggingFace CMMLU row into StandardQuestion format."""
    choices = [row.get(c, "") for c in ["A", "B", "C", "D"]]
    return {
        "question": row.get("Question", ""),
        "choices": choices,
        "correct_answer": str(row.get("Answer", "")),
        "subject": row.get("Subject", ""),
        "category": "knowledge",
        "subcategory": "chinese",
        "evaluation_type": "exact_match",
    }


# ============================================================
# Benchmark Configuration Registry
# ============================================================

_BENCHMARK_CONFIGS = {
    "mmlu": {
        "hf_id": "cais/mmlu",
        "hf_subset": "all",
        "hf_split": "test",
        "transform": mmlu_transform,
        "builtin_file": "mmlu_sample.jsonl",
        "source_name": "mmlu",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "hf_subset": "main",
        "hf_split": "test",
        "transform": gsm8k_transform,
        "builtin_file": "gsm8k_sample.jsonl",
        "source_name": "gsm8k",
    },
    "truthfulqa": {
        "hf_id": "truthfulqa/truthful_qa",
        "hf_subset": "generation",
        "hf_split": "validation",
        "transform": truthfulqa_transform,
        "builtin_file": "truthfulqa_sample.jsonl",
        "source_name": "truthfulqa",
    },
    "humaneval": {
        "hf_id": "openai/openai_humaneval",
        "hf_subset": None,
        "hf_split": "test",
        "transform": humaneval_transform,
        "builtin_file": "humaneval_sample.jsonl",
        "source_name": "humaneval",
    },
    "cmmlu": {
        "hf_id": "haonan-li/cmmlu",
        "hf_subset": "all",
        "hf_split": "test",
        "transform": cmmlu_transform,
        "builtin_file": "cmmlu_sample.jsonl",
        "source_name": "cmmlu",
    },
}

_ATTACK_FILES = {
    "jailbreak": "jailbreak_prompts.jsonl",
    "injection": "injection_prompts.jsonl",
    "leakage": "leakage_prompts.jsonl",
    "encoding": "encoding_payloads.jsonl",
    "multi_turn": "multi_turn_attacks.jsonl",
}

_ALIGNMENT_FILES = {
    "hhh": "hhh_scenarios.jsonl",
    "refusal": "refusal_calibration.jsonl",
    "bias": "bias_scenarios.jsonl",
}


# ============================================================
# DatasetLoader — High-Level Entry Point
# ============================================================

class DatasetLoader:
    """High-level orchestrator for loading datasets across all modules.

    Features:
    - Automatic HuggingFace → built-in fallback
    - Unified API for benchmarks, attacks, and alignment data
    - In-memory caching
    - Filtering, sampling, and shuffling

    Usage::

        loader = DatasetLoader()
        questions = loader.load_benchmark("mmlu", max_samples=50)
        attacks   = loader.load_attacks("jailbreak", severity="high")
        scenarios = loader.load_alignment("refusal")
    """

    def __init__(self, prefer_hf=True, cache_dir=None):
        # type: (bool, Optional[str]) -> None
        """
        Args:
            prefer_hf: Try HuggingFace datasets before built-in fallback.
            cache_dir: Cache directory for HuggingFace downloads.
        """
        self._prefer_hf = prefer_hf
        self._cache_dir = cache_dir
        self._pkg_loader = PackageDataLoader()

        # Allow runtime registration of extra datasets
        self._extra_benchmark_configs = {}     # type: Dict[str, Dict]
        self._extra_attack_files = {}          # type: Dict[str, str]
        self._extra_alignment_files = {}       # type: Dict[str, str]

    # -- Registration -------------------------------------------------------

    def register_benchmark(self, name, config):
        # type: (str, Dict[str, Any]) -> None
        """Register a custom benchmark config at runtime."""
        self._extra_benchmark_configs[name] = config

    def register_attack_file(self, category, filename):
        # type: (str, str) -> None
        """Register a custom attack file at runtime."""
        self._extra_attack_files[category] = filename

    def register_alignment_file(self, category, filename):
        # type: (str, str) -> None
        """Register a custom alignment file at runtime."""
        self._extra_alignment_files[category] = filename

    # -- Discovery ----------------------------------------------------------

    def available_benchmarks(self):
        # type: () -> List[str]
        keys = set(_BENCHMARK_CONFIGS.keys())
        keys.update(self._extra_benchmark_configs.keys())
        return sorted(keys)

    def available_attack_categories(self):
        # type: () -> List[str]
        keys = set(_ATTACK_FILES.keys())
        keys.update(self._extra_attack_files.keys())
        return sorted(keys)

    def available_alignment_categories(self):
        # type: () -> List[str]
        keys = set(_ALIGNMENT_FILES.keys())
        keys.update(self._extra_alignment_files.keys())
        return sorted(keys)

    def dataset_info(self, name):
        # type: (str) -> Dict[str, Any]
        """Get metadata about a benchmark dataset."""
        cfg = self._get_benchmark_config(name)
        return {
            "name": name,
            "hf_dataset_id": cfg.get("hf_id", ""),
            "hf_subset": cfg.get("hf_subset", ""),
            "hf_split": cfg.get("hf_split", ""),
            "builtin_file": cfg.get("builtin_file", ""),
            "has_builtin_fallback": bool(cfg.get("builtin_file")),
        }

    # -- Benchmark Loading --------------------------------------------------

    def load_benchmark(self, name, max_samples=None, shuffle=False,
                       seed=42, force_builtin=False):
        # type: (str, Optional[int], bool, int, bool) -> List[StandardQuestion]
        """Load benchmark questions by name.

        Tries HuggingFace first, then falls back to built-in sample data.
        """
        cfg = self._get_benchmark_config(name)

        # Try HuggingFace
        if self._prefer_hf and not force_builtin and cfg.get("hf_id"):
            try:
                questions = self._load_hf_benchmark(cfg, max_samples)
                logger.info(
                    "Loaded %d questions for '%s' from HuggingFace",
                    len(questions), name,
                )
                if shuffle:
                    rng = random.Random(seed)
                    rng.shuffle(questions)
                return questions
            except Exception as exc:
                logger.warning(
                    "HuggingFace load failed for '%s' (%s); "
                    "falling back to built-in data.", name, exc,
                )

        # Fallback to built-in
        builtin_file = cfg.get("builtin_file", "")
        if builtin_file:
            questions = self._pkg_loader.load_questions(
                builtin_file, max_items=max_samples,
                shuffle=shuffle, seed=seed,
            )
            logger.info(
                "Loaded %d questions for '%s' from built-in data",
                len(questions), name,
            )
            return questions

        raise RuntimeError(
            "No data source available for benchmark '%s'. "
            "Install `datasets` or check built-in data files." % name
        )

    def _get_benchmark_config(self, name):
        # type: (str) -> Dict[str, Any]
        cfg = self._extra_benchmark_configs.get(name)
        if cfg is not None:
            return cfg
        cfg = _BENCHMARK_CONFIGS.get(name)
        if cfg is not None:
            return cfg
        available = ", ".join(self.available_benchmarks())
        raise KeyError(
            "Unknown benchmark: '%s'. Available: %s" % (name, available)
        )

    def _load_hf_benchmark(self, cfg, max_samples):
        # type: (Dict[str, Any], Optional[int]) -> List[StandardQuestion]
        loader = HuggingFaceDatasetLoader(
            dataset_id=cfg["hf_id"],
            split=cfg.get("hf_split", "test"),
            config_name=cfg.get("hf_subset"),
            transform_fn=cfg.get("transform"),
            source_name=cfg.get("source_name", cfg["hf_id"]),
        )
        return loader.load(max_items=max_samples)

    # -- Attack Loading -----------------------------------------------------

    def load_attacks(self, category="all", severity=None,
                     max_samples=None, shuffle=False, seed=42):
        # type: (str, Optional[str], Optional[int], bool, int) -> List[AttackPrompt]
        """Load attack prompts by category.

        Args:
            category: Attack category or ``"all"`` for everything.
            severity: Filter by severity (low/medium/high/critical).
            max_samples: Maximum items to return.
            shuffle: Shuffle results.
            seed: Random seed.
        """
        if category == "all":
            attacks = []  # type: List[AttackPrompt]
            for cat in self.available_attack_categories():
                attacks.extend(self._load_attack_category(cat))
        else:
            attacks = self._load_attack_category(category)

        if severity:
            attacks = [a for a in attacks if a.severity == severity]
        if shuffle:
            rng = random.Random(seed)
            attacks = list(attacks)
            rng.shuffle(attacks)
        if max_samples is not None and max_samples > 0:
            attacks = attacks[:max_samples]
        return attacks

    def _load_attack_category(self, category):
        # type: (str) -> List[AttackPrompt]
        filename = self._extra_attack_files.get(
            category, _ATTACK_FILES.get(category)
        )
        if filename is None:
            available = ", ".join(self.available_attack_categories())
            raise KeyError(
                "Unknown attack category: '%s'. Available: %s"
                % (category, available)
            )
        return self._pkg_loader.load_attacks(filename)

    # -- Alignment Loading --------------------------------------------------

    def load_alignment(self, category="all", max_samples=None,
                       shuffle=False, seed=42):
        # type: (str, Optional[int], bool, int) -> List[AlignmentScenario]
        """Load alignment test scenarios by category."""
        if category == "all":
            scenarios = []  # type: List[AlignmentScenario]
            for cat in self.available_alignment_categories():
                scenarios.extend(self._load_alignment_category(cat))
        else:
            scenarios = self._load_alignment_category(category)

        if shuffle:
            rng = random.Random(seed)
            scenarios = list(scenarios)
            rng.shuffle(scenarios)
        if max_samples is not None and max_samples > 0:
            scenarios = scenarios[:max_samples]
        return scenarios

    def _load_alignment_category(self, category):
        # type: (str) -> List[AlignmentScenario]
        filename = self._extra_alignment_files.get(
            category, _ALIGNMENT_FILES.get(category)
        )
        if filename is None:
            available = ", ".join(self.available_alignment_categories())
            raise KeyError(
                "Unknown alignment category: '%s'. Available: %s"
                % (category, available)
            )
        return self._pkg_loader.load_scenarios(filename)

    # -- Hallucination Loading ----------------------------------------------

    def load_hallucination_prompts(self, max_samples=None, shuffle=False,
                                   seed=42):
        # type: (Optional[int], bool, int) -> List[AlignmentScenario]
        """Load hallucination detection prompts."""
        return self._pkg_loader.load_hallucination_prompts(
            max_items=max_samples, shuffle=shuffle, seed=seed,
        )

    # -- User-Supplied Local Files ------------------------------------------

    def load_local_questions(self, filepath, max_samples=None):
        # type: (str, Optional[int]) -> List[StandardQuestion]
        """Load questions from a user-provided JSONL file."""
        raw = load_jsonl(filepath)
        questions = [StandardQuestion.from_dict(r) for r in raw]
        if max_samples:
            questions = questions[:max_samples]
        return questions

    def load_local_attacks(self, filepath, max_samples=None):
        # type: (str, Optional[int]) -> List[AttackPrompt]
        """Load attack prompts from a user-provided JSONL file."""
        raw = load_jsonl(filepath)
        attacks = [AttackPrompt.from_dict(r) for r in raw]
        if max_samples:
            attacks = attacks[:max_samples]
        return attacks

    def load_local_scenarios(self, filepath, max_samples=None):
        # type: (str, Optional[int]) -> List[AlignmentScenario]
        """Load alignment scenarios from a user-provided JSONL file."""
        raw = load_jsonl(filepath)
        scenarios = [AlignmentScenario.from_dict(r) for r in raw]
        if max_samples:
            scenarios = scenarios[:max_samples]
        return scenarios


# ============================================================
# Convenience Factory
# ============================================================

def create_hf_loader(dataset_id, split="test", config_name=None, transform_fn=None):
    # type: (str, str, Optional[str], Optional[Callable]) -> HuggingFaceDatasetLoader
    """Shortcut to create a :class:`HuggingFaceDatasetLoader`.

    If no ``transform_fn`` is given, tries to auto-detect from built-in
    transforms for known datasets.
    """
    _auto_transforms = {
        "cais/mmlu": mmlu_transform,
        "openai/gsm8k": gsm8k_transform,
        "truthfulqa/truthful_qa": truthfulqa_transform,
        "openai/openai_humaneval": humaneval_transform,
        "haonan-li/cmmlu": cmmlu_transform,
    }
    if transform_fn is None:
        transform_fn = _auto_transforms.get(dataset_id)
    return HuggingFaceDatasetLoader(
        dataset_id=dataset_id,
        split=split,
        config_name=config_name,
        transform_fn=transform_fn,
    )
