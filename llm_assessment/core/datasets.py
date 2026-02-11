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
# Transform Functions for Common Datasets
# ============================================================

def mmlu_transform(row):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Transform a HuggingFace MMLU row into StandardQuestion format.

    Expected HF fields: ``question``, ``choices``, ``answer`` (int 0‑3).
    """
    answer_idx = row.get("answer", 0)
    if isinstance(answer_idx, int):
        answer_letter = chr(65 + answer_idx)
    else:
        answer_letter = str(answer_idx)
    return {
        "question": row.get("question", ""),
        "choices": row.get("choices", []),
        "correct_answer": answer_letter,
        "subject": row.get("subject", "general"),
        "evaluation_type": "exact_match",
    }


def gsm8k_transform(row):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Transform a HuggingFace GSM8K row into StandardQuestion format.

    Expected HF fields: ``question``, ``answer`` (contains ``####`` final answer).
    """
    raw_answer = row.get("answer", "")
    # GSM8K answers end with "#### <number>"
    final_answer = raw_answer
    if "####" in raw_answer:
        final_answer = raw_answer.split("####")[-1].strip()
    return {
        "question": row.get("question", ""),
        "correct_answer": final_answer,
        "evaluation_type": "numeric_match",
        "metadata": {"full_solution": raw_answer},
    }


def truthfulqa_transform(row):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Transform a HuggingFace TruthfulQA row into StandardQuestion format."""
    return {
        "question": row.get("question", ""),
        "correct_answer": row.get("best_answer", ""),
        "evaluation_type": "llm_judge",
        "metadata": {
            "correct_answers": row.get("correct_answers", []),
            "incorrect_answers": row.get("incorrect_answers", []),
            "category": row.get("category", ""),
        },
    }


# ============================================================
# Convenience Factory
# ============================================================

def create_hf_loader(dataset_id, split="test", config_name=None, transform_fn=None):
    # type: (str, str, Optional[str], Optional[Callable]) -> HuggingFaceDatasetLoader
    """Shortcut to create a :class:`HuggingFaceDatasetLoader`.

    If no ``transform_fn`` is given, the function tries to detect one
    from a built‑in registry of known datasets.
    """
    _auto_transforms = {
        "cais/mmlu": mmlu_transform,
        "openai/gsm8k": gsm8k_transform,
        "truthfulqa/truthful_qa": truthfulqa_transform,
    }
    if transform_fn is None:
        transform_fn = _auto_transforms.get(dataset_id)
    return HuggingFaceDatasetLoader(
        dataset_id=dataset_id,
        split=split,
        config_name=config_name,
        transform_fn=transform_fn,
    )
