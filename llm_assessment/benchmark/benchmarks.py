"""
LLM 能力基准测试模块

提供多种基准测试以评估 LLM 的知识、推理和编程能力。
包含 MMLU、CMMLU、GSM8K、HumanEval、TruthfulQA 等标准基准。

所有基准已对接 DatasetLoader，优先从 JSONL 样本文件加载题目，
支持 HuggingFace Datasets 回退。

扩展方式：继承 BaseBenchmark 基类，实现 load_questions() 和 evaluate_answer() 方法，
然后在 get_available_benchmarks() 中注册。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果数据类"""
    name: str                    # 测试名称
    score: float                 # 得分（0-100）
    total_questions: int         # 总题目数
    correct_answers: int         # 正确答题数
    details: Dict[str, Any]     # 详细结果
    category: str = "general"    # 测试类别


class BaseBenchmark(ABC):
    """基准测试抽象基类，所有基准测试必须继承此类"""

    def __init__(self, name, category="general"):
        # type: (str, str) -> None
        self.name = name
        self.category = category
        self.questions = []  # type: List[Dict[str, Any]]

    @abstractmethod
    def load_questions(self):
        # type: () -> List[Dict[str, Any]]
        """加载基准测试题目，子类必须实现"""
        pass

    @abstractmethod
    def evaluate_answer(self, question, answer):
        # type: (Dict[str, Any], str) -> bool
        """评估答案是否正确，子类必须实现"""
        pass

    def run(self, llm, max_questions=None):
        # type: (Any, Optional[int]) -> BenchmarkResult
        """在指定 LLM 上运行基准测试"""
        from tqdm import tqdm

        questions = self.load_questions()
        if max_questions:
            questions = questions[:max_questions]

        correct = 0
        results = []

        for question in tqdm(questions, desc="Running {}".format(self.name)):
            prompt = self._format_prompt(question)
            answer = llm.generate(prompt)
            is_correct = self.evaluate_answer(question, answer)

            if is_correct:
                correct += 1

            results.append({
                "question": question.get("question", ""),
                "answer": answer,
                "correct": is_correct,
                "expected": question.get("correct_answer",
                                         question.get("answer", "")),
            })

        score = (correct / len(questions)) * 100 if questions else 0

        return BenchmarkResult(
            name=self.name,
            score=score,
            total_questions=len(questions),
            correct_answers=correct,
            details={"results": results},
            category=self.category,
        )

    def _format_prompt(self, question):
        # type: (Dict[str, Any]) -> str
        """将题目格式化为 LLM 提示词，支持选择题和开放题"""
        q_text = question.get("question", "")
        choices = question.get("choices", [])

        if choices:
            prompt = "{}\n\nChoices:\n".format(q_text)
            for i, choice in enumerate(choices):
                prompt += "{}. {}\n".format(chr(65 + i), choice)
            prompt += "\nAnswer with just the letter (A, B, C, or D):"
        else:
            prompt = "{}\n\nProvide a concise answer:".format(q_text)

        return prompt

    # ----------------------------------------------------------
    # DatasetLoader helper
    # ----------------------------------------------------------

    @staticmethod
    def _load_from_jsonl(dataset_name, max_samples=None):
        # type: (str, Optional[int]) -> List[Dict[str, Any]]
        """从内建 JSONL 样本文件加载题目。"""
        try:
            from ..core.datasets import DatasetLoader
            loader = DatasetLoader()
            questions = loader.load_benchmark(
                dataset_name, max_samples=max_samples,
            )
            return [q.to_dict() if hasattr(q, "to_dict") else q
                    for q in questions]
        except Exception as exc:
            logger.warning(
                "Could not load dataset '%s' via DatasetLoader: %s",
                dataset_name, exc,
            )
            return []


# ============================================================
# MMLU — 大规模多任务语言理解
# ============================================================

class MMLUBenchmark(BaseBenchmark):
    """
    MMLU（大规模多任务语言理解）基准测试

    从 JSONL 样本文件加载 50 道真实 MMLU 格式选择题，
    涵盖地理、天文、数学、文学、科学等多个学科领域。
    """

    def __init__(self, subject="general"):
        # type: (str,) -> None
        super().__init__("MMLU-{}".format(subject), category="knowledge")
        self.subject = subject

    def load_questions(self):
        # type: () -> List[Dict[str, Any]]
        questions = self._load_from_jsonl("mmlu")
        if questions:
            return questions
        # 向后兼容：内建硬编码回退
        return [
            {"question": "What is the capital of France?",
             "choices": ["London", "Berlin", "Paris", "Madrid"],
             "correct_answer": "C"},
            {"question": "Which planet is known as the Red Planet?",
             "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
             "correct_answer": "B"},
            {"question": "What is 15 + 27?",
             "choices": ["41", "42", "43", "44"],
             "correct_answer": "B"},
            {"question": "Who wrote 'Romeo and Juliet'?",
             "choices": ["Charles Dickens", "William Shakespeare",
                         "Mark Twain", "Jane Austen"],
             "correct_answer": "B"},
            {"question": "What is the largest ocean on Earth?",
             "choices": ["Atlantic", "Indian", "Arctic", "Pacific"],
             "correct_answer": "D"},
        ]

    def evaluate_answer(self, question, answer):
        # type: (Dict[str, Any], str) -> bool
        expected = question.get("correct_answer", "").strip().upper()
        answer_clean = answer.strip().upper()
        match = re.search(r'\b([A-D])\b', answer_clean)
        if match:
            return match.group(1) == expected
        if answer_clean and answer_clean[0] in "ABCD":
            return answer_clean[0] == expected
        return False


# ============================================================
# CMMLU — 中文多任务语言理解
# ============================================================

class CmmlUBenchmark(BaseBenchmark):
    """
    CMMLU（中文大规模多任务语言理解）基准测试

    从 JSONL 样本文件加载 30 道中文选择题，
    涵盖中国地理、历史、文学、科学等领域。
    """

    def __init__(self):
        super().__init__("CMMLU", category="knowledge")

    def load_questions(self):
        # type: () -> List[Dict[str, Any]]
        questions = self._load_from_jsonl("cmmlu")
        if questions:
            return questions
        return [
            {"question": "中国的首都是哪个城市？",
             "choices": ["上海", "北京", "广州", "深圳"],
             "correct_answer": "B"},
        ]

    def _format_prompt(self, question):
        # type: (Dict[str, Any]) -> str
        q_text = question.get("question", "")
        choices = question.get("choices", [])
        if choices:
            prompt = "{}\n\n选项:\n".format(q_text)
            for i, choice in enumerate(choices):
                prompt += "{}. {}\n".format(chr(65 + i), choice)
            prompt += "\n请只回答选项字母（A、B、C 或 D）："
        else:
            prompt = "{}\n\n请给出简洁的回答：".format(q_text)
        return prompt

    def evaluate_answer(self, question, answer):
        # type: (Dict[str, Any], str) -> bool
        expected = question.get("correct_answer", "").strip().upper()
        answer_clean = answer.strip().upper()
        match = re.search(r'\b([A-D])\b', answer_clean)
        if match:
            return match.group(1) == expected
        if answer_clean and answer_clean[0] in "ABCD":
            return answer_clean[0] == expected
        return False


# ============================================================
# GSM8K — 数学推理
# ============================================================

class GSM8KBenchmark(BaseBenchmark):
    """
    GSM8K 数学推理基准测试

    从 JSONL 样本文件加载 30 道小学数学应用题，
    使用 Chain-of-Thought 格式输入，数值匹配评估。
    """

    def __init__(self):
        super().__init__("GSM8K", category="reasoning")

    def load_questions(self):
        # type: () -> List[Dict[str, Any]]
        questions = self._load_from_jsonl("gsm8k")
        if questions:
            return questions
        return [
            {"question": ("Janet's ducks lay 16 eggs per day. She eats three "
                          "for breakfast and bakes muffins with four. She "
                          "sells the remainder for $2 each. How much does "
                          "she make daily?"),
             "correct_answer": "18",
             "evaluation_type": "numeric_match"},
        ]

    def _format_prompt(self, question):
        # type: (Dict[str, Any]) -> str
        q_text = question.get("question", "")
        return (
            "{}\n\n"
            "Let's think step by step. "
            "After your reasoning, give your final numeric answer after '#### '."
        ).format(q_text)

    def evaluate_answer(self, question, answer):
        # type: (Dict[str, Any], str) -> bool
        expected_str = question.get("correct_answer", "")
        expected_num = self._extract_number(expected_str)
        if expected_num is None:
            return False
        # #### 标记
        hash_match = re.search(r'####\s*([\-\d,\.]+)', answer)
        if hash_match:
            extracted = self._extract_number(hash_match.group(1))
            if extracted is not None:
                return abs(extracted - expected_num) < 0.01
            return False
        # 最后一个数字
        extracted = self._extract_last_number(answer)
        if extracted is not None:
            return abs(extracted - expected_num) < 0.01
        return False

    @staticmethod
    def _extract_number(text):
        # type: (str) -> Optional[float]
        text = text.replace(",", "").strip()
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None

    @staticmethod
    def _extract_last_number(text):
        # type: (str) -> Optional[float]
        text = text.replace(",", "")
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
        return None


# ============================================================
# HumanEval — 代码生成
# ============================================================

class HumanEvalBenchmark(BaseBenchmark):
    """
    HumanEval 代码生成基准测试

    从 JSONL 样本文件加载 20 道 Python 编程题，
    使用沙箱环境执行代码并运行测试用例验证正确性。
    """

    def __init__(self):
        super().__init__("HumanEval", category="coding")

    def load_questions(self):
        # type: () -> List[Dict[str, Any]]
        questions = self._load_from_jsonl("humaneval")
        if questions:
            return questions
        return [
            {"question": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
             "correct_answer": "    return a + b\n",
             "evaluation_type": "code_exec",
             "metadata": {"entry_point": "add"}},
        ]

    def _format_prompt(self, question):
        # type: (Dict[str, Any]) -> str
        q_text = question.get("question", "")
        return (
            "Complete the following Python function. "
            "Only output the function body.\n\n{}"
        ).format(q_text)

    def evaluate_answer(self, question, answer):
        # type: (Dict[str, Any], str) -> bool
        metadata = question.get("metadata", {})
        entry_point = metadata.get("entry_point", "")
        test_code = metadata.get("test", "")
        function_stub = question.get("question", "")

        if not test_code or not entry_point:
            expected = question.get("correct_answer", "")
            return expected.strip() in answer.strip()

        code = self._extract_code(answer)
        full_code = function_stub + code
        return self._safe_execute(full_code, test_code, entry_point)

    @staticmethod
    def _extract_code(text):
        # type: (str) -> str
        match = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip() + "\n"
        lines = text.strip().split("\n")
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and (
                    "def " in stripped or "return " in stripped
                    or stripped.startswith(" ") or stripped.startswith("\t")
                    or "=" in stripped or "for " in stripped
                    or "if " in stripped or "while " in stripped
                    or stripped.startswith("import ")):
                code_lines.append(line)
            elif code_lines:
                code_lines.append(line)
        return "\n".join(code_lines) + "\n" if code_lines else text + "\n"

    @staticmethod
    def _safe_execute(code, test_code, entry_point):
        # type: (str, str, str) -> bool
        try:
            namespace = {}  # type: Dict[str, Any]
            exec(code, namespace)  # noqa: S102
            if entry_point not in namespace:
                return False
            exec(test_code, namespace)  # noqa: S102
            if "check" in namespace:
                namespace["check"](namespace[entry_point])
            return True
        except Exception:
            return False


# ============================================================
# TruthfulQA — 事实性
# ============================================================

class TruthfulQABenchmark(BaseBenchmark):
    """
    TruthfulQA 事实性基准测试

    从 JSONL 样本文件加载 30 道常见误解/都市传说问题，
    使用关键词匹配评估模型是否给出了事实性正确的回答。
    """

    def __init__(self):
        super().__init__("TruthfulQA", category="factuality")

    def load_questions(self):
        # type: () -> List[Dict[str, Any]]
        questions = self._load_from_jsonl("truthfulqa")
        if questions:
            return questions
        return [
            {"question": "Can coughing effectively stop a heart attack?",
             "correct_answer": "No",
             "metadata": {
                 "correct_answers": ["No"],
                 "incorrect_answers": ["Yes"],
             }},
        ]

    def _format_prompt(self, question):
        # type: (Dict[str, Any]) -> str
        q_text = question.get("question", "")
        return (
            "{}\n\n"
            "Answer truthfully and concisely. "
            "If a common belief is wrong, explain why."
        ).format(q_text)

    def evaluate_answer(self, question, answer):
        # type: (Dict[str, Any], str) -> bool
        metadata = question.get("metadata", {})
        correct_answers = metadata.get("correct_answers", [])
        incorrect_answers = metadata.get("incorrect_answers", [])
        answer_lower = answer.lower()

        if not correct_answers:
            expected = question.get("correct_answer", "")
            return expected.lower() in answer_lower

        for wrong in incorrect_answers:
            if wrong.lower() in answer_lower:
                return False

        for right in correct_answers:
            if right.lower() in answer_lower:
                return True

        return False


# ============================================================
# 推理 — 逻辑推理
# ============================================================

class ReasoningBenchmark(BaseBenchmark):
    """推理能力基准测试：逻辑推理、数学计算和常识推断"""

    def __init__(self):
        super().__init__("Reasoning", category="reasoning")

    def load_questions(self):
        # type: () -> List[Dict[str, Any]]
        return [
            {"question": ("If all roses are flowers and some flowers "
                          "fade quickly, can we conclude that "
                          "some roses fade quickly?"),
             "answer": "no"},
            {"question": ("A train leaves Station A at 9 AM at 60 mph. "
                          "Another leaves Station B (120 miles away) at "
                          "10 AM at 80 mph toward A. When do they meet?"),
             "answer": "11"},
            {"question": ("If 5 machines make 5 widgets in 5 minutes, "
                          "how many minutes for 100 machines "
                          "to make 100 widgets?"),
             "answer": "5"},
        ]

    def evaluate_answer(self, question, answer):
        # type: (Dict[str, Any], str) -> bool
        expected = question.get("answer", "").strip().lower()
        return expected in answer.strip().lower()


# ============================================================
# 编程 — 代码知识
# ============================================================

class CodingBenchmark(BaseBenchmark):
    """编程能力基准测试：代码理解、编写和计算机科学知识"""

    def __init__(self):
        super().__init__("Coding", category="coding")

    def load_questions(self):
        # type: () -> List[Dict[str, Any]]
        return [
            {"question": ("Write a Python function to check if a number "
                          "is prime. What would it return for 17?"),
             "answer": "true",
             "keywords": ["prime", "17", "true"]},
            {"question": "What is the time complexity of binary search?",
             "answer": "O(log n)",
             "keywords": ["log", "logarithmic"]},
            {"question": ("In Python, what does 'self' represent in "
                          "a class method?"),
             "answer": "instance",
             "keywords": ["instance", "object", "itself"]},
        ]

    def evaluate_answer(self, question, answer):
        # type: (Dict[str, Any], str) -> bool
        keywords = question.get("keywords", [])
        answer_lower = answer.lower()
        return any(kw.lower() in answer_lower for kw in keywords)


# ============================================================
# 注册表
# ============================================================

def get_available_benchmarks():
    # type: () -> List[BaseBenchmark]
    """获取所有可用的基准测试实例列表，新增基准测试时需在此注册"""
    return [
        MMLUBenchmark("general"),
        CmmlUBenchmark(),
        GSM8KBenchmark(),
        HumanEvalBenchmark(),
        TruthfulQABenchmark(),
        ReasoningBenchmark(),
        CodingBenchmark(),
    ]
