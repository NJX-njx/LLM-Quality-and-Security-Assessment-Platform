"""
LLM 能力基准测试模块

提供多种基准测试以评估 LLM 的知识、推理和编程能力。
包含 MMLU（多任务语言理解）、逻辑推理和代码能力三大类别。

扩展方式：继承 BaseBenchmark 基类，实现 load_questions() 和 evaluate_answer() 方法，
然后在 get_available_benchmarks() 中注册。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


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
    
    def __init__(self, name: str, category: str = "general"):
        self.name = name          # 测试名称
        self.category = category  # 测试类别
        self.questions = []       # 题目列表
        
    @abstractmethod
    def load_questions(self) -> List[Dict[str, Any]]:
        """加载基准测试题目，子类必须实现"""
        pass
    
    @abstractmethod
    def evaluate_answer(self, question: Dict[str, Any], answer: str) -> bool:
        """评估答案是否正确，子类必须实现"""
        pass
    
    def run(self, llm, max_questions: Optional[int] = None) -> BenchmarkResult:
        """在指定 LLM 上运行基准测试"""
        from tqdm import tqdm
        
        questions = self.load_questions()  # 加载题目
        if max_questions:
            questions = questions[:max_questions]  # 限制题目数量
        
        correct = 0  # 正确答题计数
        results = []
        
        for question in tqdm(questions, desc=f"Running {self.name}"):
            prompt = self._format_prompt(question)  # 格式化题目为提示词
            answer = llm.generate(prompt)            # 获取 LLM 回答
            is_correct = self.evaluate_answer(question, answer)  # 评估答案
            
            if is_correct:
                correct += 1
            
            results.append({
                "question": question.get("question", ""),
                "answer": answer,
                "correct": is_correct,
                "expected": question.get("answer", "")
            })
        
        # 计算得分百分比
        score = (correct / len(questions)) * 100 if questions else 0
        
        return BenchmarkResult(
            name=self.name,
            score=score,
            total_questions=len(questions),
            correct_answers=correct,
            details={"results": results},
            category=self.category
        )
    
    def _format_prompt(self, question: Dict[str, Any]) -> str:
        """将题目格式化为 LLM 提示词，支持选择题和开放题"""
        q_text = question.get("question", "")  # 题目文本
        choices = question.get("choices", [])   # 选项列表
        
        if choices:
            prompt = f"{q_text}\n\nChoices:\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "\nAnswer with just the letter (A, B, C, or D):"
        else:
            prompt = f"{q_text}\n\nProvide a concise answer:"
        
        return prompt


class MMLUBenchmark(BaseBenchmark):
    """
    MMLU（大规模多任务语言理解）基准测试
    
    测试模型在各种知识领域的多选题回答能力。
    """
    
    def __init__(self, subject: str = "general"):
        super().__init__(f"MMLU-{subject}", category="knowledge")
        self.subject = subject  # 学科类别
        
    def load_questions(self) -> List[Dict[str, Any]]:
        """加载 MMLU 风格的多选题"""
        # 示例 MMLU 题目（涵盖地理、科学、数学、文学等领域）
        questions = [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": "C"
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
                "answer": "B"
            },
            {
                "question": "What is 15 + 27?",
                "choices": ["41", "42", "43", "44"],
                "answer": "B"
            },
            {
                "question": "Who wrote 'Romeo and Juliet'?",
                "choices": ["Charles Dickens", "William Shakespeare", "Mark Twain", "Jane Austen"],
                "answer": "B"
            },
            {
                "question": "What is the largest ocean on Earth?",
                "choices": ["Atlantic", "Indian", "Arctic", "Pacific"],
                "answer": "D"
            },
        ]
        return questions
    
    def evaluate_answer(self, question: Dict[str, Any], answer: str) -> bool:
        """评估 MMLU 答案，提取首字母与正确答案比较"""
        expected = question.get("answer", "").strip().upper()
        # 从回答中提取第一个字母作为答案
        answer_clean = answer.strip().upper()
        if answer_clean and answer_clean[0] in "ABCD":
            return answer_clean[0] == expected
        return False


class ReasoningBenchmark(BaseBenchmark):
    """
    推理能力基准测试
    
    测试模型的逻辑推理、数学计算和常识推断能力。
    """
    
    def __init__(self):
        super().__init__("Reasoning", category="reasoning")
        
    def load_questions(self) -> List[Dict[str, Any]]:
        """加载推理测试题目（包含逻辑、数学、常识推理）"""
        questions = [
            {
                "question": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                "answer": "no",
                "explanation": "The statement doesn't specify which flowers fade quickly"
            },
            {
                "question": "A train leaves Station A at 9 AM traveling at 60 mph. Another train leaves Station B (120 miles away) at 10 AM traveling at 80 mph toward Station A. At what time do they meet?",
                "answer": "11",
                "explanation": "They meet at 11 AM"
            },
            {
                "question": "If 5 machines can make 5 widgets in 5 minutes, how many minutes would it take 100 machines to make 100 widgets?",
                "answer": "5",
                "explanation": "Each machine makes 1 widget in 5 minutes"
            },
        ]
        return questions
    
    def evaluate_answer(self, question: Dict[str, Any], answer: str) -> bool:
        """评估推理答案，检查期望答案是否包含在回答中"""
        expected = question.get("answer", "").strip().lower()
        answer_lower = answer.strip().lower()
        return expected in answer_lower


class CodingBenchmark(BaseBenchmark):
    """
    编程能力基准测试
    
    测试模型的代码理解、编写和计算机科学知识。
    """
    
    def __init__(self):
        super().__init__("Coding", category="coding")
        
    def load_questions(self) -> List[Dict[str, Any]]:
        """加载编程测试题目（包含编程实践、算法复杂度、语言特性等）"""
        questions = [
            {
                "question": "Write a Python function to check if a number is prime. What would the function return for input 17?",
                "answer": "true",
                "keywords": ["prime", "17", "true"]
            },
            {
                "question": "What is the time complexity of binary search?",
                "answer": "O(log n)",
                "keywords": ["log", "logarithmic"]
            },
            {
                "question": "In Python, what does the 'self' keyword represent in a class method?",
                "answer": "instance",
                "keywords": ["instance", "object", "itself"]
            },
        ]
        return questions
    
    def evaluate_answer(self, question: Dict[str, Any], answer: str) -> bool:
        """评估编程答案，检查关键词是否出现在回答中"""
        keywords = question.get("keywords", [])  # 期望的关键词列表
        answer_lower = answer.lower()
        # 检查任意关键词是否存在
        return any(keyword.lower() in answer_lower for keyword in keywords)


def get_available_benchmarks() -> List[BaseBenchmark]:
    """获取所有可用的基准测试实例列表，新增基准测试时需在此注册"""
    return [
        MMLUBenchmark("general"),
        ReasoningBenchmark(),
        CodingBenchmark(),
    ]
