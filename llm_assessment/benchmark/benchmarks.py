"""
Benchmark tests for LLM capability evaluation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result from a benchmark test"""
    name: str
    score: float
    total_questions: int
    correct_answers: int
    details: Dict[str, Any]
    category: str = "general"


class BaseBenchmark(ABC):
    """Base class for benchmarks"""
    
    def __init__(self, name: str, category: str = "general"):
        self.name = name
        self.category = category
        self.questions = []
        
    @abstractmethod
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load benchmark questions"""
        pass
    
    @abstractmethod
    def evaluate_answer(self, question: Dict[str, Any], answer: str) -> bool:
        """Evaluate if answer is correct"""
        pass
    
    def run(self, llm, max_questions: Optional[int] = None) -> BenchmarkResult:
        """Run benchmark on LLM"""
        from tqdm import tqdm
        
        questions = self.load_questions()
        if max_questions:
            questions = questions[:max_questions]
        
        correct = 0
        results = []
        
        for question in tqdm(questions, desc=f"Running {self.name}"):
            prompt = self._format_prompt(question)
            answer = llm.generate(prompt)
            is_correct = self.evaluate_answer(question, answer)
            
            if is_correct:
                correct += 1
            
            results.append({
                "question": question.get("question", ""),
                "answer": answer,
                "correct": is_correct,
                "expected": question.get("answer", "")
            })
        
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
        """Format question as prompt"""
        q_text = question.get("question", "")
        choices = question.get("choices", [])
        
        if choices:
            prompt = f"{q_text}\n\nChoices:\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "\nAnswer with just the letter (A, B, C, or D):"
        else:
            prompt = f"{q_text}\n\nProvide a concise answer:"
        
        return prompt


class MMLUBenchmark(BaseBenchmark):
    """MMLU (Massive Multitask Language Understanding) benchmark"""
    
    def __init__(self, subject: str = "general"):
        super().__init__(f"MMLU-{subject}", category="knowledge")
        self.subject = subject
        
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load MMLU questions"""
        # Sample MMLU-style questions
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
        """Evaluate MMLU answer"""
        expected = question.get("answer", "").strip().upper()
        # Extract first letter from answer
        answer_clean = answer.strip().upper()
        if answer_clean and answer_clean[0] in "ABCD":
            return answer_clean[0] == expected
        return False


class ReasoningBenchmark(BaseBenchmark):
    """Reasoning capability benchmark"""
    
    def __init__(self):
        super().__init__("Reasoning", category="reasoning")
        
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load reasoning questions"""
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
        """Evaluate reasoning answer"""
        expected = question.get("answer", "").strip().lower()
        answer_lower = answer.strip().lower()
        return expected in answer_lower


class CodingBenchmark(BaseBenchmark):
    """Coding capability benchmark"""
    
    def __init__(self):
        super().__init__("Coding", category="coding")
        
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load coding questions"""
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
        """Evaluate coding answer"""
        keywords = question.get("keywords", [])
        answer_lower = answer.lower()
        # Check if any keyword is present
        return any(keyword.lower() in answer_lower for keyword in keywords)


def get_available_benchmarks() -> List[BaseBenchmark]:
    """Get list of available benchmarks"""
    return [
        MMLUBenchmark("general"),
        ReasoningBenchmark(),
        CodingBenchmark(),
    ]
