"""
Example: Custom benchmark implementation
"""

from llm_assessment.benchmark.benchmarks import BaseBenchmark, BenchmarkResult
from llm_assessment.core.llm_wrapper import create_llm


class CustomMathBenchmark(BaseBenchmark):
    """Custom math benchmark"""
    
    def __init__(self):
        super().__init__("Custom Math", category="mathematics")
        
    def load_questions(self):
        return [
            {
                "question": "What is 123 + 456?",
                "answer": "579"
            },
            {
                "question": "What is 12 * 8?",
                "answer": "96"
            },
            {
                "question": "What is the square root of 144?",
                "answer": "12"
            },
        ]
    
    def evaluate_answer(self, question, answer):
        expected = question["answer"]
        return expected in answer


def main():
    # Create LLM
    llm = create_llm("mock")
    
    # Run custom benchmark
    benchmark = CustomMathBenchmark()
    result = benchmark.run(llm)
    
    print(f"Benchmark: {result.name}")
    print(f"Score: {result.score:.1f}%")
    print(f"Correct: {result.correct_answers}/{result.total_questions}")


if __name__ == "__main__":
    main()
