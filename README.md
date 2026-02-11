# LLM Quality & Security Assessment Platform

**一键体检报告** - 统一的 LLM 质量与安全评估平台

Build a unified LLM quality and safety assessment platform that integrates capability benchmarking, security red teaming, and alignment checking into a single framework, providing a "one-click health check report".

## ✨ Features

- 🎯 **Capability Benchmarks**: Evaluate LLM performance on knowledge, reasoning, and coding tasks
- 🛡️ **Security Red Teaming**: Test for vulnerabilities including jailbreaks, prompt injection, and data leakage
- ✅ **Alignment Verification**: Check for helpfulness, harmlessness, honesty, and bias
- 📊 **Comprehensive Reports**: Generate HTML and text reports with visualizations
- 🚀 **One-Click Assessment**: Run complete evaluation with a single command
- 🔌 **Extensible**: Easy to add custom benchmarks and tests

## 🏗️ Architecture

The platform consists of three main modules:

1. **Benchmark Module** (`llm_assessment.benchmark`)
   - MMLU (Massive Multitask Language Understanding)
   - Reasoning benchmarks
   - Coding capability tests

2. **Red Teaming Module** (`llm_assessment.red_teaming`)
   - Jailbreak detection
   - Prompt injection tests
   - Data leakage prevention
   - Toxic content filtering

3. **Alignment Module** (`llm_assessment.alignment`)
   - Helpfulness evaluation
   - Harmlessness verification
   - Honesty assessment
   - Bias and fairness testing

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/NJX-njx/LLM-Quality-and-Security-Assessment-Platform.git
cd LLM-Quality-and-Security-Assessment-Platform

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Run complete assessment with mock LLM (for demo)
llm-assess assess --provider mock

# Run with OpenAI
export OPENAI_API_KEY="your-api-key"
llm-assess assess --provider openai --model gpt-3.5-turbo

# Run individual modules
llm-assess benchmark --provider mock --max-questions 5
llm-assess security --provider mock
llm-assess alignment --provider mock

# Generate report from existing results
llm-assess report assessment_results.json --format html
```

### Python API

```python
from llm_assessment.core.llm_wrapper import create_llm
from llm_assessment.core.assessment import AssessmentPlatform
from llm_assessment.core.report import ReportGenerator

# Create LLM instance
llm = create_llm("mock")  # or "openai" with api_key

# Run assessment
platform = AssessmentPlatform(llm)
results = platform.run_all(max_benchmark_questions=5)

# Generate reports
report_gen = ReportGenerator(results)
report_gen.save_report("report.html", format="html")

# Print summary
summary = results["summary"]
print(f"Overall Health Score: {summary['overall_health_score']:.1f}/100")
print(f"Health Rating: {summary['health_rating']}")
```

## 📖 Usage Examples

### Example 1: Basic Usage

```python
from llm_assessment.core.llm_wrapper import create_llm
from llm_assessment.core.assessment import AssessmentPlatform

# Create and assess an LLM
llm = create_llm("mock", model_name="demo-model")
platform = AssessmentPlatform(llm)
results = platform.run_all(max_benchmark_questions=5)

# Save results
platform.save_results("results.json")
```

### Example 2: Individual Modules

```python
from llm_assessment.core.llm_wrapper import create_llm
from llm_assessment.core.assessment import AssessmentPlatform

llm = create_llm("mock")
platform = AssessmentPlatform(llm)

# Run only benchmarks
benchmark_results = platform.run_benchmarks(max_questions=3)

# Run only security tests
security_results = platform.run_red_teaming()

# Run only alignment tests
alignment_results = platform.run_alignment()
```

### Example 3: Custom Benchmark

```python
from llm_assessment.benchmark.benchmarks import BaseBenchmark

class CustomBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("Custom Test", category="custom")
    
    def load_questions(self):
        return [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"}
        ]
    
    def evaluate_answer(self, question, answer):
        return question["answer"].lower() in answer.lower()

# Use the custom benchmark
benchmark = CustomBenchmark()
result = benchmark.run(llm)
```

## 📊 Report Example

The platform generates comprehensive reports including:

- **Overall Health Score**: Weighted average of all assessments
- **Health Rating**: Excellent, Good, Fair, or Needs Improvement
- **Detailed Scores**: Individual scores for each test category
- **Vulnerability List**: Security issues found during red teaming
- **Recommendations**: Actionable suggestions for improvement

### Sample Output

```
======================================================================
LLM QUALITY & SECURITY ASSESSMENT REPORT
======================================================================

Overall Health Score: 85.5/100
Rating: Good

  Capability (Benchmark): 82.3/100
  Security: 91.2/100
  Alignment: 84.0/100

Total Vulnerabilities Found: 2
```

## 🔧 Configuration

Create a `.env` file for API keys:

```bash
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

## 🧪 Testing

Run the examples:

```bash
# Basic usage example
python examples/basic_usage.py

# Individual modules example
python examples/individual_modules.py

# Custom benchmark example
python examples/custom_benchmark.py
```

## 📚 Documentation

### LLM Providers

Currently supported providers:
- **mock**: Mock LLM for testing and demonstration (no API key required)
- **openai**: OpenAI models (GPT-3.5, GPT-4, etc.)

### Adding Custom Providers

Extend the `BaseLLM` class in `llm_assessment/core/llm_wrapper.py`:

```python
class CustomLLM(BaseLLM):
    def generate(self, prompt: str, **kwargs) -> str:
        # Implement your LLM's generation logic
        pass
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Implement your LLM's chat logic
        pass
```

### Extending the Platform

You can extend any of the three modules:

1. **Custom Benchmarks**: Extend `BaseBenchmark`
2. **Custom Security Tests**: Extend `BaseRedTeamTest`
3. **Custom Alignment Tests**: Extend `BaseAlignmentTest`

See `examples/custom_benchmark.py` for a complete example.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This platform integrates best practices from:
- MMLU benchmark methodology
- OWASP LLM security guidelines
- Anthropic's Constitutional AI principles
- OpenAI's alignment research

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
