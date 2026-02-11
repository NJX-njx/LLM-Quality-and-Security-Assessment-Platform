# Contributing to LLM Quality & Security Assessment Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/NJX-njx/LLM-Quality-and-Security-Assessment-Platform.git
cd LLM-Quality-and-Security-Assessment-Platform
```

2. Install in development mode:
```bash
pip install -e .
```

3. Install additional development dependencies:
```bash
pip install pytest black flake8
```

## Project Structure

```
llm_assessment/
├── core/               # Core platform components
│   ├── llm_wrapper.py  # LLM provider integrations
│   ├── assessment.py   # Main assessment platform
│   └── report.py       # Report generation
├── benchmark/          # Capability benchmark tests
│   └── benchmarks.py   # Benchmark implementations
├── red_teaming/        # Security red team tests
│   └── tests.py        # Security test implementations
├── alignment/          # Alignment verification tests
│   └── tests.py        # Alignment test implementations
└── cli.py             # Command-line interface

examples/               # Usage examples
```

## Adding New Features

### Adding a New LLM Provider

Extend the `BaseLLM` class in `llm_assessment/core/llm_wrapper.py`:

```python
class CustomLLM(BaseLLM):
    def __init__(self, model_name: str = "custom-model", **kwargs):
        super().__init__(model_name, **kwargs)
        # Initialize your provider
        
    def generate(self, prompt: str, **kwargs) -> str:
        # Implement generation logic
        pass
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Implement chat logic
        pass
```

Then register it in the `create_llm` factory function.

### Adding a New Benchmark

Extend the `BaseBenchmark` class in `llm_assessment/benchmark/benchmarks.py`:

```python
class CustomBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("Benchmark Name", category="category_name")
        
    def load_questions(self) -> List[Dict[str, Any]]:
        # Return list of questions
        return [
            {"question": "...", "answer": "..."},
        ]
    
    def evaluate_answer(self, question: Dict[str, Any], answer: str) -> bool:
        # Return True if answer is correct
        pass
```

Add it to the `get_available_benchmarks()` function.

### Adding a New Security Test

Extend the `BaseRedTeamTest` class in `llm_assessment/red_teaming/tests.py`:

```python
class CustomSecurityTest(BaseRedTeamTest):
    def __init__(self):
        super().__init__("Test Name", category="security")
        
    def generate_attacks(self) -> List[Dict[str, Any]]:
        # Return list of attack prompts
        return [
            {"prompt": "...", "type": "...", "severity": "high"},
        ]
    
    def evaluate_response(self, attack: Dict[str, Any], response: str) -> bool:
        # Return True if vulnerable, False if secure
        pass
```

Add it to the `get_available_red_team_tests()` function.

### Adding a New Alignment Test

Extend the `BaseAlignmentTest` class in `llm_assessment/alignment/tests.py`:

```python
class CustomAlignmentTest(BaseAlignmentTest):
    def __init__(self):
        super().__init__("Test Name", category="alignment")
        
    def generate_scenarios(self) -> List[Dict[str, Any]]:
        # Return list of test scenarios
        return [
            {"prompt": "...", "type": "...", "expected": "..."},
        ]
    
    def evaluate_response(self, scenario: Dict[str, Any], response: str) -> bool:
        # Return True if aligned, False if misaligned
        pass
```

Add it to the `get_available_alignment_tests()` function.

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Add docstrings to all classes and public methods
- Keep functions focused and single-purpose

## Testing

Before submitting changes:

1. Test your code with the mock LLM:
```bash
python examples/basic_usage.py
```

2. Run the CLI to ensure it works:
```bash
llm-assess assess --provider mock --max-questions 3
```

3. Verify examples still work:
```bash
python examples/individual_modules.py
python examples/custom_benchmark.py
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the technical aspects
- Help others learn and grow

## Questions?

Open an issue for questions or discussions about potential contributions.
