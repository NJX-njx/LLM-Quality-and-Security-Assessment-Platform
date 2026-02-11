# Copilot Instructions — LLM Assessment Platform

## Project Overview

A Python platform for unified LLM evaluation across three dimensions: **capability benchmarks**, **security red teaming**, and **alignment checking**. Outputs a weighted health score (40% benchmark, 30% security, 30% alignment) and generates HTML/text reports.

## Architecture

- **`llm_assessment/core/`** — Orchestration layer. `AssessmentPlatform` in `assessment.py` coordinates all modules via `run_all()`, calls `get_available_*()` registry functions from each module. `llm_wrapper.py` defines the `BaseLLM` ABC (`generate()`, `chat()`, `get_stats()`) and a `create_llm()` factory.
- **`llm_assessment/benchmark/benchmarks.py`** — Capability tests. Subclass `BaseBenchmark` → implement `load_questions()` + `evaluate_answer()`. Register in `get_available_benchmarks()`.
- **`llm_assessment/red_teaming/tests.py`** — Security tests. Subclass `BaseRedTeamTest` → implement `generate_attacks()` + `evaluate_response()` (returns `True` = vulnerable). Register in `get_available_red_team_tests()`.
- **`llm_assessment/alignment/tests.py`** — Alignment tests. Subclass `BaseAlignmentTest` → implement `generate_scenarios()` + `evaluate_response()` (returns `True` = aligned). Register in `get_available_alignment_tests()`.
- **`llm_assessment/cli.py`** — Click-based CLI. Entry point: `llm-assess`. Subcommands: `assess`, `benchmark`, `security`, `alignment`, `report`.

## Key Patterns

### Adding a new test/benchmark

Every module follows the same 3-step pattern:
1. Subclass the base ABC (`BaseBenchmark`, `BaseRedTeamTest`, or `BaseAlignmentTest`)
2. Implement the required abstract methods (data generation + evaluation)
3. Register the class in the module's `get_available_*()` function at the bottom of the file

### Adding a new LLM provider

Subclass `BaseLLM` in `core/llm_wrapper.py`, implement `generate()` and `chat()`, then add to the `providers` dict in `create_llm()`. Current providers: `MockLLM` (testing), `OpenAILLM`.

### Result dataclasses

Each module uses a typed dataclass for results: `BenchmarkResult` (fields: `score`, `correct_answers`, `total_questions`), `RedTeamResult` (fields: `security_score`, `vulnerabilities_found`), `AlignmentResult` (fields: `alignment_score`, `passed_tests`). All include a `details` dict and `category` string.

### Evaluation approach

Current evaluation uses **regex pattern matching** against response text. Safety/vulnerability checks follow a two-pass pattern: check safety refusal patterns first → then check violation patterns → default to safe. See `JailbreakTest.evaluate_response()` for the canonical example.

## Development Commands

```bash
pip install -e .                    # Install in dev mode (only click + tqdm required)
pip install pytest black flake8     # Dev tools
llm-assess assess --provider mock   # Quick smoke test with MockLLM
llm-assess benchmark --provider mock --max-questions 3   # Fast benchmark only
```

## Conventions

- **Scores**: All scores are 0-100 floats (higher = better). Security scores invert vulnerability counts.
- **Python ≥ 3.8** compatibility required. No walrus operator or newer-only features.
- **Lazy imports**: Heavy deps (`tqdm`, `openai`) are imported inside methods, not at module top, to keep the package importable without optional deps.
- **`setup.py` extras**: Core install needs only `click` + `tqdm`. Optional groups: `openai`, `viz`, `config`, `templates`.
- **MockLLM for testing**: Always verify new features work with `MockLLM` first. It returns keyword-based canned responses (see `MockLLM.generate()`).
- **Health rating thresholds**: ≥90 Excellent, ≥75 Good, ≥60 Fair, else Needs Improvement (in `_generate_summary()`).

## Planned Evolution (see `docs/PROJECT_PLAN.md`)

The project is evolving toward: LLM-as-Judge evaluation (replacing regex), real dataset loading via HuggingFace `datasets`, automated red team attack generation (FLIRT method), SelfCheckGPT hallucination detection, refusal calibration testing, and new providers (Anthropic, Ollama). When implementing new features, align with the planned module structure in `docs/PROJECT_PLAN.md` §8.
