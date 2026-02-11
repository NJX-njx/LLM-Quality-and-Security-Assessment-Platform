"""
Built-in data package for LLM Assessment Platform.

Contains benchmark samples, attack libraries, alignment test scenarios,
and hallucination detection prompts as JSONL data files.

Directory structure:
    benchmarks/     - Fallback sample questions for MMLU, GSM8K, etc.
    attacks/        - Static attack prompt libraries for red teaming
    alignment/      - Alignment and safety test scenarios
    hallucination/  - Factual prompt sets for hallucination detection
"""

import os

# Root path of the data directory (for resolving data file paths)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS_DIR = os.path.join(DATA_DIR, "benchmarks")
ATTACKS_DIR = os.path.join(DATA_DIR, "attacks")
ALIGNMENT_DIR = os.path.join(DATA_DIR, "alignment")
HALLUCINATION_DIR = os.path.join(DATA_DIR, "hallucination")
