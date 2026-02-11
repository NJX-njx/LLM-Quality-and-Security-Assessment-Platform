"""
LLM 评估平台内置数据包。

包含基准测试样本、攻击库、对齐测试场景、
以及幻觉检测提示词，以 JSONL 数据文件形式存储。

目录结构：
    benchmarks/     - MMLU、GSM8K 等的回退样本题目
    attacks/        - 红队测试的静态攻击提示库
    alignment/      - 对齐和安全测试场景
    hallucination/  - 幻觉检测的事实提示集
"""

import os

# 数据目录的根路径（用于解析数据文件路径）
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS_DIR = os.path.join(DATA_DIR, "benchmarks")      # 基准测试数据
ATTACKS_DIR = os.path.join(DATA_DIR, "attacks")            # 攻击提示数据
ALIGNMENT_DIR = os.path.join(DATA_DIR, "alignment")        # 对齐测试数据
HALLUCINATION_DIR = os.path.join(DATA_DIR, "hallucination") # 幻觉检测数据
