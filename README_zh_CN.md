# LLM 质量与安全评估平台

**一键体检报告** - 统一的 LLM 质量与安全评估平台

构建一个统一的 LLM 质量与安全评估平台，集成能力基准测试、安全红队测试与对齐性检查，提供“一键体检报告”。

## ✨ 功能

- 🎯 **能力基准**：评估 LLM 在知识、推理与编程任务上的表现
- 🛡️ **安全红队**：检测越狱、提示注入与数据泄露等漏洞
- ✅ **对齐性验证**：检查有用性、无害性、诚实性与偏见
- 📊 **完整报告**：生成包含可视化的 HTML 与文本报告
- 🚀 **一键评估**：使用单条命令运行完整评估流程
- 🔌 **可扩展**：方便添加自定义基准与测试

## 🏗️ 架构

平台由三个主要模块组成：

1. **基准模块**（`llm_assessment.benchmark`）
   - MMLU（大规模多任务语言理解）
   - 推理基准
   - 编程能力测试

2. **红队模块**（`llm_assessment.red_teaming`）
   - 越狱检测
   - 提示注入测试
   - 数据泄露检测
   - 有害内容过滤

3. **对齐模块**（`llm_assessment.alignment`）
   - 有用性评估
   - 无害性验证
   - 诚实性评估
   - 偏见与公平性测试

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/NJX-njx/LLM-Quality-and-Security-Assessment-Platform.git
cd LLM-Quality-and-Security-Assessment-Platform

# 安装依赖
pip install -r requirements.txt

# 安装包（可选，便于命令行使用）
pip install -e .
```

## 🚀 快速开始

### 命令行接口

```bash
# 使用示例 mock LLM 运行完整评估（演示）
llm-assess assess --provider mock

# 使用 OpenAI
export OPENAI_API_KEY="your-api-key"
llm-assess assess --provider openai --model gpt-3.5-turbo

# 运行单模块
llm-assess benchmark --provider mock --max-questions 5
llm-assess security --provider mock
llm-assess alignment --provider mock

# 从已有结果生成报告
llm-assess report assessment_results.json --format html
```

### Python API

```python
from llm_assessment.core.llm_wrapper import create_llm
from llm_assessment.core.assessment import AssessmentPlatform
from llm_assessment.core.report import ReportGenerator

# 创建 LLM 实例
llm = create_llm("mock")  # 或者使用 "openai" 并提供 api_key

# 运行评估
platform = AssessmentPlatform(llm)
results = platform.run_all(max_benchmark_questions=5)

# 生成报告
report_gen = ReportGenerator(results)
report_gen.save_report("report.html", format="html")

# 打印摘要
summary = results["summary"]
print(f"Overall Health Score: {summary['overall_health_score']:.1f}/100")
print(f"Health Rating: {summary['health_rating']}")
```

## 📖 使用示例

### 示例 1：基本用法

```python
from llm_assessment.core.llm_wrapper import create_llm
from llm_assessment.core.assessment import AssessmentPlatform

# 创建并评估 LLM
llm = create_llm("mock", model_name="demo-model")
platform = AssessmentPlatform(llm)
results = platform.run_all(max_benchmark_questions=5)

# 保存结果
platform.save_results("results.json")
```

### 示例 2：单独运行模块

```python
from llm_assessment.core.llm_wrapper import create_llm
from llm_assessment.core.assessment import AssessmentPlatform

llm = create_llm("mock")
platform = AssessmentPlatform(llm)

# 仅运行基准
benchmark_results = platform.run_benchmarks(max_questions=3)

# 仅运行安全测试
security_results = platform.run_red_teaming()

# 仅运行对齐测试
alignment_results = platform.run_alignment()
```

### 示例 3：自定义基准

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

# 使用自定义基准
benchmark = CustomBenchmark()
result = benchmark.run(llm)
```

## 📊 报告示例

平台会生成包含以下内容的完整报告：

- **总体健康分**：各项评估的加权平均分
- **健康评级**：Excellent, Good, Fair, 或 Needs Improvement
- **详细得分**：每个测试类别的独立得分
- **漏洞列表**：红队测试中发现的安全问题
- **改进建议**：可执行的改进建议

### 示例输出

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

## 🔧 配置

创建 `.env` 文件以存放 API 密钥：

```bash
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

## 🧪 测试

运行示例：

```bash
# 基本使用示例
python examples/basic_usage.py

# 单独模块示例
python examples/individual_modules.py

# 自定义基准示例
python examples/custom_benchmark.py
```

## 📚 文档

### LLM 提供者

当前支持的提供者：
- **mock**：用于测试与演示的模拟 LLM（无需 API Key）
- **openai**：OpenAI 模型（GPT-3.5、GPT-4 等）

### 添加自定义提供者

可扩展 `llm_assessment/core/llm_wrapper.py` 中的 `BaseLLM`：

```python
class CustomLLM(BaseLLM):
    def generate(self, prompt: str, **kwargs) -> str:
        # 实现你的 LLM 生成逻辑
        pass
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # 实现你的 LLM 聊天逻辑
        pass
```

### 扩展平台

你可以扩展以下模块：

1. **自定义基准**：继承 `BaseBenchmark`
2. **自定义安全测试**：继承 `BaseRedTeamTest`
3. **自定义对齐测试**：继承 `BaseAlignmentTest`

参见 `examples/custom_benchmark.py` 获取完整示例。

## 🤝 参与贡献

欢迎贡献！欢迎提交 Pull Request。

## 📄 许可

本项目采用 MIT 许可，详见 LICENSE 文件。

## 🙏 致谢

本平台借鉴并整合了以下最佳实践：
- MMLU 基准方法论
- OWASP LLM 安全指南
- Anthropic 的 Constitutional AI 原则
- OpenAI 的对齐研究

## 📧 联系方式

如有问题或反馈，请在 GitHub 上打开 issue。
