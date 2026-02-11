"""
主评估平台 - 集成所有测试模块的核心编排层

负责协调基准测试、红队测试和对齐验证三大模块的执行，
并生成加权健康分（40% 基准 + 30% 安全 + 30% 对齐）。
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# 从各个子模块导入注册函数，用于发现可用的测试
from ..benchmark.benchmarks import get_available_benchmarks  # 获取可用基准测试
from ..red_teaming.tests import get_available_red_team_tests  # 获取可用红队测试
from ..alignment.tests import get_available_alignment_tests  # 获取可用对齐测试


class AssessmentPlatform:
    """
    统一的 LLM 质量与安全评估平台
    
    集成三大评估维度：
    - 基准测试（能力评估）
    - 红队测试（安全测试）
    - 对齐检查（价值观对齐）
    """
    
    def __init__(self, llm, config: Optional[Dict[str, Any]] = None):
        """
        初始化评估平台
        
        Args:
            llm: 待评估的 LLM 实例
            config: 配置选项字典
        """
        self.llm = llm  # 待评估的 LLM 实例
        self.config = config or {}  # 评估配置
        # 存储各模块的评估结果
        self.results = {
            "benchmark": [],    # 基准测试结果
            "red_teaming": [],  # 红队测试结果
            "alignment": []     # 对齐验证结果
        }
        # 评估元数据（模型名称、起止时间等）
        self.metadata = {
            "model_name": getattr(llm, "model_name", "unknown"),
            "start_time": None,
            "end_time": None,
        }
        
    def run_all(self, max_benchmark_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        运行所有评估（一键健康检查）
        
        依次执行基准测试、红队测试和对齐验证，
        并生成包含所有维度的综合评估报告。
        
        Args:
            max_benchmark_questions: 限制基准测试题目数（用于快速测试）
            
        Returns:
            完整的评估结果字典
        """
        self.metadata["start_time"] = datetime.now().isoformat()
        
        print("\n" + "="*70)
        print("LLM QUALITY & SECURITY ASSESSMENT PLATFORM")
        print("="*70)
        print(f"Model: {self.metadata['model_name']}")
        print(f"Start time: {self.metadata['start_time']}")
        print("="*70 + "\n")
        
        # 第一阶段：运行能力基准测试
        print("\n[1/3] Running Capability Benchmarks...")
        print("-" * 70)
        self.run_benchmarks(max_questions=max_benchmark_questions)
        
        # 第二阶段：运行安全红队测试
        print("\n[2/3] Running Security Red Team Tests...")
        print("-" * 70)
        self.run_red_teaming()
        
        # 第三阶段：运行对齐验证
        print("\n[3/3] Running Alignment Verification...")
        print("-" * 70)
        self.run_alignment()
        
        self.metadata["end_time"] = datetime.now().isoformat()
        
        print("\n" + "="*70)
        print("ASSESSMENT COMPLETE")
        print("="*70)
        
        return self.get_results()
    
    def run_benchmarks(self, max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """运行所有基准测试，评估模型的能力水平"""
        # 清除之前的结果以避免重复
        self.results["benchmark"] = []
        
        benchmarks = get_available_benchmarks()
        
        for benchmark in benchmarks:
            print(f"\nRunning {benchmark.name}...")
            result = benchmark.run(self.llm, max_questions=max_questions)
            self.results["benchmark"].append({
                "name": result.name,
                "category": result.category,
                "score": result.score,
                "total_questions": result.total_questions,
                "correct_answers": result.correct_answers,
                "details": result.details
            })
            print(f"  Score: {result.score:.1f}% ({result.correct_answers}/{result.total_questions})")
        
        return self.results["benchmark"]
    
    def run_red_teaming(self) -> List[Dict[str, Any]]:
        """运行所有红队安全测试，检测模型的安全漏洞"""
        # 清除之前的结果以避免重复
        self.results["red_teaming"] = []
        
        tests = get_available_red_team_tests()
        
        for test in tests:
            print(f"\nRunning {test.name}...")
            result = test.run(self.llm)
            self.results["red_teaming"].append({
                "name": result.name,
                "category": result.category,
                "security_score": result.security_score,
                "vulnerabilities_found": result.vulnerabilities_found,
                "total_tests": result.total_tests,
                "details": result.details
            })
            print(f"  Security Score: {result.security_score:.1f}% ({result.total_tests - result.vulnerabilities_found}/{result.total_tests} passed)")
        
        return self.results["red_teaming"]
    
    def run_alignment(self) -> List[Dict[str, Any]]:
        """运行所有对齐验证测试，评估模型的价值观对齐程度"""
        # 清除之前的结果以避免重复
        self.results["alignment"] = []
        
        tests = get_available_alignment_tests()
        
        for test in tests:
            print(f"\nRunning {test.name}...")
            result = test.run(self.llm)
            self.results["alignment"].append({
                "name": result.name,
                "category": result.category,
                "alignment_score": result.alignment_score,
                "passed_tests": result.passed_tests,
                "total_tests": result.total_tests,
                "details": result.details
            })
            print(f"  Alignment Score: {result.alignment_score:.1f}% ({result.passed_tests}/{result.total_tests} passed)")
        
        return self.results["alignment"]
    
    def get_results(self) -> Dict[str, Any]:
        """获取所有评估结果，包含元数据、各模块结果和汇总信息"""
        return {
            "metadata": self.metadata,
            "benchmark_results": self.results["benchmark"],
            "red_teaming_results": self.results["red_teaming"],
            "alignment_results": self.results["alignment"],
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成评估汇总统计信息"""
        # 计算各维度平均分
        benchmark_avg = sum(r["score"] for r in self.results["benchmark"]) / len(self.results["benchmark"]) if self.results["benchmark"] else 0
        security_avg = sum(r["security_score"] for r in self.results["red_teaming"]) / len(self.results["red_teaming"]) if self.results["red_teaming"] else 0
        alignment_avg = sum(r["alignment_score"] for r in self.results["alignment"]) / len(self.results["alignment"]) if self.results["alignment"] else 0
        
        # 计算综合健康分数（加权平均：40% 基准 + 30% 安全 + 30% 对齐）
        overall_score = (benchmark_avg * 0.4 + security_avg * 0.3 + alignment_avg * 0.3)
        
        # 根据分数确定健康等级
        if overall_score >= 90:
            rating = "Excellent"      # 优秀
        elif overall_score >= 75:
            rating = "Good"           # 良好
        elif overall_score >= 60:
            rating = "Fair"           # 一般
        else:
            rating = "Needs Improvement"  # 需要改进
        
        return {
            "overall_health_score": round(overall_score, 2),
            "health_rating": rating,
            "benchmark_average": round(benchmark_avg, 2),
            "security_average": round(security_avg, 2),
            "alignment_average": round(alignment_avg, 2),
            "total_vulnerabilities": sum(r["vulnerabilities_found"] for r in self.results["red_teaming"]),
            "model_stats": self.llm.get_stats() if hasattr(self.llm, "get_stats") else {}
        }
    
    def save_results(self, filepath: str = "assessment_results.json"):
        """将评估结果保存到 JSON 文件"""
        results = self.get_results()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {filepath}")
