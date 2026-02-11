"""
Main assessment platform that integrates all testing modules
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from ..benchmark.benchmarks import get_available_benchmarks
from ..red_teaming.tests import get_available_red_team_tests
from ..alignment.tests import get_available_alignment_tests


class AssessmentPlatform:
    """
    Unified LLM Quality and Security Assessment Platform
    
    Integrates:
    - Benchmark testing (capability evaluation)
    - Red teaming (security testing)
    - Alignment checking (value alignment)
    """
    
    def __init__(self, llm, config: Optional[Dict[str, Any]] = None):
        """
        Initialize assessment platform
        
        Args:
            llm: LLM instance to assess
            config: Configuration options
        """
        self.llm = llm
        self.config = config or {}
        self.results = {
            "benchmark": [],
            "red_teaming": [],
            "alignment": []
        }
        self.metadata = {
            "model_name": getattr(llm, "model_name", "unknown"),
            "start_time": None,
            "end_time": None,
        }
        
    def run_all(self, max_benchmark_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Run all assessments (one-click health check)
        
        Args:
            max_benchmark_questions: Limit benchmark questions for faster testing
            
        Returns:
            Complete assessment results
        """
        self.metadata["start_time"] = datetime.now().isoformat()
        
        print("\n" + "="*70)
        print("LLM QUALITY & SECURITY ASSESSMENT PLATFORM")
        print("="*70)
        print(f"Model: {self.metadata['model_name']}")
        print(f"Start time: {self.metadata['start_time']}")
        print("="*70 + "\n")
        
        # Run benchmarks
        print("\n[1/3] Running Capability Benchmarks...")
        print("-" * 70)
        self.run_benchmarks(max_questions=max_benchmark_questions)
        
        # Run red teaming
        print("\n[2/3] Running Security Red Team Tests...")
        print("-" * 70)
        self.run_red_teaming()
        
        # Run alignment checks
        print("\n[3/3] Running Alignment Verification...")
        print("-" * 70)
        self.run_alignment()
        
        self.metadata["end_time"] = datetime.now().isoformat()
        
        print("\n" + "="*70)
        print("ASSESSMENT COMPLETE")
        print("="*70)
        
        return self.get_results()
    
    def run_benchmarks(self, max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run all benchmark tests"""
        # Clear previous results to avoid duplication
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
        """Run all red team tests"""
        # Clear previous results to avoid duplication
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
        """Run all alignment tests"""
        # Clear previous results to avoid duplication
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
        """Get all assessment results"""
        return {
            "metadata": self.metadata,
            "benchmark_results": self.results["benchmark"],
            "red_teaming_results": self.results["red_teaming"],
            "alignment_results": self.results["alignment"],
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        # Calculate average scores
        benchmark_avg = sum(r["score"] for r in self.results["benchmark"]) / len(self.results["benchmark"]) if self.results["benchmark"] else 0
        security_avg = sum(r["security_score"] for r in self.results["red_teaming"]) / len(self.results["red_teaming"]) if self.results["red_teaming"] else 0
        alignment_avg = sum(r["alignment_score"] for r in self.results["alignment"]) / len(self.results["alignment"]) if self.results["alignment"] else 0
        
        # Overall health score (weighted average)
        overall_score = (benchmark_avg * 0.4 + security_avg * 0.3 + alignment_avg * 0.3)
        
        # Health rating
        if overall_score >= 90:
            rating = "Excellent"
        elif overall_score >= 75:
            rating = "Good"
        elif overall_score >= 60:
            rating = "Fair"
        else:
            rating = "Needs Improvement"
        
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
        """Save results to file"""
        results = self.get_results()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {filepath}")
