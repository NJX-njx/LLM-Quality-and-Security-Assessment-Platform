"""
Example: Basic usage of LLM Assessment Platform
"""

from llm_assessment.core.llm_wrapper import create_llm
from llm_assessment.core.assessment import AssessmentPlatform
from llm_assessment.core.report import ReportGenerator


def main():
    # Create a mock LLM for demonstration
    # In production, use: create_llm("openai", model_name="gpt-3.5-turbo", api_key="your-key")
    llm = create_llm("mock", model_name="demo-model")
    
    # Initialize the assessment platform
    platform = AssessmentPlatform(llm)
    
    # Run complete assessment (one-click health check)
    print("Running complete LLM assessment...")
    results = platform.run_all(max_benchmark_questions=5)
    
    # Save results
    platform.save_results("demo_results.json")
    
    # Generate reports
    report_gen = ReportGenerator(results)
    report_gen.save_report("demo_report.html", format="html")
    report_gen.save_report("demo_report.txt", format="text")
    
    # Print summary
    summary = results["summary"]
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Overall Health Score: {summary['overall_health_score']:.1f}/100")
    print(f"Health Rating: {summary['health_rating']}")
    print(f"  - Capability: {summary['benchmark_average']:.1f}%")
    print(f"  - Security: {summary['security_average']:.1f}%")
    print(f"  - Alignment: {summary['alignment_average']:.1f}%")
    print(f"Total Vulnerabilities: {summary['total_vulnerabilities']}")
    print("="*70)


if __name__ == "__main__":
    main()
