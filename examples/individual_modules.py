"""
Example: Running individual test modules
"""

from llm_assessment.core.llm_wrapper import create_llm
from llm_assessment.core.assessment import AssessmentPlatform


def main():
    # Create LLM instance
    llm = create_llm("mock")
    platform = AssessmentPlatform(llm)
    
    print("="*70)
    print("Running Individual Test Modules")
    print("="*70)
    
    # 1. Run only benchmarks
    print("\n1. CAPABILITY BENCHMARKS")
    print("-"*70)
    benchmark_results = platform.run_benchmarks(max_questions=3)
    for result in benchmark_results:
        print(f"  {result['name']}: {result['score']:.1f}%")
    
    # 2. Run only security tests
    print("\n2. SECURITY RED TEAM TESTS")
    print("-"*70)
    security_results = platform.run_red_teaming()
    for result in security_results:
        print(f"  {result['name']}: {result['security_score']:.1f}% ({result['vulnerabilities_found']} vulnerabilities)")
    
    # 3. Run only alignment tests
    print("\n3. ALIGNMENT VERIFICATION")
    print("-"*70)
    alignment_results = platform.run_alignment()
    for result in alignment_results:
        print(f"  {result['name']}: {result['alignment_score']:.1f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
