"""
Command-line interface for LLM Assessment Platform
"""

import click
import json
import os
from pathlib import Path

from .core.llm_wrapper import create_llm
from .core.assessment import AssessmentPlatform
from .core.report import ReportGenerator


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    LLM Quality and Security Assessment Platform
    
    A unified platform for evaluating LLM capabilities, security, and alignment.
    """
    pass


@main.command()
@click.option("--provider", default="mock", help="LLM provider (mock, openai)")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key (or set via environment)")
@click.option("--max-questions", default=None, type=int, help="Limit benchmark questions")
@click.option("--output", default="assessment_results.json", help="Output file path")
@click.option("--report-format", default="html", type=click.Choice(["html", "text"]), help="Report format")
def assess(provider, model, api_key, max_questions, output, report_format):
    """
    Run complete LLM assessment (one-click health check)
    """
    click.echo("🚀 Starting LLM Assessment Platform...")
    
    # Get API key from environment if not provided
    if not api_key and provider != "mock":
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else None
    
    # Create LLM instance
    try:
        llm_kwargs = {}
        if model:
            llm_kwargs["model_name"] = model
        if api_key:
            llm_kwargs["api_key"] = api_key
        
        llm = create_llm(provider=provider, **llm_kwargs)
        click.echo(f"✓ Initialized {provider} LLM")
    except Exception as e:
        click.echo(f"❌ Error initializing LLM: {e}", err=True)
        return
    
    # Run assessment
    try:
        platform = AssessmentPlatform(llm)
        results = platform.run_all(max_benchmark_questions=max_questions)
        
        # Save JSON results
        platform.save_results(output)
        
        # Generate and save report
        report_file = output.replace(".json", f".{report_format}")
        report_gen = ReportGenerator(results)
        report_gen.save_report(report_file, format=report_format)
        
        # Print summary
        summary = results["summary"]
        click.echo("\n" + "="*70)
        click.echo("📊 ASSESSMENT SUMMARY")
        click.echo("="*70)
        click.echo(f"Overall Health Score: {summary['overall_health_score']:.1f}/100")
        click.echo(f"Rating: {summary['health_rating']}")
        click.echo(f"  • Capability: {summary['benchmark_average']:.1f}/100")
        click.echo(f"  • Security: {summary['security_average']:.1f}/100")
        click.echo(f"  • Alignment: {summary['alignment_average']:.1f}/100")
        click.echo(f"\n⚠️  Total Vulnerabilities: {summary['total_vulnerabilities']}")
        click.echo("="*70)
        
    except Exception as e:
        click.echo(f"❌ Error during assessment: {e}", err=True)
        import traceback
        traceback.print_exc()
        return


@main.command()
@click.option("--provider", default="mock", help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
@click.option("--max-questions", default=5, type=int, help="Questions per benchmark")
def benchmark(provider, model, api_key, max_questions):
    """
    Run only capability benchmarks
    """
    if not api_key and provider != "mock":
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else None
    
    llm_kwargs = {}
    if model:
        llm_kwargs["model_name"] = model
    if api_key:
        llm_kwargs["api_key"] = api_key
    
    llm = create_llm(provider=provider, **llm_kwargs)
    platform = AssessmentPlatform(llm)
    
    click.echo("Running benchmarks...")
    results = platform.run_benchmarks(max_questions=max_questions)
    
    click.echo("\n📊 Benchmark Results:")
    for result in results:
        click.echo(f"  {result['name']}: {result['score']:.1f}%")


@main.command()
@click.option("--provider", default="mock", help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
def security(provider, model, api_key):
    """
    Run only security red team tests
    """
    if not api_key and provider != "mock":
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else None
    
    llm_kwargs = {}
    if model:
        llm_kwargs["model_name"] = model
    if api_key:
        llm_kwargs["api_key"] = api_key
    
    llm = create_llm(provider=provider, **llm_kwargs)
    platform = AssessmentPlatform(llm)
    
    click.echo("Running security tests...")
    results = platform.run_red_teaming()
    
    click.echo("\n🛡️ Security Test Results:")
    for result in results:
        click.echo(f"  {result['name']}: {result['security_score']:.1f}% ({result['vulnerabilities_found']} vulnerabilities)")


@main.command()
@click.option("--provider", default="mock", help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
def alignment(provider, model, api_key):
    """
    Run only alignment verification tests
    """
    if not api_key and provider != "mock":
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else None
    
    llm_kwargs = {}
    if model:
        llm_kwargs["model_name"] = model
    if api_key:
        llm_kwargs["api_key"] = api_key
    
    llm = create_llm(provider=provider, **llm_kwargs)
    platform = AssessmentPlatform(llm)
    
    click.echo("Running alignment tests...")
    results = platform.run_alignment()
    
    click.echo("\n✨ Alignment Test Results:")
    for result in results:
        click.echo(f"  {result['name']}: {result['alignment_score']:.1f}%")


@main.command()
@click.argument("results_file")
@click.option("--format", default="html", type=click.Choice(["html", "text"]), help="Report format")
@click.option("--output", default=None, help="Output file path")
def report(results_file, format, output):
    """
    Generate report from existing results file
    """
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        
        if not output:
            output = results_file.replace(".json", f".{format}")
        
        report_gen = ReportGenerator(results)
        report_gen.save_report(output, format=format)
        
        click.echo(f"✓ Report generated: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error generating report: {e}", err=True)


if __name__ == "__main__":
    main()
