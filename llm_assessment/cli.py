"""
Command-line interface for LLM Assessment Platform
"""

import click
import json
import os

from .core.llm_wrapper import create_llm
from .core.assessment import AssessmentPlatform
from .core.report import ReportGenerator

# All supported providers
_PROVIDERS = [
    "mock", "openai", "azure", "anthropic", "ollama",
    "huggingface", "huggingface-local", "vllm", "custom",
]

# Env var lookup per provider (for API keys)
_ENV_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "huggingface": "HF_TOKEN",
    "huggingface-local": None,
    "ollama": None,
    "vllm": None,
    "mock": None,
    "custom": None,
}


def _resolve_api_key(provider, api_key):
    """Resolve API key from argument or environment variable."""
    if api_key:
        return api_key
    env_var = _ENV_KEY_MAP.get(provider)
    if env_var:
        return os.getenv(env_var)
    return None


def _build_llm_kwargs(provider, model, api_key, **extra):
    """Build kwargs dict for create_llm()."""
    kwargs = {}
    if model:
        kwargs["model_name"] = model
    resolved_key = _resolve_api_key(provider, api_key)
    if resolved_key:
        kwargs["api_key"] = resolved_key

    # Azure-specific env vars
    if provider == "azure":
        endpoint = extra.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        if endpoint:
            kwargs["azure_endpoint"] = endpoint

    # Ollama / vLLM / custom base_url
    base_url = extra.get("base_url")
    if base_url:
        kwargs["base_url"] = base_url

    return kwargs


@click.group()
@click.version_option(version="0.2.0")
def main():
    """
    LLM Quality and Security Assessment Platform

    A unified platform for evaluating LLM capabilities, security, and alignment.

    Supported providers: mock, openai, azure, anthropic, ollama,
    huggingface, huggingface-local, vllm, custom.
    """
    pass


@main.command()
@click.option("--provider", default="mock", type=click.Choice(_PROVIDERS), help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key (or set via environment)")
@click.option("--base-url", default=None, help="API base URL (for ollama/vllm/custom)")
@click.option("--max-questions", default=None, type=int, help="Limit benchmark questions")
@click.option("--output", default="assessment_results.json", help="Output file path")
@click.option("--report-format", default="html", type=click.Choice(["html", "text"]), help="Report format")
def assess(provider, model, api_key, base_url, max_questions, output, report_format):
    """
    Run complete LLM assessment (one-click health check)
    """
    click.echo("🚀 Starting LLM Assessment Platform...")

    # Create LLM instance
    try:
        llm_kwargs = _build_llm_kwargs(provider, model, api_key, base_url=base_url)
        llm = create_llm(provider=provider, **llm_kwargs)
        click.echo(f"✓ Initialized {provider} LLM")
    except Exception as e:
        raise click.ClickException(f"Error initializing LLM: {e}")
    
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
        raise click.ClickException(f"Error during assessment: {e}")


@main.command()
@click.option("--provider", default="mock", type=click.Choice(_PROVIDERS), help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
@click.option("--base-url", default=None, help="API base URL")
@click.option("--max-questions", default=5, type=int, help="Questions per benchmark")
def benchmark(provider, model, api_key, base_url, max_questions):
    """
    Run only capability benchmarks
    """
    llm_kwargs = _build_llm_kwargs(provider, model, api_key, base_url=base_url)

    try:
        llm = create_llm(provider=provider, **llm_kwargs)
    except Exception as e:
        raise click.ClickException("Error initializing LLM: {}".format(e))
    
    platform = AssessmentPlatform(llm)
    
    click.echo("Running benchmarks...")
    results = platform.run_benchmarks(max_questions=max_questions)
    
    click.echo("\n📊 Benchmark Results:")
    for result in results:
        click.echo(f"  {result['name']}: {result['score']:.1f}%")


@main.command()
@click.option("--provider", default="mock", type=click.Choice(_PROVIDERS), help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
@click.option("--base-url", default=None, help="API base URL")
def security(provider, model, api_key, base_url):
    """
    Run only security red team tests
    """
    llm_kwargs = _build_llm_kwargs(provider, model, api_key, base_url=base_url)

    try:
        llm = create_llm(provider=provider, **llm_kwargs)
    except Exception as e:
        raise click.ClickException("Error initializing LLM: {}".format(e))
    
    platform = AssessmentPlatform(llm)
    
    click.echo("Running security tests...")
    results = platform.run_red_teaming()
    
    click.echo("\n🛡️ Security Test Results:")
    for result in results:
        click.echo(f"  {result['name']}: {result['security_score']:.1f}% ({result['vulnerabilities_found']} vulnerabilities)")


@main.command()
@click.option("--provider", default="mock", type=click.Choice(_PROVIDERS), help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
@click.option("--base-url", default=None, help="API base URL")
def alignment(provider, model, api_key, base_url):
    """
    Run only alignment verification tests
    """
    llm_kwargs = _build_llm_kwargs(provider, model, api_key, base_url=base_url)

    try:
        llm = create_llm(provider=provider, **llm_kwargs)
    except Exception as e:
        raise click.ClickException("Error initializing LLM: {}".format(e))
    
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
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        if not output:
            output = results_file.replace(".json", f".{format}")
        
        report_gen = ReportGenerator(results)
        report_gen.save_report(output, format=format)
        
        click.echo(f"✓ Report generated: {output}")
        
    except Exception as e:
        raise click.ClickException("Error generating report: {}".format(e))


@main.command("list-providers")
def list_providers():
    """
    List all available LLM providers and their status.
    """
    from .providers import list_providers as _list, list_aliases

    click.echo("Available LLM Providers:")
    click.echo("-" * 40)
    for name in _list():
        click.echo("  {}".format(name))
    click.echo("")
    click.echo("Aliases:")
    for alias, target in sorted(list_aliases().items()):
        click.echo("  {} -> {}".format(alias, target))


@main.command("health-check")
@click.option("--provider", required=True, type=click.Choice(_PROVIDERS), help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
@click.option("--base-url", default=None, help="API base URL")
def health_check(provider, model, api_key, base_url):
    """
    Check connectivity and availability of an LLM provider.
    """
    llm_kwargs = _build_llm_kwargs(provider, model, api_key, base_url=base_url)

    try:
        llm = create_llm(provider=provider, **llm_kwargs)
    except Exception as e:
        raise click.ClickException("Error initializing LLM: {}".format(e))

    click.echo("Checking {} provider...".format(provider))
    result = llm.health_check()

    status = result.get("status", "unknown")
    icon = "✓" if status == "healthy" else "✗"
    click.echo("{} Status: {}".format(icon, status))
    for key, value in result.items():
        if key != "status":
            click.echo("  {}: {}".format(key, value))


if __name__ == "__main__":
    main()
