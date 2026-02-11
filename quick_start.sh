#!/bin/bash

# Quick start script for LLM Assessment Platform
# This demonstrates how to run a complete assessment

echo "🚀 LLM Quality & Security Assessment Platform"
echo "=============================================="
echo ""

# Check if package is installed
if ! command -v llm-assess &> /dev/null; then
    echo "Installing LLM Assessment Platform..."
    pip install -e . --quiet
fi

echo "Running complete assessment with mock LLM..."
echo ""

# Run assessment
llm-assess assess \
    --provider mock \
    --max-questions 3 \
    --output quick_start_results.json \
    --report-format html

echo ""
echo "✅ Assessment complete!"
echo ""
echo "Generated files:"
echo "  - quick_start_results.json (JSON results)"
echo "  - quick_start_results.html (HTML report)"
echo ""
echo "To view the HTML report, open: quick_start_results.html"
echo ""
echo "For more options, run: llm-assess --help"
