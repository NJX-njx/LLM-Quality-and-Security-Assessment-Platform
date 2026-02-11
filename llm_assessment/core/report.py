"""
Report generation for assessment results
"""

from typing import Dict, Any
import html


class ReportGenerator:
    """Generate comprehensive assessment reports"""
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize report generator
        
        Args:
            results: Assessment results from AssessmentPlatform
        """
        self.results = results
        
    def generate_text_report(self) -> str:
        """Generate plain text report"""
        report = []
        report.append("=" * 80)
        report.append("LLM QUALITY & SECURITY ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Metadata
        metadata = self.results.get("metadata", {})
        report.append(f"Model: {metadata.get('model_name', 'Unknown')}")
        report.append(f"Assessment Date: {metadata.get('start_time', 'Unknown')}")
        report.append("")
        
        # Summary
        summary = self.results.get("summary", {})
        report.append("OVERALL HEALTH CHECK")
        report.append("-" * 80)
        report.append(f"Overall Score: {summary.get('overall_health_score', 0):.1f}/100")
        report.append(f"Rating: {summary.get('health_rating', 'Unknown')}")
        report.append("")
        report.append(f"  Capability (Benchmark): {summary.get('benchmark_average', 0):.1f}/100")
        report.append(f"  Security: {summary.get('security_average', 0):.1f}/100")
        report.append(f"  Alignment: {summary.get('alignment_average', 0):.1f}/100")
        report.append("")
        report.append(f"Total Vulnerabilities Found: {summary.get('total_vulnerabilities', 0)}")
        report.append("")
        
        # Benchmark Results
        report.append("=" * 80)
        report.append("1. CAPABILITY BENCHMARKS")
        report.append("=" * 80)
        for result in self.results.get("benchmark_results", []):
            report.append("")
            report.append(f"{result['name']} ({result['category']})")
            report.append(f"  Score: {result['score']:.1f}%")
            report.append(f"  Correct: {result['correct_answers']}/{result['total_questions']}")
        
        # Red Teaming Results
        report.append("")
        report.append("=" * 80)
        report.append("2. SECURITY RED TEAM TESTS")
        report.append("=" * 80)
        for result in self.results.get("red_teaming_results", []):
            report.append("")
            report.append(f"{result['name']} ({result['category']})")
            report.append(f"  Security Score: {result['security_score']:.1f}%")
            report.append(f"  Vulnerabilities: {result['vulnerabilities_found']}/{result['total_tests']}")
            
            # Show vulnerable cases
            if result['vulnerabilities_found'] > 0:
                vulnerable_cases = [r for r in result['details']['results'] if r['vulnerable']]
                report.append(f"  Vulnerable cases:")
                for i, case in enumerate(vulnerable_cases[:3], 1):  # Show first 3
                    report.append(f"    {i}. {case['attack_type']} (severity: {case['severity']})")
        
        # Alignment Results
        report.append("")
        report.append("=" * 80)
        report.append("3. ALIGNMENT VERIFICATION")
        report.append("=" * 80)
        for result in self.results.get("alignment_results", []):
            report.append("")
            report.append(f"{result['name']} ({result['category']})")
            report.append(f"  Alignment Score: {result['alignment_score']:.1f}%")
            report.append(f"  Passed: {result['passed_tests']}/{result['total_tests']}")
        
        # Recommendations
        report.append("")
        report.append("=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        report.extend(self._generate_recommendations())
        
        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_html_report(self) -> str:
        """Generate HTML report with Chart.js radar charts"""
        summary = self.results.get("summary", {})
        metadata = self.results.get("metadata", {})
        
        # Escape HTML to prevent injection
        model_name = html.escape(metadata.get('model_name', 'Unknown'))
        start_time = html.escape(metadata.get('start_time', 'Unknown'))
        health_rating = html.escape(summary.get('health_rating', 'Unknown'))

        # Rating color
        score = summary.get('overall_health_score', 0)
        if score >= 90:
            rating_color = "#4CAF50"
        elif score >= 75:
            rating_color = "#8BC34A"
        elif score >= 60:
            rating_color = "#FF9800"
        else:
            rating_color = "#f44336"
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Assessment Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h3 {{ margin-top: 0; color: #333; }}
        .score {{
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }}
        .rating {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            background-color: {rating_color};
            color: white;
            font-weight: bold;
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
            height: 350px;
        }}
        .chart-container h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .test-result {{
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            background-color: #f9f9f9;
            border-radius: 0 6px 6px 0;
        }}
        .progress-bar {{
            background-color: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.3s ease;
        }}
        .progress-fill.security {{ background: linear-gradient(90deg, #4CAF50, #8BC34A); }}
        .progress-fill.danger {{ background: linear-gradient(90deg, #f44336, #ff5722); }}
        .vulnerability {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 0 4px 4px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{ background-color: #f5f5f5; }}
        .severity-critical {{ color: #d32f2f; font-weight: bold; }}
        .severity-high {{ color: #f44336; }}
        .severity-medium {{ color: #ff9800; }}
        .severity-low {{ color: #4caf50; }}
        @media (max-width: 768px) {{
            .chart-row {{ grid-template-columns: 1fr; }}
            .summary-cards {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>&#128269; LLM Quality &amp; Security Assessment Report</h1>
        <p>Model: {model_name}</p>
        <p>Assessment Date: {start_time}</p>
    </div>
    
    <div class="summary-cards">
        <div class="card">
            <h3>Overall Health</h3>
            <div class="score">{summary.get('overall_health_score', 0):.1f}</div>
            <div class="rating">{health_rating}</div>
        </div>
        <div class="card">
            <h3>Capability</h3>
            <div class="score">{summary.get('benchmark_average', 0):.1f}</div>
            <small>Benchmark Score (40%)</small>
        </div>
        <div class="card">
            <h3>Security</h3>
            <div class="score">{summary.get('security_average', 0):.1f}</div>
            <small>Security Score (30%)</small>
        </div>
        <div class="card">
            <h3>Alignment</h3>
            <div class="score">{summary.get('alignment_average', 0):.1f}</div>
            <small>Alignment Score (30%)</small>
        </div>
    </div>
"""
        # ---- Radar Chart + Bar Chart ----
        benchmark_labels = []
        benchmark_scores = []
        for r in self.results.get("benchmark_results", []):
            benchmark_labels.append(r['name'])
            benchmark_scores.append(round(r['score'], 1))

        security_labels = []
        security_scores = []
        for r in self.results.get("red_teaming_results", []):
            security_labels.append(r['name'])
            security_scores.append(round(r['security_score'], 1))

        alignment_labels = []
        alignment_scores = []
        for r in self.results.get("alignment_results", []):
            alignment_labels.append(r['name'])
            score_val = r.get('alignment_score', r.get('calibration_score', r.get('toxicity_score', r.get('hallucination_score', 0))))
            alignment_scores.append(round(score_val, 1))

        # Radar chart data
        radar_labels = ["Capability", "Security", "Alignment"]
        radar_data = [
            round(summary.get('benchmark_average', 0), 1),
            round(summary.get('security_average', 0), 1),
            round(summary.get('alignment_average', 0), 1),
        ]

        import json
        html_content += f"""
    <div class="chart-row">
        <div class="chart-container">
            <h3>&#128202; Overall Radar</h3>
            <canvas id="radarChart"></canvas>
        </div>
        <div class="chart-container">
            <h3>&#128200; Benchmark Scores</h3>
            <canvas id="benchmarkChart"></canvas>
        </div>
    </div>
    <div class="chart-row">
        <div class="chart-container">
            <h3>&#128737; Security Scores</h3>
            <canvas id="securityChart"></canvas>
        </div>
        <div class="chart-container">
            <h3>&#10024; Alignment Scores</h3>
            <canvas id="alignmentChart"></canvas>
        </div>
    </div>

    <script>
    // Radar Chart - Overall
    new Chart(document.getElementById('radarChart'), {{
        type: 'radar',
        data: {{
            labels: {json.dumps(radar_labels)},
            datasets: [{{
                label: '{model_name}',
                data: {json.dumps(radar_data)},
                borderColor: 'rgba(102, 126, 234, 1)',
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(102, 126, 234, 1)',
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                r: {{
                    beginAtZero: true,
                    max: 100,
                    ticks: {{ stepSize: 20 }}
                }}
            }}
        }}
    }});

    // Benchmark Bar Chart
    new Chart(document.getElementById('benchmarkChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(benchmark_labels)},
            datasets: [{{
                label: 'Score',
                data: {json.dumps(benchmark_scores)},
                backgroundColor: 'rgba(102, 126, 234, 0.7)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{ y: {{ beginAtZero: true, max: 100 }} }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});

    // Security Bar Chart
    new Chart(document.getElementById('securityChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(security_labels)},
            datasets: [{{
                label: 'Security Score',
                data: {json.dumps(security_scores)},
                backgroundColor: {json.dumps(security_scores)}.map(v => v >= 80 ? 'rgba(76,175,80,0.7)' : v >= 60 ? 'rgba(255,152,0,0.7)' : 'rgba(244,67,54,0.7)'),
                borderWidth: 1
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{ y: {{ beginAtZero: true, max: 100 }} }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});

    // Alignment Bar Chart
    new Chart(document.getElementById('alignmentChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(alignment_labels)},
            datasets: [{{
                label: 'Alignment Score',
                data: {json.dumps(alignment_scores)},
                backgroundColor: 'rgba(118, 75, 162, 0.7)',
                borderColor: 'rgba(118, 75, 162, 1)',
                borderWidth: 1
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{ y: {{ beginAtZero: true, max: 100 }} }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});
    </script>
"""
        
        # Benchmark section
        html_content += """
    <div class="section">
        <h2>&#128202; Capability Benchmarks</h2>
"""
        for result in self.results.get("benchmark_results", []):
            name = html.escape(result['name'])
            category = html.escape(result['category'])
            html_content += f"""
        <div class="test-result">
            <h4>{name} ({category})</h4>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {result['score']:.0f}%"></div>
            </div>
            <p>Score: {result['score']:.1f}% ({result['correct_answers']}/{result['total_questions']} correct)</p>
        </div>
"""
        html_content += "    </div>\n"
        
        # Security section with vulnerability table
        html_content += """
    <div class="section">
        <h2>&#128737; Security Red Team Tests</h2>
"""
        for result in self.results.get("red_teaming_results", []):
            name = html.escape(result['name'])
            category = html.escape(result['category'])
            fill_cls = "security" if result['security_score'] >= 80 else ("" if result['security_score'] >= 50 else "danger")
            html_content += f"""
        <div class="test-result">
            <h4>{name} ({category})</h4>
            <div class="progress-bar">
                <div class="progress-fill {fill_cls}" style="width: {result['security_score']:.0f}%"></div>
            </div>
            <p>Security Score: {result['security_score']:.1f}%</p>
            <p>Vulnerabilities Found: {result['vulnerabilities_found']}/{result['total_tests']}</p>
"""
            if result['vulnerabilities_found'] > 0:
                html_content += """            <div class="vulnerability">
                <strong>&#9888;&#65039; Vulnerabilities detected!</strong>
            </div>
            <table><tr><th>#</th><th>Attack Type</th><th>Severity</th></tr>
"""
                vuln_cases = [r for r in result['details'].get('results', []) if r.get('vulnerable')]
                for idx, case in enumerate(vuln_cases[:10], 1):
                    sev = html.escape(case.get('severity', 'medium'))
                    atype = html.escape(case.get('attack_type', 'unknown'))
                    html_content += f"            <tr><td>{idx}</td><td>{atype}</td><td class=\"severity-{sev}\">{sev}</td></tr>\n"
                if len(vuln_cases) > 10:
                    html_content += f"            <tr><td colspan='3'>... and {len(vuln_cases) - 10} more</td></tr>\n"
                html_content += "            </table>\n"
            html_content += "        </div>\n"
        html_content += "    </div>\n"
        
        # Alignment section
        html_content += """
    <div class="section">
        <h2>&#10024; Alignment Verification</h2>
"""
        for result in self.results.get("alignment_results", []):
            name = html.escape(result['name'])
            category = html.escape(result['category'])
            score_val = result.get('alignment_score', result.get('calibration_score', result.get('toxicity_score', result.get('hallucination_score', 0))))
            passed = result.get('passed_tests', result.get('total_tests', 0))
            total = result.get('total_tests', 0)
            html_content += f"""
        <div class="test-result">
            <h4>{name} ({category})</h4>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {score_val:.0f}%"></div>
            </div>
            <p>Score: {score_val:.1f}%</p>
            <p>Passed: {passed}/{total}</p>
        </div>
"""
        html_content += "    </div>\n"
        
        # Recommendations
        html_content += """
    <div class="section">
        <h2>&#128161; Recommendations</h2>
        <ul>
"""
        for rec in self._generate_recommendations():
            escaped_rec = html.escape(rec)
            html_content += f"            <li>{escaped_rec}</li>\n"
        html_content += """        </ul>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        return html_content
    
    def _generate_recommendations(self) -> list:
        """Generate recommendations based on results"""
        recommendations = []
        summary = self.results.get("summary", {})
        
        # Benchmark recommendations
        if summary.get("benchmark_average", 0) < 70:
            recommendations.append("Consider additional training on general knowledge and reasoning tasks.")
        
        # Security recommendations
        if summary.get("total_vulnerabilities", 0) > 0:
            recommendations.append(f"Address {summary['total_vulnerabilities']} security vulnerabilities found during red team testing.")
        
        if summary.get("security_average", 0) < 80:
            recommendations.append("Implement stronger safety filters and guardrails against adversarial inputs.")
        
        # Alignment recommendations
        if summary.get("alignment_average", 0) < 80:
            recommendations.append("Enhance alignment training to improve helpfulness, harmlessness, and honesty.")
        
        # Check specific tests
        for result in self.results.get("alignment_results", []):
            if result["alignment_score"] < 70:
                recommendations.append(f"Focus on improving {result['name']} alignment.")
        
        if not recommendations:
            recommendations.append("Model shows good overall performance. Continue monitoring and testing regularly.")
        
        return recommendations
    
    def save_report(self, filepath: str, format: str = "html"):
        """
        Save report to file
        
        Args:
            filepath: Output file path
            format: Report format ('html' or 'text')
        """
        if format == "html":
            content = self.generate_html_report()
        else:
            content = self.generate_text_report()
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"\nReport saved to: {filepath}")
