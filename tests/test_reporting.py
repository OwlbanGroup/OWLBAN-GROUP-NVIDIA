#!/usr/bin/env python3
"""
Automated Test Reporting for JPMorgan Financial APIs
Generates comprehensive test reports with metrics and dashboards
"""
import json
import os
from datetime import datetime, timezone
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template
import glob

class ReportGenerator:
    """Generate comprehensive test reports"""

    def __init__(self, results_dir="test_results"):
        self.results_dir = results_dir
        self.reports = {}
        os.makedirs(results_dir, exist_ok=True)

    def load_test_results(self):
        """Load all test result files"""
        result_files = [
            'performance_benchmark_results.json',
            'chaos_test_results.json',
            'performance_monitoring_report.json',
            'api_contract_test_results.json'
        ]

        for filename in result_files:
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        self.reports[filename.replace('.json', '')] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")

        # Also check for results in current directory
        for filename in result_files:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        self.reports[filename.replace('.json', '')] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")

    def generate_summary_report(self):
        """Generate overall test summary"""
        summary = {
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'test_categories': {},
            'overall_metrics': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'success_rate': 0.0
            }
        }

        # Process performance benchmarks
        if 'performance_benchmark_results' in self.reports:
            perf_data = self.reports['performance_benchmark_results']
            summary['test_categories']['performance_benchmarking'] = {
                'status': 'completed',
                'metrics': perf_data.get('summary', {})
            }

        # Process chaos engineering results
        if 'chaos_test_results' in self.reports:
            chaos_results = self.reports['chaos_test_results']
            passed_chaos = sum(1 for r in chaos_results if r.get('success', False))
            total_chaos = len(chaos_results)

            summary['test_categories']['chaos_engineering'] = {
                'status': 'completed',
                'total_tests': total_chaos,
                'passed_tests': passed_chaos,
                'failed_tests': total_chaos - passed_chaos,
                'success_rate': (passed_chaos / total_chaos * 100) if total_chaos > 0 else 0
            }

        # Process performance monitoring
        if 'performance_monitoring_report' in self.reports:
            perf_report = self.reports['performance_monitoring_report']
            overall = perf_report.get('overall_summary', {})

            summary['test_categories']['performance_monitoring'] = {
                'status': 'completed',
                'total_tests': overall.get('total_tests', 0),
                'passed_tests': overall.get('passed_tests', 0),
                'failed_tests': overall.get('failed_tests', 0),
                'execution_time': overall.get('total_execution_time', 0),
                'insights': perf_report.get('performance_insights', {})
            }

        # Process API contract tests
        if 'api_contract_test_results' in self.reports:
            contract_results = self.reports['api_contract_test_results']
            passed_contract = sum(1 for r in contract_results if r.get('success', False))
            total_contract = len(contract_results)

            summary['test_categories']['api_contract_testing'] = {
                'status': 'completed',
                'total_tests': total_contract,
                'passed_tests': passed_contract,
                'failed_tests': total_contract - passed_contract,
                'success_rate': (passed_contract / total_contract * 100) if total_contract > 0 else 0
            }

        # Calculate overall metrics
        total_tests = sum(cat.get('total_tests', 0) for cat in summary['test_categories'].values()
                        if 'total_tests' in cat)
        passed_tests = sum(cat.get('passed_tests', 0) for cat in summary['test_categories'].values()
                            if 'passed_tests' in cat)

        summary['overall_metrics']['total_tests'] = total_tests
        summary['overall_metrics']['passed_tests'] = passed_tests
        summary['overall_metrics']['failed_tests'] = total_tests - passed_tests
        summary['overall_metrics']['success_rate'] = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        return summary

    def generate_html_report(self, summary):
        """Generate HTML test report"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JPMorgan Financial APIs - Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .summary { background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .category { background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 3px; }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .error { color: #e74c3c; }
        .chart { margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #2c3e50; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 JPMorgan Financial APIs - Comprehensive Test Report</h1>
        <p>Generated: {{ summary.report_generated }}</p>
    </div>

    <div class="summary">
        <h2>📊 Overall Summary</h2>
        <div class="metric success">Total Tests: {{ summary.overall_metrics.total_tests }}</div>
        <div class="metric success">Passed: {{ summary.overall_metrics.passed_tests }}</div>
        <div class="metric {% if summary.overall_metrics.failed_tests > 0 %}error{% else %}success{% endif %}">
            Failed: {{ summary.overall_metrics.failed_tests }}
        </div>
        <div class="metric {% if summary.overall_metrics.success_rate >= 90 %}success{% elif summary.overall_metrics.success_rate >= 75 %}warning{% else %}error{% endif %}">
            Success Rate: {{ "%.1f"|format(summary.overall_metrics.success_rate) }}%
        </div>
    </div>

    <h2>📈 Test Categories</h2>
    {% for category_name, category_data in summary.test_categories.items() %}
    <div class="category">
        <h3>{{ category_name.replace('_', ' ').title() }}</h3>
        <p>Status: <strong class="{% if category_data.status == 'completed' %}success{% else %}warning{% endif %}">{{ category_data.status.title() }}</strong></p>

        {% if category_data.total_tests is defined %}
        <div class="metric">Tests: {{ category_data.total_tests }}</div>
        <div class="metric success">Passed: {{ category_data.passed_tests }}</div>
        <div class="metric {% if category_data.failed_tests > 0 %}error{% else %}success{% endif %}">Failed: {{ category_data.failed_tests }}</div>
        {% if category_data.success_rate is defined %}
        <div class="metric {% if category_data.success_rate >= 90 %}success{% elif category_data.success_rate >= 75 %}warning{% else %}error{% endif %}">
            Success Rate: {{ "%.1f"|format(category_data.success_rate) }}%
        </div>
        {% endif %}
        {% endif %}

        {% if category_data.execution_time is defined %}
        <div class="metric">Total Execution Time: {{ "%.2f"|format(category_data.execution_time) }}s</div>
        {% endif %}

        {% if category_data.insights %}
        <h4>Performance Insights:</h4>
        <ul>
        {% for key, value in category_data.insights.items() %}
            <li><strong>{{ key.replace('_', ' ').title() }}:</strong> {{ value }}</li>
        {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endfor %}

    <div class="summary">
        <h2>🎯 Recommendations</h2>
        {% if summary.overall_metrics.success_rate >= 95 %}
        <p class="success">✅ Excellent! All systems are performing well. Ready for production deployment.</p>
        {% elif summary.overall_metrics.success_rate >= 85 %}
        <p class="warning">⚠️ Good performance, but some issues need attention before production.</p>
        {% else %}
        <p class="error">❌ Critical issues detected. Thorough review required before deployment.</p>
        {% endif %}
    </div>
</body>
</html>
        """

        template = Template(html_template)
        html_content = template.render(summary=summary)

        report_path = os.path.join(self.results_dir, 'comprehensive_test_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 HTML report generated: {report_path}")
        return report_path

    def generate_performance_charts(self):
        """Generate performance visualization charts"""
        if 'performance_monitoring_report' not in self.reports:
            return

        perf_data = self.reports['performance_monitoring_report']

        # Create CPU usage chart
        if perf_data.get('test_results'):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Performance Monitoring Results')

            # CPU Usage
            ax1.set_title('CPU Usage Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('CPU %')

            # Memory Usage
            ax2.set_title('Memory Usage Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Memory %')

            # Test Execution Times
            test_names = [r['test_name'] for r in perf_data['test_results']]
            execution_times = [r['execution_time'] for r in perf_data['test_results']]

            ax3.bar(range(len(test_names)), execution_times)
            ax3.set_title('Test Execution Times')
            ax3.set_xlabel('Test')
            ax3.set_ylabel('Time (s)')
            ax3.set_xticks(range(len(test_names)))
            ax3.set_xticklabels(test_names, rotation=45, ha='right')

            # Success/Failure Chart
            success_count = sum(1 for r in perf_data['test_results'] if r['success'])
            failure_count = len(perf_data['test_results']) - success_count

            ax4.pie([success_count, failure_count], labels=['Passed', 'Failed'],
                    autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'])
            ax4.set_title('Test Results Distribution')

            plt.tight_layout()
            chart_path = os.path.join(self.results_dir, 'performance_charts.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 Performance charts generated: {chart_path}")
            return chart_path

    def generate_junit_xml(self, summary):
        """Generate JUnit XML report for CI/CD integration"""
        xml_template = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="JPMorgan Financial APIs Test Suite"
                tests="{{ summary.overall_metrics.total_tests }}"
                failures="{{ summary.overall_metrics.failed_tests }}"
                time="0">
        {% for category_name, category_data in summary.test_categories.items() %}
        <testcase name="{{ category_name }}"
                    time="{% if category_data.execution_time %}{{ category_data.execution_time }}{% else %}0{% endif %}">
            {% if category_data.failed_tests > 0 %}
            <failure message="{{ category_data.failed_tests }} tests failed">
                Failed tests in {{ category_name }}: {{ category_data.failed_tests }}/{{ category_data.total_tests }}
            </failure>
            {% endif %}
        </testcase>
        {% endfor %}
    </testsuite>
</testsuites>"""

        template = Template(xml_template)
        xml_content = template.render(summary=summary)

        xml_path = os.path.join(self.results_dir, 'junit_report.xml')
        with open(xml_path, 'w') as f:
            f.write(xml_content)

        print(f"🔧 JUnit XML report generated: {xml_path}")
        return xml_path

    def generate_all_reports(self):
        """Generate all types of reports"""
        print("📋 Generating Comprehensive Test Reports")
        print("=" * 50)

        # Load test results
        self.load_test_results()

        # Generate summary
        summary = self.generate_summary_report()

        # Generate reports
        html_report = self.generate_html_report(summary)
        charts = self.generate_performance_charts()
        junit_xml = self.generate_junit_xml(summary)

        # Save summary JSON
        summary_path = os.path.join(self.results_dir, 'test_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 50)
        print("📋 Report Generation Complete!")
        print("=" * 50)
        print(f"📄 HTML Report: {html_report}")
        if charts:
            print(f"📊 Charts: {charts}")
        print(f"🔧 JUnit XML: {junit_xml}")
        print(f"📋 Summary JSON: {summary_path}")

        # Print summary
        overall = summary['overall_metrics']
        print("\n📊 Overall Results:")
        print(f"  Total Tests: {overall['total_tests']}")
        print(f"  Passed: {overall['passed_tests']}")
        print(f"  Failed: {overall['failed_tests']}")
        print(".1f")
        print(f"  Success Rate: {overall['success_rate']:.1f}%")

        return summary

def run_test_reporting():
    """Main function to generate test reports"""
    reporter = TestReportGenerator()
    summary = reporter.generate_all_reports()

    success = summary['overall_metrics']['success_rate'] >= 85  # 85% success threshold
    return success

if __name__ == "__main__":
    success = run_test_reporting()
    exit(0 if success else 1)
