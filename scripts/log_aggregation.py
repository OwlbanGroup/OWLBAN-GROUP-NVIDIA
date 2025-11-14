#!/usr/bin/env python3
"""
Log Aggregation Setup and Analysis Script for JPMorgan Financial APIs

This script sets up centralized logging configuration and provides
analysis tools for log data aggregation and monitoring.
"""

import os
import sys
import json
import logging
import logging.config
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import re

class LogAggregator:
    def __init__(self, log_dir="logs", config_file="logging_config.json"):
        self.log_dir = Path(log_dir)
        self.config_file = Path(config_file)
        self.log_dir.mkdir(exist_ok=True)

    def setup_logging_config(self):
        """Set up centralized logging configuration"""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "json": {
                    "format": json.dumps({
                        "timestamp": "%(asctime)s",
                        "level": "%(levelname)s",
                        "logger": "%(name)s",
                        "message": "%(message)s"
                    })
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "detailed"
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "filename": str(self.log_dir / "app.log")
                },
                "error_file": {
                    "class": "logging.FileHandler",
                    "level": "ERROR",
                    "formatter": "json",
                    "filename": str(self.log_dir / "error.log")
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console", "file", "error_file"]
            },
            "loggers": {
                "jpmorgan_financial_apis": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            }
        }

        # Save config to file
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Apply configuration
        logging.config.dictConfig(config)
        logger = logging.getLogger("jpmorgan_financial_apis")
        logger.info("Log aggregation configuration applied successfully")

        return config

    def analyze_logs(self, hours=24):
        """Analyze recent log files for patterns and issues"""
        analysis = {
            "period": f"Last {hours} hours",
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": [],
            "summary": {
                "total_lines": 0,
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "debug_count": 0
            },
            "top_errors": [],
            "recent_errors": []
        }

        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Analyze log files
        log_files = list(self.log_dir.glob("*.log"))
        for log_file in log_files:
            if log_file.exists():
                analysis["files_analyzed"].append(str(log_file.name))
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            analysis["summary"]["total_lines"] += 1

                            # Extract timestamp and level
                            match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .* - (ERROR|WARNING|INFO|DEBUG)', line)
                            if match:
                                timestamp_str, level = match.groups()
                                try:
                                    line_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                                    if line_time >= cutoff_time:
                                        if level == "ERROR":
                                            analysis["summary"]["error_count"] += 1
                                            analysis["recent_errors"].append({
                                                "timestamp": timestamp_str,
                                                "message": line.strip()
                                            })
                                        elif level == "WARNING":
                                            analysis["summary"]["warning_count"] += 1
                                        elif level == "INFO":
                                            analysis["summary"]["info_count"] += 1
                                        elif level == "DEBUG":
                                            analysis["summary"]["debug_count"] += 1
                                except ValueError:
                                    continue
                except Exception as e:
                    print(f"Error reading {log_file}: {e}")

        # Get top errors (simplified - just count occurrences)
        error_messages = {}
        for error in analysis["recent_errors"]:
            msg = error["message"]
            error_messages[msg] = error_messages.get(msg, 0) + 1

        analysis["top_errors"] = sorted(
            [{"message": msg, "count": count} for msg, count in error_messages.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]

        return analysis

    def generate_report(self, analysis):
        """Generate a human-readable analysis report"""
        report = f"""
Log Analysis Report
===================
Period: {analysis['period']}
Generated: {analysis['timestamp']}
Files Analyzed: {', '.join(analysis['files_analyzed'])}

Summary Statistics:
- Total Lines Processed: {analysis['summary']['total_lines']}
- Errors: {analysis['summary']['error_count']}
- Warnings: {analysis['summary']['warning_count']}
- Info Messages: {analysis['summary']['info_count']}
- Debug Messages: {analysis['summary']['debug_count']}

Top Error Patterns:
"""
        for i, error in enumerate(analysis['top_errors'][:5], 1):
            report += f"{i}. {error['message']} (occurrences: {error['count']})\n"

        if analysis['recent_errors']:
            report += "\nRecent Errors:\n"
            for error in analysis['recent_errors'][-5:]:
                report += f"- {error['timestamp']}: {error['message']}\n"

        return report

def main():
    parser = argparse.ArgumentParser(description='Log Aggregation Setup and Analysis')
    parser.add_argument('--setup', action='store_true', help='Set up logging configuration')
    parser.add_argument('--analyze', action='store_true', help='Analyze recent logs')
    parser.add_argument('--hours', type=int, default=24, help='Hours to analyze (default: 24)')
    parser.add_argument('--log-dir', default='logs', help='Log directory path')
    parser.add_argument('--output', choices=['text', 'json'], default='text', help='Output format')

    args = parser.parse_args()

    aggregator = LogAggregator(args.log_dir)

    if args.setup:
        print("Setting up log aggregation configuration...")
        config = aggregator.setup_logging_config()
        print(f"Configuration saved to {aggregator.config_file}")
        if args.output == 'json':
            print(json.dumps(config, indent=2))

    if args.analyze:
        print(f"Analyzing logs from the last {args.hours} hours...")
        analysis = aggregator.analyze_logs(args.hours)
        if args.output == 'json':
            print(json.dumps(analysis, indent=2))
        else:
            report = aggregator.generate_report(analysis)
            print(report)

    if not args.setup and not args.analyze:
        parser.print_help()

if __name__ == '__main__':
    main()
