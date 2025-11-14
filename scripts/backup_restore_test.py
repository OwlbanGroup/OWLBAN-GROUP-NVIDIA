#!/usr/bin/env python3
"""
Backup and Restore Test Script for JPMorgan Financial APIs

This script validates backup and restore processes for databases and files.
"""

import os
import sys
import json
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path
import argparse
import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackupRestoreTester:
    """Backup and Restore Tester class for JPMorgan Financial APIs."""
    def __init__(self, db_url=None, backup_dir=None):
        self.db_url = db_url or os.getenv(
            'DATABASE_URL',
            'postgresql://user:pass@localhost:5432/jpmorgan_financial_apis'
        )
        self.backup_dir = Path(backup_dir or os.getenv('BACKUP_DIR', './backups'))
        self.backup_dir.mkdir(exist_ok=True)
        self.results = []

    def log_result(self, test_name, status, message, details=None):
        """Log a test result"""
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.results.append(result)
        logger.info("%s: %s - %s", test_name, status.upper(), message)

    def test_database_backup(self):
        """Test database backup functionality"""
        try:
            # Connect to database
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()

            # Get table count before backup
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
            )
            _ = cursor.fetchone()[0]

            # Create backup
            backup_file = self.backup_dir / f"test_backup_{int(time.time())}.sql"
            os.system(f"pg_dump {self.db_url} > {backup_file}")

            if backup_file.exists() and backup_file.stat().st_size > 0:
                self.log_result(
                    'database_backup', 'pass',
                    f'Backup created successfully: {backup_file}'
                )
            else:
                self.log_result('database_backup', 'fail', 'Backup file not created or empty')

            conn.close()
        except (psycopg2.Error, OSError) as e:
            self.log_result(
                'database_backup', 'fail',
                f'Database backup failed: {str(e)}'
            )

    def test_database_restore(self):
        """Test database restore functionality"""
        try:
            # Create test database
            _ = self.db_url.replace('jpmorgan_financial_apis', 'test_restore_db')

            # Create test data
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS test_restore (id SERIAL PRIMARY KEY, data TEXT)"
            )
            cursor.execute("INSERT INTO test_restore (data) VALUES ('test_data')")
            conn.commit()

            # Backup test data
            backup_file = self.backup_dir / f"restore_test_{int(time.time())}.sql"
            os.system(f"pg_dump {self.db_url} --table=test_restore > {backup_file}")

            # Drop and recreate table
            cursor.execute("DROP TABLE test_restore")
            conn.commit()
            conn.close()

            # Restore from backup
            os.system(f"psql {self.db_url} < {backup_file}")

            # Verify restoration
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test_restore WHERE data = 'test_data'")
            count = cursor.fetchone()[0]

            if count > 0:
                self.log_result(
                    'database_restore', 'pass',
                    'Database restore successful'
                )
            else:
                self.log_result(
                    'database_restore', 'fail',
                    'Database restore failed - data not found'
                )

            # Cleanup
            cursor.execute("DROP TABLE test_restore")
            conn.commit()
            conn.close()
            backup_file.unlink(missing_ok=True)

        except (psycopg2.Error, OSError) as e:
            self.log_result(
                'database_restore', 'fail',
                f'Database restore test failed: {str(e)}'
            )

    def test_file_backup(self):
        """Test file system backup"""
        try:
            test_dir = Path('./test_backup_source')
            test_dir.mkdir(exist_ok=True)

            # Create test files
            (test_dir / 'test1.txt').write_text('test content 1')
            (test_dir / 'test2.txt').write_text('test content 2')
            (test_dir / 'subdir').mkdir(exist_ok=True)
            (test_dir / 'subdir' / 'test3.txt').write_text('test content 3')

            # Create backup
            backup_file = self.backup_dir / f"file_backup_{int(time.time())}.tar.gz"
            os.system(f"tar -czf {backup_file} {test_dir}")

            if backup_file.exists() and backup_file.stat().st_size > 0:
                self.log_result(
                    'file_backup', 'pass',
                    f'File backup created: {backup_file}'
                )
            else:
                self.log_result('file_backup', 'fail', 'File backup not created')

            # Cleanup
            shutil.rmtree(test_dir)
            backup_file.unlink(missing_ok=True)

        except OSError as e:
            self.log_result(
                'file_backup', 'fail',
                f'File backup test failed: {str(e)}'
            )

    def test_file_restore(self):
        """Test file system restore"""
        try:
            # Create backup first
            test_dir = Path('./test_restore_source')
            test_dir.mkdir(exist_ok=True)
            (test_dir / 'restore_test.txt').write_text('restore test content')

            backup_file = self.backup_dir / f"restore_file_test_{int(time.time())}.tar.gz"
            os.system(f"tar -czf {backup_file} {test_dir}")

            # Remove original
            shutil.rmtree(test_dir)

            # Restore
            restore_dir = Path('./test_restore_target')
            restore_dir.mkdir(exist_ok=True)
            os.system(f"tar -xzf {backup_file} -C {restore_dir}")

            restored_file = restore_dir / 'test_restore_source' / 'restore_test.txt'
            if restored_file.exists() and restored_file.read_text() == 'restore test content':
                self.log_result(
                    'file_restore', 'pass',
                    'File restore successful'
                )
            else:
                self.log_result(
                    'file_restore', 'fail',
                    'File restore failed'
                )

            # Cleanup
            shutil.rmtree(restore_dir)
            backup_file.unlink(missing_ok=True)

        except OSError as e:
            self.log_result(
                'file_restore', 'fail',
                f'File restore test failed: {str(e)}'
            )

    def test_backup_integrity(self):
        """Test backup integrity"""
        try:
            # Create a backup
            test_file = self.backup_dir / 'integrity_test.txt'
            test_file.write_text('integrity test data')

            backup_file = self.backup_dir / f"integrity_backup_{int(time.time())}.tar.gz"
            os.system(f"tar -czf {backup_file} {test_file}")

            # Test integrity
            result = os.system(f"tar -tzf {backup_file} > /dev/null 2>&1")
            if result == 0:
                self.log_result(
                    'backup_integrity', 'pass',
                    'Backup integrity check passed'
                )
            else:
                self.log_result(
                    'backup_integrity', 'fail',
                    'Backup integrity check failed'
                )

            # Cleanup
            test_file.unlink(missing_ok=True)
            backup_file.unlink(missing_ok=True)

        except OSError as e:
            self.log_result(
                'backup_integrity', 'fail',
                f'Backup integrity test failed: {str(e)}'
            )

    def run_all_tests(self):
        """Run all backup and restore tests"""
        logger.info("Starting backup and restore tests...")

        tests = [
            self.test_database_backup,
            self.test_database_restore,
            self.test_file_backup,
            self.test_file_restore,
            self.test_backup_integrity
        ]

        for test in tests:
            test()

        return self.results

    def generate_report(self):
        """Generate test report"""
        total_tests = len(self.results)
        passed = len([r for r in self.results if r['status'] == 'pass'])
        failed = len([r for r in self.results if r['status'] == 'fail'])

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'success_rate': (passed / total_tests * 100) if total_tests > 0 else 0
            },
            'results': self.results
        }

        return report

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Backup and Restore Test Script')
    parser.add_argument('--db-url', help='Database connection URL')
    parser.add_argument('--backup-dir', help='Backup directory')
    parser.add_argument('--output', choices=['json', 'text'], default='text', help='Output format')

    args = parser.parse_args()

    tester = BackupRestoreTester(args.db_url, args.backup_dir)
    tester.run_all_tests()

    if args.output == 'json':
        report = tester.generate_report()
        print(json.dumps(report, indent=2))
    else:
        report = tester.generate_report()
        print("Backup and Restore Test Report")
        print("=" * 40)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print("\nDetailed Results:")
        for result in report['results']:
            status_icon = {'pass': '✓', 'fail': '✗'}[result['status']]
            print(f"{status_icon} {result['test']}: {result['message']}")

        # Exit with appropriate code
        if report['summary']['failed'] > 0:
            sys.exit(1)

if __name__ == '__main__':
    main()
