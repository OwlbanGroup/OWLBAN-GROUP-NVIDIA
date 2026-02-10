"""
JPMorgan Data Synchronization Scheduler
Component 4: Cron Job Scripts (Python Schedule Module)

This module implements automated synchronization jobs for JPMorgan financial data
using the Python schedule library. It handles transactions, balances, and accounts sync.
"""

import os
import time
import logging
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our JPMorgan connector and database models
from jpmorgan_connector import JPMorganConnector, create_jpmorgan_connector
import psycopg2
from psycopg2.extras import execute_values
import json

# Import enrichment and AI services for post-sync processing
try:
    from apollo_connector import ApolloConnector, create_apollo_connector
    APOLLO_AVAILABLE = True
except ImportError:
    APOLLO_AVAILABLE = False
    print("Warning: Apollo connector not available for enrichment")

try:
    from src.ai_service import ai_service
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: AI service not available for analysis")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sync_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SyncJob:
    """Represents a single synchronization job"""

    def __init__(self, job_id: str, job_type: str, interval_minutes: int,
                 job_function: Callable, description: str = ""):
        self.job_id = job_id
        self.job_type = job_type
        self.interval_minutes = interval_minutes
        self.job_function = job_function
        self.description = description
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.is_running = False
        self.success_count = 0
        self.failure_count = 0
        self.last_error: Optional[str] = None

    def run(self):
        """Execute the job"""
        if self.is_running:
            logger.warning(f"Job {self.job_id} is already running, skipping")
            return

        self.is_running = True
        self.last_run = datetime.now()

        try:
            logger.info(f"Starting job: {self.job_id} - {self.description}")
            result = self.job_function()
            self.success_count += 1
            logger.info(f"Job {self.job_id} completed successfully")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_error = str(e)
            logger.error(f"Job {self.job_id} failed: {e}")
            raise
        finally:
            self.is_running = False
            self.next_run = datetime.now() + timedelta(minutes=self.interval_minutes)

class DatabaseManager:
    """Database operations for sync jobs"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.connection_string)

    def log_sync_start(self, sync_type: str, metadata: dict = None) -> str:
        """Log the start of a sync operation"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO sync_logs (sync_type, status, sync_metadata)
                    VALUES (%s, 'started', %s)
                    RETURNING id
                """, (sync_type, json.dumps(metadata or {})))
                sync_id = cursor.fetchone()[0]
                conn.commit()
                return str(sync_id)

    def log_sync_complete(self, sync_id: str, records_processed: int,
                         records_failed: int, error_message: str = None):
        """Log the completion of a sync operation"""
        status = 'failed' if error_message else 'completed'
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE sync_logs
                    SET status = %s, records_processed = %s, records_failed = %s,
                        error_message = %s, completed_at = CURRENT_TIMESTAMP,
                        duration_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at))
                    WHERE id = %s
                """, (status, records_processed, records_failed, error_message, sync_id))
                conn.commit()

    def upsert_accounts(self, accounts_data: list) -> tuple[int, int]:
        """Upsert JPMorgan accounts data"""
        processed = 0
        failed = 0

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                for account in accounts_data:
                    try:
                        cursor.execute("""
                            INSERT INTO jpmorgan_accounts (
                                jpmorgan_id, name, type, currency_code, status,
                                account_metadata, last_sync
                            ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                            ON CONFLICT (jpmorgan_id) DO UPDATE SET
                                name = EXCLUDED.name,
                                type = EXCLUDED.type,
                                currency_code = EXCLUDED.currency_code,
                                status = EXCLUDED.status,
                                account_metadata = EXCLUDED.account_metadata,
                                last_sync = CURRENT_TIMESTAMP,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            account['id'], account['name'], account['type'],
                            account.get('currency', 'USD'), account.get('status', 'active'),
                            json.dumps(account.get('metadata', {}))
                        ))
                        processed += 1
                    except Exception as e:
                        logger.error(f"Failed to upsert account {account.get('id')}: {e}")
                        failed += 1

                conn.commit()
        return processed, failed

    def upsert_balances(self, balances_data: list) -> tuple[int, int]:
        """Upsert JPMorgan balances data"""
        processed = 0
        failed = 0

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                for balance in balances_data:
                    try:
                        cursor.execute("""
                            INSERT INTO jpmorgan_balances (
                                account_id, available_balance, ledger_balance,
                                currency_code, balance_date, balance_metadata
                            )
                            SELECT
                                a.id, %s, %s, %s, %s, %s
                            FROM jpmorgan_accounts a
                            WHERE a.jpmorgan_id = %s
                            ON CONFLICT (account_id, balance_date) DO UPDATE SET
                                available_balance = EXCLUDED.available_balance,
                                ledger_balance = EXCLUDED.ledger_balance,
                                balance_metadata = EXCLUDED.balance_metadata
                        """, (
                            balance['available'], balance['ledger'],
                            balance.get('currency', 'USD'), balance['timestamp'].date(),
                            json.dumps(balance.get('metadata', {})),
                            balance['account_id']
                        ))
                        processed += 1
                    except Exception as e:
                        logger.error(f"Failed to upsert balance for account {balance.get('account_id')}: {e}")
                        failed += 1

                conn.commit()
        return processed, failed

    def upsert_transactions(self, transactions_data: list) -> tuple[int, int]:
        """Upsert JPMorgan transactions data"""
        processed = 0
        failed = 0

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                for transaction in transactions_data:
                    try:
                        cursor.execute("""
                            INSERT INTO jpmorgan_transactions (
                                jpmorgan_id, account_id, amount, currency_code,
                                transaction_type, description, transaction_date,
                                posting_date, reference_number, status,
                                transaction_metadata, last_sync
                            )
                            SELECT
                                %s, a.id, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
                            FROM jpmorgan_accounts a
                            WHERE a.jpmorgan_id = %s
                            ON CONFLICT (jpmorgan_id) DO UPDATE SET
                                amount = EXCLUDED.amount,
                                description = EXCLUDED.description,
                                transaction_date = EXCLUDED.transaction_date,
                                posting_date = EXCLUDED.posting_date,
                                status = EXCLUDED.status,
                                transaction_metadata = EXCLUDED.transaction_metadata,
                                last_sync = CURRENT_TIMESTAMP,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            transaction['id'], transaction['amount'],
                            transaction.get('currency', 'USD'), transaction.get('type', 'unknown'),
                            transaction.get('description', ''), transaction['timestamp'],
                            transaction.get('posting_date'), transaction.get('reference'),
                            transaction.get('status', 'posted'),
                            json.dumps(transaction.get('metadata', {})),
                            transaction['account_id']
                        ))
                        processed += 1
                    except Exception as e:
                        logger.error(f"Failed to upsert transaction {transaction.get('id')}: {e}")
                        failed += 1

                conn.commit()
        return processed, failed

class JPMorganSyncScheduler:
    """
    Main scheduler for JPMorgan data synchronization jobs
    """

    def __init__(self, db_connection_string: str, max_workers: int = 3):
        self.db_manager = DatabaseManager(db_connection_string)
        self.connector = create_jpmorgan_connector()

        # Initialize enrichment and AI services if available
        self.apollo_connector = None
        if APOLLO_AVAILABLE:
            try:
                self.apollo_connector = create_apollo_connector()
                logger.info("Apollo connector initialized for enrichment")
            except Exception as e:
                logger.warning(f"Failed to initialize Apollo connector: {e}")

        self.ai_service = None
        if AI_AVAILABLE:
            try:
                self.ai_service = ai_service
                logger.info("AI service initialized for analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize AI service: {e}")

        self.jobs: Dict[str, SyncJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # Initialize sync jobs
        self._setup_jobs()

    def _setup_jobs(self):
        """Set up all synchronization jobs"""

        # Transactions sync job (every 1 minute)
        self.jobs['transactions_sync'] = SyncJob(
            job_id='transactions_sync',
            job_type='transactions',
            interval_minutes=1,
            job_function=self._sync_transactions,
            description='Sync recent transactions from JPMorgan API'
        )

        # Balances sync job (every 5 minutes)
        self.jobs['balances_sync'] = SyncJob(
            job_id='balances_sync',
            job_type='balances',
            interval_minutes=5,
            job_function=self._sync_balances,
            description='Sync account balances from JPMorgan API'
        )

        # Accounts sync job (every hour)
        self.jobs['accounts_sync'] = SyncJob(
            job_id='accounts_sync',
            job_type='accounts',
            interval_minutes=60,
            job_function=self._sync_accounts,
            description='Sync account information from JPMorgan API'
        )

        # Full sync job (daily at 2 AM)
        self.jobs['full_sync'] = SyncJob(
            job_id='full_sync',
            job_type='full',
            interval_minutes=1440,  # 24 hours
            job_function=self._full_sync,
            description='Complete synchronization of all JPMorgan data'
        )

    def _sync_accounts(self) -> dict:
        """Synchronize accounts data"""
        sync_id = self.db_manager.log_sync_start('accounts')

        try:
            # Get accounts from JPMorgan API
            accounts = self.connector.get_accounts()

            # Convert to dict format for database
            accounts_data = []
            for account in accounts:
                accounts_data.append({
                    'id': account.id,
                    'name': account.name,
                    'type': account.type,
                    'currency': account.currency,
                    'status': account.status,
                    'metadata': {
                        'available_balance': account.available_balance,
                        'ledger_balance': account.balance
                    }
                })

            # Upsert to database
            processed, failed = self.db_manager.upsert_accounts(accounts_data)

            self.db_manager.log_sync_complete(sync_id, processed, failed)

            return {
                'sync_type': 'accounts',
                'records_processed': processed,
                'records_failed': failed,
                'total_accounts': len(accounts)
            }

        except Exception as e:
            error_msg = f"Accounts sync failed: {str(e)}"
            self.db_manager.log_sync_complete(sync_id, 0, 0, error_msg)
            raise

    def _sync_balances(self) -> dict:
        """Synchronize balances data"""
        sync_id = self.db_manager.log_sync_start('balances')

        try:
            # Get balances from JPMorgan API
            balances = self.connector.get_balances()

            # Convert to dict format for database
            balances_data = []
            for balance in balances:
                balances_data.append({
                    'account_id': balance.account_id,
                    'available': balance.available,
                    'ledger': balance.ledger,
                    'currency': balance.currency,
                    'timestamp': balance.timestamp,
                    'metadata': {}
                })

            # Upsert to database
            processed, failed = self.db_manager.upsert_balances(balances_data)

            self.db_manager.log_sync_complete(sync_id, processed, failed)

            return {
                'sync_type': 'balances',
                'records_processed': processed,
                'records_failed': failed,
                'total_balances': len(balances)
            }

        except Exception as e:
            error_msg = f"Balances sync failed: {str(e)}"
            self.db_manager.log_sync_complete(sync_id, 0, 0, error_msg)
            raise

    def _sync_transactions(self) -> dict:
        """Synchronize recent transactions"""
        sync_id = self.db_manager.log_sync_start('transactions', {
            'time_range': 'last_24_hours'
        })

        try:
            # Get recent transactions (last 24 hours)
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=24)

            transactions = self.connector.get_transactions(
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )

            # Convert to dict format for database
            transactions_data = []
            for tx in transactions:
                transactions_data.append({
                    'id': tx.id,
                    'account_id': tx.account_id,
                    'amount': tx.amount,
                    'currency': tx.currency,
                    'type': tx.type,
                    'description': tx.description,
                    'timestamp': tx.timestamp,
                    'status': tx.status,
                    'reference': tx.reference,
                    'metadata': {}
                })

            # Upsert to database
            processed, failed = self.db_manager.upsert_transactions(transactions_data)

            # Post-sync processing: enrichment and AI analysis
            if processed > 0:
                self._post_sync_processing(transactions_data, 'transactions')

            self.db_manager.log_sync_complete(sync_id, processed, failed)

            return {
                'sync_type': 'transactions',
                'records_processed': processed,
                'records_failed': failed,
                'total_transactions': len(transactions),
                'time_range': 'last_24_hours'
            }

        except Exception as e:
            error_msg = f"Transactions sync failed: {str(e)}"
            self.db_manager.log_sync_complete(sync_id, 0, 0, error_msg)
            raise

    def _full_sync(self) -> dict:
        """Perform complete synchronization of all data"""
        logger.info("Starting full synchronization")

        # Run all sync jobs sequentially
        results = {}

        try:
            results['accounts'] = self._sync_accounts()
            time.sleep(2)  # Brief pause between jobs

            results['balances'] = self._sync_balances()
            time.sleep(2)

            # Get transactions for the last 7 days for full sync
            sync_id = self.db_manager.log_sync_start('transactions', {
                'time_range': 'last_7_days'
            })

            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            transactions = self.connector.get_transactions(
                start_date=start_date,
                end_date=end_date,
                limit=5000
            )

            transactions_data = []
            for tx in transactions:
                transactions_data.append({
                    'id': tx.id,
                    'account_id': tx.account_id,
                    'amount': tx.amount,
                    'currency': tx.currency,
                    'type': tx.type,
                    'description': tx.description,
                    'timestamp': tx.timestamp,
                    'status': tx.status,
                    'reference': tx.reference,
                    'metadata': {}
                })

            processed, failed = self.db_manager.upsert_transactions(transactions_data)
            self.db_manager.log_sync_complete(sync_id, processed, failed)

            results['transactions'] = {
                'sync_type': 'transactions',
                'records_processed': processed,
                'records_failed': failed,
                'total_transactions': len(transactions),
                'time_range': 'last_7_days'
            }

            logger.info("Full synchronization completed successfully")
            return results

        except Exception as e:
            logger.error(f"Full synchronization failed: {e}")
            raise

    def start_scheduler(self):
        """Start the scheduler in a background thread"""
        if self.running:
            logger.warning("Scheduler is already running")
            return

        self.running = True
        logger.info("Starting JPMorgan sync scheduler")

        # Schedule jobs using python-schedule
        schedule.every(1).minutes.do(self._run_job_async, 'transactions_sync')
        schedule.every(5).minutes.do(self._run_job_async, 'balances_sync')
        schedule.every().hour.do(self._run_job_async, 'accounts_sync')
        schedule.every().day.at("02:00").do(self._run_job_async, 'full_sync')

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        logger.info("JPMorgan sync scheduler started successfully")

    def stop_scheduler(self):
        """Stop the scheduler"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping JPMorgan sync scheduler")

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)

        self.executor.shutdown(wait=True)
        logger.info("JPMorgan sync scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def _run_job_async(self, job_id: str):
        """Run a job asynchronously"""
        if job_id not in self.jobs:
            logger.error(f"Unknown job: {job_id}")
            return

        job = self.jobs[job_id]
        if job.is_running:
            logger.warning(f"Job {job_id} is already running")
            return

        # Submit job to thread pool
        future = self.executor.submit(job.run)
        future.add_done_callback(lambda f: self._handle_job_completion(job_id, f))

    def _handle_job_completion(self, job_id: str, future):
        """Handle job completion"""
        try:
            result = future.result()
            logger.info(f"Job {job_id} completed with result: {result}")
        except Exception as e:
            logger.error(f"Job {job_id} failed with error: {e}")

    def run_job_now(self, job_id: str) -> dict:
        """Run a specific job immediately"""
        if job_id not in self.jobs:
            raise ValueError(f"Unknown job: {job_id}")

        job = self.jobs[job_id]
        return job.run()

    def _post_sync_processing(self, data: list, data_type: str):
        """
        Perform post-sync processing: enrichment and AI analysis

        Args:
            data: List of synced data records
            data_type: Type of data ('transactions', 'accounts', 'balances')
        """
        try:
            logger.info(f"Starting post-sync processing for {len(data)} {data_type} records")

            # Step 1: Apollo.io enrichment (if available)
            if self.apollo_connector and data_type == 'transactions':
                logger.info("Starting Apollo.io enrichment")
                enriched_count = 0

                for record in data:
                    try:
                        # Extract contact information from transaction description
                        # This is a simplified example - in practice, you'd parse the description
                        # to extract names, emails, companies, etc.
                        description = record.get('description', '')

                        # Search for contacts based on transaction description
                        # This is a placeholder - actual implementation would depend on data structure
                        if len(description) > 5:  # Basic filter
                            search_results = self.apollo_connector.search_contacts(
                                q=description[:50],  # Limit search query length
                                page=1,
                                per_page=1
                            )

                            if search_results and len(search_results) > 0:
                                # Enrich the transaction record with contact data
                                record['enriched_contact'] = search_results[0]
                                enriched_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to enrich transaction {record.get('id')}: {e}")
                        continue

                logger.info(f"Apollo.io enrichment completed: {enriched_count}/{len(data)} records enriched")

            # Step 2: AI Analysis (if available)
            if self.ai_service and data_type == 'transactions':
                logger.info("Starting AI-powered financial analysis")

                # Prepare data for AI analysis
                financial_data = {
                    'transactions': data[:50],  # Limit for analysis
                    'summary': {
                        'total_transactions': len(data),
                        'data_type': data_type,
                        'sync_timestamp': datetime.now().isoformat()
                    }
                }

                # Perform AI analysis
                try:
                    # Analyze financial patterns
                    analysis_result = self.ai_service.analyze_financial_data(
                        financial_data,
                        question="Analyze these recent financial transactions for patterns, anomalies, and insights",
                        context="Automated post-sync analysis of JPMorgan financial data"
                    )

                    # Perform risk assessment
                    risk_result = self.ai_service.assess_transaction_risk(
                        {
                            'transactions': data[:20],  # Sample for risk assessment
                            'analysis_context': 'Post-sync automated risk assessment'
                        },
                        historical_patterns=[],
                        market_conditions={}
                    )

                    logger.info("AI analysis completed successfully")
                    logger.info(f"Analysis summary: {analysis_result.get('summary', 'N/A')}")
                    logger.info(f"Risk assessment: {risk_result.get('risk_level', 'N/A')}")

                except Exception as e:
                    logger.error(f"AI analysis failed: {e}")

            logger.info("Post-sync processing completed")

        except Exception as e:
            logger.error(f"Post-sync processing failed: {e}")

    def get_job_status(self) -> dict:
        """Get status of all jobs"""
        return {
            job_id: {
                'job_type': job.job_type,
                'description': job.description,
                'last_run': job.last_run.isoformat() if job.last_run else None,
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'is_running': job.is_running,
                'success_count': job.success_count,
                'failure_count': job.failure_count,
                'last_error': job.last_error
            }
            for job_id, job in self.jobs.items()
        }

def create_scheduler() -> JPMorganSyncScheduler:
    """
    Factory function to create scheduler with environment-based configuration

    Returns:
        Configured JPMorganSyncScheduler instance
    """
    # Database connection string from environment
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_port = os.environ.get('DB_PORT', '5432')
    db_name = os.environ.get('DB_NAME', 'jpmorgan_financial')
    db_user = os.environ.get('DB_USER', 'jpmorgan_user')
    db_password = os.environ.get('DB_PASSWORD', '')

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    return JPMorganSyncScheduler(connection_string)

# Example usage and CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='JPMorgan Data Synchronization Scheduler')
    parser.add_argument('--start', action='store_true', help='Start the scheduler')
    parser.add_argument('--stop', action='store_true', help='Stop the scheduler')
    parser.add_argument('--run-job', type=str, help='Run a specific job immediately')
    parser.add_argument('--status', action='store_true', help='Show job status')

    args = parser.parse_args()

    scheduler = create_scheduler()

    if args.start:
        scheduler.start_scheduler()
        print("Scheduler started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop_scheduler()
            print("Scheduler stopped.")

    elif args.run_job:
        try:
            result = scheduler.run_job_now(args.run_job)
            print(f"Job {args.run_job} completed: {result}")
        except Exception as e:
            print(f"Job {args.run_job} failed: {e}")

    elif args.status:
        status = scheduler.get_job_status()
        print("Job Status:")
        for job_id, job_status in status.items():
            print(f"  {job_id}: {job_status}")

    else:
        parser.print_help()
