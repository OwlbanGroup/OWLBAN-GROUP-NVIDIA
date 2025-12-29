import time
import random
import threading
from datetime import datetime, timezone
from src.logger import telemetry_logger
from src.revenue_service import revenue_service
from src.models.revenue import RevenueType, TransactionStatus
from src.payments_service import payments_service
from src.models.payments import PaymentStatus, PaymentType
from prometheus_client import Counter, Gauge, Histogram

# Metrics for the processor
transactions_processed = Counter(
    'jpmorgan_transactions_processed_total',
    'Total number of JP Morgan transactions processed',
    ['type', 'status']
)

ops_jobs_processed = Counter(
    'jpmorgan_ops_jobs_processed_total',
    'Total number of JP Morgan ops jobs processed',
    ['job_type', 'status']
)

processing_latency = Histogram(
    'jpmorgan_processing_latency_seconds',
    'Processing latency for JP Morgan operations',
    ['operation_type']
)

active_jobs = Gauge(
    'jpmorgan_active_jobs',
    'Number of active JP Morgan processing jobs'
)

class JPMorganProcessor:
    def __init__(self):
        self.logger = telemetry_logger.get_logger()
        self.running = False
        self.thread = None

    def start_processing(self):
        """Start the background processing thread"""
        if self.running:
            self.logger.warning("Processor already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        self.logger.info("JP Morgan processor started")

    def stop_processing(self):
        """Stop the background processing"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info("JP Morgan processor stopped")

    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Process transactions
                self._process_transactions()

                # Process ops jobs
                self._process_ops_jobs()

                # Sleep between processing cycles
                time.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(5)  # Back off on errors

    def _process_transactions(self):
        """Process JP Morgan transactions"""
        start_time = time.time()

        try:
            # Simulate processing different types of transactions
            transaction_types = ['card', 'ach', 'wire', 'internal']

            for tx_type in transaction_types:
                # Simulate transaction processing
                success = random.random() > 0.05  # 95% success rate

                if success:
                    # Create a revenue transaction
                    transaction = revenue_service.create_transaction(
                        user_id=f"user_{random.randint(1, 100)}",
                        revenue_type=RevenueType.PAYMENT,
                        amount=random.uniform(10, 10000),
                        currency='USD',
                        description=f'JP Morgan {tx_type} transaction',
                        merchant_name='JP Morgan',
                        payment_method=tx_type,
                        metadata={'source': 'JP Morgan', 'type': tx_type}
                    )

                    # Process the transaction
                    revenue_service.process_transaction(transaction.id, True)

                    transactions_processed.labels(type=tx_type, status='success').inc()
                else:
                    transactions_processed.labels(type=tx_type, status='failure').inc()

                # Simulate processing time
                time.sleep(random.uniform(0.01, 0.1))

            processing_latency.labels(operation_type='transactions').observe(time.time() - start_time)

        except Exception as e:
            self.logger.error(f"Error processing transactions: {e}")
            transactions_processed.labels(type='unknown', status='error').inc()

    def _process_ops_jobs(self):
        """Process JP Morgan operational jobs"""
        start_time = time.time()

        try:
            # Simulate different types of ops jobs
            job_types = ['settlement', 'reconciliation', 'fraud_detection', 'reporting']

            active_jobs.inc()

            for job_type in job_types:
                # Simulate job processing
                success = random.random() > 0.1  # 90% success rate

                if success:
                    ops_jobs_processed.labels(job_type=job_type, status='success').inc()

                    # Simulate job-specific operations
                    if job_type == 'settlement':
                        # Process pending payments
                        pending_payments = payments_service.get_payments_by_status(PaymentStatus.PENDING)
                        for payment in pending_payments[:5]:  # Process up to 5
                            payments_service.update_payment_status(payment.id, PaymentStatus.COMPLETED)

                    elif job_type == 'reconciliation':
                        # Simulate reconciliation checks
                        pass

                    elif job_type == 'fraud_detection':
                        # Simulate fraud checks
                        pass

                    elif job_type == 'reporting':
                        # Generate reports
                        pass

                else:
                    ops_jobs_processed.labels(job_type=job_type, status='failure').inc()

                # Simulate job processing time
                time.sleep(random.uniform(0.1, 0.5))

            active_jobs.dec()
            processing_latency.labels(operation_type='ops_jobs').observe(time.time() - start_time)

        except Exception as e:
            self.logger.error(f"Error processing ops jobs: {e}")
            active_jobs.dec()
            ops_jobs_processed.labels(job_type='unknown', status='error').inc()

    def get_status(self):
        """Get current processor status"""
        return {
            'running': self.running,
            'thread_alive': self.thread.is_alive() if self.thread else False,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# Global processor instance
jpmorgan_processor = JPMorganProcessor()

def start_jpmorgan_processor():
    """Start the JP Morgan processor"""
    jpmorgan_processor.start_processing()

def stop_jpmorgan_processor():
    """Stop the JP Morgan processor"""
    jpmorgan_processor.stop_processing()

def get_processor_status():
    """Get processor status"""
    return jpmorgan_processor.get_status()
