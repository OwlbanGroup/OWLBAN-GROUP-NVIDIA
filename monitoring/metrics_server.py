# metrics_server.py
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import random
import time

# Transaction counters
transactions_total = Counter(
    "owlban_transactions_total",
    "Total number of Owlban transactions processed",
    ["env", "status"]  # labels: env=prod/stage, status=success/fail
)

# In-flight ops
inflight_jobs = Gauge(
    "owlban_inflight_jobs",
    "Number of Owlban operational jobs currently running",
    ["env", "job_type"]
)

# Latency
transaction_latency_seconds = Histogram(
    "owlban_transaction_latency_seconds",
    "Latency of Owlban transactions in seconds",
    ["env", "transaction_type"]
)

ENV = "prod"

def process_transaction(transaction_type: str, success: bool, latency: float):
    status = "success" if success else "failure"
    transactions_total.labels(env=ENV, status=status).inc()
    transaction_latency_seconds.labels(env=ENV, transaction_type=transaction_type).observe(latency)

def simulate_ops_loop():
    while True:
        # Simulated job mix
        job_type = random.choice(["settlement", "recon", "fraud_check"])
        inflight_jobs.labels(env=ENV, job_type=job_type).inc()

        # Simulated transaction
        ttype = random.choice(["card", "ach", "internal"])
        success = random.random() > 0.05
        latency = random.uniform(0.05, 1.2)
        process_transaction(ttype, success, latency)

        time.sleep(0.5)
        inflight_jobs.labels(env=ENV, job_type=job_type).dec()

if __name__ == "__main__":
    # Expose metrics at :8000/metrics
    start_http_server(8000)
    simulate_ops_loop()
