import json
import time
from locust import HttpUser, task, between
from faker import Faker
import random

fake = Faker()

class JPMorganAPITestUser(HttpUser):
    """
    Locust user class for load testing JPMorgan Financial APIs
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def generate_telemetry_data(self):
        """
        Generate realistic telemetry data using Faker
        """
        return {
            "timestamp": fake.date_time_this_year().isoformat() + "Z",
            "operation": random.choice(['CREATE', 'UPDATE', 'DELETE', 'READ', 'EXECUTE']),
            "pfn": fake.bothify(text='????????????????'),  # 16 character alphanumeric
            "version": fake.bothify(text='?.?.?'),  # Semver-like version
            "event_name": random.choice(['app_launch', 'user_action', 'system_event', 'error_occurred', 'performance_metric']),
            "shell_id": random.randint(1, 1000),
            "event_flags": random.randint(0, 255),
            "pg_name": fake.domain_name(),
            "dvc_sample": random.uniform(0, 1),
            "flags": random.randint(0, 65535),
            "edition": random.randint(1, 10),
            "epoch": str(int(time.time())),
            "seq": random.randint(1, 1000000),
            "data_type": random.randint(1, 100),
            "is_required": random.choice([True, False]),
            "data_category": random.randint(1, 50),
            "product": random.randint(1, 100),
            "priv_tags": random.randint(0, 4294967295),
            "policies": random.randint(0, 4294967295),
            "cv": fake.bothify(text='????????'),  # 8 character alphanumeric
            "boot_id": random.randint(1, 1000000),
            "os_name": random.choice(['Windows', 'macOS', 'Linux', 'iOS', 'Android']),
            "os_version": fake.bothify(text='?.?.?'),
            "exp_id": fake.bothify(text='????????????'),  # 12 character alphanumeric
            "app_id": fake.bothify(text='????????????????'),  # 16 character alphanumeric
            "app_version": fake.bothify(text='?.?.?'),
            "is_1p": random.randint(0, 1),
            "as_id": random.randint(1, 1000),
            "local_id": fake.bothify(text='????????????????????'),  # 20 character alphanumeric
            "device_class": random.choice(['desktop', 'mobile', 'tablet', 'server']),
            "dev_make": fake.company(),
            "dev_model": random.choice(['Model A', 'Model B', 'Model C', 'Professional', 'Enterprise']),
            "ticket_keys": json.dumps({
                "ticket1": fake.bothify(text='????????????????????????????????'),  # 32 character
                "ticket2": fake.bothify(text='????????????????????????????????')   # 32 character
            }),
            "user_local_id": fake.bothify(text='????????????????????????'),  # 24 character alphanumeric
            "tz": random.choice(['UTC', 'EST', 'PST', 'GMT', 'CET']),
            "pn1": fake.word(),
            "p1": fake.bothify(text='??????????'),  # 10 character alphanumeric
            "pn2": fake.word(),
            "p2": fake.bothify(text='??????????'),
            "pn3": fake.word(),
            "p3": fake.bothify(text='??????????'),
            "pn4": fake.word(),
            "p4": fake.bothify(text='??????????')
        }

    @task(7)  # 70% weight
    def submit_telemetry(self):
        """
        Submit telemetry data to the API
        """
        telemetry_data = self.generate_telemetry_data()

        with self.client.post("/api/telemetry",
                            json=telemetry_data,
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to submit telemetry: {response.status_code}")

    @task(2)  # 20% weight
    def health_check(self):
        """
        Perform health check on the API
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)  # 10% weight
    def get_telemetry_metrics(self):
        """
        Retrieve telemetry metrics from the API
        """
        with self.client.get("/api/telemetry/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get metrics: {response.status_code}")

    @task(1)  # Additional task for GPU telemetry if available
    def submit_gpu_telemetry(self):
        """
        Submit GPU-specific telemetry data
        """
        gpu_data = {
            "timestamp": fake.date_time_this_year().isoformat() + "Z",
            "gpu_name": random.choice(["NVIDIA A100", "NVIDIA V100", "NVIDIA RTX 3080", "NVIDIA RTX 3090"]),
            "gpu_utilization": random.uniform(0, 100),
            "memory_used": random.uniform(0, 81920),  # MB
            "memory_total": 81920,  # MB
            "temperature": random.uniform(30, 90),  # Celsius
            "power_usage": random.uniform(50, 300),  # Watts
            "power_limit": 300,  # Watts
            "fan_speed": random.uniform(0, 100),  # Percentage
            "clock_speed": random.uniform(1000, 2000),  # MHz
            "processes": random.randint(0, 10)
        }

        with self.client.post("/api/telemetry/gpu",
                            json=gpu_data,
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed to submit GPU telemetry: {response.status_code}")

    @task(1)  # Stress testing task
    def stress_test_large_batch(self):
        """
        Submit large batches of telemetry data for stress testing
        """
        # Generate a large batch (100-500 items)
        batch_size = random.randint(100, 500)
        large_batch = {
            "telemetry_data": []
        }

        for i in range(batch_size):
            telemetry = self.generate_telemetry_data()
            large_batch["telemetry_data"].append(telemetry)

        with self.client.post("/telemetry/batch",
                            json=large_batch,
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed stress test batch: {response.status_code}")

    @task(1)  # Resource exhaustion test
    def resource_exhaustion_test(self):
        """
        Test resource exhaustion scenarios
        """
        # Send very large payload
        large_payload = {
            "telemetry_data": [{
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.StressTest",
                "time": fake.date_time_this_year().isoformat() + "Z",
                "data": {
                    "Op": "StoreConfigurationServer::StressTestAsync",
                    "PFN": "x" * 1000,  # Large string
                    "OS": "Windows 11",
                    "DeviceModel": "Stress Test Device",
                    "UserId": f"stress_user_{random.randint(1, 1000)}",
                    "large_field": "x" * 50000  # 50KB string
                },
                "ext": {
                    "flags": 1,
                    "privacy": "public"
                }
            }]
        }

        with self.client.post("/telemetry/batch",
                            json=large_payload,
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            # Accept both success and expected failures due to size limits
            if response.status_code in [200, 413, 400]:
                response.success()
            else:
                response.failure(f"Unexpected response in resource exhaustion test: {response.status_code}")

    @task(1)  # Failure injection test
    def failure_injection_test(self):
        """
        Test system behavior under failure conditions
        """
        # Send malformed data to test error handling
        malformed_data = {
            "invalid_field": "test",
            "missing_required": None
        }

        with self.client.post("/telemetry",
                            json=malformed_data,
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            # Expect 400 Bad Request for malformed data
            if response.status_code == 400:
                response.success()
            else:
                response.failure(f"Expected 400 for malformed data, got {response.status_code}")

    @task(1)  # Mixed operations test
    def mixed_operations_test(self):
        """
        Test mixed read/write operations
        """
        # Mix of GET and POST operations
        operations = [
            ("GET", "/health"),
            ("GET", "/telemetry/metrics?hours=1"),
            ("POST", "/telemetry", self.generate_telemetry_data()),
            ("GET", "/telemetry/export?limit=5&format=json"),
        ]

        for method, endpoint, *data in operations:
            if method == "GET":
                with self.client.get(endpoint, catch_response=True) as response:
                    if response.status_code == 200:
                        response.success()
                    else:
                        response.failure(f"Failed {method} {endpoint}: {response.status_code}")
            elif method == "POST":
                with self.client.post(endpoint, json=data[0] if data else None,
                                    headers={"Content-Type": "application/json"},
                                    catch_response=True) as response:
                    if response.status_code == 200:
                        response.success()
                    else:
                        response.failure(f"Failed {method} {endpoint}: {response.status_code}")
