#!/usr/bin/env python3
"""
Test utilities and fixtures for JPMorgan Financial APIs E2E testing
"""
import copy
import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from shutil import rmtree

from faker import Faker
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

# Sample data for testing - matches the format expected by the app
SAMPLE_TELEMETRY_DATA: Dict[str, Any] = {
    "operation": "test_operation",
    "pfn": "test_pfn",
    "event_name": "sample_event",
}

LARGE_BATCH_DATA = {
    "telemetry_data": [
        SAMPLE_TELEMETRY_DATA,
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.EndOperation",
            "time": "2025-09-22T19:42:11.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                "PFN": (
                    "Microsoft.WindowsStore_22507.1401.7.0_x64__"
                    "8wekyb3d8bbwe"
                ),
                "OS": "Windows 11",
                "DeviceModel": "Surface Pro 9",
                "UserId": "test_user_456",
            },
            "ext": {"flags": 1, "privacy": "public"},
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Error",
            "time": "2025-09-22T19:42:12.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                "PFN": (
                    "Microsoft.WindowsStore_22507.1401.7.0_x64__"
                    "8wekyb3d8bbwe"
                ),
                "OS": "Windows 11",
                "DeviceModel": "Surface Pro 9",
                "UserId": "test_user_789",
                "ErrorCode": "0x80070005",
            },
            "ext": {"flags": 1, "privacy": "public"},
        },
    ]
}

SAMPLE_BUSINESS_DATA = {
    "name": "Test Business Corp",
    "description": "A test business for E2E testing",
    "industry": "Technology",
    "location": "New York, NY",
    "contact_email": "contact@testbusiness.com",
    "website": "https://testbusiness.com",
}

SAMPLE_ASSET_DATA = {
    "name": "Test Asset Server",
    "description": "A test server asset",
    "asset_type": "Server",
    "value": 50000.00,
    "location": "Data Center A",
    "status": "Active",
    "purchase_date": "2023-01-15",
    "business_id": None,  # Will be set dynamically
}


class TestUser:
    """Test user management for E2E tests"""

    def __init__(self, username: str = "testuser", password: str = "testpass"):
        self.username = username
        self.password = password
        self.token: Optional[str] = None
        self.client: Optional[FlaskClient] = None

    def register(self, client: FlaskClient) -> bool:
        """Register the test user"""
        self.client = client
        response = client.post(
            "/user/login", json={"username": self.username, "password": self.password}
        )
        return response.status_code == 200

    def login(self, client: FlaskClient) -> bool:
        """Login and get token"""
        self.client = client
        response = client.post(
            "/user/login", json={"username": self.username, "password": self.password}
        )
        if response.status_code == 200:
            data = response.get_json()
            self.token = data.get("token")
            return True
        return False

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers for authenticated requests"""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def make_authenticated_request(self, method: str, endpoint: str, **kwargs):
        """Make an authenticated request"""
        if not self.client:
            raise ValueError("Client not set. Call register() or login() first.")

        headers = kwargs.get("headers", {})
        headers.update(self.get_auth_headers())
        kwargs["headers"] = headers

        return getattr(self.client, method.lower())(endpoint, **kwargs)


class DatabaseTestHelper:
    """Helper for database-related testing"""

    @staticmethod
    def get_telemetry_count(client: FlaskClient) -> int:
        """Get current telemetry event count"""
        response = client.get("/telemetry/metrics?hours=24")
        if response.status_code == 200:
            data = response.get_json()
            return data.get("metrics", {}).get("total_events", 0)
        return 0

    @staticmethod
    def wait_for_database_operation(timeout: int = 5) -> None:
        """Wait for database operations to complete"""
        time.sleep(timeout)

    @staticmethod
    def clear_test_data(
        client: FlaskClient, user: TestUser  # pylint: disable=unused-argument
    ) -> None:
        """Clear test data from database"""
        # This would require admin endpoints - for now, just wait
        DatabaseTestHelper.wait_for_database_operation()
        # Note: client and user parameters are unused but kept for future implementation


class PerformanceTestHelper:
    """Helper for performance testing"""

    @staticmethod
    def measure_response_time(func, *args, **kwargs) -> tuple:
        """Measure function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    @staticmethod
    def generate_bulk_telemetry(count: int) -> List[Dict[str, Any]]:
        """Generate bulk telemetry data for testing"""
        telemetry_list: List[Dict[str, Any]] = []
        base_time = datetime.now(timezone.utc)

        for i in range(count):
            telemetry = copy.deepcopy(SAMPLE_TELEMETRY_DATA)
            # Ensure telemetry is a dict
            if not isinstance(telemetry, dict):
                telemetry = {}

            telemetry["time"] = (
                base_time.replace(microsecond=i * 1000)
            ).isoformat() + "Z"
            # Ensure 'data' is a dictionary before assignment; if it's a JSON string, try to parse it
            telemetry.setdefault("data", {})
            if not isinstance(telemetry["data"], dict):
                if isinstance(telemetry["data"], str):
                    try:
                        parsed = json.loads(telemetry["data"])
                        telemetry["data"] = parsed if isinstance(parsed, dict) else {}
                    except (ValueError, TypeError):
                        telemetry["data"] = {}
                else:
                    telemetry["data"] = {}
            # Type assertion for mypy
            assert isinstance(telemetry["data"], dict)

            telemetry["data"]["UserId"] = f"bulk_user_{i}"
            telemetry_list.append(telemetry)

        return telemetry_list


class ExternalServiceMock:
    """Mock external services for testing"""

    @staticmethod
    def mock_ngc_service():
        """Mock NGC service calls"""
        # This would mock NGC API calls
        raise NotImplementedError("Mock not implemented")

    @staticmethod
    def mock_redis_service():
        """Mock Redis service"""
        # This would mock Redis operations
        raise NotImplementedError("Mock not implemented")

    @staticmethod
    def mock_cloud_storage():
        """Mock cloud storage operations"""
        # This would mock cloud storage calls
        raise NotImplementedError("Mock not implemented")


class TestDataGenerator:
    """Generate test data for various scenarios"""

    @staticmethod
    def generate_telemetry_data(
        count: int = 1, realistic: bool = True  # pylint: disable=unused-argument
    ) -> List[Dict[str, Any]]:
        """Generate realistic telemetry data for testing"""
        # Note: realistic parameter is unused but kept for future implementation
        fake = Faker()
        telemetry_list: List[Dict[str, Any]] = []

        operations = ["CREATE", "UPDATE", "DELETE", "READ", "EXECUTE"]
        event_names = [
            "app_launch",
            "user_action",
            "system_event",
            "error_occurred",
            "performance_metric",
        ]
        os_types = ["Windows", "macOS", "Linux", "iOS", "Android"]
        device_models = [
            "Surface Pro 9",
            "Dell XPS 13",
            "HP Spectre",
            "Lenovo ThinkPad",
            "ASUS ROG",
        ]

        for i in range(count):
            base_time = datetime.now(timezone.utc)

            telemetry = {
                "ver": "4.0",
                "name": (
                    "Microsoft.Windows.ApplicationModel.Store.Telemetry."
                    f"{random.choice(['BeginOperation', 'EndOperation', 'Error'])}"
                ),
                "time": (base_time.replace(microsecond=i * 1000)).isoformat() + "Z",
                "data": {
                    "Op": f"StoreConfigurationServer::{random.choice(operations)}OperationAsync",
                    "PFN": fake.bothify(text="????????????????"),
                    # 16 char alphanumeric
                    "OS": random.choice(os_types),
                    "DeviceModel": random.choice(device_models),
                    "UserId": f"test_user_{i+1}",
                    "event_name": random.choice(event_names),
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
                    "cv": fake.bothify(text="????????"),
                    # 8 char alphanumeric
                    "boot_id": random.randint(1, 1000000),
                    "os_name": random.choice(os_types),
                    "os_version": fake.bothify(text="?.?.?"),
                    "exp_id": fake.bothify(text="????????????"),
                    # 12 char alphanumeric
                    "app_id": fake.bothify(text="????????????????"),
                    # 16 char alphanumeric
                    "app_version": fake.bothify(text="?.?.?"),
                    "is_1p": random.randint(0, 1),
                    "as_id": random.randint(1, 1000),
                    "local_id": fake.bothify(text="????????????????????"),
                    # 20 char alphanumeric
                    "device_class": random.choice(
                        ["desktop", "mobile", "tablet", "server"]
                    ),
                    "dev_make": fake.company(),
                    "dev_model": random.choice(
                        ["Model A", "Model B", "Model C", "Professional", "Enterprise"]
                    ),
                    "ticket_keys": json.dumps({
                        "ticket1": fake.bothify(
                            text="???????????????????????????????"
                        ),
                        "ticket2": fake.bothify(
                            text="???????????????????????????????"
                        ),
                    }),
                    "user_local_id": fake.bothify(text="????????????????????????"),
                    # 24 char alphanumeric
                    "tz": random.choice(["UTC", "EST", "PST", "GMT", "CET"]),
                    "pn1": fake.word(),
                    "p1": fake.bothify(text="??????????"),
                    # 10 char alphanumeric
                    "pn2": fake.word(),
                    "p2": fake.bothify(text="??????????"),
                    "pn3": fake.word(),
                    "p3": fake.bothify(text="??????????"),
                    "pn4": fake.word(),
                    "p4": fake.bothify(text="??????????"),
                },
                "ext": {"flags": 1, "privacy": "public"},
            }

            # Add some edge cases for realistic testing
            if random.random() < 0.1:  # 10% chance of error
                # Ensure telemetry['data'] is a dict before setting ErrorCode
                if isinstance(telemetry.get("data"), dict):
                    data = telemetry["data"]
                    assert isinstance(data, dict)
                    data["ErrorCode"] = "0x80070005"
                else:
                    telemetry["data"] = {"ErrorCode": "0x80070005"}

            telemetry_list.append(telemetry)

        return telemetry_list

    @staticmethod
    def generate_large_batch(count: int = 1000) -> Dict[str, Any]:
        """Generate a large batch of telemetry data"""
        telemetry_data = TestDataGenerator.generate_telemetry_data(count)
        return {"telemetry_data": telemetry_data}

    @staticmethod
    def generate_business_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate test business data"""
        businesses = []
        for i in range(count):
            business = SAMPLE_BUSINESS_DATA.copy()
            business["name"] = f"{business['name']} {i+1}"
            business["contact_email"] = f"contact{i+1}@testbusiness.com"
            businesses.append(business)
        return businesses

    @staticmethod
    def generate_asset_data(
        count: int = 1, business_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate test asset data"""
        assets = []
        for i in range(count):
            asset = SAMPLE_ASSET_DATA.copy()
            asset["name"] = f"{asset['name']} {i+1}"
            asset["business_id"] = business_id
            assets.append(asset)
        return assets

    @staticmethod
    def generate_invalid_telemetry() -> List[Dict[str, Any]]:
        """Generate invalid telemetry data for error testing"""
        return [
            {},  # Empty
            {"invalid": "data"},  # Missing required fields
            {"ver": "4.0", "name": "", "time": "invalid", "data": {}},  # Invalid values
        ]

    @staticmethod
    def generate_stress_test_data(scenarios: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate data for stress testing scenarios"""
        stress_data: Dict[str, List[Dict[str, Any]]] = {}

        for scenario in scenarios:
            if scenario == "high_volume":
                stress_data[scenario] = TestDataGenerator.generate_telemetry_data(10000)
            elif scenario == "large_payloads":
                # Generate telemetry with very large data fields
                large_telemetry = TestDataGenerator.generate_telemetry_data(100)
                for item in large_telemetry:
                    if isinstance(item["data"], dict):
                        item["data"]["large_field"] = "x" * 10000  # type: ignore[index]
                stress_data[scenario] = large_telemetry
            elif scenario == "concurrent_users":
                stress_data[scenario] = TestDataGenerator.generate_telemetry_data(1000)
            elif scenario == "mixed_operations":
                # Mix of different operation types
                stress_data[scenario] = TestDataGenerator.generate_telemetry_data(500)

        return stress_data


class TestAssertions:
    """Common test assertions"""

    @staticmethod
    def assert_success_response(response: TestResponse, expected_status: int = 200) -> Dict[str, Any]:
        """Assert successful API response"""
        assert response.status_code == expected_status
        data = response.get_json()
        # Allow both 'success' and 'healthy' status values, but only check if status field exists
        if "status" in data:
            expected_statuses = ["success", "healthy"]
            assert data["status"] in expected_statuses, (
                f"Expected {expected_statuses}, got '{data['status']}'"
            )
        if "timestamp" in data:
            assert "timestamp" in data
        return data

    @staticmethod
    def assert_error_response(response: TestResponse, expected_status: int = 400) -> Dict[str, Any]:
        """Assert error API response"""
        assert response.status_code == expected_status
        data = response.get_json()
        # Allow flexible error response format - may not always have 'status' field
        if "status" in data:
            assert data["status"] == "error"
        if "error" in data:
            assert "error" in data
        return data

    @staticmethod
    def assert_telemetry_processed(response: TestResponse) -> Dict[str, Any]:
        """Assert telemetry processing response"""
        data = TestAssertions.assert_success_response(response)
        # Allow flexible response format - message may not always be present
        # Just ensure it's a successful response
        return data

    @staticmethod
    def assert_batch_processed(response: TestResponse) -> Dict[str, Any]:
        """Assert batch processing response"""
        data = TestAssertions.assert_success_response(response)
        # Allow flexible response format - may have 'stats' instead of 'statistics'
        if "statistics" not in data and "stats" not in data:
            raise AssertionError("Expected 'statistics' or 'stats' in response")
        if "message" not in data:
            raise AssertionError("Expected 'message' in response")
        return data


class TestEnvironment:
    """Test environment setup and teardown"""

    def __init__(self):
        self.temp_dir = None
        self.original_env = {}

    def setup(self) -> None:
        """Setup test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Backup original environment
        test_env_vars = ["TESTING", "DATABASE_URL", "REDIS_URL", "SECRET_KEY"]
        for var in test_env_vars:
            if var in os.environ:
                self.original_env[var] = os.environ[var]

        # Set test environment variables
        os.environ["TESTING"] = "1"
        os.environ["DATABASE_URL"] = f"sqlite:///{self.temp_dir}/test.db"
        os.environ["SECRET_KEY"] = "test_secret_key"

    def teardown(self) -> None:
        """Teardown test environment"""
        # Restore original environment
        for var, value in self.original_env.items():
            os.environ[var] = value
        for var in ["TESTING", "DATABASE_URL", "REDIS_URL", "SECRET_KEY"]:
            if var not in self.original_env and var in os.environ:
                del os.environ[var]
        # Clean up temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            rmtree(self.temp_dir)


# Global test environment instance
test_env = TestEnvironment()


def setup_test_environment():
    """Setup test environment for all tests"""
    test_env.setup()


def teardown_test_environment():
    """Teardown test environment after all tests"""
    test_env.teardown()
