"""
Test utilities for JPMorgan Financial APIs comprehensive testing
Provides test helpers, data generators, and assertion utilities
"""

import json
import time
import random
import string
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import unittest
from unittest.mock import Mock, patch

# Sample telemetry data for testing
SAMPLE_TELEMETRY_DATA = {
    "ver": "4.0",
    "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.NetworkRequest",
    "time": "2025-09-22T19:42:13.2549325Z",
    "data": {
        "Op": "StoreConfigurationServer::DownloadConfigurationAsync",
        "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
        "OS": "Windows 10",
        "DeviceModel": "Dell XPS 13",
        "UserId": "test_user_123",
        "URL": "https://config.store.microsoft.com/config",
        "ResponseTime": 150
    },
    "ext": {
        "flags": 1,
        "privacy": "public"
    }
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
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 11",
                "DeviceModel": "Surface Pro 9",
                "UserId": "test_user_456"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Error",
            "time": "2025-09-22T19:42:12.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 11",
                "DeviceModel": "Surface Pro 9",
                "UserId": "test_user_789",
                "ErrorCode": "0x80070005"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        }
    ]
}

class TestUser:
    """Test user helper for authentication testing"""

    def __init__(self, username: str = None, password: str = None, role: str = "USER"):
        self.username = username or f"test_user_{random.randint(1000, 9999)}"
        self.password = password or "testpass123"
        self.role = role
        self.token = None
        self.user_id = None

    def register(self, client) -> Dict[str, Any]:
        """Register the test user"""
        response = client.post('/user/register', json={
            'username': self.username,
            'password': self.password,
            'email': f"{self.username}@test.com",
            'role': self.role
        })
        if response.status_code == 201:
            data = json.loads(response.data)
            self.user_id = data['user']['id']
        return json.loads(response.data)

    def login(self, client) -> Dict[str, Any]:
        """Login the test user and get token"""
        response = client.post('/user/login', json={
            'username': self.username,
            'password': self.password
        })
        if response.status_code == 200:
            data = json.loads(response.data)
            self.token = data['token']
        return json.loads(response.data)

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests"""
        return {'Authorization': f'Bearer {self.token}'} if self.token else {}

class DatabaseTestHelper:
    """Helper for database testing operations"""

    @staticmethod
    def cleanup_test_data(db_manager):
        """Clean up test data from database"""
        try:
            # Clean up test users
            db_manager.execute_query("DELETE FROM users WHERE username LIKE 'test_%'")
            # Clean up test businesses
            db_manager.execute_query("DELETE FROM businesses WHERE name LIKE 'Test%'")
            # Clean up test assets
            db_manager.execute_query("DELETE FROM assets WHERE name LIKE 'Test%'")
            # Clean up test transactions
            db_manager.execute_query("DELETE FROM revenue_transactions WHERE description LIKE 'Test%'")
        except Exception as e:
            print(f"Warning: Could not cleanup test data: {e}")

    @staticmethod
    def get_table_counts(db_manager) -> Dict[str, int]:
        """Get row counts for main tables"""
        counts = {}
        try:
            tables = ['users', 'businesses', 'assets', 'revenue_transactions', 'audit_logs']
            for table in tables:
                result = db_manager.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                counts[table] = result[0]['count'] if result else 0
        except Exception as e:
            print(f"Warning: Could not get table counts: {e}")
        return counts

class PerformanceTestHelper:
    """Helper for performance testing"""

    @staticmethod
    def measure_response_time(func, *args, **kwargs) -> float:
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result

    @staticmethod
    def run_load_test(client, endpoint: str, num_requests: int = 100, concurrent: bool = False) -> Dict[str, Any]:
        """Run basic load test on an endpoint"""
        import concurrent.futures

        def make_request():
            start = time.time()
            response = client.get(endpoint)
            end = time.time()
            return response.status_code, end - start

        results = []
        if concurrent:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
        else:
            for _ in range(num_requests):
                results.append(make_request())

        status_codes = [r[0] for r in results]
        response_times = [r[1] for r in results]

        return {
            'total_requests': num_requests,
            'successful_requests': status_codes.count(200),
            'failed_requests': len([s for s in status_codes if s != 200]),
            'avg_response_time': sum(response_times) / len(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'status_codes': {code: status_codes.count(code) for code in set(status_codes)}
        }

class TestDataGenerator:
    """Generator for test data"""

    @staticmethod
    def generate_telemetry_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate sample telemetry data"""
        data = []
        operations = [
            "StoreConfigurationServer::DownloadConfigurationAsync",
            "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
            "StoreConfigurationServer::GetCachedConfiguration",
            "StoreConfigurationServer::UpdateConfiguration"
        ]
        devices = ["Dell XPS 13", "Surface Pro 9", "HP Spectre", "MacBook Pro", "ThinkPad X1"]
        os_versions = ["Windows 10", "Windows 11", "macOS 12", "macOS 13"]

        for i in range(count):
            telemetry = {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.NetworkRequest",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "Op": random.choice(operations),
                    "PFN": f"Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe_{i}",
                    "OS": random.choice(os_versions),
                    "DeviceModel": random.choice(devices),
                    "UserId": f"test_user_{i}",
                    "URL": "https://config.store.microsoft.com/config",
                    "ResponseTime": random.randint(50, 500)
                },
                "ext": {
                    "flags": 1,
                    "privacy": "public"
                }
            }
            data.append(telemetry)
        return data

    @staticmethod
    def generate_business_data() -> Dict[str, Any]:
        """Generate sample business data"""
        return {
            "name": f"Test Business Corp {random.randint(1000, 9999)}",
            "type": random.choice(["corporation", "llc", "partnership"]),
            "registration_number": f"REG{random.randint(100000, 999999)}",
            "address": f"{random.randint(100, 9999)} Test Street, {random.choice(['New York', 'London', 'Tokyo'])}, NY",
            "contact_info": {
                "email": f"contact{random.randint(1000, 9999)}@testbusiness.com",
                "phone": f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            }
        }

    @staticmethod
    def generate_asset_data(business_id: int = None) -> Dict[str, Any]:
        """Generate sample asset data"""
        return {
            "business_id": business_id or random.randint(1, 100),
            "name": f"Test Asset {random.choice(['Server', 'Software', 'Equipment', 'Vehicle'])} {random.randint(1000, 9999)}",
            "type": random.choice(["equipment", "software", "property", "vehicle"]),
            "value": round(random.uniform(1000, 100000), 2),
            "acquisition_date": (datetime.now(timezone.utc) - timedelta(days=random.randint(1, 365))).isoformat(),
            "ownership_percentage": random.choice([100.0, 50.0, 75.0, 25.0]),
            "description": f"Test asset description {random.randint(1000, 9999)}"
        }

class TestAssertions:
    """Custom assertions for testing"""

    @staticmethod
    def assert_response_success(response, expected_status: int = 200):
        """Assert successful API response"""
        assert response.status_code == expected_status, f"Expected status {expected_status}, got {response.status_code}"
        data = json.loads(response.data)
        assert data['status'] == 'success', f"Expected success status, got {data.get('status')}"

    @staticmethod
    def assert_response_error(response, expected_status: int = 400):
        """Assert error API response"""
        assert response.status_code == expected_status, f"Expected status {expected_status}, got {response.status_code}"
        data = json.loads(response.data)
        assert data['status'] == 'error', f"Expected error status, got {data.get('status')}"
        assert 'error' in data, "Error response should contain 'error' field"

    @staticmethod
    def assert_telemetry_processed(response):
        """Assert telemetry was processed successfully"""
        TestAssertions.assert_response_success(response)
        data = json.loads(response.data)
        assert 'message' in data, "Response should contain message"
        assert 'timestamp' in data, "Response should contain timestamp"

    @staticmethod
    def assert_business_created(response):
        """Assert business was created successfully"""
        TestAssertions.assert_response_success(response, 201)
        data = json.loads(response.data)
        assert 'business' in data, "Response should contain business data"
        assert 'id' in data['business'], "Business should have ID"

    @staticmethod
    def assert_asset_created(response):
        """Assert asset was created successfully"""
        TestAssertions.assert_response_success(response, 201)
        data = json.loads(response.data)
        assert 'asset' in data, "Response should contain asset data"
        assert 'id' in data['asset'], "Asset should have ID"

    @staticmethod
    def assert_user_authenticated(response):
        """Assert user authentication was successful"""
        TestAssertions.assert_response_success(response, 200)
        data = json.loads(response.data)
        assert 'token' in data, "Response should contain token"
        assert 'user' in data, "Response should contain user data"

# Additional test data constants
SAMPLE_BUSINESS_DATA = TestDataGenerator.generate_business_data()
SAMPLE_ASSET_DATA = TestDataGenerator.generate_asset_data()

# Test configuration
TEST_CONFIG = {
    'database_url': 'sqlite:///test.db',
    'redis_url': None,  # Use in-memory for tests
    'secret_key': 'test_secret_key_12345',
    'testing': True,
    'log_level': 'WARNING'  # Reduce log noise during tests
}
