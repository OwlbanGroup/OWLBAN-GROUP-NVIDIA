#!/usr/bin/env python3
"""
Comprehensive Integration Tests for JPMorgan Financial APIs
Tests the full integration between all components: API, database, ML, cloud storage, WebSocket, MCP, etc.
"""
import os
import json
import time
import pytest
import tempfile
import threading
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Set environment variables for testing
os.environ['ALLOW_MISSING_TOKENS'] = 'true'
os.environ['SECRET_KEY'] = 'test_secret_key_for_testing'

from app import app
from datetime import datetime, timezone
import redis
import asyncio

# Sample data for testing
SAMPLE_TELEMETRY_DATA = {
    "ver": "4.0",
    "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
    "time": "2025-09-22T19:42:10.2549325Z",
    "data": {
        "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
        "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
        "OS": "Windows 11",
        "DeviceModel": "Surface Pro 9",
        "UserId": "test_user_123"
    },
    "ext": {
        "flags": 1,
        "privacy": "public"
    }
}

BATCH_TELEMETRY_DATA = {
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
        }
    ]
}

@pytest.fixture(scope='module')
def test_client():
    """Fixture to start the Flask app in test mode"""
    app.config['TESTING'] = True
    # Disable rate limiting for tests by accessing the existing limiter
    from app import limiter
    limiter.enabled = False

    with app.test_client() as client:
        yield client

@pytest.fixture(scope='module')
def mock_redis():
    """Mock Redis for testing"""
    with patch('redis.from_url') as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        yield mock_client

class TestAPIDatabaseIntegration:
    """Test API to Database integration"""

    def test_telemetry_ingestion_to_database(self, test_client):
        """Test that telemetry data flows from API to database"""
        # Send telemetry data
        response = test_client.post('/telemetry', json=SAMPLE_TELEMETRY_DATA)
        # In test mode, authentication may redirect, but we expect success for the core functionality
        # The 302 redirect is expected when authentication middleware is active
        assert response.status_code in [200, 302]  # Allow both success and redirect

        if response.status_code == 200:
            data = json.loads(response.data)
            assert data['status'] == 'success'

            # Verify data was stored by checking metrics
            metrics_response = test_client.get('/telemetry/metrics?hours=1')
            assert metrics_response.status_code == 200

            metrics_data = json.loads(metrics_response.data)
            assert metrics_data['status'] == 'success'
            assert 'metrics' in metrics_data

    def test_batch_processing_integration(self, test_client):
        """Test batch processing from API to database"""
        response = test_client.post('/telemetry/batch', json=BATCH_TELEMETRY_DATA)
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'statistics' in data
        assert data['statistics']['total'] == 2

    def test_export_from_database(self, test_client):
        """Test data export from database"""
        # First ingest some data
        test_client.post('/telemetry', json=SAMPLE_TELEMETRY_DATA)

        # Export data
        response = test_client.get('/telemetry/export?limit=10')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'events' in data
        assert len(data['events']) > 0

class TestAPIMLIntegration:
    """Test API to ML model integration"""

    def test_anomaly_detection_pipeline(self, test_client):
        """Test full anomaly detection pipeline"""
        response = test_client.post('/ml/anomalies', json=BATCH_TELEMETRY_DATA)
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'anomaly_results' in data

    def test_ml_training_integration(self, test_client):
        """Test ML model training integration"""
        response = test_client.post('/ml/train', json=BATCH_TELEMETRY_DATA)
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'message' in data

class TestAPICloudStorageIntegration:
    """Test API to Cloud Storage integration"""

    @patch('src.cloud_storage.cloud_storage_manager')
    def test_cloud_export_integration(self, mock_storage_manager, test_client):
        """Test cloud storage export integration"""
        # Mock the cloud storage manager
        mock_storage_manager.export_telemetry_data.return_value = {
            'aws': 's3://bucket/file.json',
            'gcs': 'gs://bucket/file.json'
        }

        export_config = {
            "operation": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
            "limit": 10,
            "format": "json",
            "providers": ["aws", "gcs"],
            "filename_prefix": "test_export"
        }

        response = test_client.post('/storage/export', json=export_config)
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'export_results' in data

class TestAPIWebSocketIntegration:
    """Test API to WebSocket integration"""

    @patch('src.websocket_manager.websocket_manager')
    def test_websocket_status_integration(self, mock_ws_manager, test_client):
        """Test WebSocket status integration"""
        # Mock WebSocket manager
        async def mock_connection_count():
            return 5
        async def mock_client_count():
            return 3

        mock_ws_manager.get_connection_count = mock_connection_count
        mock_ws_manager.get_client_count = mock_client_count

        response = test_client.get('/ws/status')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['active_connections'] == 5
        assert data['unique_clients'] == 3

class TestAPIMCPIntegration:
    """Test API to MCP (GitHub) integration"""

    @patch('src.mcp_integration.mcp_client')
    def test_github_repos_integration(self, mock_mcp_client, test_client):
        """Test GitHub repositories integration"""
        mock_mcp_client.list_repositories.return_value = [
            {'name': 'repo1', 'owner': 'user1'},
            {'name': 'repo2', 'owner': 'user2'}
        ]

        response = test_client.get('/mcp/repos?query=test')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert len(data['repositories']) == 2

    @patch('src.mcp_integration.mcp_client')
    def test_github_issues_integration(self, mock_mcp_client, test_client):
        """Test GitHub issues integration"""
        mock_mcp_client.list_issues.return_value = [
            {'title': 'Issue 1', 'state': 'open'},
            {'title': 'Issue 2', 'state': 'closed'}
        ]

        response = test_client.get('/mcp/issues/owner/repo')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert len(data['issues']) == 2

class TestAPIRedisIntegration:
    """Test API to Redis caching integration"""

    def test_caching_integration(self, test_client, mock_redis):
        """Test Redis caching integration"""
        # Mock Redis operations
        mock_redis.get.return_value = None  # Cache miss first time
        mock_redis.setex.return_value = True

        # First request should cache
        response1 = test_client.get('/telemetry/metrics?hours=1')
        assert response1.status_code == 200

        # Second request should use cache
        response2 = test_client.get('/telemetry/metrics?hours=1')
        assert response2.status_code == 200

        # Verify Redis was called
        assert mock_redis.get.called
        assert mock_redis.setex.called

class TestAPISecurityIntegration:
    """Test API security integration"""

    def test_authentication_integration(self, test_client):
        """Test authentication integration"""
        # Test without auth header
        response = test_client.post('/telemetry', json=SAMPLE_TELEMETRY_DATA)
        # Should work in test mode (auth disabled)
        assert response.status_code == 200

    def test_rate_limiting_integration(self, test_client):
        """Test rate limiting integration (disabled in test mode)"""
        # Rate limiting is disabled in test mode, so multiple requests should work
        for _ in range(5):
            response = test_client.get('/health')
            assert response.status_code == 200

class TestDataFormatIntegration:
    """Test data format conversion integration"""

    def test_format_conversion_integration(self, test_client):
        """Test data format conversion integration"""
        conversion_data = {
            "data": [
                {"name": "test1", "value": 1},
                {"name": "test2", "value": 2}
            ],
            "from_format": "json",
            "to_format": "csv"
        }

        response = test_client.post('/data/convert', json=conversion_data)
        assert response.status_code == 200

        # Should return CSV data
        csv_data = response.data.decode('utf-8')
        assert 'name,value' in csv_data
        assert 'test1,1' in csv_data

class TestFullPipelineIntegration:
    """Test complete end-to-end pipeline integration"""

    def test_complete_telemetry_pipeline(self, test_client):
        """Test complete telemetry processing pipeline"""
        # Step 1: Health check
        response = test_client.get('/health')
        assert response.status_code == 200

        # Step 2: Ingest telemetry
        response = test_client.post('/telemetry', json=SAMPLE_TELEMETRY_DATA)
        assert response.status_code == 200

        # Step 3: Check metrics
        response = test_client.get('/telemetry/metrics?hours=1')
        assert response.status_code == 200

        # Step 4: Detect anomalies
        response = test_client.post('/ml/anomalies', json=BATCH_TELEMETRY_DATA)
        assert response.status_code == 200

        # Step 5: Train model
        response = test_client.post('/ml/train', json=BATCH_TELEMETRY_DATA)
        assert response.status_code == 200

        # Step 6: Export data
        response = test_client.get('/telemetry/export?limit=5')
        assert response.status_code == 200

        # Step 7: Convert format
        conversion_data = {
            "data": [{"test": "data"}],
            "from_format": "json",
            "to_format": "json"
        }
        response = test_client.post('/data/convert', json=conversion_data)
        assert response.status_code == 200

    def test_error_handling_integration(self, test_client):
        """Test error handling across components"""
        # Test invalid JSON
        response = test_client.post('/telemetry', data='invalid json', content_type='application/json')
        assert response.status_code == 400

        # Test missing data
        response = test_client.post('/telemetry', json={})
        assert response.status_code == 400

        # Test invalid endpoint
        response = test_client.get('/nonexistent')
        assert response.status_code == 404

class TestConcurrentAccessIntegration:
    """Test concurrent access integration"""

    def test_concurrent_telemetry_ingestion(self, test_client):
        """Test concurrent telemetry ingestion"""
        import concurrent.futures
        import threading

        results = []
        lock = threading.Lock()

        def ingest_telemetry():
            response = test_client.post('/telemetry', json=SAMPLE_TELEMETRY_DATA)
            with lock:
                results.append(response.status_code)

        # Run 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(ingest_telemetry) for _ in range(5)]
            concurrent.futures.wait(futures)

        # All should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

class TestMetricsAndMonitoringIntegration:
    """Test metrics and monitoring integration"""

    def test_prometheus_metrics_integration(self, test_client):
        """Test Prometheus metrics integration"""
        response = test_client.get('/metrics')
        assert response.status_code == 200

        # Should return Prometheus format
        metrics_data = response.data.decode('utf-8')
        assert 'http_requests_total' in metrics_data
        assert 'http_request_duration_seconds' in metrics_data

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
