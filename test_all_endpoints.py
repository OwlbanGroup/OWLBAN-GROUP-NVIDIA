import unittest
import json
import os
from app import app

# Set dummy env vars for testing
os.environ['TOKEN_CLIENT_ID'] = 'dummy_client_id'
os.environ['TOKEN_CLIENT_SECRET'] = 'dummy_client_secret'
os.environ['SECRET_KEY'] = 'dummy_secret_key'

class AllEndpointsTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.auth_header = {
            'Authorization': 'Bearer dummy_token'  # Assuming token validation is mocked or dummy
        }

    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')

    def test_metrics_endpoint(self):
        response = self.app.get('/metrics')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/plain', response.content_type)

    def test_websocket_status_endpoint(self):
        response = self.app.get('/ws/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('active_connections', data)

    def test_data_formats_endpoint(self):
        response = self.app.get('/data/formats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('import_formats', data)
        self.assertIn('export_formats', data)

    # Security tests
    def test_missing_auth_header(self):
        response = self.app.post('/telemetry')
        self.assertEqual(response.status_code, 401)

    def test_rate_limiting(self):
        # Test rate limiting on health endpoint (10 per minute)
        for _ in range(12):
            response = self.app.get('/health')
        self.assertEqual(response.status_code, 429)

    # Telemetry endpoints
    def test_telemetry_post_missing_data(self):
        response = self.app.post('/telemetry', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)

    def test_telemetry_post_invalid_json(self):
        response = self.app.post('/telemetry', headers=self.auth_header, data='invalid json')
        self.assertEqual(response.status_code, 400)

    def test_telemetry_batch_missing_data(self):
        response = self.app.post('/telemetry/batch', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)

    def test_telemetry_metrics(self):
        response = self.app.get('/telemetry/metrics', headers=self.auth_header)
        self.assertEqual(response.status_code, 200)

    def test_telemetry_export_invalid_limit(self):
        response = self.app.get('/telemetry/export?limit=10001', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)

    # ML endpoints
    def test_anomalies_missing_data(self):
        response = self.app.post('/ml/anomalies', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)

    def test_train_ml_missing_data(self):
        response = self.app.post('/ml/train', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)

    # Data conversion
    def test_data_convert_missing_data(self):
        response = self.app.post('/data/convert', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)

    def test_data_convert_invalid_format(self):
        data = {'data': [{'test': 'data'}], 'from_format': 'json', 'to_format': 'invalid'}
        response = self.app.post('/data/convert', headers=self.auth_header, json=data)
        self.assertEqual(response.status_code, 400)

    # MCP endpoints
    def test_mcp_repos_missing_auth(self):
        response = self.app.get('/mcp/repos')
        self.assertEqual(response.status_code, 401)

    def test_mcp_repos_invalid_per_page(self):
        response = self.app.get('/mcp/repos?per_page=101', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)

    def test_mcp_issues_invalid_state(self):
        response = self.app.get('/mcp/issues/test/test?state=invalid', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)

    def test_mcp_create_issue_missing_title(self):
        response = self.app.post('/mcp/issues/test/test', headers=self.auth_header, json={})
        self.assertEqual(response.status_code, 400)

    def test_mcp_create_issue_invalid_assignees(self):
        data = {'title': 'Test', 'assignees': 'not_list'}
        response = self.app.post('/mcp/issues/test/test', headers=self.auth_header, json=data)
        self.assertEqual(response.status_code, 400)

    # Cloud storage
    def test_cloud_export_missing_data(self):
        response = self.app.post('/storage/export', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)

    # Error handling
    def test_404_error(self):
        response = self.app.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
