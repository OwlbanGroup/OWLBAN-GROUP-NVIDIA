import unittest
from app_final import app
token_manager = type('MockTokenManager', (), {'validate_token': lambda self, t: True})()
from flask import json

class TestAuthentication(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')

    def test_telemetry_without_auth(self):
        response = self.app.post('/telemetry', data=json.dumps({'test': 'data'}), content_type='application/json')
        self.assertEqual(response.status_code, 401)

    def test_telemetry_with_invalid_token(self):
        response = self.app.post('/telemetry', data=json.dumps({'test': 'data'}), content_type='application/json', headers={'Authorization': 'Bearer invalid_token'})
        self.assertEqual(response.status_code, 401)

    def test_telemetry_with_valid_token(self):
        # Generate a valid token for testing
        valid_token = token_manager.generate_token()
        response = self.app.post('/telemetry', data=json.dumps({'test': 'data'}), content_type='application/json', headers={'Authorization': f'Bearer {valid_token}'})
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
