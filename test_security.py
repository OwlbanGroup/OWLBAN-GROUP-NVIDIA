import unittest
import json
from app import app

class SecurityTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.auth_header = {
            'Authorization': 'Bearer valid_test_token'
        }

    def test_missing_auth_header(self):
        response = self.app.get('/mcp/repos')
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_invalid_auth_token(self):
        response = self.app.get('/mcp/repos', headers={'Authorization': 'Bearer invalid_token'})
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_rate_limiting(self):
        # Exceed rate limit for /mcp/repos endpoint
        for _ in range(11):
            response = self.app.get('/mcp/repos', headers=self.auth_header)
        self.assertEqual(response.status_code, 429)  # Too Many Requests

    def test_invalid_per_page_parameter(self):
        response = self.app.get('/mcp/repos?per_page=101', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_invalid_state_parameter(self):
        response = self.app.get('/mcp/issues/test/test?state=invalid', headers=self.auth_header)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_create_issue_missing_title(self):
        response = self.app.post('/mcp/issues/test/test', headers=self.auth_header, json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_create_issue_invalid_assignees(self):
        response = self.app.post('/mcp/issues/test/test', headers=self.auth_header, json={
            'title': 'Test Issue',
            'assignees': 'not_a_list'
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
