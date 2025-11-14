import os
os.environ['ALLOW_IMPORTS_SUCCESSFULLY'] = 'true'

from app import app

def test_health_endpoint():
    with app.test_client() as client:
        response = client.get('/health')
        print('Health endpoint response:', response.get_json())
        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data
        assert data['status'] == 'healthy'
        print('Health endpoint test passed!')

if __name__ == '__main__':
    test_health_endpoint()
