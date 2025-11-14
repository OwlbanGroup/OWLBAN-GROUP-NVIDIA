import os
import time
os.environ['ALLOW_IMPORTS_SUCCESSFULLY'] = 'true'

from app import app

def test_user_register_endpoint():
    with app.test_client() as client:
        # Test successful registration
        username = f'testuser_{int(time.time())}'
        response = client.post('/user/register', json={
            'username': username,
            'password': 'testpass123',
            'email': 'test@example.com'
        })
        print('User register response:', response.get_json())
        assert response.status_code == 201
        data = response.get_json()
        assert 'status' in data
        assert data['status'] == 'success'
        assert 'user' in data
        assert data['user']['username'] == username
        assert data['user']['role'] == 'user'
        print('User register endpoint test passed!')

if __name__ == '__main__':
    test_user_register_endpoint()
