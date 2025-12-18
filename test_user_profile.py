import requests
import time

BASE_URL = 'http://localhost:5000'

# Simple session with SSL verification disabled
session = requests.Session()
session.verify = False
requests.packages.urllib3.disable_warnings()

def test_user_profile():
    try:
        # Register user
        register_data = {
            'username': 'testuser',
            'password': 'testpass',
            'email': 'test@example.com',
            'role': 'user'
        }
        response = session.post(f'{BASE_URL}/user/register', json=register_data, timeout=10)
        print('Register response status:', response.status_code)
        print('Register response:', response.json())

        if response.status_code != 201:
            print('Registration failed')
            return False

        # Login
        login_data = {
            'username': 'testuser',
            'password': 'testpass'
        }
        response = session.post(f'{BASE_URL}/user/login', json=login_data, timeout=10)
        print('Login response status:', response.status_code)
        print('Login response:', response.json())

        if response.status_code != 200:
            print('Login failed')
            return False

        token = response.json()['token']

        # Get profile
        headers = {'Authorization': f'Bearer {token}'}
        response = session.get(f'{BASE_URL}/user/profile', headers=headers, timeout=10)
        print('Profile response status:', response.status_code)
        print('Profile response:', response.json())

        if response.status_code != 200:
            print('Profile request failed')
            return False

        # Check response structure
        data = response.json()
        if 'user' not in data:
            print('No user data in response')
            return False

        user = data['user']
        expected_fields = ['id', 'username', 'email', 'role', 'business_id', 'is_active', 'created_at', 'updated_at', 'last_login_at']

        missing_fields = []
        for field in expected_fields:
            if field not in user:
                missing_fields.append(field)
            else:
                print(f'✓ Field {field}: {user[field]}')

        if missing_fields:
            print(f'✗ Missing fields: {missing_fields}')
            return False

        if 'permissions' not in data:
            print('✗ Missing permissions')
            return False

        print('✓ Permissions:', data['permissions'])
        print('✓ All checks passed')
        return True

    except requests.exceptions.RequestException as e:
        print(f'Request error: {e}')
        return False
    except Exception as e:
        print(f'Error: {e}')
        return False

if __name__ == '__main__':
    success = test_user_profile()
    if success:
        print('TEST PASSED: User profile endpoint works correctly')
    else:
        print('TEST FAILED: Issues found with user profile endpoint')
