import urllib.request
import json

BASE_URL = 'http://localhost:5000'

def test_health():
    try:
        with urllib.request.urlopen(f'{BASE_URL}/health') as response:
            data = json.loads(response.read().decode())
            print('Health check successful:', data)
            return True
    except Exception as e:
        print(f'Health check failed: {e}')
        return False

if __name__ == '__main__':
    test_health()
