import pytest
from unittest.mock import patch, MagicMock
from src.token_manager import TokenManager

@pytest.fixture
def token_manager():
    return TokenManager(
        client_id='test_client_id',
        client_secret='test_client_secret',
        token_url='https://auth.example.com/token',
        scope='openid'
    )

def test_get_token_first_time(token_manager):
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'access_token': 'test_access_token',
            'expires_in': 3600
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        token = token_manager.get_token()
        assert token == 'test_access_token'
        assert token_manager.access_token == 'test_access_token'
        assert token_manager.token_expires_at > 0

def test_get_token_cached(token_manager):
    token_manager.access_token = 'cached_token'
    token_manager.token_expires_at = 9999999999  # Future time

    token = token_manager.get_token()
    assert token == 'cached_token'

def test_refresh_token_error(token_manager):
    with patch('requests.post') as mock_post:
        mock_post.side_effect = Exception('Network error')

        with pytest.raises(Exception):
            token_manager._refresh_token()
