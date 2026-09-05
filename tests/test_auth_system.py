"""
Comprehensive tests for the OWLBAN GROUP Authentication System.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_lib import (
    AuthManager, authenticate_user, verify_token, create_user,
    request_password_reset, reset_password, generate_api_key, verify_api_key,
    AuthConfig
)


@pytest.fixture(autouse=True)
def temp_auth_store(tmp_path, monkeypatch):
    user_file = str(tmp_path / "users.json")
    session_file = str(tmp_path / "sessions.json")
    monkeypatch.setattr(AuthManager, '__init__', lambda self, **kw: None)
    am = AuthManager.__new__(AuthManager)
    am.user_store_file = user_file
    am.session_store_file = session_file
    am.config = AuthConfig()
    am.users = {}
    am.sessions = {}
    am._password_reset_tokens = {}
    am._audit_log = []
    am._api_keys = {}
    monkeypatch.setattr('auth_lib.auth_manager', am)
    return am


class TestUserRegistration:
    def test_create_user_success(self, temp_auth_store):
        success, msg = create_user(
            'newuser@owlban.com', 'newuser', 'SecurePass1!', 'user', 'OWLBAN_GROUP'
        )
        assert success is True
        assert 'created successfully' in msg

    def test_create_user_duplicate_email(self, temp_auth_store):
        create_user('dup@owlban.com', 'user1', 'SecurePass1!', 'user', 'OWLBAN_GROUP')
        success, msg = create_user('dup@owlban.com', 'user2', 'SecurePass1!', 'user', 'OWLBAN_GROUP')
        assert success is False
        assert 'already exists' in msg

    def test_create_user_invalid_email(self, temp_auth_store):
        success, msg = create_user('notanemail', 'user1', 'SecurePass1!')
        assert success is False
        assert 'email' in msg.lower()

    def test_create_user_weak_password(self, temp_auth_store):
        success, msg = create_user('weak@owlban.com', 'user1', 'short')
        assert success is False
        assert 'password' in msg.lower()

    def test_create_user_invalid_company(self, temp_auth_store):
        success, msg = create_user('test@owlban.com', 'user1', 'SecurePass1!', 'user', 'INVALID_CO')
        assert success is False
        assert 'company' in msg.lower()

    def test_create_user_invalid_role(self, temp_auth_store):
        success, msg = create_user('test@owlban.com', 'user1', 'SecurePass1!', 'superadmin')
        assert success is False
        assert 'role' in msg.lower()

    def test_password_policy_min_length(self, temp_auth_store):
        success, msg = create_user('test@owlban.com', 'user1', 'Ab1!', 'user')
        assert success is False
        assert '8 characters' in msg


class TestUserAuthentication:
    def test_authenticate_valid_user(self, temp_auth_store):
        create_user('auth@owlban.com', 'authuser', 'TestPass123!', 'user', 'OWLBAN_GROUP')
        success, msg, user = authenticate_user('auth@owlban.com', 'TestPass123!')
        assert success is True
        assert user is not None
        assert user.email == 'auth@owlban.com'

    def test_authenticate_wrong_password(self, temp_auth_store):
        create_user('auth2@owlban.com', 'authuser2', 'TestPass123!')
        success, msg, user = authenticate_user('auth2@owlban.com', 'WrongPass123!')
        assert success is False
        assert user is None

    def test_authenticate_nonexistent_user(self, temp_auth_store):
        success, msg, user = authenticate_user('ghost@owlban.com', 'TestPass123!')
        assert success is False
        assert user is None

    def test_account_lockout_after_max_attempts(self, temp_auth_store):
        create_user('lock@owlban.com', 'lockuser', 'TestPass123!')
        for _ in range(AuthConfig.MAX_LOGIN_ATTEMPTS):
            authenticate_user('lock@owlban.com', 'wrongpass')
        success, msg, user = authenticate_user('lock@owlban.com', 'TestPass123!')
        assert success is False
        assert 'locked' in msg.lower() or 'attempts' in msg.lower()


class TestTokenManagement:
    def test_generate_and_verify_access_token(self, temp_auth_store):
        create_user('token@owlban.com', 'tokenuser', 'TestPass123!')
        _, _, user = authenticate_user('token@owlban.com', 'TestPass123!')
        access_token, refresh_token = temp_auth_store.generate_tokens(user)
        assert access_token is not None
        payload = verify_token(access_token)
        assert payload is not None
        assert payload['email'] == 'token@owlban.com'

    def test_verify_invalid_token(self, temp_auth_store):
        result = verify_token('invalid.token.here')
        assert result is None


class TestPasswordReset:
    def test_create_reset_token(self, temp_auth_store):
        create_user('reset@owlban.com', 'resetuser', 'TestPass123!')
        token = request_password_reset('reset@owlban.com')
        assert token is not None

    def test_reset_password_success(self, temp_auth_store):
        create_user('reset2@owlban.com', 'resetuser2', 'TestPass123!')
        token = request_password_reset('reset2@owlban.com')
        success, msg = reset_password(token, 'NewSecurePass1!')
        assert success is True

    def test_reset_password_invalid_token(self, temp_auth_store):
        success, msg = reset_password('badtoken', 'NewSecurePass1!')
        assert success is False


class TestAPIKeys:
    def test_generate_api_key(self, temp_auth_store):
        create_user('api@owlban.com', 'apiuser', 'TestPass123!')
        key = generate_api_key('api@owlban.com', 'my-app')
        assert key is not None
        assert key.startswith('owlban_')

    def test_verify_valid_api_key(self, temp_auth_store):
        create_user('api2@owlban.com', 'apiuser2', 'TestPass123!')
        key = generate_api_key('api2@owlban.com')
        result = verify_api_key(key)
        assert result is not None
        assert result['email'] == 'api2@owlban.com'

    def test_verify_invalid_api_key(self, temp_auth_store):
        result = verify_api_key('owlban_invalidkey123')
        assert result is None


class TestE2EAuthenticationFlow:
    def test_full_registration_to_login_flow(self, temp_auth_store):
        success, msg = create_user('e2e@owlban.com', 'e2euser', 'SecurePass1!', 'user', 'OWLBAN_GROUP')
        assert success is True
        success, msg, user = authenticate_user('e2e@owlban.com', 'SecurePass1!')
        assert success is True
        access_token, _ = temp_auth_store.generate_tokens(user)
        payload = verify_token(access_token)
        assert payload is not None
        assert payload['email'] == 'e2e@owlban.com'

    def test_full_password_reset_flow(self, temp_auth_store):
        create_user('reset_e2e@owlban.com', 'resetuser', 'OldPass123!')
        success, _, _ = authenticate_user('reset_e2e@owlban.com', 'OldPass123!')
        assert success is True
        token = request_password_reset('reset_e2e@owlban.com')
        success, _ = reset_password(token, 'NewPass123!')
        assert success is True
        success, _, _ = authenticate_user('reset_e2e@owlban.com', 'OldPass123!')
        assert success is False
        success, _, _ = authenticate_user('reset_e2e@owlban.com', 'NewPass123!')
        assert success is True
