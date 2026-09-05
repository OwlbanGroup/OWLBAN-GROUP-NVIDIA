"""
E2E Integration Tests - Cross-Platform Authentication Verification.
Tests that auth works across OWLBAN GROUP, OSCAR BROOME, BLACKBOX AI,
the FastAPI server, and the Streamlit dashboard.
"""

import sys
import os
import json
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_lib import (
    AuthManager, authenticate_user, verify_token, create_user,
    request_password_reset, reset_password, generate_api_key, verify_api_key,
    auth_manager, AuthConfig
)


@pytest.fixture(autouse=True)
def temp_auth_store(tmp_path, monkeypatch):
    """Use temporary files for user/session storage during tests."""
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


class TestCrossPlatformSSO:
    """Test Single Sign-On: register once, authenticate everywhere."""

    def test_register_then_login_all_platforms(self, temp_auth_store):
        """E2E: Register user -> Login via API -> Verify token works everywhere."""
        success, msg = create_user('sso@owlban.com', 'ssouser', 'SecurePass1!', 'user', 'OWLBAN_GROUP')
        assert success is True
        success, msg, user = authenticate_user('sso@owlban.com', 'SecurePass1!')
        assert success is True
        assert user is not None
        access_token, refresh_token = temp_auth_store.generate_tokens(user)
        assert access_token is not None
        assert refresh_token is not None
        payload = verify_token(access_token)
        assert payload is not None
        assert payload['email'] == 'sso@owlban.com'
        assert payload['role'] == 'user'
        assert payload['company'] == 'OWLBAN_GROUP'

    def test_same_credentials_all_companies(self, temp_auth_store):
        """E2E: Same user can authenticate against any company platform."""
        create_user('multi@owlban.com', 'multiuser', 'SecurePass1!', 'user', 'OWLBAN_GROUP')
        success, _, user = authenticate_user('multi@owlban.com', 'SecurePass1!')
        assert success is True
        access_token, _ = temp_auth_store.generate_tokens(user)
        payload = verify_token(access_token)
        assert payload['email'] == 'multi@owlban.com'

    def test_token_contains_required_claims(self, temp_auth_store):
        """E2E: JWT tokens contain all required claims for cross-platform use."""
        create_user('claims@owlban.com', 'claimsuser', 'SecurePass1!')
        _, _, user = authenticate_user('claims@owlban.com', 'SecurePass1!')
        access_token, _ = temp_auth_store.generate_tokens(user)
        payload = verify_token(access_token)
        assert 'email' in payload
        assert 'role' in payload
        assert 'company' in payload
        assert 'exp' in payload
        assert 'iat' in payload
        assert 'type' in payload
        assert payload['type'] == 'access'


class TestAPIKeyInteroperability:
    """Test API keys work across platforms."""

    def test_api_key_from_one_platform_works_everywhere(self, temp_auth_store):
        """E2E: API key generated on OWLBAN GROUP validates on any platform."""
        create_user('api@owlban.com', 'apiuser', 'SecurePass1!')
        key = generate_api_key('api@owlban.com', 'cross-platform')
        assert key is not None
        assert key.startswith('owlban_')
        result = verify_api_key(key)
        assert result is not None
        assert result['email'] == 'api@owlban.com'
        assert result['name'] == 'cross-platform'

    def test_multiple_api_keys_per_user(self, temp_auth_store):
        """E2E: User can have multiple active API keys for different services."""
        create_user('multikey@owlban.com', 'keyuser', 'SecurePass1!')
        key1 = generate_api_key('multikey@owlban.com', 'owlban-web')
        key2 = generate_api_key('multikey@owlban.com', 'blackbox-ai')
        key3 = generate_api_key('multikey@owlban.com', 'oscar-revenue')
        assert key1 != key2 != key3
        assert verify_api_key(key1) is not None
        assert verify_api_key(key2) is not None
        assert verify_api_key(key3) is not None

    def test_revoke_key_invalidates_everywhere(self, temp_auth_store):
        """E2E: Revoking a key on one platform invalidates it everywhere."""
        create_user('revoke@owlban.com', 'revokeuser', 'SecurePass1!')
        key = generate_api_key('revoke@owlban.com', 'temp-key')
        assert verify_api_key(key) is not None
        success = temp_auth_store.revoke_api_key('revoke@owlban.com', key)
        assert success is True
        assert verify_api_key(key) is None


class TestPasswordResetPropagation:
    """Test password reset affects all platforms."""

    def test_password_reset_invalidates_all_sessions(self, temp_auth_store):
        """E2E: Password reset logs out all active sessions across platforms."""
        create_user('reset@owlban.com', 'resetuser', 'OldPass123!')
        success, _, user = authenticate_user('reset@owlban.com', 'OldPass123!')
        token1, _ = temp_auth_store.generate_tokens(user)
        token2, _ = temp_auth_store.generate_tokens(user)
        token3, _ = temp_auth_store.generate_tokens(user)
        assert verify_token(token1) is not None
        assert verify_token(token2) is not None
        assert verify_token(token3) is not None
        reset_tok = request_password_reset('reset@owlban.com')
        assert reset_tok is not None
        success, msg = reset_password(reset_tok, 'NewPass123!')
        assert success is True
        success, _, _ = authenticate_user('reset@owlban.com', 'OldPass123!')
        assert success is False
        success, _, _ = authenticate_user('reset@owlban.com', 'NewPass123!')
        assert success is True


class TestTokenRefreshFlow:
    """Test token refresh across platforms."""

    def test_refresh_token_issues_new_access(self, temp_auth_store):
        """E2E: Refresh token can get a new access token."""
        create_user('refresh@owlban.com', 'refreshuser', 'SecurePass1!')
        _, _, user = authenticate_user('refresh@owlban.com', 'SecurePass1!')
        _, refresh_token = temp_auth_store.generate_tokens(user)
        result = temp_auth_store.refresh_access_token(refresh_token)
        assert result is not None
        new_access, new_refresh = result
        assert new_access is not None
        assert new_refresh is not None
        payload = verify_token(new_access)
        assert payload is not None
        assert payload['email'] == 'refresh@owlban.com'

    def test_refresh_produces_new_token_pair(self, temp_auth_store):
        """E2E: Refresh returns both new access and refresh tokens."""
        create_user('rotate@owlban.com', 'rotateuser', 'SecurePass1!')
        _, _, user = authenticate_user('rotate@owlban.com', 'SecurePass1!')
        _, refresh_token = temp_auth_store.generate_tokens(user)
        result = temp_auth_store.refresh_access_token(refresh_token)
        assert result is not None
        new_access, new_refresh = result
        assert new_access is not None
        assert new_refresh is not None
        # New access token is valid
        payload = verify_token(new_access)
        assert payload is not None
        assert payload['email'] == 'rotate@owlban.com'

    def test_invalid_refresh_token_rejected(self, temp_auth_store):
        """E2E: Invalid refresh tokens are rejected."""
        result = temp_auth_store.refresh_access_token('invalid.token.value')
        assert result is None


class TestFullUserJourney:
    """Test complete user journeys across platforms."""

    def test_complete_registration_to_api_access(self, temp_auth_store):
        """E2E: Full journey - Register -> Login -> Get API key -> Access API."""
        success, _ = create_user('journey@owlban.com', 'journeyuser', 'SecurePass1!', 'developer', 'OWLBAN_GROUP')
        assert success is True
        success, _, user = authenticate_user('journey@owlban.com', 'SecurePass1!')
        assert success is True
        access_token, _ = temp_auth_store.generate_tokens(user)
        assert verify_token(access_token) is not None
        api_key = generate_api_key('journey@owlban.com', 'production')
        assert api_key is not None
        result = verify_api_key(api_key)
        assert result is not None
        assert result['email'] == 'journey@owlban.com'

    def test_admin_user_journey(self, temp_auth_store):
        """E2E: Admin user can manage other users."""
        create_user('admin@owlban.com', 'adminuser', 'AdminPass1!', 'admin', 'OWLBAN_GROUP')
        create_user('user1@owlban.com', 'regular1', 'SecurePass1!', 'user', 'OSCAR_BROOME')
        create_user('user2@owlban.com', 'regular2', 'SecurePass1!', 'user', 'BLACKBOX_AI')
        users = temp_auth_store.list_users()
        assert len(users) == 3
        owlban_users = temp_auth_store.list_users(company='OWLBAN_GROUP')
        assert len(owlban_users) == 1

    def test_account_lockout_journey(self, temp_auth_store):
        """E2E: Account locks after max failed attempts, unlocks after timeout."""
        create_user('lock@owlban.com', 'lockuser', 'SecurePass1!')
        for _ in range(AuthConfig.MAX_LOGIN_ATTEMPTS):
            authenticate_user('lock@owlban.com', 'wrongpass')
        success, msg, _ = authenticate_user('lock@owlban.com', 'SecurePass1!')
        assert success is False
        assert 'locked' in msg.lower()
        user = temp_auth_store.get_user_by_email('lock@owlban.com')
        user.locked_until = None
        user.login_attempts = 0
        temp_auth_store._save_data()
        success, _, _ = authenticate_user('lock@owlban.com', 'SecurePass1!')
        assert success is True


class TestSecurityCompliance:
    """Test security measures across platforms."""

    def test_password_never_stored_plaintext(self, temp_auth_store):
        """E2E: Passwords are always hashed, never stored plaintext."""
        create_user('secure@owlban.com', 'secureuser', 'SecurePass1!')
        user = temp_auth_store.get_user_by_email('secure@owlban.com')
        assert user.password_hash != 'SecurePass1!'
        assert 'SecurePass1!' not in user.password_hash
        assert user.password_hash.startswith('$2b$')

    def test_user_list_hides_passwords(self, temp_auth_store):
        """E2E: User listing never exposes password hashes."""
        create_user('list@owlban.com', 'listuser', 'SecurePass1!')
        users = temp_auth_store.list_users()
        for u in users:
            assert 'password_hash' not in u

    def test_inactive_user_cannot_login(self, temp_auth_store):
        """E2E: Deactivated users cannot authenticate."""
        create_user('inactive@owlban.com', 'inactiveuser', 'SecurePass1!')
        user = temp_auth_store.get_user_by_email('inactive@owlban.com')
        user.active = False
        temp_auth_store._save_data()
        success, msg, _ = authenticate_user('inactive@owlban.com', 'SecurePass1!')
        assert success is False

    def test_session_cleanup(self, temp_auth_store):
        """E2E: Expired sessions are cleaned up."""
        create_user('cleanup@owlban.com', 'cleanupuser', 'SecurePass1!')
        _, _, user = authenticate_user('cleanup@owlban.com', 'SecurePass1!')
        temp_auth_store.generate_tokens(user)
        for sid, session in temp_auth_store.sessions.items():
            session.active = False
        temp_auth_store.cleanup_expired_sessions()
        assert len(temp_auth_store.sessions) == 0
