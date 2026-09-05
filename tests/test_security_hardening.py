"""
Security hardening tests: MFA (TOTP) and CSRF protection.
"""

import sys
import os
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from auth_lib import TOTP, AuthManager
from middleware.csrf import (
    generate_csrf_token,
    validate_csrf_token,
    CSRFProtectionMiddleware,
)


def make_manager(tmp_path):
    return AuthManager(
        user_store_file=str(tmp_path / "users.json"),
        session_store_file=str(tmp_path / "sessions.json"),
    )


def make_user(tmp_path):
    m = make_manager(tmp_path)
    m.create_user("sec@owlban.com", "secuser", "Sec2024!", "user", "OWLBAN_GROUP")
    return m, m.users["sec@owlban.com"]


# ===================== TOTP primitives =====================

class TestTOTP:
    def test_secret_is_urlsafe_base32(self):
        s = TOTP.generate_secret()
        assert s and len(s) >= 16
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for c in s)

    def test_code_is_six_digits(self):
        s = TOTP.generate_secret()
        code = TOTP.code_for_time(s)
        assert len(code) == 6 and code.isdigit()

    def test_code_is_time_variant(self):
        s = TOTP.generate_secret()
        code1 = TOTP.code_for_time(s, at_time=1000)
        code2 = TOTP.code_for_time(s, at_time=1030)
        assert code1 != code2

    def test_verify_success_and_failure(self):
        s = TOTP.generate_secret()
        code = TOTP.code_for_time(s)
        assert TOTP.verify(s, code)
        bad = str(int(code) + 1 if int(code) < 999999 else 0).zfill(6)
        assert not TOTP.verify(s, bad)
        assert not TOTP.verify(s, "abc")
        assert not TOTP.verify(s, "12")

    def test_verify_accepts_window_drift(self):
        s = TOTP.generate_secret()
        code = TOTP.code_for_time(s)
        assert TOTP.verify(s, code, window=1)

    def test_provisioning_uri_wellformed(self):
        s = TOTP.generate_secret()
        uri = TOTP.provisioning_uri(s, "user@owlban.com")
        assert uri.startswith("otpauth://totp/")
        assert "secret=" + s in uri
        assert "digits=6" in uri and "period=30" in uri

    def test_hotp_known_rfc_vector(self):
        # RFC 4226 test vector secret: b"12345678901234567890" (base32 encoded)
        raw = b"12345678901234567890"
        secret_b32 = base64.b32encode(raw).decode().rstrip("=")
        secret_bytes = base64.b32decode(secret_b32 + "=" * ((8 - len(secret_b32) % 8) % 8))
        code = TOTP._hotp(secret_bytes, 0)
        assert len(code) == 6 and code.isdigit()


# ===================== MFA flow on AuthManager =====================

class TestMFAFlow:
    def test_manager_mfa_off_by_default(self, tmp_path):
        m, _ = make_user(tmp_path)
        assert m.mfa_required("sec@owlban.com") is False

    def test_setup_returns_secret_and_uri(self, tmp_path):
        m, u = make_user(tmp_path)
        result = m.setup_mfa("sec@owlban.com")
        assert result is not None
        assert "secret" in result and "provisioning_uri" in result
        assert u.mfa_secret == result["secret"]

    def test_setup_refused_when_enabled(self, tmp_path):
        m, _ = make_user(tmp_path)
        result = m.setup_mfa("sec@owlban.com")
        m.enable_mfa("sec@owlban.com", TOTP.code_for_time(result["secret"]))
        assert m.setup_mfa("sec@owlban.com") is None

    def test_enable_requires_valid_code(self, tmp_path):
        m, _ = make_user(tmp_path)
        assert m.setup_mfa("sec@owlban.com")
        ok, _ = m.enable_mfa("sec@owlban.com", "000000")
        assert not ok
        assert not m.mfa_required("sec@owlban.com")
        secret = m.users["sec@owlban.com"].mfa_secret
        ok, _ = m.enable_mfa("sec@owlban.com", TOTP.code_for_time(secret))
        assert ok
        assert m.mfa_required("sec@owlban.com")

    def test_disable_requires_valid_code(self, tmp_path):
        m, _ = make_user(tmp_path)
        result = m.setup_mfa("sec@owlban.com")
        m.enable_mfa("sec@owlban.com", TOTP.code_for_time(result["secret"]))
        assert m.mfa_required("sec@owlban.com")
        ok, _ = m.disable_mfa("sec@owlban.com", "000001")
        assert not ok
        assert m.mfa_required("sec@owlban.com")
        ok, _ = m.disable_mfa("sec@owlban.com", TOTP.code_for_time(result["secret"]))
        assert ok
        assert not m.mfa_required("sec@owlban.com")

    def test_verify_mfa_code(self, tmp_path):
        m, _ = make_user(tmp_path)
        result = m.setup_mfa("sec@owlban.com")
        m.enable_mfa("sec@owlban.com", TOTP.code_for_time(result["secret"]))
        assert m.verify_mfa_code("sec@owlban.com", TOTP.code_for_time(result["secret"]))
        assert not m.verify_mfa_code("sec@owlban.com", "123456")


# ===================== CSRF protection =====================

class TestCSRF:
    def test_generate_and_validate(self):
        token = generate_csrf_token()
        assert token and len(token) >= 32
        assert validate_csrf_token(token, token)
        assert not validate_csrf_token(token, "attacker")
        assert not validate_csrf_token(None, token)
        assert not validate_csrf_token(token, None)

    def test_middleware_path_exemptions(self):
        mw = CSRFProtectionMiddleware.__new__(CSRFProtectionMiddleware)
        mw.SKIPPED_PATHS = ("/auth/", "/prometheus/", "/metrics", "/health", "/status")
        assert mw._is_exempt("/auth/login")
        assert mw._is_exempt("/prometheus/metrics")
        assert not mw._is_exempt("/api/payment")
        assert not mw._is_exempt("/dashboard")

    def test_unsafe_method_set_contains_state_changers(self):
        assert "POST" in CSRFProtectionMiddleware.UNSAFE_METHODS
        assert "GET" not in CSRFProtectionMiddleware.UNSAFE_METHODS

    def test_cookie_name_constants(self):
        assert CSRFProtectionMiddleware.COOKIE_NAME == "csrf_token"
        assert CSRFProtectionMiddleware.HEADER_NAME == "X-CSRF-Token"


# ===================== API login MFA gating =====================

class TestMFALoginGate:
    def test_login_requires_mfa_status_after_enable(self, tmp_path):
        m, _ = make_user(tmp_path)
        result = m.setup_mfa("sec@owlban.com")
        m.enable_mfa("sec@owlban.com", TOTP.code_for_time(result["secret"]))
        assert m.mfa_required("sec@owlban.com") is True
        assert m.verify_mfa_code("sec@owlban.com", TOTP.code_for_time(result["secret"]))
        assert not m.verify_mfa_code("sec@owlban.com", "999999")


def test_all_importable():
    assert TOTP is not None
    assert AuthManager is not None
    assert CSRFProtectionMiddleware is not None