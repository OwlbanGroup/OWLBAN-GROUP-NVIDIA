"""
Tests for the OWLBAN GROUP auth Prometheus metrics collector.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.auth_metrics import AuthMetrics


def make_metrics():
    return AuthMetrics()


def test_render_helpers_present():
    m = make_metrics()
    out = m.render()
    assert "# HELP owlban_auth_login_total" in out
    assert "# TYPE owlban_auth_active_sessions gauge" in out
    assert "# TYPE owlban_auth_token_generated_total counter" in out


def test_record_login_renders_with_labels():
    m = make_metrics()
    m.record_login("success", "OWLBAN_GROUP")
    m.record_login("invalid_credentials", "OSCAR_BROOME")
    out = m.render()
    assert 'owlban_auth_login_total{outcome="success",company="OWLBAN_GROUP"} 1' in out
    assert 'owlban_auth_login_total{outcome="invalid_credentials",company="OSCAR_BROOME"} 1' in out


def test_token_and_refresh_counters():
    m = make_metrics()
    m.record_token_generated("access")
    m.record_token_generated("refresh")
    m.record_token_refreshed()
    out = m.render()
    assert 'owlban_auth_token_generated_total{type="access"} 1' in out
    assert 'owlban_auth_token_generated_total{type="refresh"} 1' in out
    assert "owlban_auth_token_refreshed_total 1" in out


def test_gauges_render_as_integers():
    m = make_metrics()
    m.set_active_sessions(7)
    m.set_account_lockouts(2)
    m.set_api_keys_active(3)
    out = m.render()
    assert "owlban_auth_active_sessions 7" in out
    assert "owlban_auth_account_lockouts 2" in out
    assert "owlban_auth_api_keys_active 3" in out


def test_rate_limit_and_audit_recorded():
    m = make_metrics()
    m.record_rate_limit("/auth/login")
    m.record_lockout()
    m.record_audit_event("user_login", "info")
    out = m.render()
    assert 'owlban_auth_rate_limit_total{action="/auth/login"} 1' in out
    assert "owlban_auth_lockout_total 1" in out
    assert 'owlban_auth_audit_events_total{event_type="user_login",severity="info"} 1' in out


def test_counters_increment():
    m = make_metrics()
    m.record_login("success")
    m.record_login("success")
    m.record_login("success")
    out = m.render()
    assert 'owlban_auth_login_total{outcome="success",company="OWLBAN_GROUP"} 3' in out


def test_thread_safe_increments():
    import threading
    m = make_metrics()
    results = []

    def worker():
        for _ in range(100):
            m.record_login("success")
            m.record_token_generated("access")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    out = m.render()
    assert 'owlban_auth_login_total{outcome="success",company="OWLBAN_GROUP"} 400' in out
    assert 'owlban_auth_token_generated_total{type="access"} 400' in out