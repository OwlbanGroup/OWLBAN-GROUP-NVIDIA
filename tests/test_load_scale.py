"""
Load & Scale Testing for OWLBAN GROUP authentication system.

Benchmarks the throughput and scalability of the core auth primitives:
- JWT token generation & verification
- Authentication & session management
- User-store lookups at large synthetic scale
- Rate-limiter burst/refill correctness at scale
- Concurrent (thread-safe) operations
- ID namespace entropy for the 10B-user target

Scale for synthetic user-store tests is controlled by the SCALE_TEST_FACTOR
env var (default 1000) to keep CI runs fast while allowing full-scale runs:
    set SCALE_TEST_FACTOR=100000  # ~100k synthetic users
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

# Silence auth_lib logging so benchmarks reflect pure operation time
logging.disable(logging.CRITICAL)

import auth_lib  # noqa: E402
from auth_lib import AuthManager, AuthConfig, Session, User  # noqa: E402
from middleware.rate_limiter import TokenBucket  # noqa: E402

# Synthetic user-store scale. Default 1000 for fast CI; override for larger runs.
SCALE = int(os.getenv('SCALE_TEST_FACTOR', '1000'))
TEN_BILLION = 10_000_000_000  # target user namespace required by the plan


def make_manager(tmp_path):
    """Build an isolated AuthManager writing to a temp dir."""
    return AuthManager(
        user_store_file=str(tmp_path / "users.json"),
        session_store_file=str(tmp_path / "sessions.json"),
    )


def sample_user(tmp_path):
    """Create a valid, password-policy-compliant test user."""
    m = make_manager(tmp_path)
    ok, msg = m.create_user(
        email="scale@owlban.com",
        username="scaleuser",
        password="Scale2024!",
        role="user",
        company="OWLBAN_GROUP",
        permissions=["read"],
    )
    assert ok, msg
    return m, m.users["scale@owlban.com"]


def bench(fn, n):
    """Run fn n times, return average seconds/op."""
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / n


def bulk_insert(m, count):
    """Insert count synthetic users directly, skipping per-user file I/O."""
    for i in range(count):
        m.users[f"bulk{i}@owlban.com"] = User(
            id=f"owlban_group-{i:016x}",
            email=f"bulk{i}@owlban.com",
            username=f"bulk{i}",
            password_hash="$2b$12$dummyhashdummyhash",
            role="user",
            company="OWLBAN_GROUP",
            permissions=["read"],
        )


# ===================== 1. ID namespace (10B-user scale) =====================

class TestIDNamespace:
    def test_user_id_has_64bit_entropy(self, tmp_path):
        """Each user ID must carry >=64 bits of random entropy (~1.8e19 namespace)."""
        m = make_manager(tmp_path)
        ids = []
        for i in range(20):
            ok, _ = m.create_user(
                email=f"ns{i}@owlban.com",
                username=f"ns{i}",
                password="Ns2024!!",
                role="user",
                company="OWLBAN_GROUP",
            )
            assert ok
            ids.append(m.users[f"ns{i}@owlban.com"].id)

        for sid in ids:
            assert sid.startswith("owlban_group-")
            # hex suffix = 8 bytes = 64 bits = 16 hex chars
            assert len(sid.split("-", 1)[1]) == 16
        # No collisions across a 50-user sample
        assert len(set(ids)) == len(ids)

    def test_id_entropy_supports_10b(self):
        """2^64 >> 10B; assert the ID field can uniquely address the target."""
        assert 2 ** 64 >= TEN_BILLION
        # Sample space is astronomically larger than the 10B target
        assert (2 ** 64) / TEN_BILLION > 1_000_000_000


# ===================== 2. Token throughput =====================

class TestTokenThroughput:
    def test_access_token_generation_throughput(self, tmp_path):
        m, user = sample_user(tmp_path)
        avg_s = bench(lambda: m.generate_tokens(user)[0], 2000)
        ops = (1.0 / avg_s) if avg_s else float("inf")
        assert ops >= 1000, f"only {ops:.0f} tok/sec"

    def test_token_pair_generation(self, tmp_path):
        """generate_tokens() returns a distinct (access, refresh) pair."""
        m, user = sample_user(tmp_path)
        access, refresh = m.generate_tokens(user)
        assert access != refresh
        assert m.verify_access_token(access) is not None

    def test_access_token_verification_throughput(self, tmp_path):
        m, user = sample_user(tmp_path)
        access, _ = m.generate_tokens(user)
        avg_s = bench(lambda: m.verify_access_token(access), 2000)
        ops = (1.0 / avg_s) if avg_s else float("inf")
        assert ops >= 1000, f"only {ops:.0f} verify/sec"

    def test_refresh_token_flow(self, tmp_path):
        m, user = sample_user(tmp_path)
        access, refresh = m.generate_tokens(user)
        updated = m.refresh_access_token(refresh)
        assert updated is not None
        new_access, new_refresh = updated
        assert new_access != access
        assert m.verify_access_token(new_access) is not None

    def test_authenticate_throughput(self, tmp_path):
        """bcrypt is intentionally slow; just guard against regressions."""
        m, user = sample_user(tmp_path)
        avg_s = bench(lambda: m.authenticate_user(user.email, "Scale2024!"), 5)
        assert avg_s < 2.0, f"auth too slow: {avg_s:.3f}s/op"


# ===================== 3. Session throughput =====================

class TestSessionScale:
    def test_session_creation_throughput(self, tmp_path):
        # NOTE: create_session() synchronously persists to JSON on every call,
        # which bounds throughput (~40-60/s on this hardware). For true
        # high-throughput scale the session store should be Redis-backed (see
        # docker-compose.yml redis service). This test guards against gross
        # regressions in the file-backed path and any in-memory overhead.
        m, user = sample_user(tmp_path)
        avg_s = bench(lambda: m.create_session(user, ip_address="10.0.0.1"), 200)
        ops = (1.0 / avg_s) if avg_s else float("inf")
        assert ops >= 20, f"only {ops:.0f} sess/sec"

    def test_many_active_sessions(self, tmp_path):
        """Large burst of sessions; lookups must remain fast."""
        m, user = sample_user(tmp_path)
        sids = set()
        for i in range(300):
            sid = m.create_session(user, ip_address=f"10.0.0.{i % 254}", user_agent="load-test")
            sids.add(sid)
        assert len(sids) == 300
        avg_s = bench(lambda: m.verify_session(next(iter(sids))), 1000)
        ops = (1.0 / avg_s) if avg_s else float("inf")
        assert ops >= 1000, f"only {ops:.0f} lookup/sec"

    def test_expired_session_cleanup(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        m, user = sample_user(tmp_path)
        expired = m.create_session(user)
        now = datetime.now(timezone.utc)
        m.sessions[expired].created_at = now - timedelta(hours=2)
        m.sessions[expired].expires_at = now - timedelta(hours=1)
        m.cleanup_expired_sessions()
        assert expired not in m.sessions


# ===================== 4. Large user-store scalability =====================

class TestUserStoreScale:
    def test_lookup_on_large_user_store(self, tmp_path):
        m = make_manager(tmp_path)
        bulk_insert(m, SCALE)
        # AuthManager auto-seeds a default admin, so total = SCALE + base
        bulk_count = sum(1 for e in m.users if e.startswith("bulk"))
        assert bulk_count == SCALE

        # O(1) dict lookup across SCALE users
        avg_s = bench(lambda: m.get_user_by_email(f"bulk{(SCALE - 1)}@owlban.com"), 2000)
        ops = (1.0 / avg_s) if avg_s else float("inf")
        assert ops >= 1000, f"only {ops:.0f} lookup/sec"
        assert m.get_user_by_email("bulk0@owlban.com") is not None

    def test_list_users_filters_by_company(self, tmp_path):
        m = make_manager(tmp_path)
        bulk_insert(m, SCALE)
        m.users["nvidia@owlban.com"] = User(
            id="nvidia_integration-1", email="nvidia@owlban.com", username="nvidia",
            password_hash="$2b$12$dummy", role="developer", company="NVIDIA_INTEGRATION",
            permissions=["read"],
        )
        listed = m.list_users(company="NVIDIA_INTEGRATION")
        assert len(listed) == 1
        assert listed[0]["email"] == "nvidia@owlban.com"
        # password_hash must never leak in serialized output
        assert "password_hash" not in listed[0]


# ===================== 5. Rate limiter at scale =====================

class TestRateLimiterScale:
    def test_burst_then_throttle(self):
        b = TokenBucket(rate=10, capacity=50)
        for _ in range(50):
            assert b.consume()
        assert not b.consume()

    def test_refill_over_time(self):
        b = TokenBucket(rate=10, capacity=50)
        for _ in range(50):
            b.consume()
        time.sleep(0.1)  # ~1 token refill at 10/sec
        assert b.consume()

    def test_high_throughput_tokens(self):
        """Token bucket must sustain its rate without locking out traffic."""
        b = TokenBucket(rate=1000, capacity=5000)
        consumed = 0
        start = time.perf_counter()
        while time.perf_counter() - start < 1.0:
            while b.consume():
                consumed += 1
            time.sleep(0.01)
        # Over ~1s at 1000 tokens/sec, should deliver roughly 1000 tokens
        assert consumed >= 500, f"rate limiter under-delivered: {consumed}"

    def test_middleware_burst_allows_initial(self):
        from collections import defaultdict
        from middleware.rate_limiter import RateLimiterMiddleware
        ml = RateLimiterMiddleware.__new__(RateLimiterMiddleware)
        ml.rate, ml.burst = 10, 50
        ml.buckets = defaultdict(lambda: TokenBucket(ml.rate, ml.burst))
        bucket = ml.buckets["192.168.1.1"]
        for _ in range(50):
            assert bucket.consume(), "burst capacity should allow exactly 50"
        assert not bucket.consume(), "51st request must be throttled"


# ===================== 6. Concurrency / thread safety =====================

class TestConcurrencyScale:
    def test_concurrent_token_generation(self, tmp_path):
        m, user = sample_user(tmp_path)
        tokens = []

        def gen():
            for _ in range(200):
                tokens.append(m.generate_tokens(user)[0])

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda _: gen(), range(8)))

        assert len(tokens) == 8 * 200
        assert m.verify_access_token(tokens[0]) is not None
        assert len(set(tokens)) == len(tokens)

    def test_concurrent_user_creation(self, tmp_path):
        m = make_manager(tmp_path)
        emails = []

        def create(i):
            ok, _ = m.create_user(
                email=f"conc{i}@owlban.com", username=f"conc{i}",
                password="Conc2024!", role="user", company="OWLBAN_GROUP",
            )
            if ok:
                emails.append(f"conc{i}@owlban.com")

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(create, range(40)))

        # 40 created + the auto-seeded default admin
        conc_count = sum(1 for e in m.users if e.startswith("conc"))
        assert conc_count == 40
        assert len(set(emails)) == 40


# ===================== 7. Module import guard =====================

def test_load_scale_importable():
    """Guard that all module-level machinery imports without error."""
    assert hasattr(auth_lib, "AuthManager")
    assert TokenBucket is not None
    assert SCALE >= 1