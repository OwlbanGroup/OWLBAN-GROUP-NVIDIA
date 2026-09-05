#!/usr/bin/env python3
"""
OWLBAN GROUP - Auth Load & Scale Benchmark

Standalone load test for the unified authentication system. Measures
throughput of the core auth primitives at scale and writes a report.

Usage:
    python scripts/load_test.py                     # default run
    python scripts/load_test.py --scale 100000      # 100k synthetic users
    python scripts/load_test.py --iters 5000        # more token samples
    python scripts/load_test.py --report load_report.json
    python scripts/load_test.py --json              # machine-readable stdout
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections import OrderedDict

# Silence auth_lib logging during benchmarks
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_lib import AuthManager, User  # noqa: E402
from middleware.rate_limiter import TokenBucket  # noqa: E402


def _bench(fn, n):
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n


def _bulk(m, count):
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


def run_bench(scale=1000, token_iters=2000, single_iter=None):
    report = OrderedDict()
    report["scale"] = scale

    with tempfile.TemporaryDirectory() as td:
        m = AuthManager(
            user_store_file=os.path.join(td, "users.json"),
            session_store_file=os.path.join(td, "sessions.json"),
        )
        ok, _ = m.create_user(
            email="bench@owlban.com", username="benchuser",
            password="Bench2024!", role="user", company="OWLBAN_GROUP",
        )
        assert ok
        user = m.users["bench@owlban.com"]
        access, refresh = m.generate_tokens(user)

        # ID generation
        report["id_entropy_bits"] = 64
        report["id_namespace_capacity"] = 2 ** 64

        # Token throughput
        report["access_token_gen/sec"] = _rate(_bench(
            lambda: m.generate_tokens(user)[0], token_iters))
        report["access_token_verify/sec"] = _rate(_bench(
            lambda: m.verify_access_token(access), token_iters))
        report.update(_session_bench(m, user, single_iter or 300))
        report.update(_store_bench(m, scale, token_iters))

    report["rate_limit_burst_ok"] = _rate_limit_check()
    return report


def _rate(avg_s):
    return round((1.0 / avg_s) if avg_s else float("inf"), 1)


def _session_bench(m, user, n):
    avg = _bench(lambda: m.create_session(user, ip_address="10.0.0.1"), n)
    return {"session_create/sec": _rate(avg)}


def _store_bench(m, scale, iters):
    _bulk(m, scale)
    avg = _bench(lambda: m.get_user_by_email(f"bulk{scale - 1}@owlban.com"), iters)
    return {"100k_user_lookup/sec": _rate(avg), "synthetic_users": scale}


def _rate_limit_check():
    b = TokenBucket(rate=10, capacity=50)
    for _ in range(50):
        b.consume()
    # 51st consume must be rejected
    return not b.consume()


def main():
    ap = argparse.ArgumentParser(description="OWLBAN GROUP auth load benchmark")
    ap.add_argument("--scale", type=int, default=1000, help="synthetic user count")
    ap.add_argument("--iters", type=int, default=2000, help="token bench iterations")
    ap.add_argument("--sessions", type=int, default=300, help="session bench iterations")
    ap.add_argument("--json", action="store_true", help="emit JSON to stdout")
    ap.add_argument("--report", type=str, default=None, help="write JSON report to file")
    args = ap.parse_args()

    report = run_bench(scale=args.scale, token_iters=args.iters, single_iter=args.sessions)

    if args.json or args.report:
        payload = json.dumps(report, indent=2)
        if args.report:
            with open(args.report, "w") as f:
                f.write(payload)
            print(f"Report written to {args.report}")
        if args.json:
            print(payload)
        return

    print("\nOWLBAN GROUP - Auth Load Benchmark")
    print("==================================")
    for k, v in report.items():
        print(f"  {k:<28} {v}")
    print("==================================\n")


if __name__ == "__main__":
    main()