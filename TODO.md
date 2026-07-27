# Coverage Boost TODO (>=78%)

- [x] Review existing coverage-focused tests for easy missed branches in `app_final.py`
- [ ] Add targeted tests in `tests/test_app_final_coverage_boost.py` for:
  - [ ] GitHub API status mapping branches (`/api/github/orgs` and `/api/github/repos` for 401/404/502/other)
  - [ ] GitHub API `requests.RequestException` error branch
  - [ ] lightweight deterministic fallback/error branch in dashboard or sync path
- [ ] Run targeted test file first without coverage:
  - `pytest -q tests/test_app_final_coverage_boost.py --no-cov`
- [ ] Run full required coverage gate:
  - `pytest -q tests/test_audit_logger_additional_coverage.py tests/test_app_final_mass_coverage.py tests/test_app_final_coverage_boost.py tests/test_audit_logging.py tests/test_revenue_service.py --cov=app_final --cov=src.audit_logger --cov=src.models.audit_log --cov=src.revenue_service --cov-report=term-missing --cov-fail-under=78`
- [ ] If still below threshold, add one more focused test for highest-yield uncovered branch and re-run
