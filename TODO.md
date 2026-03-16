# E2E Perfection Execution TODO

- [x] Review core app contracts in `app.py` and `app_async.py`
- [x] Review E2E harness in `comprehensive_e2e_test.py` and detect contract mismatches
- [x] Review supporting runner in `test_runner.py`
- [ ] Refactor `comprehensive_e2e_test.py` to align with FastAPI async app contracts and stable status codes
- [ ] Run targeted comprehensive E2E test script and capture failures
- [ ] Patch remaining mismatches in tests (and app only if truly defective)
- [ ] Re-run until comprehensive E2E passes consistently
- [ ] Run broader validation (`pytest`) for regression confidence
- [ ] Finalize and report verified end-to-end readiness
