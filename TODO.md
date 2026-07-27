# TODO - Docker-unavailable graceful handling for model runner tests

- [x] Analyze failing test output and identify root cause (Docker daemon unavailable).
- [ ] Update `tests/test_model_runner.py` fixture to skip when Docker is unavailable.
- [ ] Run `pytest tests/test_model_runner.py -q --no-cov` and verify behavior.
- [ ] Summarize results and remaining optional coverage.
