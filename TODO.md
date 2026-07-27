# Test Stabilization TODO

- [x] Investigate failing suites and identify root causes from traceback
- [x] Inspect relevant test files and application/blueprint code
- [x] Draft and confirm remediation plan with user

## Implementation Steps
- [x] Refactor `tests/test_mcp_tools.py` to avoid hard dependency on `localhost:8080`
- [x] Fix `tests/test_pfm_full_coverage.py` misplaced tests/signatures causing runtime 500/mispatch issues
- [ ] Patch `blueprints/pfm.py` runtime 500 paths (`/accounts/link` + safe bill date rollover)
- [ ] Evaluate `tests/test_model_runner.py` and add conditional Docker skip if daemon unavailable (if needed)
- [ ] Run targeted pytest for updated files
- [ ] Summarize fixes and remaining environment-only failures
