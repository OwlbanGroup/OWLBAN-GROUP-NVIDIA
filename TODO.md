# TODO - Internal Operations Orchestration (Payroll + Personal Banking + Company Bills)

- [x] Analyze existing payroll, PFM, payments, and app registration files
- [x] Confirm implementation plan with user
- [x] Add `blueprints/internal_ops.py` with unified orchestration endpoint(s)
- [x] Register `internal_ops` blueprint in `app_final.py`
- [x] Add/extend unit tests in `tests/test_phase8_units.py`
- [x] Update `README.md` with internal operations usage examples
- [ ] Fix `/internal-ops/execute` happy-path behavior for payroll-only minimal payload
- [ ] Run relevant tests and summarize outcomes

## Execution Plan (Approved)
1. Introduce a dedicated internal operations blueprint that orchestrates:
   - Internal team payroll processing
   - Internal personal banking account/budget/bill interactions
   - Company bill payment execution
2. Wire this blueprint into `app_final.py` with a stable URL prefix.
3. Add tests for success and validation/error scenarios.
4. Document request/response examples in README.
