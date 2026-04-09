# JPMorgan Financial APIs - 100% Test Coverage Plan
## Status: [IN PROGRESS] 33% → 100%

## Step 1: Import Fixes (Current)
- [x] tests/conftest.py: module-level sys.path
- [x] pyproject.toml: pythonpath + --cov-fail-under=100
- [ ] Verify: python -c "from blueprints.pfm import pfm_bp"

## Step 2: test_phase8_units.py (50+ NEW tests)
1. Mock src.logger/auth (MagicMock)
2. Test categorize_transaction (25 cases) 
3. Test ALL 50+ @pfm_bp.route(): POST/GET each endpoint
4. Branch coverage: every if/try path
5. Edge cases: validation errors, empty data

## Step 3: Run Coverage
```
cd jpmorgan_financial_apis
pytest tests/ -v --cov-fail-under=100 --cov-branch --cov-report=html
```

## Step 4: Full Project (All Blueprints)
- Expand tests/test_new_modules.py
- tests/conftest.py test_app covers all blueprints
- Exclude: venv/, backups/, docs/

## Step 5: Production
```
docker compose -f docker-compose.production.yml up -d --build
pytest tests/ --cov-fail-under=100  # Must pass
```

## Progress Tracker
```
[ ] 50% (imports fixed)
[ ] 80% (pfm.py full coverage) 
[ ] 100% (all blueprints + src)
```

