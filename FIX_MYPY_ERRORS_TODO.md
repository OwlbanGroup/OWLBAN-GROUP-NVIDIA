# Fix Mypy Errors TODO

## Issues to Fix

1. **Blueprint assignment errors** (lines 51, 60, 68, 76, 84, 92, 100, 108, 116)
   - Fix by adding Optional type hints and proper initialization

2. **Config attributes** - Ensure proper import of Config class
   - Fixed by adding **init** method with instance attributes to config.py

3. **Undefined "g" variable** (line ~1002)
   - Add import for `g` from Flask

4. **Undefined "sync_service"** (lines ~1567, 1588, 1612)
   - Add import for `sync_service` from src.sync_service

## Implementation Steps

- [x] 1. Add `Optional` import from typing
- [x] 2. Add `Blueprint` import from flask_restx
- [x] 3. Fix blueprint variable declarations with Optional[Blueprint] type hints
- [x] 4. Add `g` import from flask
- [x] 5. Add `sync_service` import from src.sync_service
- [x] 6. Add **init** method to Config class with instance attributes
- [x] 7. Verify fixes resolve all mypy errors

## Summary of Changes Made to app_final.py

1. Added `from typing import Optional` import
2. Added `g` to Flask import: `from flask import Flask, request, jsonify, render_template, g`
3. Added `Blueprint` to flask_restx import: `from flask_restx import Api, Blueprint`
4. Added `from src.sync_service import sync_service` import
5. Added type annotations to all 9 blueprint variables:
   - pfm_bp: Optional[Blueprint] = None
   - payments_bp: Optional[Blueprint] = None
   - payroll_bp: Optional[Blueprint] = None
   - user_bp: Optional[Blueprint] = None
   - asset_bp: Optional[Blueprint] = None
   - business_bp: Optional[Blueprint] = None
   - ml_bp: Optional[Blueprint] = None
   - data_bp: Optional[Blueprint] = None
   - ai_bp: Optional[Blueprint] = None

## Summary of Changes Made to config.py

1. Added `__init__` method with all configuration attributes as instance attributes
2. Changed classmethods to instance methods (get_database_url, get_jpmorgan_endpoint_url, get_all_settings)
3. Added proper type annotations to all instance attributes
4. Global `config = Config()` instance at the end of the file

## Errors Fixed

- 9 "Incompatible types in assignment (expression has type 'None', variable has type 'Blueprint')" errors
- 1 "g is not defined" error
- 3 "sync_service is not defined" errors
- Multiple "Config has no attribute" errors (get_all_settings, TOKEN_CLIENT_ID, TOKEN_CLIENT_SECRET, TOKEN_URL, TOKEN_SCOPE, REDIS_URL, LOG_LEVEL)

## All Mypy Errors Resolved

All mypy errors in app_final.py have been resolved by:

1. Adding proper imports and type hints in app_final.py
2. Updating config.py to have all attributes as instance attributes with proper type annotations
