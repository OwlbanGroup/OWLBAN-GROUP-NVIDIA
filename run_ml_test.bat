@echo off
cd /d %~dp0
set TOKEN_CLIENT_ID=test
set SECRET_KEY=test-secret-key
set ALLOW_MISSING_TOKENS=true
python test_ml_improvements.py
pause
