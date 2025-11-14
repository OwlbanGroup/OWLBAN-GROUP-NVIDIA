@echo off
echo Running unit tests...

cd jpmorgan_financial_apis

python -m unittest test_security.py

if %errorlevel% neq 0 (
    echo Unit tests failed.
    exit /b 1
)

echo Unit tests completed successfully!
