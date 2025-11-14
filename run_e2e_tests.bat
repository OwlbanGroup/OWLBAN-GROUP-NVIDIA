@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
pip install -r requirements.txt
python -m pytest e2e_test.py -v --tb=short
pause
