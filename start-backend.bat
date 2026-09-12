@echo off
echo Starting Alpha Harness Backend on http://localhost:8000 ...
cd /d "%~dp0backend"
python -m uv run uvicorn alpha_harness.main:app --port 8000 --reload
pause
