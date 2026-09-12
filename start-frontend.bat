@echo off
echo Starting Alpha Harness Frontend on http://localhost:5173 ...
cd /d "%~dp0frontend"
pnpm dev
pause
