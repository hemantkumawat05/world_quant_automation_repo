@echo off
echo Starting Alpha Harness (Backend + Frontend)...
start "Alpha Harness Backend" "%~dp0start-backend.bat"
start "Alpha Harness Frontend" "%~dp0start-frontend.bat"
echo Services are starting in separate terminal windows.
echo Frontend will be accessible at: http://localhost:5173
echo Backend API will be accessible at: http://localhost:8000
