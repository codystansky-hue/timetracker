@echo off
:: Re-launch as Administrator if not already elevated
net session >nul 2>&1
if %errorLevel% neq 0 (
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: Run the PowerShell start script from this file's directory
powershell -ExecutionPolicy Bypass -File "%~dp0timetracker-start.ps1"
pause
