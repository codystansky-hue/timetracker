# Self-elevate to Administrator if not already elevated
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

$repoRoot = $PSScriptRoot
$pythonPath = "$repoRoot\venv\Scripts\python.exe"
$scriptPath = "$repoRoot\TimeTracker.py"

# Bootstrap venv if it doesn't exist
if (-not (Test-Path $pythonPath)) {
    Write-Host "Setting up virtual environment..."
    python -m venv "$repoRoot\venv"
    & "$repoRoot\venv\Scripts\pip.exe" install -r "$repoRoot\requirements.txt"
}

Start-Process -FilePath $pythonPath -ArgumentList $scriptPath -WorkingDirectory $repoRoot -WindowStyle Hidden

Write-Host "TimeTracker server started on http://127.0.0.1:5001"
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5001"
