$repoRoot = $PSScriptRoot
$pythonPath = "$repoRoot\venv\Scripts\python.exe"
$scriptPath = "$repoRoot\TimeTracker.py"

# Bootstrap venv if it doesn't exist (e.g. fresh clone on a new machine)
if (-not (Test-Path $pythonPath)) {
    Write-Host "Setting up virtual environment..."
    python -m venv "$repoRoot\venv"
    & "$repoRoot\venv\Scripts\pip.exe" install -r "$repoRoot\requirements.txt"
}

Start-Process -FilePath $pythonPath -ArgumentList $scriptPath -WorkingDirectory $repoRoot -WindowStyle Hidden

Write-Host "TimeTracker server started on http://127.0.0.1:5001"
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5001"
