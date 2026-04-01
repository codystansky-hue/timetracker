$pythonPath = "$PSScriptRoot\venv\Scripts\python.exe"
$scriptPath = "$PSScriptRoot\TimeTracker.py"

Start-Process -FilePath $pythonPath -ArgumentList $scriptPath -WorkingDirectory $PSScriptRoot -WindowStyle Hidden

Write-Host "TimeTracker server started on http://127.0.0.1:5001"
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5001"
