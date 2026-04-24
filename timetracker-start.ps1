$repoRoot   = $PSScriptRoot
$pythonPath = "$repoRoot\venv\Scripts\python.exe"
$scriptPath = "$repoRoot\TimeTracker.py"
$url        = "http://127.0.0.1:5001"
$logPath    = "$repoRoot\timetracker.log"
$errLogPath = "$repoRoot\timetracker-error.log"
$maxWait    = 30   # seconds to wait for server to come up

# --- 1. Already running? Open the browser and bail. ---
$portInUse = $false
try {
    $conn = New-Object System.Net.Sockets.TcpClient
    $conn.Connect("127.0.0.1", 5001)
    $conn.Close()
    $portInUse = $true
} catch {}

if ($portInUse) {
    Write-Host "TimeTracker is already running. Opening browser..."
    Start-Process $url
    exit 0
}

# --- 2. Bootstrap venv if it doesn't exist (fresh clone on a new machine). ---
if (-not (Test-Path $pythonPath)) {
    Write-Host "Setting up virtual environment..."
    python -m venv "$repoRoot\venv"
    & "$repoRoot\venv\Scripts\pip.exe" install -r "$repoRoot\requirements.txt"
}

# --- 3. Validate paths. ---
if (-not (Test-Path $pythonPath)) {
    Write-Error "Python venv not found at: $pythonPath"
    exit 1
}
if (-not (Test-Path $scriptPath)) {
    Write-Error "TimeTracker.py not found at: $scriptPath"
    exit 1
}

# --- 4. Start server with stdout/stderr captured for debugging. ---
Write-Host "Starting TimeTracker server..."
$proc = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList "`"$scriptPath`"" `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError  $errLogPath `
    -PassThru

Write-Host "Server PID: $($proc.Id) -- logs: $logPath"

# --- 5. Poll until the port is accepting connections (or timeout). ---
$ready   = $false
$elapsed = 0
Write-Host -NoNewline "Waiting for server"

while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds 1
    $elapsed++
    Write-Host -NoNewline "."

    if ($proc.HasExited) {
        Write-Host ""
        Write-Error "Server process exited early (code $($proc.ExitCode)). Check $errLogPath"
        exit 1
    }

    try {
        $conn = New-Object System.Net.Sockets.TcpClient
        $conn.Connect("127.0.0.1", 5001)
        $conn.Close()
        $ready = $true
        break
    } catch {}
}

Write-Host ""

if (-not $ready) {
    Write-Error "Server did not start within $maxWait seconds. Check $errLogPath"
    exit 1
}

# --- 6. Open browser. ---
Write-Host "TimeTracker ready at $url"
Start-Process $url
