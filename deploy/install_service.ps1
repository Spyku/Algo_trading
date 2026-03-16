# ============================================================
# Install Crypto Trader as a Windows Scheduled Task (auto-start on boot)
# Run as Administrator: powershell -ExecutionPolicy Bypass -File deploy\install_service.ps1
# ============================================================

$TaskName = "CryptoTrader"

# Detect machine
if (Test-Path "G:\Autres ordinateurs\My laptop\engine") {
    $EngineDir = "G:\Autres ordinateurs\My laptop\engine"
    $PythonExe = "C:\algo_trading\venv\Scripts\python.exe"
} elseif (Test-Path "C:\Users\Alex\algo_trading\engine") {
    $EngineDir = "C:\Users\Alex\algo_trading\engine"
    $PythonExe = "C:\Users\Alex\algo_trading\venv\Scripts\python.exe"
} else {
    Write-Host "ERROR: Could not find engine directory" -ForegroundColor Red
    exit 1
}

$Script = Join-Path $EngineDir "crypto_revolut_trader.py"

# Check prerequisites
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Python not found at $PythonExe" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Script)) {
    Write-Host "ERROR: Trader script not found at $Script" -ForegroundColor Red
    exit 1
}

Write-Host "=== Crypto Trader - Task Scheduler Setup ==="
Write-Host "  Engine:  $EngineDir"
Write-Host "  Python:  $PythonExe"
Write-Host ""

# Remove existing task if present
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task..."
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Start-Sleep -Seconds 1
}

# Create the task
$Action = New-ScheduledTaskAction `
    -Execute "`"$PythonExe`"" `
    -Argument "`"$Script`" --loop" `
    -WorkingDirectory "`"$EngineDir`""

# Trigger: at system startup
$Trigger = New-ScheduledTaskTrigger -AtStartup

# Settings: restart on failure, never stop, run whether logged in or not
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

# Run as current user with highest privileges
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -RunLevel Highest `
    -LogonType S4U

Write-Host "Registering scheduled task..."
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Revolut X multi-asset auto-trader with Telegram commands"

# Start it now
Write-Host "Starting task..."
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3

# Verify
$task = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName
if ($task.State -eq "Running") {
    Write-Host ""
    Write-Host "=== Crypto Trader installed and running ===" -ForegroundColor Green
    Write-Host ""
    Write-Host 'Commands (run as Admin):'
    Write-Host "  Status:   Get-ScheduledTask -TaskName CryptoTrader"
    Write-Host "  Stop:     Stop-ScheduledTask -TaskName CryptoTrader"
    Write-Host "  Start:    Start-ScheduledTask -TaskName CryptoTrader"
    Write-Host "  Remove:   Unregister-ScheduledTask -TaskName CryptoTrader"
    Write-Host ""
    Write-Host 'Auto-starts on boot. Restarts on crash, 1 min delay.' -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Task registered but not yet running." -ForegroundColor Yellow
    Write-Host "It will start automatically on next boot."
    Write-Host "To start now: Start-ScheduledTask -TaskName CryptoTrader"
}
