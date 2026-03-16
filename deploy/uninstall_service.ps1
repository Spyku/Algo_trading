# ============================================================
# Remove Crypto Trader scheduled task
# Run as Administrator: powershell -ExecutionPolicy Bypass -File deploy\uninstall_service.ps1
# ============================================================

$TaskName = "CryptoTrader"

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Stopping task..."
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Task removed." -ForegroundColor Green
} else {
    Write-Host "Task '$TaskName' not found."
}
