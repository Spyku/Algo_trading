# deploy_v3.ps1 — Push Version 3 to GitHub
# Usage: .\deploy_v3.ps1
# Or:    .\deploy_v3.ps1 "custom commit message"

param(
    [string]$Message = "v3: Multi-asset trading + Revolut X + interactive charts"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DEPLOY v3 to GitHub" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Files to track
$tracked = @(
    # Core system
    "crypto_trading_system.py",
    "crypto_live_trader.py",
    "crypto_revolut_trader.py",

    # Testing
    "crypto_strategy_test.py",
    "crypto_horizon_test.py",
    "crypto_trading_threshold_system.py",

    # Infrastructure
    "detect_hardware.py",
    "download_macro_data.py",
    "migrate_folders.py",

    # IB system (kept)
    "broly.py",
    "ib_auto_trader.py",
    "ib_auto_trader_test.py",
    "ib_test_connection.py",

    # Docs
    "README.md",
    "requirements.txt",
    ".gitignore"
)

# Check we're in the right folder
if (-not (Test-Path "crypto_trading_system.py")) {
    Write-Host "  ERROR: Run from Algo_trading folder!" -ForegroundColor Red
    exit 1
}

# Check git
if (-not (Test-Path ".git")) {
    Write-Host "  Initializing git repo..." -ForegroundColor Yellow
    git init
    git branch -M main
}

# Update .gitignore
$gitignore = @"
# Python
__pycache__/
*.pyc
*.pyo
venv/
.env

# Data (too large for git)
data/
charts/
models/
config/

# Secrets
private.pem
telegram_config.json
revolut_x_config.json
revolut_position.json
*.json

# System
*.csv
*.png
*.html
.DS_Store
Thumbs.db

# Allow specific files
!requirements.txt
!.gitignore
"@
$gitignore | Set-Content -Path ".gitignore" -Encoding UTF8
Write-Host "  Updated .gitignore" -ForegroundColor Green

# Stage tracked files
Write-Host ""
Write-Host "  Staging files:" -ForegroundColor Yellow
foreach ($f in $tracked) {
    if (Test-Path $f) {
        git add $f
        Write-Host "    + $f" -ForegroundColor Green
    } else {
        Write-Host "    - $f (not found, skipping)" -ForegroundColor DarkGray
    }
}

# Show status
Write-Host ""
Write-Host "  Git status:" -ForegroundColor Yellow
git status --short

# Tag version
$tag = "v3.0"
$tagExists = git tag -l $tag
if (-not $tagExists) {
    Write-Host ""
    Write-Host "  Will create tag: $tag" -ForegroundColor Cyan
}

# Commit
Write-Host ""
$confirm = Read-Host "  Commit and push? (y/n)"
if ($confirm -ne "y") {
    Write-Host "  Cancelled." -ForegroundColor Yellow
    exit 0
}

git commit -m $Message

# Tag
if (-not $tagExists) {
    git tag -a $tag -m "Version 3: Multi-asset + Revolut X + interactive charts"
    Write-Host "  Tagged: $tag" -ForegroundColor Green
}

# Push
Write-Host ""
Write-Host "  Pushing to origin..." -ForegroundColor Yellow
git push origin main
git push origin --tags

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  v3 DEPLOYED" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Commit: $Message"
Write-Host "  Tag: $tag"
Write-Host ""
