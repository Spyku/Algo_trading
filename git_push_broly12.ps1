# ============================================================
#  GIT PUSH — Broly 1.2 Major Update
# ============================================================
#  Run from: C:\Users\alexa\algo_trading\
#
#  What this does:
#    1. Creates folder structure (data/, output/, docs/)
#    2. Stages all updated files
#    3. Commits with descriptive message
#    4. Pushes to GitHub
#
#  Usage:
#    cd C:\Users\alexa\algo_trading
#    .\git_push_broly12.ps1
# ============================================================

Set-Location "C:\Users\alexa\algo_trading"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  BROLY 1.2 — GIT PUSH" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# ---- Step 1: Ensure folder structure exists ----
Write-Host "[1/5] Ensuring folder structure..." -ForegroundColor Green

$dirs = @(
    "data\indices",
    "data\crypto",
    "output\charts",
    "output\dashboards",
    "output\diagnostics",
    "output\backtests",
    "docs"
)

foreach ($d in $dirs) {
    if (!(Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        Write-Host "  Created: $d" -ForegroundColor DarkGray
    }
}

# Add .gitkeep to empty folders so Git tracks them
foreach ($d in $dirs) {
    $gitkeep = Join-Path $d ".gitkeep"
    if (!(Test-Path $gitkeep)) {
        New-Item -ItemType File -Path $gitkeep -Force | Out-Null
    }
}

# ---- Step 2: Check git status ----
Write-Host "`n[2/5] Current git status:" -ForegroundColor Green
git status --short

# ---- Step 3: Stage files ----
Write-Host "`n[3/5] Staging files..." -ForegroundColor Green

# Core scripts (updated)
git add daily_setup.py
git add generate_signals.py
git add ib_auto_trader.py
git add ib_test_connection.py

# Supporting modules
git add features_v2.py
git add hardware_config.py
git add broly.py

# Project files
git add .gitignore
git add README.md

# Folder structure (.gitkeep files)
git add data/ output/ docs/

# Also add any other .py files in root
git add *.py 2>$null

Write-Host "  Files staged." -ForegroundColor DarkGray

# ---- Step 4: Show what will be committed ----
Write-Host "`n[4/5] Changes to commit:" -ForegroundColor Green
git diff --cached --stat

# ---- Step 5: Commit and push ----
Write-Host "`n[5/5] Committing..." -ForegroundColor Green

$commitMsg = @"
Broly 1.2: Major refactor — modular pipeline + IB auto-trader

BREAKING CHANGES:
- Removed hourly_trading_system.py dependency
- All scripts now use V2 features + setup_config.json pipeline

New architecture:
  daily_setup.py       → data update + 75-config diagnostic (run once/morning)
  generate_signals.py  → interactive backtest dashboard
  ib_auto_trader.py    → live hourly trading via Interactive Brokers
  ib_test_connection.py → IB connection test utility

Key changes:
- daily_setup.py: 15 optimal V2 features, parallel model training, GPU support
- generate_signals.py: fixed leverage (1:1, 1:5, 1:10 on same chart),
  last month + last week view toggle with rebased equity,
  interactive zoom + legend click-to-toggle
- ib_auto_trader.py: rebuilt signal generation with V2 features pipeline,
  lightweight yfinance data update (5s vs full redownload),
  reads setup_config.json instead of importing hourly_trading_system
- Folder structure: data/indices/, data/crypto/, output/, docs/
- All data paths updated to data/indices/ subfolder
"@

git commit -m $commitMsg

Write-Host "`nPushing to GitHub..." -ForegroundColor Green
git push

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  DONE" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan
