# deploy.ps1 - Copy updated files to repo and git push with versioning
# =====================================================
# Usage:
#   .\deploy.ps1                          # Normal deploy (code files only)
#   .\deploy.ps1 -Tag "v2.0"             # Deploy + create git tag
#   .\deploy.ps1 -Full                    # Deploy code + market data files
#   .\deploy.ps1 -Full -Tag "v2.0"       # Full deploy + version tag
#   .\deploy.ps1 -Message "my message"   # Custom commit message

param(
    [string]$Source = "$env:USERPROFILE\Downloads",
    [string]$Repo = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$Message = "",
    [string]$Tag = "",
    [switch]$Full = $false
)

$ErrorActionPreference = "Stop"

# --- FILE LISTS ---
$CODE_FILES = @(
    "crypto_trading_system.py",
    "crypto_live_trader.py",
    "crypto_trading_threshold_system.py",
    "crypto_feature_analysis.py",
    "download_macro_data.py",
    "detect_hardware.py",
    "hardware_config.py",
    "telegram_alerts.py",
    "broly.py",
    "ib_auto_trader.py",
    "ib_auto_trader_test.py",
    "README.md",
    "requirements.txt",
    "deploy.ps1",
    "setup_algo_trading.ps1",
    "INSTALL_GUIDE.md"
)

$MARKET_FILES = @(
    "btc_hourly_data.csv",
    "eth_hourly_data.csv",
    "sol_hourly_data.csv",
    "xrp_hourly_data.csv",
    "doge_hourly_data.csv",
    "dax_hourly_data.csv",
    "smi_hourly_data.csv",
    "cac40_hourly_data.csv",
    "crypto_hourly_best_models.csv",
    "crypto_hourly_best_models_mode_d.csv",
    "crypto_feature_set_comparison.csv",
    "crypto_hourly_chart_data.json"
)

# --- HEADER ---
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  DEPLOY - Copy and Git Push" -ForegroundColor Cyan
if ($Tag) {
    Write-Host "  VERSION TAG: $Tag" -ForegroundColor Yellow
}
if ($Full) {
    Write-Host "  MODE: Full (code + market data)" -ForegroundColor Yellow
} else {
    Write-Host "  MODE: Code only" -ForegroundColor Yellow
}
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- CHECK REPO ---
if (-not (Test-Path "$Repo\.git")) {
    Write-Host "  ERROR: Git repo not found at $Repo" -ForegroundColor Red
    Write-Host "  Run: cd $Repo; git init; git remote add origin https://github.com/Spyku/Algo_trading.git"
    exit 1
}

# --- COPY FILES FROM DOWNLOADS ---
$filesToCopy = $CODE_FILES
if ($Full) {
    $filesToCopy = $CODE_FILES + $MARKET_FILES
}

$copied = @()
foreach ($file in $filesToCopy) {
    $sourcePath = Join-Path $Source $file
    $destPath = Join-Path $Repo $file

    if (Test-Path $sourcePath) {
        $sourceTime = (Get-Item $sourcePath).LastWriteTime
        $shouldCopy = $true

        if (Test-Path $destPath) {
            $destTime = (Get-Item $destPath).LastWriteTime
            if ($sourceTime -le $destTime) {
                $shouldCopy = $false
            }
        }

        if ($shouldCopy) {
            Copy-Item $sourcePath $destPath -Force
            $copied += $file
            Write-Host "  COPIED: $file" -ForegroundColor Green
        }
    }
}

# Include macro_data folder and extra files if -Full
if ($Full) {
    $macroPath = Join-Path $Repo "macro_data"
    if (Test-Path $macroPath) {
        Write-Host "  INCLUDED: macro_data/" -ForegroundColor Green
        $copied += "macro_data/"
    }

    $extraPatterns = @("crypto_feature_analysis_*_auto.csv", "*_backtest.png", "*_threshold_backtest.png", "*_threshold_results.csv")
    foreach ($pattern in $extraPatterns) {
        $matches = Get-ChildItem -Path $Repo -Filter $pattern -ErrorAction SilentlyContinue
        foreach ($match in $matches) {
            Write-Host "  INCLUDED: $($match.Name)" -ForegroundColor DarkGreen
        }
    }
}

# --- CHECK STATUS ---
Set-Location $Repo

if ($copied.Count -eq 0) {
    Write-Host "  No new files found in $Source" -ForegroundColor Yellow
    Write-Host ""

    $status = git status --porcelain 2>&1
    if ($status) {
        Write-Host "  Uncommitted changes in repo:" -ForegroundColor Yellow
        git status --short
        Write-Host ""
        $push = Read-Host "  Push these changes? (y/n)"
        if ($push -ne "y") {
            exit 0
        }
    } else {
        Write-Host "  Repo is clean. Nothing to do." -ForegroundColor Green
        exit 0
    }
}

# --- ENSURE .gitignore ---
$gitignorePath = Join-Path $Repo ".gitignore"

if (-not (Test-Path $gitignorePath)) {
    $lines = @(
        "# Python",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "venv/",
        ".venv/",
        "*.egg-info/",
        "",
        "# IDE",
        ".vscode/",
        ".idea/",
        "*.swp",
        "",
        "# Secrets",
        "telegram_config.json",
        "",
        "# OS",
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini"
    )
    $lines | Set-Content -Path $gitignorePath
    Write-Host "  CREATED .gitignore" -ForegroundColor Green
} else {
    $existing = Get-Content $gitignorePath -Raw
    if ($existing -notmatch "telegram_config") {
        Add-Content -Path $gitignorePath -Value "telegram_config.json"
        Write-Host "  UPDATED .gitignore (added telegram_config.json)" -ForegroundColor Green
    }
}

# --- GIT ADD, COMMIT, PUSH ---
Write-Host ""
Write-Host "  Git status:" -ForegroundColor Cyan
git add -A
git status --short

# Auto-generate commit message
if (-not $Message) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    if ($Tag) {
        $Message = "$Tag - Mode E, dual horizon, threshold system, derivative features [$timestamp]"
    } else {
        $fileList = ($copied | Select-Object -First 5) -join ", "
        if ($copied.Count -gt 5) { $fileList += ", +$($copied.Count - 5) more" }
        $Message = "Update: $fileList [$timestamp]"
    }
}

Write-Host ""
Write-Host "  Commit: $Message" -ForegroundColor Cyan
git commit -m $Message

# --- TAG ---
if ($Tag) {
    Write-Host ""
    Write-Host "  Creating tag: $Tag" -ForegroundColor Yellow

    $existingTag = git tag -l $Tag 2>&1
    if ($existingTag) {
        Write-Host "  Tag $Tag already exists. Overwriting..." -ForegroundColor Yellow
        git tag -d $Tag 2>&1 | Out-Null
        git push origin ":refs/tags/$Tag" 2>&1 | Out-Null
    }

    git tag -a $Tag -m $Message
    Write-Host "  Tag created: $Tag" -ForegroundColor Green
}

# --- PUSH ---
Write-Host ""
Write-Host "  Pushing to GitHub..." -ForegroundColor Cyan
git push origin main 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "  Trying master branch..." -ForegroundColor Yellow
    git push origin master 2>&1
}

if ($Tag) {
    Write-Host "  Pushing tag $Tag..." -ForegroundColor Cyan
    git push origin $Tag 2>&1
}

# --- DONE ---
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  DEPLOYED SUCCESSFULLY" -ForegroundColor Green
if ($Tag) {
    Write-Host "  Version: $Tag" -ForegroundColor Green
}
Write-Host "  Files: $($copied.Count) updated" -ForegroundColor Green
Write-Host "  Repo: https://github.com/Spyku/Algo_trading" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
