# ============================================================
# setup_algo_trading.ps1 - Fresh Install Script
# Run in PowerShell as Administrator
# Usage: powershell -ExecutionPolicy Bypass -File setup_algo_trading.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$PROJECT_DIR = "C:\Users\$env:USERNAME\algo_trading"

Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  ALGO TRADING SYSTEM - FRESH INSTALL" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan

# -----------------------------------------------------------
# STEP 1: Check/Install Python
# -----------------------------------------------------------
Write-Host ""
Write-Host "[1/7] Checking Python..." -ForegroundColor Yellow

$pythonInstalled = $false
try {
    $pyVersion = & python --version 2>&1
    if ($pyVersion -match "3\.1[3-4]") {
        Write-Host "  Found: $pyVersion" -ForegroundColor Green
        $pythonInstalled = $true
    } else {
        Write-Host "  Found $pyVersion but need 3.13 or 3.14" -ForegroundColor Red
    }
} catch {
    Write-Host "  Python not found in PATH" -ForegroundColor Red
}

if (-not $pythonInstalled) {
    Write-Host ""
    Write-Host "  Python 3.13+ is required." -ForegroundColor Magenta
    Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Magenta
    Write-Host "  Install with these options:" -ForegroundColor White
    Write-Host "    [x] Add Python to PATH" -ForegroundColor White
    Write-Host "    [x] Install for all users" -ForegroundColor White
    Write-Host "    [x] pip included" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "  Have you installed Python 3.13+? (y/n)"
    if ($response -ne "y") {
        Write-Host "  Please install Python first, then re-run this script." -ForegroundColor Red
        exit 1
    }
}

# -----------------------------------------------------------
# STEP 2: Check NVIDIA GPU and CUDA
# -----------------------------------------------------------
Write-Host ""
Write-Host "[2/7] Checking NVIDIA GPU..." -ForegroundColor Yellow

$gpuAvailable = $false
try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  GPU: $nvidiaSmi" -ForegroundColor Green
        $gpuAvailable = $true
    }
} catch {
    Write-Host "  No NVIDIA GPU detected or drivers not installed" -ForegroundColor Red
}

if (-not $gpuAvailable) {
    Write-Host ""
    Write-Host "  GPU SETUP REQUIRED:" -ForegroundColor Magenta
    Write-Host "  1. Install NVIDIA drivers: https://www.nvidia.com/drivers" -ForegroundColor White
    Write-Host "  2. Install CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
    Write-Host "  3. Restart and re-run this script" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "  Continue without GPU? LightGBM will use CPU only. (y/n)"
    if ($response -ne "y") { exit 1 }
}

# Check CUDA toolkit
$cudaInstalled = $false
try {
    $cudaVersion = & nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  CUDA Toolkit: installed" -ForegroundColor Green
        $cudaInstalled = $true
    }
} catch {
    if ($gpuAvailable) {
        Write-Host "  CUDA Toolkit NOT found." -ForegroundColor Red
        Write-Host "  Install from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    }
}

# -----------------------------------------------------------
# STEP 3: Create project directory
# -----------------------------------------------------------
Write-Host ""
Write-Host "[3/7] Setting up project directory..." -ForegroundColor Yellow

if (-not (Test-Path $PROJECT_DIR)) {
    New-Item -ItemType Directory -Path $PROJECT_DIR -Force | Out-Null
    Write-Host "  Created: $PROJECT_DIR" -ForegroundColor Green
} else {
    Write-Host "  Exists: $PROJECT_DIR" -ForegroundColor Green
}

Set-Location $PROJECT_DIR

# -----------------------------------------------------------
# STEP 4: Create virtual environment
# -----------------------------------------------------------
Write-Host ""
Write-Host "[4/7] Creating virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path "venv")) {
    & python -m venv venv
    Write-Host "  Created venv" -ForegroundColor Green
} else {
    Write-Host "  Exists: venv" -ForegroundColor Green
}

# Activate
& "$PROJECT_DIR\venv\Scripts\Activate.ps1"
Write-Host "  Activated venv" -ForegroundColor Green

# Upgrade pip
Write-Host "  Upgrading pip..." -ForegroundColor Gray
& python -m pip install --upgrade pip --quiet

# -----------------------------------------------------------
# STEP 5: Install Python packages
# -----------------------------------------------------------
Write-Host ""
Write-Host "[5/7] Installing Python packages..." -ForegroundColor Yellow

Write-Host "  Installing numpy, pandas, scipy..." -ForegroundColor Gray
& pip install numpy pandas scipy --quiet

Write-Host "  Installing scikit-learn..." -ForegroundColor Gray
& pip install scikit-learn --quiet

Write-Host "  Installing ccxt, yfinance..." -ForegroundColor Gray
& pip install ccxt yfinance --quiet

Write-Host "  Installing ib_insync, nest_asyncio..." -ForegroundColor Gray
& pip install ib_insync nest_asyncio --quiet

Write-Host "  Installing joblib, matplotlib, pytz, requests..." -ForegroundColor Gray
& pip install joblib matplotlib pytz requests --quiet

# LightGBM - GPU or CPU
if ($gpuAvailable -and $cudaInstalled) {
    Write-Host "  Installing LightGBM (GPU build)..." -ForegroundColor Yellow
    Write-Host "  Trying pip with USE_GPU=ON..." -ForegroundColor Gray

    $lgbmResult = & pip install lightgbm --config-settings=cmake.define.USE_GPU=ON 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  GPU build failed. Installing CPU version as fallback..." -ForegroundColor Red
        Write-Host "  For GPU later: conda install -c conda-forge lightgbm-gpu" -ForegroundColor Magenta
        & pip install lightgbm --quiet
    } else {
        Write-Host "  LightGBM GPU build installed" -ForegroundColor Green
    }
} else {
    Write-Host "  Installing LightGBM (CPU)..." -ForegroundColor Gray
    & pip install lightgbm --quiet
    if ($gpuAvailable) {
        Write-Host "  TIP: Install CUDA Toolkit to enable GPU acceleration" -ForegroundColor Magenta
    }
}

# -----------------------------------------------------------
# STEP 6: Verify installations
# -----------------------------------------------------------
Write-Host ""
Write-Host "[6/7] Verifying installations..." -ForegroundColor Yellow

$allGood = $true

$packages = @("numpy", "pandas", "scipy", "sklearn", "lightgbm", "ccxt", "yfinance", "ib_insync", "joblib", "matplotlib")

foreach ($pkg in $packages) {
    $checkScript = "import " + $pkg + "; print(" + $pkg + ".__version__)"
    $result = & python -c $checkScript 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host ("  OK    " + $pkg + " " + $result) -ForegroundColor Green
    } else {
        Write-Host ("  FAIL  " + $pkg) -ForegroundColor Red
        $allGood = $false
    }
}

# -----------------------------------------------------------
# STEP 7: Run hardware detection
# -----------------------------------------------------------
Write-Host ""
Write-Host "[7/7] Running hardware detection..." -ForegroundColor Yellow

$detectScript = Join-Path $PROJECT_DIR "detect_hardware.py"
if (Test-Path $detectScript) {
    & python $detectScript
} else {
    Write-Host "  detect_hardware.py not found in project dir" -ForegroundColor Magenta
    Write-Host "  Copy it there and run: python detect_hardware.py" -ForegroundColor Magenta
}

# -----------------------------------------------------------
# DONE
# -----------------------------------------------------------
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  INSTALLATION COMPLETE" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""
Write-Host "  Project dir: $PROJECT_DIR" -ForegroundColor White
Write-Host "  Venv dir:    $PROJECT_DIR\venv" -ForegroundColor White
Write-Host ""
Write-Host "  NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Copy your .py and .csv files into $PROJECT_DIR" -ForegroundColor White
Write-Host "  2. Copy detect_hardware.py into $PROJECT_DIR" -ForegroundColor White
Write-Host "  3. Run: python detect_hardware.py" -ForegroundColor White
Write-Host "  4. This generates hardware_config.py with core count and GPU settings" -ForegroundColor White
Write-Host ""
Write-Host "  ACTIVATE VENV (each new terminal):" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\activate" -ForegroundColor White
Write-Host ""

if (-not $allGood) {
    Write-Host "  WARNING: Some packages failed. Check errors above." -ForegroundColor Red
}
