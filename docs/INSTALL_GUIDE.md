# Fresh Install Guide — Algo Trading System

## Quick Start (3 steps)

### Step 1: Run the setup script
```powershell
# Open PowerShell as Administrator, then:
powershell -ExecutionPolicy Bypass -File setup_algo_trading.ps1
```

This installs Python 3.14, creates the venv, and installs all packages.

### Step 2: Run hardware detection
```cmd
cd C:\Users\YourName\algo_trading
venv\Scripts\activate
python detect_hardware.py
```

This detects your CPU cores, GPU, tests LightGBM GPU, and generates `hardware_config.py`.

### Step 3: Integrate into your trading systems
Add this import to the top of `crypto_trading_system.py`, `hourly_trading_system.py`, and the diagnostic files:

```python
from hardware_config import (
    N_JOBS_PARALLEL,
    LGBM_DEVICE,
    get_cpu_models,
    get_gpu_models,
    get_all_models,
)
```

Then replace the hardcoded model definitions and `n_jobs` values with the config values.

---

## What Each File Does

| File | Purpose |
|---|---|
| `setup_algo_trading.ps1` | PowerShell installer — Python, venv, all pip packages, GPU setup |
| `detect_hardware.py` | Detects CPU cores + GPU → generates `hardware_config.py` |
| `hardware_config.py` | **Auto-generated** — contains core count, GPU flag, model definitions |
| `requirements.txt` | All Python dependencies (for manual `pip install -r requirements.txt`) |

---

## Integration Changes in Your Code

### 1. Diagnostic files (model_diagnostic.py, hourly_model_diagnostic.py)

**Before** (hardcoded):
```python
# Phase 1 — CPU parallel
results = Parallel(n_jobs=-1)(...)  # used all cores blindly
```

**After** (adapted):
```python
from hardware_config import N_JOBS_PARALLEL, get_cpu_models, get_gpu_models

# Phase 1 — CPU parallel (auto-adapted to this machine's core count)
results = Parallel(n_jobs=N_JOBS_PARALLEL)(...)

# Phase 2 — GPU sequential
gpu_models = get_gpu_models()  # LGBM device auto-set to 'gpu' or 'cpu'
```

### 2. Trading system files

**Before**:
```python
import lightgbm as lgb
lgb.LGBMClassifier(..., device='gpu', ...)
```

**After**:
```python
from hardware_config import get_all_models, LGBM_DEVICE
models = get_all_models()  # LGBM_DEVICE already set correctly
```

### 3. Python 3.14 asyncio fix (keep this!)
```python
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())  # Required for Python 3.14
from ib_insync import *
```

---

## GPU Setup Details

LightGBM GPU requires:
1. **NVIDIA drivers** — https://www.nvidia.com/drivers
2. **CUDA Toolkit 12.x** — https://developer.nvidia.com/cuda-downloads
3. **LightGBM GPU build**:
   ```cmd
   pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
   ```

If the pip GPU build fails (common on Windows), alternatives:
- Use conda: `conda install -c conda-forge lightgbm-gpu`
- Build from source: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
- The system falls back to CPU automatically via `hardware_config.py`

---

## Manual Install (if setup script doesn't work)

```cmd
:: 1. Install Python 3.14 from python.org (add to PATH!)

:: 2. Create project & venv
mkdir C:\Users\%USERNAME%\algo_trading
cd C:\Users\%USERNAME%\algo_trading
python -m venv venv
venv\Scripts\activate

:: 3. Install packages
pip install --upgrade pip
pip install -r requirements.txt

:: 4. For GPU LightGBM (optional, needs CUDA):
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON

:: 5. Detect hardware
python detect_hardware.py
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure venv is activated: `venv\Scripts\activate` |
| LightGBM GPU fails | Install CUDA Toolkit, or use CPU fallback (auto-detected) |
| Python 3.14 asyncio error | Add `asyncio.set_event_loop(asyncio.new_event_loop())` before `import ib_insync` |
| IB connection fails | Check IB Gateway running on port 7497, API enabled in settings |
| `sklearn` warnings | Already patched in trading systems (monkey-patch `warnings.warn`) |
| Yahoo volume = 0 for indices | Known — volume features set to neutral (1.0/0.0) in feature engineering |
| UTF-8 encoding errors | Add `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` at top |
