"""
hardware_config.py -- Auto-detects LAPTOP vs DESKTOP at import time.
LAPTOP:  16 logical cores (old machine)
DESKTOP: 28 logical cores, i7-14700KF, RTX 4080

Usage in other files:
    from hardware_config import (
        MACHINE, N_JOBS_PARALLEL, LGBM_DEVICE,
        get_cpu_models, get_gpu_models, get_all_models,
    )
"""

import multiprocessing
import subprocess

# ============================================================
# AUTO-DETECT MACHINE
# ============================================================
LOGICAL_CORES = multiprocessing.cpu_count()

if LOGICAL_CORES >= 24:
    MACHINE = 'DESKTOP'
    PHYSICAL_CORES = 20
    N_JOBS_PARALLEL = 26       # 28 - 2 for system
else:
    MACHINE = 'LAPTOP'
    PHYSICAL_CORES = LOGICAL_CORES // 2 or 8
    N_JOBS_PARALLEL = max(1, LOGICAL_CORES - 2)  # 14 for 16-core

# ============================================================
# GPU DETECTION
# ============================================================
GPU_AVAILABLE = False
GPU_NAME = None
GPU_VRAM_MB = None

try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        parts = result.stdout.strip().split(',')
        GPU_AVAILABLE = True
        GPU_NAME = parts[0].strip()
        GPU_VRAM_MB = int(float(parts[1].strip()))
except Exception:
    pass

# LightGBM GPU: test once at import
LGBM_DEVICE = 'cpu'  # default fallback
if GPU_AVAILABLE:
    try:
        import lightgbm as _lgb
        import numpy as _np
        _X = _np.random.rand(50, 3)
        _y = _np.random.randint(0, 2, 50)
        _ds = _lgb.Dataset(_X, label=_y)
        _lgb.train({'objective': 'binary', 'device': 'gpu', 'verbose': -1,
                     'num_iterations': 2}, _ds, num_boost_round=2)
        LGBM_DEVICE = 'gpu'
        del _lgb, _np, _X, _y, _ds
    except Exception:
        pass

# ============================================================
# MODEL FACTORIES
# ============================================================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def get_cpu_models():
    """Phase 1: CPU parallel workers. RF n_jobs=1 to avoid nested parallelism."""
    return {
        'RF': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced',
            random_state=42, n_jobs=1
        ),
        'GB': lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42
        ),
        'LR': lambda: LogisticRegression(
            max_iter=1000, class_weight='balanced', C=0.1,
            random_state=42
        ),
    }


def get_diagnostic_models():
    """All models on CPU for parallel diagnostic (GPU overhead kills small data)."""
    import lightgbm as lgb
    return {
        'RF': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced',
            random_state=42, n_jobs=1
        ),
        'GB': lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42
        ),
        'LR': lambda: LogisticRegression(
            max_iter=1000, class_weight='balanced', C=0.1,
            random_state=42
        ),
        'LGBM': lambda: lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=42,
            device='cpu'
        ),
    }


def get_gpu_models():
    """Phase 2: GPU sequential. RF gets full CPU, LGBM gets GPU."""
    import lightgbm as lgb
    return {
        'RF': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'GB': lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42
        ),
        'LR': lambda: LogisticRegression(
            max_iter=1000, class_weight='balanced', C=0.1,
            random_state=42
        ),
        'LGBM': lambda: lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=42,
            device=LGBM_DEVICE
        ),
    }


def get_all_models():
    """Signal generation (not in parallel). All models with full resources."""
    import lightgbm as lgb
    return {
        'RF': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'GB': lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42
        ),
        'LR': lambda: LogisticRegression(
            max_iter=1000, class_weight='balanced', C=0.1,
            random_state=42
        ),
        'LGBM': lambda: lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=42,
            device=LGBM_DEVICE
        ),
    }


# ============================================================
# STARTUP BANNER (printed once on import)
# ============================================================
print(f"  [{MACHINE}] {LOGICAL_CORES} cores | "
      f"GPU: {GPU_NAME or 'None'} | "
      f"LGBM: {LGBM_DEVICE} | "
      f"Parallel workers: {N_JOBS_PARALLEL}")
