# hardware_config.py - Auto-configured for DESKTOP
# CPU: Intel i7-14700KF (28 logical cores)
# GPU: NVIDIA RTX 4080 (16GB VRAM)

MACHINE = 'DESKTOP'
N_JOBS_PARALLEL = 26
LGBM_DEVICE = 'gpu'

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def get_cpu_models():
    return {
        'RF': lambda: RandomForestClassifier(n_estimators=300, max_depth=4, class_weight='balanced', random_state=42, n_jobs=-1),
        'GB': lambda: GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=42),
        'LR': lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    }

def get_gpu_models():
    from lightgbm import LGBMClassifier
    return {
        'LGBM': lambda: LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, class_weight='balanced', verbose=-1, random_state=42, device='gpu'),
    }

def get_all_models():
    models = get_cpu_models()
    models.update(get_gpu_models())
    return models

def get_diagnostic_models():
    return get_all_models()
