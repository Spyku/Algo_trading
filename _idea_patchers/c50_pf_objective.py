
"""C50 raw profit factor as primary Optuna metric (instead of APF).

Sets eng.OPTUNA_METRIC = 'rawpf' BEFORE the engine's main() runs. The engine
already supports this branch in _compute_optuna_score (see crypto_trading_system_ed.py:534).
With OPTUNA_METRIC='rawpf', best_models_rawpf.csv path is used, but with
--no-persist the writes go to _noprod copies anyway.
"""
import crypto_trading_system_ed as eng

_orig = getattr(eng, 'OPTUNA_METRIC', 'apf')
eng.OPTUNA_METRIC = 'rawpf'
print(f'[C50] OPTUNA_METRIC: {_orig!r} -> {eng.OPTUNA_METRIC!r} (raw profit factor)')
