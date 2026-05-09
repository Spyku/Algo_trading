
"""C42 CatBoost as 4th ensemble model. Adds CB+LGBM combo to grid."""
import crypto_trading_system_ed as eng

# Add CatBoost to model dispatch
try:
    from catboost import CatBoostClassifier
    _CB_AVAILABLE = True
except ImportError:
    print('[C42] catboost not installed: pip install catboost')
    _CB_AVAILABLE = False

if _CB_AVAILABLE:
    _orig_models = eng.ALL_MODELS

    def _make_cb():
        return CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.05,
            random_seed=42, verbose=0, thread_count=1,
            auto_class_weights='Balanced',
        )

    eng.ALL_MODELS = dict(_orig_models)
    eng.ALL_MODELS['CB'] = _make_cb

    # Patch DIAG_MODELS (used in Mode D grid via _get_deku_diagnostic_models)
    if hasattr(eng, 'DIAG_MODELS'):
        eng.DIAG_MODELS = dict(eng.DIAG_MODELS)
        eng.DIAG_MODELS['CB'] = lambda: CatBoostClassifier(
            iterations=100, depth=4, learning_rate=0.05,
            random_seed=42, verbose=0, thread_count=1,
            auto_class_weights='Balanced',
        )

    # Inject CB+LGBM into GRID_COMBOS (replace one or add)
    _orig_combos = list(eng.GRID_COMBOS)
    if 'CB+LGBM' not in _orig_combos:
        eng.GRID_COMBOS = ['RF+LGBM', 'CB+LGBM']  # cap at 2 for time budget
        print(f'[C42] GRID_COMBOS: {_orig_combos} -> {eng.GRID_COMBOS}')
    print('[C42] CatBoost added to ALL_MODELS and DIAG_MODELS')
