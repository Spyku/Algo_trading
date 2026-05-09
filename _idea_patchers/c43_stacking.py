
"""C43 Stacking meta-learner. Adds STACK pseudo-combo using sklearn StackingClassifier."""
import crypto_trading_system_ed as eng

_orig_models = eng.ALL_MODELS

def _make_stack():
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=4,
                                       class_weight='balanced', random_state=42, n_jobs=1)),
        ('lgbm', LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  class_weight='balanced', verbose=-1, random_state=42)),
    ]
    if XGBClassifier is not None:
        estimators.append(('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                                  random_state=42, tree_method='hist',
                                                  verbosity=0, n_jobs=1)))
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
        cv=3, n_jobs=1, passthrough=False,
    )

eng.ALL_MODELS = dict(_orig_models)
eng.ALL_MODELS['STACK'] = _make_stack
if hasattr(eng, 'DIAG_MODELS'):
    eng.DIAG_MODELS = dict(eng.DIAG_MODELS)
    eng.DIAG_MODELS['STACK'] = _make_stack

# Use STACK alone (it's already an ensemble — paired with itself or LGBM as second voter)
_orig_combos = list(eng.GRID_COMBOS)
eng.GRID_COMBOS = ['STACK+LGBM', 'STACK+RF']  # Stack votes with another base
print(f'[C43] GRID_COMBOS: {_orig_combos} -> {eng.GRID_COMBOS}')
print('[C43] StackingClassifier added to ALL_MODELS / DIAG_MODELS')
