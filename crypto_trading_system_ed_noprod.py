"""NO-PRODUCTION version of crypto_trading_system_ed.

Writes to *_noprod.csv / *_noprod.json files that the live trader does NOT read.
Safe for experimentation — nothing can leak to production.

File mapping:
  models/crypto_ed_best_models.csv    -> models/crypto_ed_best_models_noprod.csv
  models/crypto_ed_production.csv     -> models/crypto_ed_production_noprod.csv
  config/regime_config_ed.json        -> config/regime_config_ed_noprod.json

Usage is identical to the real script, just the filename changes:
  python -u crypto_trading_system_ed_noprod.py D ETH 6,7,8h
  python -u crypto_trading_system_ed_noprod.py V ETH 6,7,8h
  python -u crypto_trading_system_ed_noprod.py R ETH 6,7,8h
"""
import os
import shutil
import sys

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))

_NOPROD_BEST_MODELS = 'models/crypto_ed_best_models_noprod.csv'
_NOPROD_PRODUCTION  = 'models/crypto_ed_production_noprod.csv'
_NOPROD_REGIME      = 'config/regime_config_ed_noprod.json'

# MODELS_CSV_OVERRIDE is captured at import time in crypto_trading_system_ed (line 836),
# so it must be set BEFORE the import below.
os.environ['MODELS_CSV_OVERRIDE'] = _NOPROD_BEST_MODELS

# Seed the noprod files from the real ones if they don't exist yet, so Mode R/V/S
# have a starting point. Existing noprod files are preserved across runs.
def _seed(real_rel, noprod_rel):
    real = os.path.join(_ENGINE_DIR, real_rel)
    noprod = os.path.join(_ENGINE_DIR, noprod_rel)
    if os.path.exists(real) and not os.path.exists(noprod):
        os.makedirs(os.path.dirname(noprod), exist_ok=True)
        shutil.copy2(real, noprod)
        print(f"  [noprod] seeded {noprod_rel} from {real_rel}")

_seed('models/crypto_ed_best_models.csv', _NOPROD_BEST_MODELS)
_seed('models/crypto_ed_production.csv',  _NOPROD_PRODUCTION)
_seed('config/regime_config_ed.json',     _NOPROD_REGIME)

import crypto_trading_system_ed as _ed

# Patch module-level paths so all reads/writes land in the noprod namespace.
_ed.PRODUCTION_CSV     = _NOPROD_PRODUCTION
_ed.REGIME_CONFIG_PATH = _NOPROD_REGIME

if __name__ == '__main__':
    print("=" * 60)
    print("  NO-PRODUCTION MODE")
    print("  Writes to *_noprod.* files — trader does NOT read these")
    print(f"  best_models: {_NOPROD_BEST_MODELS}")
    print(f"  production:  {_NOPROD_PRODUCTION}")
    print(f"  regime:      {_NOPROD_REGIME}")
    print("=" * 60)
    _ed.main()
