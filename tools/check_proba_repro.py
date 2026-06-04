"""
check_proba_repro.py — cross-machine determinism probe for the live signal model.

Rebuilds ETH features for a given closed bar, trains the production bear-5h config
(RF + GPU-LGBM via the live factory) TWICE, and prints per-model probabilities.

  - run1 == run2 on the SAME machine  -> the model is deterministic (expected).
  - laptop-B values == laptop-A values -> GPU LightGBM is CROSS-MACHINE deterministic
    (=> the sanity check can safely use GPU and drop --cpu-lgbm).
  - laptop-B != laptop-A on LGBM only  -> GPU differs across GPUs; keep --cpu-lgbm for
    portable/deterministic validation (but note it then != the GPU live trader).

Reference (laptop = DESKTOP, 2026-06-04, bar 2026-06-04 04:00, lagged ED engine):
    RF = 0.51463784    LGBM = 0.05573897

Usage:  python tools/check_proba_repro.py [BAR]    e.g. "2026-06-04 04:00:00"
"""
import os
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import crypto_trading_system_ed as e          # noqa: E402
import crypto_live_trader_ed as lt            # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BAR = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2026-06-04 04:00:00")
ASSET, HORIZON = "ETH", 5

prod = pd.read_csv("models/crypto_ed_production.csv")
row = prod[(prod.coin == ASSET) & (prod.horizon == HORIZON)].iloc[0]
models = row["best_combo"].split("+")
window = int(row["best_window"])
gamma = float(row["gamma"])
feats = [f.strip() for f in str(row["optimal_features"]).split(",") if f.strip() and f.strip() != "nan"]

eth = pd.read_csv(f"data/{ASSET.lower()}_hourly_data.csv", parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
dft = eth[eth.datetime <= BAR].reset_index(drop=True)
f, c = e.build_all_features(dft, asset_name=ASSET, horizon=HORIZON, verbose=False, keep_label_nan_tail=True)
e._compute_pysr_features(f, c, ASSET, HORIZON, verbose=False)
fc = [x for x in feats if x in c]
dtr = f.dropna(subset=["label"]).reset_index(drop=True)
df = f.reset_index(drop=True).copy()
df[fc] = df[fc].ffill().fillna(0.0)
dtr[fc] = dtr[fc].ffill().fillna(0.0)
ts = max(0, len(dtr) - window)
Xtr = dtr.iloc[ts:][fc].values
ytr = dtr.iloc[ts:]["label"].values
Xte = df.iloc[len(df) - 1:len(df)][fc].values
sw = lt.get_decay_weights(len(ytr), gamma)
sc = StandardScaler()
A = sc.fit_transform(Xtr)
B = sc.transform(Xte)

print(f"  machine={os.environ.get('COMPUTERNAME', '?')}  bar={BAR}  config={'+'.join(models)} w={window} g={gamma} "
      f"nfeat={len(fc)} train_rows={len(ytr)}  DAILY_LAG={e.DAILY_MERGE_LAG_DAYS} OC_LAG={e.ONCHAIN_MERGE_LAG_DAYS}")
for run in (1, 2):
    out = []
    for name in models:
        m = lt.ALL_MODELS[name]()
        m.fit(A, ytr, sample_weight=sw)
        out.append(f"{name}={m.predict_proba(B)[0][1]:.8f}")
    print(f"    run {run}: " + "  ".join(out))
print("  reference (DESKTOP): RF=0.51463784  LGBM=0.05573897")
