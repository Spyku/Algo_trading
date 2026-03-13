"""
+==============================================================+
|                      BROLY 1.2                                |
|       Advanced Trading Enhancement Layer                      |
|                                                               |
|  WHAT'S NEW in 1.2 (vs 1.0):                                |
|   - Optimal 15 V2 features (76.1% accuracy, from analysis)   |
|   - 5-tier graduated signals: STRONG_BUY / BUY / HOLD /      |
|     SELL / STRONG_SELL with proportional position sizing       |
|   - Regime-aware SL/TP adapts to signal strength              |
|   - Enhanced alerts with signal tier + confidence              |
|   - Updated paths for reorganized folder structure             |
|                                                               |
|  Components:                                                  |
|   1. Market Regime Detector (BULL / BEAR / SIDEWAYS)          |
|   2. Graduated Signal Engine (5-tier from ML confidence)      |
|   3. Regime-Specific Model Training & Selection               |
|   4. Full Stop-Loss / Take-Profit System                      |
|   5. Discord + Telegram Alert System                          |
|                                                               |
|  Integrates with:                                             |
|   - hourly_trading_system.py (signal generation)              |
|   - features_v2.py (macro features including VIX, DXY, etc)  |
|   - ib_auto_trader.py (order execution)                       |
|                                                               |
|  Usage:                                                       |
|   from broly import Broly                                     |
|   broly = Broly()                                             |
|   regime = broly.detect_regime()                              |
|   signal = broly.classify_signal('DAX', ml_probability=0.72)  |
|   broly.alert(f"DAX: {signal.tier} @ {signal.position_size_pct}%") |
+==============================================================+
"""

import os, json, time, warnings, logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger('broly')


# ============================================================
# OPTIMAL V2 FEATURES (from feature_analysis_v2.py results)
# 15 features -> 76.1% accuracy (best subset)
# ============================================================

FEATURE_COLS_V2 = [
    'logret_240h',          # BASE - 10-day momentum
    'm_sp500_vol20d',       # MACRO - S&P500 volatility
    'm_vix_zscore',         # MACRO - VIX normalized
    'm_sp500_zscore',       # MACRO - S&P500 normalized
    'logret_24h',           # BASE - 1-day return
    'volatility_48h',       # BASE - 2-day volatility
    'xa_sp500_relstr5d',    # CROSS-ASSET - relative strength vs S&P
    'atr_pct_14h',          # BASE - ATR as % of price
    'sma20_to_sma50h',      # BASE - MA crossover ratio
    'zscore_50h',           # BASE - price z-score
    'xa_sp500_corr30d',     # CROSS-ASSET - 30d correlation with S&P
    'spread_120h_8h',       # BASE - fast/slow spread
    'xa_nasdaq_corr10d',    # CROSS-ASSET - 10d correlation with Nasdaq
    'm_gold_vol20d',        # MACRO - gold volatility
    'fg_zscore',            # SENTIMENT - Fear & Greed normalized
]

CONFIDENCE_THRESHOLD = 70  # % for STRONG signals


# ============================================================
# SIGNAL TIER SYSTEM (5-tier graduated)
# ============================================================

class SignalTier(Enum):
    STRONG_BUY  = 'STRONG_BUY'
    BUY         = 'BUY'
    HOLD        = 'HOLD'
    SELL        = 'SELL'
    STRONG_SELL = 'STRONG_SELL'


@dataclass
class SignalResult:
    """Complete signal output from Broly."""
    asset: str
    tier: str                  # SignalTier value
    ml_probability: float      # Raw ML probability (0-1)
    confidence: float          # Adjusted confidence
    position_size_pct: float   # Recommended position size (% of capital)
    regime: str                # Current market regime
    strategy: str              # Strategy name
    timestamp: str
    features_used: int = 15    # Number of features


# ============================================================
# 1. MARKET REGIME DETECTION
# ============================================================

class Regime(Enum):
    BULL = 'BULL'
    BEAR = 'BEAR'
    SIDEWAYS = 'SIDEWAYS'


@dataclass
class RegimeSnapshot:
    """Complete regime analysis snapshot."""
    regime: str
    confidence: float
    vix_level: float
    vix_zscore: float
    sp500_vs_sma200: float
    sp500_trend_6m: float
    dxy_zscore: float
    timestamp: str
    details: Dict = field(default_factory=dict)


class MarketRegimeDetector:
    """
    Detects market regime using macro data from download_macro_data.py.

    Classification:
      BULL:     VIX < 20 AND S&P500 > SMA200 AND 6m trend > +5%
      BEAR:     VIX > 28 AND S&P500 < SMA200 AND 6m trend < -5%
      SIDEWAYS: Everything else

    Regime score [-1 to +1]: -1 = extreme bear, 0 = neutral, +1 = extreme bull
    """

    def __init__(self, macro_data_dir='macro_data'):
        self.macro_dir = Path(macro_data_dir)
        self._cache = None
        self._cache_time = None
        self.CACHE_TTL = 3600

    def _load_macro_data(self):
        macro_file = self.macro_dir / 'macro_daily.csv'
        if not macro_file.exists():
            logger.warning(f"Macro data not found: {macro_file}")
            return None
        df = pd.read_csv(macro_file, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    def detect(self):
        if self._cache and self._cache_time:
            if (datetime.now() - self._cache_time).seconds < self.CACHE_TTL:
                return self._cache

        df = self._load_macro_data()
        if df is None:
            return self._default_snapshot()

        latest = df.iloc[-1]

        # VIX
        vix_col = [c for c in df.columns if 'vix' in c.lower()]
        if not vix_col:
            vix, vix_zscore = 18.0, 0.0
        else:
            vix_series = df[vix_col[0]].dropna()
            vix = float(vix_series.iloc[-1])
            vix_mean = vix_series.rolling(50, min_periods=10).mean().iloc[-1]
            vix_std = vix_series.rolling(50, min_periods=10).std().iloc[-1]
            vix_zscore = (vix - vix_mean) / max(vix_std, 0.01)

        # S&P500
        sp_col = [c for c in df.columns if 'sp500' in c.lower() or 'gspc' in c.lower()]
        if not sp_col:
            sp500_vs_sma200, sp500_trend_6m = 0.0, 0.0
        else:
            sp_series = df[sp_col[0]].dropna()
            sp_current = float(sp_series.iloc[-1])
            sp_sma200 = float(sp_series.rolling(200, min_periods=50).mean().iloc[-1])
            sp500_vs_sma200 = (sp_current / sp_sma200 - 1) * 100
            lookback = min(126, len(sp_series) - 1)
            sp_6m_ago = float(sp_series.iloc[-lookback - 1]) if lookback > 0 else sp_current
            sp500_trend_6m = (sp_current / sp_6m_ago - 1) * 100

        # DXY
        dxy_col = [c for c in df.columns if 'dxy' in c.lower() or 'dollar' in c.lower()]
        if not dxy_col:
            dxy_zscore = 0.0
        else:
            dxy_series = df[dxy_col[0]].dropna()
            dxy_mean = dxy_series.rolling(50, min_periods=10).mean().iloc[-1]
            dxy_std = dxy_series.rolling(50, min_periods=10).std().iloc[-1]
            dxy_zscore = (float(dxy_series.iloc[-1]) - dxy_mean) / max(dxy_std, 0.01)

        # Classification
        bull_score, bear_score = 0, 0

        if vix < 15: bull_score += 3
        elif vix < 20: bull_score += 2
        elif vix < 25: bull_score += 1
        elif vix < 30: bear_score += 1
        elif vix < 35: bear_score += 2
        else: bear_score += 3

        if sp500_vs_sma200 > 5: bull_score += 2
        elif sp500_vs_sma200 > 0: bull_score += 1
        elif sp500_vs_sma200 > -5: bear_score += 1
        else: bear_score += 2

        if sp500_trend_6m > 10: bull_score += 2
        elif sp500_trend_6m > 5: bull_score += 1
        elif sp500_trend_6m > -5: pass
        elif sp500_trend_6m > -10: bear_score += 1
        else: bear_score += 2

        if dxy_zscore > 1.5: bear_score += 1
        elif dxy_zscore < -1.5: bull_score += 1

        total = bull_score + bear_score
        if total == 0:
            regime, confidence = Regime.SIDEWAYS, 0.5
        elif bull_score >= bear_score + 3:
            regime = Regime.BULL
            confidence = min(0.95, 0.6 + (bull_score - bear_score) * 0.05)
        elif bear_score >= bull_score + 3:
            regime = Regime.BEAR
            confidence = min(0.95, 0.6 + (bear_score - bull_score) * 0.05)
        else:
            regime = Regime.SIDEWAYS
            confidence = 0.5 + abs(bull_score - bear_score) * 0.05

        regime_score = (bull_score - bear_score) / total if total > 0 else 0.0

        snapshot = RegimeSnapshot(
            regime=regime.value, confidence=round(confidence, 3),
            vix_level=round(vix, 2), vix_zscore=round(vix_zscore, 3),
            sp500_vs_sma200=round(sp500_vs_sma200, 2),
            sp500_trend_6m=round(sp500_trend_6m, 2),
            dxy_zscore=round(dxy_zscore, 3),
            timestamp=datetime.now().isoformat(),
            details={'bull_score': bull_score, 'bear_score': bear_score,
                     'regime_score': round(regime_score, 3)},
        )
        self._cache = snapshot
        self._cache_time = datetime.now()
        return snapshot

    def _default_snapshot(self):
        return RegimeSnapshot(
            regime=Regime.SIDEWAYS.value, confidence=0.3,
            vix_level=0, vix_zscore=0, sp500_vs_sma200=0, sp500_trend_6m=0,
            dxy_zscore=0, timestamp=datetime.now().isoformat(),
            details={'warning': 'no_macro_data'},
        )

    def get_regime_features(self):
        snap = self.detect()
        return {
            'regime_bull': 1.0 if snap.regime == 'BULL' else 0.0,
            'regime_bear': 1.0 if snap.regime == 'BEAR' else 0.0,
            'regime_sideways': 1.0 if snap.regime == 'SIDEWAYS' else 0.0,
            'regime_score': snap.details.get('regime_score', 0.0),
            'regime_confidence': snap.confidence,
        }


# ============================================================
# 2. GRADUATED SIGNAL ENGINE (NEW in 1.2)
# ============================================================

@dataclass
class RegimeStrategy:
    """Strategy parameters that change per regime."""
    name: str
    regime: str
    max_position_pct: float
    entry_confidence_min: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_sl_pct: float
    trailing_tp_pct: float
    # 5-tier thresholds
    strong_buy_threshold: float
    buy_threshold: float
    sell_threshold: float
    strong_sell_threshold: float
    # Position size multipliers per tier
    tier_sizes: Dict = field(default_factory=dict)


REGIME_STRATEGIES = {
    'BULL': RegimeStrategy(
        name='BULL_AGGRESSIVE', regime='BULL',
        max_position_pct=100.0, entry_confidence_min=0.55,
        stop_loss_pct=2.5, take_profit_pct=4.0,
        trailing_sl_pct=5.0, trailing_tp_pct=12.0,
        strong_buy_threshold=0.70, buy_threshold=0.58,
        sell_threshold=0.60, strong_sell_threshold=0.70,
        tier_sizes={
            'STRONG_BUY': 1.00, 'BUY': 0.65, 'HOLD': 0.00,
            'SELL': 0.40, 'STRONG_SELL': 0.60,
        },
    ),
    'BEAR': RegimeStrategy(
        name='BEAR_DEFENSIVE', regime='BEAR',
        max_position_pct=50.0, entry_confidence_min=0.65,
        stop_loss_pct=1.5, take_profit_pct=3.0,
        trailing_sl_pct=3.0, trailing_tp_pct=8.0,
        strong_buy_threshold=0.75, buy_threshold=0.65,
        sell_threshold=0.55, strong_sell_threshold=0.65,
        tier_sizes={
            'STRONG_BUY': 0.60, 'BUY': 0.40, 'HOLD': 0.00,
            'SELL': 0.65, 'STRONG_SELL': 1.00,
        },
    ),
    'SIDEWAYS': RegimeStrategy(
        name='SIDEWAYS_NEUTRAL', regime='SIDEWAYS',
        max_position_pct=75.0, entry_confidence_min=0.60,
        stop_loss_pct=2.0, take_profit_pct=3.0,
        trailing_sl_pct=4.0, trailing_tp_pct=10.0,
        strong_buy_threshold=0.72, buy_threshold=0.60,
        sell_threshold=0.60, strong_sell_threshold=0.72,
        tier_sizes={
            'STRONG_BUY': 0.85, 'BUY': 0.55, 'HOLD': 0.00,
            'SELL': 0.55, 'STRONG_SELL': 0.85,
        },
    ),
}


def classify_signal_5tier(buy_votes, total_votes, avg_proba):
    """Compatibility wrapper: vote-based 5-tier classification."""
    if total_votes == 0:
        return 'HOLD', 50.0
    buy_ratio = buy_votes / total_votes
    if buy_ratio == 1.0:
        conf = avg_proba * 100
        return ('STRONG BUY', conf) if conf >= 70 else ('BUY', conf)
    elif buy_ratio > 0.5:
        return 'BUY', avg_proba * 100
    elif buy_ratio == 0:
        conf = (1 - avg_proba) * 100
        return ('STRONG SELL', conf) if conf >= 70 else ('SELL', conf)
    else:
        return 'HOLD', max(avg_proba, 1 - avg_proba) * 100


class GraduatedSignalEngine:
    """
    Converts raw ML probability into a 5-tier signal with position sizing.

    ML model outputs P(up) in [0, 1]:
      P(up) very high  -> STRONG_BUY  -> large long
      P(up) high       -> BUY         -> medium long
      P(up) neutral    -> HOLD        -> no position
      P(up) low        -> SELL        -> medium short
      P(up) very low   -> STRONG_SELL -> large short

    Position size = max_position_pct * tier_multiplier * confidence_boost
    Thresholds and multipliers change per regime.
    """

    def __init__(self, regime_detector):
        self.detector = regime_detector

    def classify(self, asset, ml_probability, regime=None):
        if regime is None:
            snap = self.detector.detect()
            regime = snap.regime

        strategy = REGIME_STRATEGIES.get(regime, REGIME_STRATEGIES['SIDEWAYS'])
        p = ml_probability

        if p >= strategy.strong_buy_threshold:
            tier = SignalTier.STRONG_BUY
            confidence = min(1.0, (p - strategy.strong_buy_threshold) /
                           (1.0 - strategy.strong_buy_threshold) * 0.5 + 0.75)
        elif p >= strategy.buy_threshold:
            tier = SignalTier.BUY
            confidence = 0.55 + (p - strategy.buy_threshold) / \
                        max(strategy.strong_buy_threshold - strategy.buy_threshold, 0.01) * 0.2
        elif (1 - p) >= strategy.strong_sell_threshold:
            tier = SignalTier.STRONG_SELL
            confidence = min(1.0, ((1 - p) - strategy.strong_sell_threshold) /
                           (1.0 - strategy.strong_sell_threshold) * 0.5 + 0.75)
        elif (1 - p) >= strategy.sell_threshold:
            tier = SignalTier.SELL
            confidence = 0.55 + ((1 - p) - strategy.sell_threshold) / \
                        max(strategy.strong_sell_threshold - strategy.sell_threshold, 0.01) * 0.2
        else:
            tier = SignalTier.HOLD
            confidence = 0.5

        tier_multiplier = strategy.tier_sizes.get(tier.value, 0.0)
        position_size = strategy.max_position_pct * tier_multiplier

        conf_boost = 0.8 + (confidence - 0.5) * 0.8
        position_size *= conf_boost
        position_size = round(min(position_size, strategy.max_position_pct), 1)

        if confidence < strategy.entry_confidence_min and tier != SignalTier.HOLD:
            tier = SignalTier.HOLD
            position_size = 0.0

        return SignalResult(
            asset=asset, tier=tier.value,
            ml_probability=round(ml_probability, 4),
            confidence=round(confidence, 4),
            position_size_pct=position_size,
            regime=regime, strategy=strategy.name,
            timestamp=datetime.now().isoformat(),
            features_used=len(FEATURE_COLS_V2),
        )

    def classify_batch(self, predictions, regime=None):
        return {asset: self.classify(asset, prob, regime)
                for asset, prob in predictions.items()}


# Alias for external imports
GraduatedPositionSizer = GraduatedSignalEngine


# ============================================================
# 3. REGIME-SPECIFIC MODEL TRAINING
# ============================================================

class RegimeModelManager:
    """
    Universal + 3 regime-specific models.
    Final prediction = weighted blend based on regime accuracy.
    """

    def __init__(self, regime_detector):
        self.detector = regime_detector
        self.models = {}
        self.model_dir = Path('models/broly')
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def label_regimes(self, df):
        macro_file = self.detector.macro_dir / 'macro_daily.csv'
        if not macro_file.exists():
            return pd.Series('SIDEWAYS', index=df.index)

        macro = pd.read_csv(macro_file, parse_dates=['date'])
        macro = macro.sort_values('date')

        vix_col = [c for c in macro.columns if 'vix' in c.lower()]
        sp_col = [c for c in macro.columns if 'sp500' in c.lower() or 'gspc' in c.lower()]
        if not vix_col or not sp_col:
            return pd.Series('SIDEWAYS', index=df.index)

        vix_col, sp_col = vix_col[0], sp_col[0]
        macro['sp_sma200'] = macro[sp_col].rolling(200, min_periods=50).mean()
        macro['sp_vs_sma200'] = (macro[sp_col] / macro['sp_sma200'] - 1) * 100
        macro['sp_trend_6m'] = macro[sp_col].pct_change(126) * 100

        def classify_row(row):
            vix = row.get(vix_col, 20)
            sp_vs = row.get('sp_vs_sma200', 0)
            trend = row.get('sp_trend_6m', 0)
            if pd.isna(vix) or pd.isna(sp_vs):
                return 'SIDEWAYS'
            bull_pts, bear_pts = 0, 0
            if vix < 20: bull_pts += 2
            elif vix > 28: bear_pts += 2
            if sp_vs > 3: bull_pts += 2
            elif sp_vs < -3: bear_pts += 2
            if not pd.isna(trend):
                if trend > 5: bull_pts += 1
                elif trend < -5: bear_pts += 1
            if bull_pts >= bear_pts + 2: return 'BULL'
            elif bear_pts >= bull_pts + 2: return 'BEAR'
            return 'SIDEWAYS'

        macro['regime'] = macro.apply(classify_row, axis=1)

        if 'datetime' in df.columns:
            df_dates = pd.to_datetime(df['datetime']).dt.date
        else:
            df_dates = pd.to_datetime(df.index).date

        macro['date_only'] = macro['date'].dt.date
        regime_map = dict(zip(macro['date_only'], macro['regime']))
        regimes = df_dates.map(lambda d: regime_map.get(d, 'SIDEWAYS'))
        return pd.Series(regimes.values, index=df.index)

    def train_regime_models(self, df, feature_cols, target_col='label',
                            model_factories=None, window=100):
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        if model_factories is None:
            model_factories = {
                'RF': lambda: RandomForestClassifier(n_estimators=200, max_depth=8,
                                                     random_state=42, n_jobs=-1),
                'GB': lambda: GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                          random_state=42),
            }
            try:
                import lightgbm as lgb
                model_factories['LGBM'] = lambda: lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    random_state=42, verbose=-1, n_jobs=-1)
            except ImportError:
                pass

        regimes = self.label_regimes(df)
        df = df.copy()
        df['_regime'] = regimes.values

        regime_counts = df['_regime'].value_counts()
        print(f"\n  Regime distribution in training data:")
        for r, c in regime_counts.items():
            print(f"    {r}: {c} rows ({c/len(df)*100:.1f}%)")

        results = {}

        print(f"\n  Training UNIVERSAL model ({len(df)} rows)...")
        results['UNIVERSAL'] = self._train_one_set(
            df, feature_cols, target_col, model_factories, window, 'UNIVERSAL')

        for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
            regime_df = df[df['_regime'] == regime].reset_index(drop=True)
            if len(regime_df) < 200:
                print(f"\n  Skipping {regime} model (only {len(regime_df)} rows, need 200)")
                results[regime] = None
                continue
            print(f"\n  Training {regime} model ({len(regime_df)} rows)...")
            results[regime] = self._train_one_set(
                regime_df, feature_cols, target_col, model_factories, window, regime)

        self.models = results
        return results

    def _train_one_set(self, df, feature_cols, target_col, model_factories, window, label):
        from sklearn.preprocessing import StandardScaler
        X = df[feature_cols].values
        y = df[target_col].values
        n = len(X)
        test_start = int(n * 0.7)
        correct, total = 0, 0
        step = max(1, (n - test_start) // 50)

        for i in range(test_start, n, step):
            train_start = max(0, i - window)
            X_train, y_train = X[train_start:i], y[train_start:i]
            if len(np.unique(y_train)) < 2:
                continue
            X_test, y_test = X[i:i+1], y[i]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            votes = []
            for name, factory in model_factories.items():
                try:
                    model = factory()
                    model.fit(X_train_s, y_train)
                    votes.append(int(model.predict(X_test_s)[0]))
                except:
                    pass
            if not votes:
                continue
            pred = 1 if sum(votes) > len(votes) / 2 else 0
            if pred == int(y_test): correct += 1
            total += 1

        acc = round(correct / total * 100, 2) if total > 0 else 0
        print(f"    [{label}] Accuracy: {acc}% ({correct}/{total})")
        return {'accuracy': acc, 'samples': total, 'correct': correct, 'label': label}

    def get_prediction_weight(self, regime):
        universal = self.models.get('UNIVERSAL')
        regime_model = self.models.get(regime)
        if regime_model is None or universal is None:
            return {'UNIVERSAL': 1.0, regime: 0.0}
        u_acc = universal.get('accuracy', 50)
        r_acc = regime_model.get('accuracy', 50)
        if r_acc > u_acc + 2: return {'UNIVERSAL': 0.4, regime: 0.6}
        elif r_acc > u_acc - 2: return {'UNIVERSAL': 0.5, regime: 0.5}
        else: return {'UNIVERSAL': 0.7, regime: 0.3}


# ============================================================
# 4. STOP-LOSS / TAKE-PROFIT SYSTEM
# ============================================================

@dataclass
class PositionState:
    """Tracks state of an open position for SL/TP monitoring."""
    asset: str
    side: str
    entry_price: float
    entry_time: str
    quantity: float
    peak_price: float
    trough_price: float
    regime: str
    strategy: str
    signal_tier: str           # NEW in 1.2
    sl_price: float
    tp_price: float
    trailing_sl_price: float
    status: str = 'OPEN'


class StopLossManager:
    """
    Full SL/TP system with tier-aware adjustments.
    STRONG signals get wider SL (more room to breathe).
    Normal signals get base SL.
    """

    TIER_SL_MULTIPLIERS = {
        'STRONG_BUY': 1.30, 'BUY': 1.00,
        'SELL': 1.00, 'STRONG_SELL': 1.30,
    }
    TIER_TP_MULTIPLIERS = {
        'STRONG_BUY': 1.40, 'BUY': 1.00,
        'SELL': 1.00, 'STRONG_SELL': 1.40,
    }

    def __init__(self, state_file='data/broly_positions.json'):
        self.state_file = Path(state_file)
        self.positions = {}
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                for asset, pdata in data.items():
                    if 'signal_tier' not in pdata:
                        pdata['signal_tier'] = 'BUY'
                    self.positions[asset] = PositionState(**pdata)
            except:
                self.positions = {}

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {asset: asdict(pos) for asset, pos in self.positions.items()}
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def open_position(self, asset, side, entry_price, quantity,
                      strategy, signal_tier='BUY'):
        sl_mult = self.TIER_SL_MULTIPLIERS.get(signal_tier, 1.0)
        tp_mult = self.TIER_TP_MULTIPLIERS.get(signal_tier, 1.0)
        sl_pct = strategy.stop_loss_pct * sl_mult
        tp_pct = strategy.take_profit_pct * tp_mult

        if side == 'LONG':
            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + tp_pct / 100)
            trailing_sl = entry_price * (1 - strategy.trailing_sl_pct / 100)
        else:
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - tp_pct / 100)
            trailing_sl = entry_price * (1 + strategy.trailing_sl_pct / 100)

        pos = PositionState(
            asset=asset, side=side,
            entry_price=round(entry_price, 4),
            entry_time=datetime.now().isoformat(),
            quantity=quantity, peak_price=entry_price, trough_price=entry_price,
            regime=strategy.regime, strategy=strategy.name,
            signal_tier=signal_tier,
            sl_price=round(sl_price, 4), tp_price=round(tp_price, 4),
            trailing_sl_price=round(trailing_sl, 4),
        )
        self.positions[asset] = pos
        self._save_state()
        logger.info(f"Position opened: {asset} {side} @ {entry_price:.2f} "
                     f"[{signal_tier}] SL={sl_price:.2f} TP={tp_price:.2f}")
        return pos

    def update(self, asset, current_price):
        if asset not in self.positions:
            return None
        pos = self.positions[asset]
        if pos.status != 'OPEN':
            return None

        if current_price > pos.peak_price:
            pos.peak_price = current_price
        if current_price < pos.trough_price:
            pos.trough_price = current_price

        strategy = REGIME_STRATEGIES.get(pos.regime, REGIME_STRATEGIES['SIDEWAYS'])

        if pos.side == 'LONG':
            new_trailing_sl = pos.peak_price * (1 - strategy.trailing_sl_pct / 100)
            pos.trailing_sl_price = max(pos.trailing_sl_price, new_trailing_sl)
            pnl_pct = (current_price / pos.entry_price - 1) * 100
        else:
            new_trailing_sl = pos.trough_price * (1 + strategy.trailing_sl_pct / 100)
            pos.trailing_sl_price = min(pos.trailing_sl_price, new_trailing_sl)
            pnl_pct = (pos.entry_price / current_price - 1) * 100

        if pos.side == 'LONG':
            if current_price <= pos.sl_price:
                return self._close(asset, 'FIXED_SL', pnl_pct)
            if current_price >= pos.tp_price:
                return self._close(asset, 'FIXED_TP', pnl_pct)
            if current_price <= pos.trailing_sl_price and pnl_pct > 0:
                return self._close(asset, 'TRAILING_SL', pnl_pct)
        else:
            if current_price >= pos.sl_price:
                return self._close(asset, 'FIXED_SL', pnl_pct)
            if current_price <= pos.tp_price:
                return self._close(asset, 'FIXED_TP', pnl_pct)
            if current_price >= pos.trailing_sl_price and pnl_pct > 0:
                return self._close(asset, 'TRAILING_SL', pnl_pct)

        self._save_state()
        return None

    def _close(self, asset, reason, pnl_pct):
        pos = self.positions[asset]
        pos.status = reason
        self._save_state()
        result = {
            'action': 'CLOSE', 'asset': asset, 'side': pos.side,
            'signal_tier': pos.signal_tier, 'reason': reason,
            'entry_price': pos.entry_price, 'pnl_pct': round(pnl_pct, 2),
            'regime': pos.regime,
            'duration': str(datetime.now() - datetime.fromisoformat(pos.entry_time)),
        }
        logger.info(f"Position closed: {asset} [{pos.signal_tier}] | "
                     f"Reason: {reason} | P&L: {pnl_pct:+.2f}%")
        return result

    def get_open_positions(self):
        return {a: p for a, p in self.positions.items() if p.status == 'OPEN'}

    def remove_closed(self):
        self.positions = {a: p for a, p in self.positions.items() if p.status == 'OPEN'}
        self._save_state()


# ============================================================
# 5. DISCORD + TELEGRAM ALERT SYSTEM
# ============================================================

class AlertManager:
    """
    Sends alerts via Discord webhook and/or Telegram bot.
    Config via environment variables or .env file.
    """

    def __init__(self, config_file='.env'):
        self.discord_url = os.environ.get('DISCORD_WEBHOOK_URL', '')
        self.telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
        if not self.discord_url and Path(config_file).exists():
            self._load_env(config_file)
        self.enabled_discord = bool(self.discord_url)
        self.enabled_telegram = bool(self.telegram_token and self.telegram_chat_id)

    def _load_env(self, filepath):
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, val = line.split('=', 1)
                        key, val = key.strip(), val.strip().strip('"').strip("'")
                        if key == 'DISCORD_WEBHOOK_URL': self.discord_url = val
                        elif key == 'TELEGRAM_BOT_TOKEN': self.telegram_token = val
                        elif key == 'TELEGRAM_CHAT_ID': self.telegram_chat_id = val
        except:
            pass

    def send(self, message, level='INFO'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"[{timestamp}] [{level}] {message}"
        if self.enabled_discord: self._send_discord(formatted)
        if self.enabled_telegram: self._send_telegram(formatted)
        if not self.enabled_discord and not self.enabled_telegram:
            logger.info(f"[ALERT] {formatted}")
            print(f"  [ALERT] {formatted}")

    def _send_discord(self, message):
        import urllib.request, urllib.error
        payload = json.dumps({'content': message}).encode('utf-8')
        req = urllib.request.Request(
            self.discord_url, data=payload,
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Discord alert failed: {e}")

    def _send_telegram(self, message):
        import urllib.request, urllib.parse, urllib.error
        url = (f"https://api.telegram.org/bot{self.telegram_token}"
               f"/sendMessage?chat_id={self.telegram_chat_id}"
               f"&text={urllib.parse.quote(message)}")
        try:
            urllib.request.urlopen(url, timeout=10)
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")

    def send_regime_report(self, snapshot):
        msg = (f"=== BROLY REGIME REPORT ===\n"
               f"Regime: {snapshot.regime} ({snapshot.confidence:.0%} confidence)\n"
               f"VIX: {snapshot.vix_level:.1f} (z={snapshot.vix_zscore:+.2f})\n"
               f"S&P vs SMA200: {snapshot.sp500_vs_sma200:+.1f}%\n"
               f"6M Trend: {snapshot.sp500_trend_6m:+.1f}%\n"
               f"DXY z-score: {snapshot.dxy_zscore:+.2f}\n"
               f"Score: {snapshot.details.get('regime_score', 0):+.2f}")
        self.send(msg, level='REGIME')

    def send_trade_alert(self, action, asset, side, price, confidence,
                         regime, strategy_name, signal_tier='BUY',
                         position_size_pct=0, sl_price=0, tp_price=0):
        msg = (f"=== BROLY TRADE ===\n"
               f"{action}: {asset} {side}\n"
               f"Signal: {signal_tier} | Confidence: {confidence:.1%}\n"
               f"Price: {price:.2f} | Size: {position_size_pct:.0f}%\n"
               f"Regime: {regime} | Strategy: {strategy_name}\n")
        if sl_price:
            msg += f"SL: {sl_price:.2f} | TP: {tp_price:.2f}\n"
        self.send(msg, level='TRADE')

    def send_sl_tp_alert(self, close_info):
        pnl = close_info['pnl_pct']
        pnl_mark = 'WIN' if pnl > 0 else 'LOSS'
        msg = (f"=== BROLY EXIT ===\n"
               f"{close_info['reason']}: {close_info['asset']} "
               f"[{close_info.get('signal_tier', '?')}]\n"
               f"Side: {close_info['side']}\n"
               f"Entry: {close_info['entry_price']:.2f}\n"
               f"[{pnl_mark}] P&L: {pnl:+.2f}%\n"
               f"Duration: {close_info.get('duration', '?')}\n"
               f"Regime: {close_info.get('regime', '?')}")
        self.send(msg, level='EXIT')

    def send_signal_summary(self, signals):
        msg = "=== BROLY SIGNALS ===\n"
        for asset, sig in signals.items():
            size = f"{sig.position_size_pct:.0f}%" if sig.position_size_pct > 0 else "---"
            msg += f"  {asset:6s}: {sig.tier:12s} | {sig.confidence:.0%} | {size}\n"
        if signals:
            msg += f"Regime: {list(signals.values())[0].regime}"
        self.send(msg, level='SIGNALS')

    def send_portfolio_summary(self, positions, equity=0, regime=''):
        msg = f"=== BROLY PORTFOLIO === [{regime}]\n"
        if equity: msg += f"Equity: {equity:,.0f}\n"
        msg += f"Positions ({len(positions)}):\n"
        for asset, pos in positions.items():
            side_mark = 'L' if pos.side == 'LONG' else 'S'
            msg += (f"  [{side_mark}] {asset} [{pos.signal_tier}] "
                    f"@ {pos.entry_price:.2f} | "
                    f"SL={pos.sl_price:.2f} TP={pos.tp_price:.2f}\n")
        self.send(msg, level='PORTFOLIO')


# ============================================================
# 6. BROLY MAIN CLASS
# ============================================================

class Broly:
    """
    Broly 1.2 orchestrator.

    NEW in 1.2:
      - classify_signal() -> 5-tier graduated output
      - Optimal 15 V2 features (76.1% accuracy)
      - Position sizing proportional to signal strength
      - Tier-adjusted SL/TP (STRONG signals get wider room)

    Usage:
        broly = Broly()
        regime = broly.detect_regime()
        signal = broly.classify_signal('DAX', ml_probability=0.72)
        broly.open_position('DAX', signal, price=21500)
    """

    VERSION = '1.2'

    def __init__(self, macro_data_dir='macro_data', config_file='.env'):
        print(f"\n  BROLY v{self.VERSION} initializing...")
        self.regime_detector = MarketRegimeDetector(macro_data_dir)
        self.signal_engine = GraduatedSignalEngine(self.regime_detector)
        self.model_manager = RegimeModelManager(self.regime_detector)
        self.sl_manager = StopLossManager()
        self.alerts = AlertManager(config_file)
        self._current_regime = None
        self._current_strategy = None
        print(f"  BROLY v{self.VERSION} ready")
        print(f"    Features: {len(FEATURE_COLS_V2)} optimal V2")
        print(f"    Signal tiers: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL")
        disc = 'ON' if self.alerts.enabled_discord else 'OFF'
        tele = 'ON' if self.alerts.enabled_telegram else 'OFF'
        print(f"    Discord: {disc} | Telegram: {tele}")

    def detect_regime(self):
        snap = self.regime_detector.detect()
        self._current_regime = snap
        self._current_strategy = REGIME_STRATEGIES.get(snap.regime, REGIME_STRATEGIES['SIDEWAYS'])
        return snap

    def print_regime(self):
        snap = self.detect_regime()
        print(f"\n  {'='*55}")
        print(f"  MARKET REGIME: {snap.regime} ({snap.confidence:.0%} confidence)")
        print(f"  {'='*55}")
        print(f"  VIX:           {snap.vix_level:>8.1f}  (z-score: {snap.vix_zscore:+.2f})")
        print(f"  S&P vs SMA200: {snap.sp500_vs_sma200:>+8.1f}%")
        print(f"  6M Trend:      {snap.sp500_trend_6m:>+8.1f}%")
        print(f"  DXY z-score:   {snap.dxy_zscore:>+8.2f}")
        print(f"  Regime score:  {snap.details.get('regime_score', 0):>+8.2f}"
              f"  (-1=bear, +1=bull)")
        print(f"  Strategy:      {self._current_strategy.name}")
        print(f"  Max position:  {self._current_strategy.max_position_pct:.0f}%")
        print(f"  SL/TP:         -{self._current_strategy.stop_loss_pct}% / "
              f"+{self._current_strategy.take_profit_pct}%")
        print(f"  Trailing SL:   -{self._current_strategy.trailing_sl_pct}%")
        print(f"  {'='*55}")

    def classify_signal(self, asset, ml_probability, regime=None):
        if regime is None and self._current_regime is None:
            self.detect_regime()
        return self.signal_engine.classify(
            asset, ml_probability,
            regime or (self._current_regime.regime if self._current_regime else None))

    def classify_batch(self, predictions):
        if self._current_regime is None:
            self.detect_regime()
        return self.signal_engine.classify_batch(predictions, self._current_regime.regime)

    def get_strategy(self, regime=None):
        if regime is None:
            if self._current_regime is None:
                self.detect_regime()
            regime = self._current_regime.regime
        return REGIME_STRATEGIES.get(regime, REGIME_STRATEGIES['SIDEWAYS'])

    def open_position(self, asset, signal, price=None, quantity=None):
        if signal.tier == 'HOLD' or signal.position_size_pct == 0:
            return None
        strategy = self.get_strategy(signal.regime)
        side = 'LONG' if signal.tier in ('STRONG_BUY', 'BUY') else 'SHORT'
        if price is None:
            logger.warning("Price required to open position")
            return None
        if quantity is None:
            quantity = 1
        pos = self.sl_manager.open_position(
            asset, side, price, quantity, strategy, signal_tier=signal.tier)
        self.alerts.send_trade_alert(
            action='OPEN', asset=asset, side=side, price=price,
            confidence=signal.confidence, regime=signal.regime,
            strategy_name=strategy.name, signal_tier=signal.tier,
            position_size_pct=signal.position_size_pct,
            sl_price=pos.sl_price, tp_price=pos.tp_price)
        return pos

    def check_position(self, asset, current_price):
        result = self.sl_manager.update(asset, current_price)
        if result: self.alerts.send_sl_tp_alert(result)
        return result

    def check_all_positions(self, prices):
        actions = []
        for asset, price in prices.items():
            result = self.check_position(asset, price)
            if result: actions.append(result)
        return actions

    def train_regime_models(self, df, feature_cols=None, target_col='label'):
        if feature_cols is None:
            feature_cols = FEATURE_COLS_V2
        return self.model_manager.train_regime_models(df, feature_cols, target_col)

    def alert(self, message, level='INFO'):
        self.alerts.send(f"BROLY | {message}", level)

    def send_regime_report(self):
        snap = self.detect_regime()
        self.alerts.send_regime_report(snap)

    def send_portfolio_summary(self, equity=0):
        positions = self.sl_manager.get_open_positions()
        regime = self._current_regime.regime if self._current_regime else 'UNKNOWN'
        self.alerts.send_portfolio_summary(positions, equity, regime)


# ============================================================
# CLI: Quick test
# ============================================================

def main():
    print("=" * 60)
    print("  BROLY 1.2 -- SYSTEM TEST")
    print("=" * 60)

    broly = Broly()

    # Test 1: Regime Detection
    print("\n" + "=" * 60)
    print("  TEST 1: REGIME DETECTION")
    print("=" * 60)
    broly.print_regime()

    # Test 2: Strategy Per Regime
    print("\n" + "=" * 60)
    print("  TEST 2: STRATEGY PER REGIME (5-tier)")
    print("=" * 60)
    for rname in ['BULL', 'BEAR', 'SIDEWAYS']:
        s = REGIME_STRATEGIES[rname]
        print(f"\n  {rname} ({s.name}):")
        print(f"    Max position:   {s.max_position_pct:.0f}%")
        print(f"    Min confidence: {s.entry_confidence_min:.0%}")
        print(f"    SL: -{s.stop_loss_pct}% | TP: +{s.take_profit_pct}%")
        print(f"    Trailing SL: -{s.trailing_sl_pct}%")
        print(f"    Thresholds: STRONG_BUY>{s.strong_buy_threshold} "
              f"BUY>{s.buy_threshold} "
              f"SELL<{1-s.sell_threshold:.2f} "
              f"STRONG_SELL<{1-s.strong_sell_threshold:.2f}")
        sizes = " ".join(f"{t}={v:.0%}" for t, v in s.tier_sizes.items())
        print(f"    Tier sizes: {sizes}")

    # Test 3: Signal Classification
    print("\n" + "=" * 60)
    print("  TEST 3: GRADUATED SIGNAL CLASSIFICATION")
    print("=" * 60)
    test_probs = [0.85, 0.72, 0.62, 0.52, 0.45, 0.35, 0.22, 0.10]
    regime = broly._current_regime.regime if broly._current_regime else 'SIDEWAYS'
    print(f"\n  Current regime: {regime}")
    print(f"  {'Prob':>6s}  {'Tier':>14s}  {'Conf':>6s}  {'Size':>6s}  {'Side':>6s}")
    print(f"  {'-'*48}")
    for p in test_probs:
        sig = broly.classify_signal('DAX', p)
        side = 'LONG' if sig.tier in ('STRONG_BUY', 'BUY') else \
               'SHORT' if sig.tier in ('SELL', 'STRONG_SELL') else '---'
        print(f"  {p:>6.2f}  {sig.tier:>14s}  {sig.confidence:>5.0%}  "
              f"{sig.position_size_pct:>5.1f}%  {side:>6s}")

    # Test 4: Tier-Adjusted SL/TP
    print("\n" + "=" * 60)
    print("  TEST 4: TIER-ADJUSTED SL/TP")
    print("=" * 60)
    strategy = broly.get_strategy()
    for tier in ['BUY', 'STRONG_BUY']:
        pos = broly.sl_manager.open_position('DAX', 'LONG', 21500.0, 10,
                                              strategy, signal_tier=tier)
        sl_d = abs(pos.entry_price - pos.sl_price)
        tp_d = abs(pos.tp_price - pos.entry_price)
        print(f"  {tier:>12s} @ 21500: SL={pos.sl_price:.2f} ({sl_d:.0f}pts) | "
              f"TP={pos.tp_price:.2f} ({tp_d:.0f}pts)")
        broly.sl_manager.positions.clear()

    # Test 5: SL/TP Price Walk
    print("\n  Simulating price movement (BUY tier)...")
    pos = broly.sl_manager.open_position('DAX', 'LONG', 21500.0, 10,
                                          strategy, signal_tier='BUY')
    for p in [21550, 21700, 21800, 21750, 21600, 21400, 21300]:
        result = broly.sl_manager.update('DAX', p)
        pos_now = broly.sl_manager.positions.get('DAX')
        trail = pos_now.trailing_sl_price if pos_now else 0
        status = pos_now.status if pos_now else '?'
        print(f"    Price {p:.0f} | trailing_sl={trail:.2f} | status={status}")
        if result:
            print(f"    *** TRIGGERED: {result['reason']} | P&L: {result['pnl_pct']:+.2f}%")
            break
    broly.sl_manager.positions.clear()

    # Test 6: V2 Features
    print("\n" + "=" * 60)
    print("  TEST 5: OPTIMAL V2 FEATURES")
    print("=" * 60)
    print(f"  {len(FEATURE_COLS_V2)} features (76.1% accuracy):")
    for i, feat in enumerate(FEATURE_COLS_V2, 1):
        print(f"    {i:>2d}. {feat}")

    # Test 7: Alerts
    print("\n" + "=" * 60)
    print("  TEST 6: ALERT SYSTEM")
    print("=" * 60)
    disc = 'ON' if broly.alerts.enabled_discord else 'OFF'
    tele = 'ON' if broly.alerts.enabled_telegram else 'OFF'
    print(f"  Discord: {disc} | Telegram: {tele}")
    broly.alert("Broly 1.2 system test complete!")

    # Test 8: Model Training
    print("\n" + "=" * 60)
    print("  TEST 7: REGIME MODEL TRAINING (optional)")
    print("=" * 60)
    try:
        from hourly_trading_system import load_data, build_hourly_features, update_all_data
        from features_v2 import build_features_v2_hourly

        print("  Loading DAX data...")
        try: update_all_data(['DAX'])
        except: pass

        df_raw = load_data('DAX')
        if df_raw is not None:
            df, all_cols = build_features_v2_hourly(df_raw,
                                                     original_builder=build_hourly_features)
            available = [c for c in FEATURE_COLS_V2 if c in df.columns]
            missing = [c for c in FEATURE_COLS_V2 if c not in df.columns]
            if missing: print(f"  WARNING: Missing features: {missing}")

            df_clean = df.dropna(subset=available + ['label']).reset_index(drop=True)
            print(f"  Data: {len(df_clean)} rows, {len(available)} features")
            results = broly.train_regime_models(df_clean, available)

            print(f"\n  Model Results:")
            for name, info in results.items():
                if info:
                    print(f"    {name:12s}: {info['accuracy']:.1f}% "
                          f"({info['samples']} samples)")
                else:
                    print(f"    {name:12s}: SKIPPED (insufficient data)")

            current_regime = broly._current_regime.regime
            weights = broly.model_manager.get_prediction_weight(current_regime)
            print(f"\n  Prediction weights for {current_regime} regime:")
            for model, w in weights.items():
                print(f"    {model}: {w:.0%}")
        else:
            print("  No DAX data available, skipping")
    except ImportError as e:
        print(f"  Cannot import: {e}")
        print("  (Requires hourly_trading_system.py + features_v2.py)")

    print(f"\n{'='*60}")
    print(f"  BROLY 1.2 -- ALL TESTS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
