# Literature V3 — Feature & Enhancement Ideas

## Status: 20/50 reviewed, 13 kept, 7 skipped, 30 pending

---

## KEPT (approved for testing)

| # | Rank | Name | Category | Notes |
|---|------|------|----------|-------|
| 1 | 1 | Asymmetric Cost Function | Methodology | Penalize wrong BUYs more via sample_weight. Test multipliers 1.5x, 2x, 3x |
| 2 | 2 | Volatility-Scaled Returns | Methodology | Divide returns by rolling vol before labeling. Re-runs full Mode D |
| 3 | 3 | US Market Hours Flag | Calendar | Binary: NYSE open (14:30-21:00 UTC). Handle DST |
| 4 | 5 | Hurst Exponent (rolling) | Fractal | DFA method, windows 48h + 120h. Trending vs mean-reverting |
| 5 | 6 | Volatility of Volatility | Volatility | std(volatility_12h) over 24h and 48h windows |
| 6 | 10 | DXY Acceleration | Cross-asset | 2nd derivative of DXY. Uses existing m_dxy_chg1d data |
| 7 | 11 | KAMA Slope | Technical | Kaufman Adaptive MA slope. **MUST compare vs sma20_to_sma50h, price_to_sma20h/50h/100h** |
| 8 | 12 | Ehlers Fisher Transform | Technical | Gaussian-normalized oscillator. **MUST compare vs rsi_14h, stoch_k_14h, bb_position_20h** |
| 9 | 14 | Approximate Entropy (ApEn) | Information Theory | Predictability measure, nolds library. Windows 48h + 120h |
| 10 | 16 | BTC Dominance ROC | Cross-asset | Rate of change of BTC market cap dominance. CoinGecko API |
| 11 | 18 | Realized Skewness & Kurtosis | Volatility | Higher moments of hourly returns. Windows 24h + 72h. 4 features |
| 12 | 20 | Connors RSI | Technical | RSI(3) + streak + percentile rank composite |
| 13 | 21 | Anchored Expanding Window | Methodology | Expanding from anchor vs fixed rolling. Conflicts with rolling window design |

---

## SKIPPED

| Rank | Name | Reason |
|------|------|--------|
| 4 | Month-of-Year Encoding | Too few samples per month in training windows (48-200h) |
| 7 | CUSUM Structural Break | Overlap with Hurst + VoV already kept |
| 8 | Amihud Illiquidity | BTC/ETH too liquid at $6k positions |
| 9 | Parkinson Range | Redundant with existing GK volatility + 5 other vol features |
| 13 | Market Stress Index | Derived from existing features, trees learn this already |
| 15 | Roll Spread | Noisy on hourly, BTC/ETH spread negligible |
| 17 | Futures Expiry Proximity | Sparse signal, only 4 quarterly events/year |
| 19 | Gold/BTC Ratio Momentum | Linear combo of existing gold + BTC features |

---

## PENDING REVIEW (ranks 22-50)

### Technical Advanced
| Rank | Name | Description |
|------|------|-------------|
| 23 | Chaikin Money Flow | Volume-weighted accumulation/distribution over N periods |
| 42 | Hilbert Transform Dominant Cycle | Extracts current dominant cycle period for adaptive oscillators |

### Cross-Asset Advanced
| Rank | Name | Description |
|------|------|-------------|
| 22 | Yield Curve Slope (10Y-2Y) | Classic recession/risk indicator from Fed data |
| 30 | Equity Put/Call Ratio | CBOE put/call, contrarian signal for crypto |

### Information Theory
| Rank | Name | Description |
|------|------|-------------|
| 24 | Lempel-Ziv Complexity | Compression-based complexity of direction sequence |
| 37 | Transfer Entropy (BTC→altcoin) | Directional info flow, computationally heavy |
| 39 | Mutual Information Decay | MI between returns at increasing lags |

### Fractal & Scaling
| Rank | Name | Description |
|------|------|-------------|
| 25 | Fractal Dimension (Higuchi) | Price path complexity measure |
| 38 | Multifractal Spectrum (MFDFA) | Singularity spectrum width, regime indicator |
| 40 | DCCA Detrended Cross-Correlation | Scale-dependent correlation between assets |

### Sentiment
| Rank | Name | Description |
|------|------|-------------|
| 26 | Funding Rate Momentum | Z-score and ROC of funding rate (extends V1 derivatives) |
| 27 | Long/Short Ratio | Binance top trader ratio, contrarian signal |
| 36 | Google Trends Momentum | ROC of "Bitcoin" search interest |
| 41 | Crypto Twitter Sentiment | LunarCrush/Santiment aggregated score (paid API) |

### Regime & State
| Rank | Name | Description |
|------|------|-------------|
| 29 | Dispersion Ratio | Cross-sectional vol / avg individual vol |
| 43 | Markov-Switching Variance | Two-state vol model, similar to HMM (V2) |

### On-Chain V2
| Rank | Name | Description |
|------|------|-------------|
| 31 | Exchange Netflow | Net BTC in/out of exchanges (Glassnode, paid?) |
| 32 | MVRV Z-Score | Market Value / Realized Value (Glassnode) |
| 33 | SOPR | Spent Output Profit Ratio (Glassnode) |
| 34 | Stablecoin Supply Ratio | BTC mcap / stablecoin supply |
| 35 | Active Address Momentum | 30d/365d active address ratio |

### Methodology
| Rank | Name | Description |
|------|------|-------------|
| 28 | Temporal Cross-Validation | Multiple non-overlapping test folds to reduce variance |

### Microstructure (likely impractical)
| Rank | Name | Description |
|------|------|-------------|
| 44 | VPIN | Needs tick data, poor on hourly |
| 45 | Kyle's Lambda | Needs signed volume at tick level |
| 46 | Trade Intensity Ratio | Needs trade count, not in OHLCV |

### Nonlinear Dynamics (computationally expensive)
| Rank | Name | Description |
|------|------|-------------|
| 47 | Recurrence Rate | O(n^2) distance matrix, slow |
| 48 | RQA Determinism | Same computational issues |
| 49 | Lyapunov Exponent | Very noisy estimates on hourly data |

### Data Limitation
| Rank | Name | Description |
|------|------|-------------|
| 50 | RV Signature | Needs sub-hourly data |
