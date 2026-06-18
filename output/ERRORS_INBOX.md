# Runtime Errors Inbox (Ed V2 trader)

Appended automatically by the live trader and helpers. Rate-limited to
1 entry per unique key per hour. Severity: info / warn / critical.

**Triage:** review → fix upstream → clear file (or move old entries to archive).

---

- `2026-05-24 17:43:34` ⚠ **WARN** `deriv_partial_BTC_perp_klines` — ⚠ BTC derivatives partial download — missing: ['perp_klines']. CSV written with 2/3 sub-sources; deriv_* features that depend on missing sources will be absent.
- `2026-05-24 17:43:38` 🚨 **CRITICAL** `deriv_total_fail_BNB` — 🚨 BNB derivatives: ALL THREE sub-sources failed (funding, OI, perp_klines). No data written. Check Binance API status.
- `2026-05-24 17:47:22` ⚠ **WARN** `deriv_partial_SOL_funding_rate` — ⚠ SOL derivatives partial download — missing: ['funding_rate']. CSV written with 2/3 sub-sources; deriv_* features that depend on missing sources will be absent.
- `2026-06-05 22:10:56` 🚨 **CRITICAL** `stale_wall_ETH_5` — 🛑 ETH 5h: REFUSING — freshest bar in df_full is 168.2h old (wall clock). Data pipeline not updating. Check downloads.
- `2026-06-08 16:00:52` ⚠ **WARN** `feat_partial_ETH_8` — ⚠ ETH 8h: 4/24 configured features missing from current build (['kama_10', 'vol_of_vol_24h', 'vol_of_vol_8h']...) — using remaining 20 features.
- `2026-06-15 23:31:50` ⚠ **WARN** `regime_named_unknown_bogus_xyz` — 🚨 Unknown regime detector name 'bogus_xyz' — refusing to trade. Valid: sma24>sma100, sma48>sma100, sma168>sma480, price>sma72, vol_calm, tsmom_672h, tsmom_168h.
