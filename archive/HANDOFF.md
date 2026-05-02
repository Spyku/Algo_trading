# ETH Trader Handoff — 2026-04-18

Switching from desktop (Drive mirror) to laptop (actual prod). **Read this first** when resuming on the laptop.

## Priority 1 — Verify Config B is live on ETH trader

**Config B** = rally_cooldown in `config/regime_config_ed.json`:
```
h_short=18, h_long=36, t_short=3.5%, t_long=5.0%, cd=16h
```

**Verification:** live trader log should print
```
ETH rally-cd: rr18=X.XX% rr36=Y.YY%
```
If it still says `rr8=...`, it loaded the old config. Restart trader via `start_ed_v2.bat`.

Quick checks:
- `/status` on Telegram
- Live terminal window
- On restart, first cycle confirms which config loaded

State at handoff:
- Trader was restarted at 08:56:59 local on 2026-04-18 (after config was set to B at 08:19)
- Log was 0 bytes on Drive (open-file, Drive doesn't sync) — couldn't verify from desktop
- Previous session (06:09) was still on OLD config (`rr8=`) — expected, pre-restart

## Priority 2 — ⚠ Mode G auto-overwrites config

Today at 06:35 local, Mode G ran and auto-wrote OLD config back:
```
rr8h≥3.0% OR rr36h≥5.5%, cd=30h
```
Config B was manually re-applied at 08:19.

**Before running Mode G again**:
- Back up `config/regime_config_ed.json`, OR
- Patch Mode G to gate the auto-write behind a `--write-config` flag

Why Config B was chosen over Mode G's output: user picked the reactive/30d winner (cd=16h), Mode G's 60d winner was the old config. Mode G doesn't know about this preference.

## Priority 3 — BTC decision pending

- `enabled=false` in config. On-chain features now wired (with SOPR from BGeometrics) but no Mode D validation yet.
- Before enabling BTC trading: run Mode D sweep (5/6/7/8h) and check if any `oc_*` or `sopr` features earn Grade 3+.

## Priority 4 — ETH Mode D (was pending from original on-chain plan)

Run Mode D on ETH 5,6,7,8h to see if any `oc_*` features earn Grade 3+ selection.

## Code + data state (good)

- Commits on `origin/main`:
  - `334bde5` — Rally-cooldown offline catch-up scan
  - `dc6562d` — On-chain features (ETH + BTC) + Engine Reference Card
- `data/macro_data/onchain_btc.csv` + `onchain_eth.csv` fresh (Apr 17 13:23)
- macro_hourly / macro_daily / fear_greed / cross_asset / derivatives — Apr 17 02:00 nightly
- `config/` is gitignored → reaches prod only via Google Drive sync, not git
- Position state: ETH `cash`, last trade 2026-04-16 12:01 +1.25%

## Known issues observed today

- Transient DNS error at 06:10 (`get_balances failed ... getaddrinfo failed`) — position sync skipped for that tick. Watch if recurring.
