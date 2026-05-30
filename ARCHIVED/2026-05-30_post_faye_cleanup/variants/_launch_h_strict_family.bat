@echo off
set V2_DATA_SNAPSHOT=data\_reliability_hrst_snapshot_desktop_20260515_154801
set RELIABILITY_K=5
C:\Users\Alex\algo_trading\venv\Scripts\python.exe crypto_trading_system_ed_h_strict_family.py DV ETH, 7h --replay 1440 --no-persist --no-data-update --grid-tag H_STRICT_FAMILY
