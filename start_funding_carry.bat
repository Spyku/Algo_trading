@echo off
title Funding-Carry Paper Bot - ETH
cd /d "%~dp0"

REM Auto-detect venv python across machines (first existing wins).
if exist "C:\Users\alexa\algo_trading\venv\Scripts\python.exe" (
    set "PYTHON=C:\Users\alexa\algo_trading\venv\Scripts\python.exe"
) else if exist "C:\algo_trading\venv\Scripts\python.exe" (
    set "PYTHON=C:\algo_trading\venv\Scripts\python.exe"
) else (
    set "PYTHON=C:\Users\Alex\algo_trading\venv\Scripts\python.exe"
)

:loop
set "ts=%date:~6,4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "ts=%ts: =0%"
if not exist "%~dp0logs\trader" mkdir "%~dp0logs\trader"
set "logfile=%~dp0logs\trader\funding_carry_%ts%.log"
echo [%date% %time%] Starting funding-carry paper bot (ETH, --loop)...
echo [%date% %time%] Log: %logfile%
"%PYTHON%" "%~dp0tools\tee_launcher.py" "%logfile%" "%PYTHON%" -u "%~dp0tools\funding_carry_eth.py" --loop
echo [%date% %time%] Funding-carry exited (code %errorlevel%). Restarting in 30s...
echo Press Ctrl+C to stop, or wait for auto-restart.
timeout /t 30
goto loop
