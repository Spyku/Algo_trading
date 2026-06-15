@echo off
title Ed Optimizer
cd /d "%~dp0"

REM Auto-detect machine: Desktop vs Laptop
if exist "C:\algo_trading\venv\Scripts\python.exe" (
    set "PYTHON=C:\algo_trading\venv\Scripts\python.exe"
) else (
    set "PYTHON=C:\Users\Alex\algo_trading\venv\Scripts\python.exe"
)

:loop
set "ts=%date:~6,4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "ts=%ts: =0%"
if not exist "%~dp0logs\trader" mkdir "%~dp0logs\trader"
set "logfile=%~dp0logs\trader\optimizer_%ts%.log"
echo [%date% %time%] Starting optimizer bot...
echo [%date% %time%] Log: %logfile%
"%PYTHON%" "%~dp0tools\tee_launcher.py" "%logfile%" "%PYTHON%" -u "%~dp0crypto_optimizer_bot.py"
echo [%date% %time%] Bot exited (code %errorlevel%). Restarting in 30s...
echo Press Ctrl+C to stop, or wait for auto-restart.
timeout /t 30
goto loop
