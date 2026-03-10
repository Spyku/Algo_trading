@echo off
cd /d C:\algo_trading\engine

echo ============================================================
echo   GIT PUSH - Algo Trading Engine
echo ============================================================
echo.

git status
echo.

git add -A

echo.
echo Files staged:
git diff --cached --name-only
echo.

set /p MSG="Commit message (or press Enter for 'Update trading system'): "
if "%MSG%"=="" set MSG=Update trading system

git commit -m "%MSG%"
echo.

git push origin main
echo.

echo ============================================================
echo   DONE - check above for any errors
echo ============================================================
pause
