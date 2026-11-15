@echo off
cd /d "%~dp0"

echo ============================================================
echo  LocalCUA selfcheck: environment only, no clicks/typing
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m tools.selfcheck
) else (
    python -m tools.selfcheck
)

echo.
echo Done. Send me the [FAIL] lines above, or the newest
echo report.txt under data\selfcheck\
pause
