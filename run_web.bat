@echo off
cd /d "%~dp0"

rem Prefer project venv, fall back to system python
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

rem Skip install when deps are already present (works offline)
"%PY%" -c "import fastapi, uvicorn, pyautogui, yaml, PIL" 2>nul
if errorlevel 1 (
    echo Missing deps, installing once ^(needs network^)...
    "%PY%" -m pip install -q -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] pip install failed. Run selfcheck.bat to see what is missing.
        pause
        exit /b 1
    )
)

echo ============================================================
echo  LocalCUA starting: http://127.0.0.1:7860
echo  Browser opens in 6 seconds. Close this window to stop.
echo ============================================================
echo.

rem Wait for the server, then open the browser
start "" /b cmd /c "timeout /t 6 >nul & start "" http://127.0.0.1:7860"

"%PY%" run.py
pause
