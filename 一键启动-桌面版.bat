@echo off
chcp 65001 >nul
title LocalCUA Desktop
cd /d "%~dp0"

rem Keep this minimized console window open: closing it closes the program.
rem The standalone app window pops up in a few seconds. Close THAT window
rem to stop the web service cleanly. Log: data\desktop.log

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%~dp0desktop.py" --check
if errorlevel 1 (
  echo [ERROR] Desktop launcher self-check failed. See data\desktop.log
  pause
  exit /b 1
)

start /min "LocalCUALog" "%PY%" "%~dp0desktop.py"
