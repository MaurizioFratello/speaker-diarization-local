@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Run setup.ps1 first.
  exit /b 1
)
set PYTHONUTF8=1
.venv\Scripts\python.exe app.py --preload
pause
