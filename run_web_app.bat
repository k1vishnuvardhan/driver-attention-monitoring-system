@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"

echo Starting Driver Attention Monitoring backend...
start "Driver Attention Backend" cmd /k %PYTHON_CMD% src\main.py --headless

timeout /t 3 /nobreak >nul

echo Starting local website server...
start "Driver Attention Website" cmd /k %PYTHON_CMD% -m http.server 5501

timeout /t 2 /nobreak >nul

echo Opening dashboard...
start "" http://127.0.0.1:5501/ui/index.html

echo.
echo If the browser shows an old version, press Ctrl+F5 to hard refresh.
echo Keep both opened terminals running while using the app.
