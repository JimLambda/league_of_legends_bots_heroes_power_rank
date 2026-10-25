@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo    LoL Bot Heroes Power Rank  -  one-click launcher
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on your PATH.
    echo.
    echo Please install Python 3.9 - 3.12 from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
)
call ".venv\Scripts\activate.bat"

echo [2/3] Installing dependencies - the first run may take 5-15 minutes...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo If paddlepaddle fails on your Python version, install Python 3.9-3.12,
    echo or edit requirements.txt to use RapidOCR only.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting the dashboard in your browser...
start "" http://localhost:8501
python -m streamlit run app.py

echo.
echo Dashboard stopped.
pause
