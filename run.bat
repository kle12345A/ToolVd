@echo off
chcp 65001 >nul
title AI Movie Short ^& Review Studio v2.0

echo ============================================
echo   AI Movie Short ^& Review Studio
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Python khong tim thay. Vui long cai Python 3.10+
    echo      Tai tai: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check venv
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Tao virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [LOI] Khong tao duoc venv.
        pause
        exit /b 1
    )
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install/update dependencies
echo [INFO] Kiem tra va cai dependencies...
"venv\Scripts\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo [LOI] Cai dat dependencies that bai.
    pause
    exit /b 1
)

:: Create needed dirs
if not exist "tools" mkdir tools
if not exist "data" mkdir data
if not exist "output" mkdir output
if not exist "logs" mkdir logs

:: Run app
echo [INFO] Khoi dong app...
echo.
"venv\Scripts\python.exe" main.py

if errorlevel 1 (
    echo.
    echo [LOI] App thoat voi loi. Kiem tra file logs\app.log de biet them chi tiet.
    pause
)
