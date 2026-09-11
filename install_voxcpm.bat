@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Cai VoxCPM cho AI Movie Studio

echo ============================================
echo   Cai VoxCPM2 cho AI Movie Studio
echo ============================================
echo.
echo Luu y:
echo - Goi PyTorch va cac phu thuoc co the chiem vai GB.
echo - Model openbmb/VoxCPM2 duoc tai o lan tao giong dau tien.
echo - VoxCPM2 can khoang 8 GB VRAM; GPU nho hon se chay CPU.
echo.

if not exist "venv\Scripts\python.exe" (
    echo [INFO] Tao virtual environment...
    python -m venv venv
    if errorlevel 1 goto :error
)

echo [INFO] Cai VoxCPM vao moi truong cua app...
"venv\Scripts\python.exe" -m pip install -r requirements-voxcpm.txt
if errorlevel 1 goto :error

echo [INFO] Kiem tra import VoxCPM...
"venv\Scripts\python.exe" -c "from voxcpm import VoxCPM; print('[OK] VoxCPM runtime san sang.')"
if errorlevel 1 goto :error

echo.
echo [OK] Da cai VoxCPM. Hay dong va mo lai AI Movie Studio.
if /I not "%~1"=="--no-pause" pause
exit /b 0

:error
echo.
echo [LOI] Khong cai duoc VoxCPM. Kiem tra ket noi mang va dung luong o dia.
if /I not "%~1"=="--no-pause" pause
exit /b 1
