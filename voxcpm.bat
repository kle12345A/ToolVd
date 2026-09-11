@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title AI Movie Studio - VoxCPM

if not exist "venv\Scripts\python.exe" (
    call install_voxcpm.bat --no-pause
    if errorlevel 1 goto :error
)

"venv\Scripts\python.exe" -c "from voxcpm import VoxCPM" >nul 2>&1
if errorlevel 1 (
    call install_voxcpm.bat --no-pause
    if errorlevel 1 goto :error
)

echo [OK] VoxCPM runtime san sang.
if /I "%~1"=="--check" exit /b 0
echo [INFO] Dang mo AI Movie Studio...
"venv\Scripts\python.exe" main.py
if errorlevel 1 goto :error
exit /b 0

:error
echo [LOI] Khong the khoi dong AI Movie Studio voi VoxCPM.
pause
exit /b 1
