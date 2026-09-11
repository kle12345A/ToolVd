@echo off
REM ============================================================
REM  Build AI Movie Studio into a standalone Windows app
REM ============================================================
echo.
echo === [1/2] Tao icon tu logo (assets\logo.png neu co) ===
python tools\make_icon.py

echo.
echo === [2/2] Dong goi bang PyInstaller (co the mat vai phut) ===
python -m PyInstaller amsr.spec --noconfirm --clean

echo.
echo ============================================================
echo  XONG! App nam trong:  dist\AI Movie Studio\
echo  Chay file:            dist\AI Movie Studio\AI Movie Studio.exe
echo ============================================================
pause
