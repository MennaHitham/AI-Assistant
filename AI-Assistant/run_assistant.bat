@echo off
setlocal

:: Force Python to ignore global site-packages in Users folder
set PYTHONNOUSERSITE=1

:: Try to find a local virtual environment first (.venv)
if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
    echo ✓ Using local environment (.venv)
) else if exist "D:\python\pytorch\ragenv\Scripts\python.exe" (
    :: Fallback to your custom path
    set PYTHON_EXE=D:\python\pytorch\ragenv\Scripts\python.exe
    echo ✓ Using custom environment (ragenv)
) else (
    :: Last resort: system python
    set PYTHON_EXE=python
    echo ! WARNING: Virtual environment not found. Using system Python...
)

echo ============================================================
echo   Starting Course Material AI Assistant (GPU Accelerated)
echo ============================================================
%PYTHON_EXE% main.py
pause
