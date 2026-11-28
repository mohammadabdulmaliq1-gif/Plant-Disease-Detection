@echo off
echo ========================================
echo   Plant Disease Detection Backend
echo ========================================
echo.

REM Determine Python command
set PYTHON_CMD=python
if exist "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
)

REM Check if any supported model exists
if not exist "models\best_model.h5" (
    if not exist "models\plant_disease_model_improved.keras" (
        if not exist "models\plant_disease_model.keras" (
            echo WARNING: No trained model found in models\
            echo You can generate an initial model by running:
            echo   cd ..\scripts
            echo   %PYTHON_CMD% create_improved_model.py
            echo   cd ..\backend
            echo.
        )
    )
)

echo Starting backend server on http://localhost:8000
echo Docs available at http://localhost:8000/docs
echo.

%PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
