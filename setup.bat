@echo off
SET "VENV_DIR=.venv"
SET "PYTHON_EXE=python"
SET "REQ_FILE=requirements.txt"
SET "REQUIRED_VERSION=3.10.4"

:: 1. Check if Python is installed
where %PYTHON_EXE% >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: %PYTHON_EXE% is not installed on your system.
    exit /b 1
)

:: 2. Strict Version Check
:: Runs inline python and captures the output into ACTUAL_VERSION
FOR /F "tokens=*" %%i IN ('%PYTHON_EXE% -c "import platform; print(platform.python_version())"') DO SET ACTUAL_VERSION=%%i

if "%ACTUAL_VERSION%" NEQ "%REQUIRED_VERSION%" (
    echo Error: Strict version requirement failed.
    echo Required: %REQUIRED_VERSION%
    echo Found:    %ACTUAL_VERSION%
    echo Please install Python %REQUIRED_VERSION% and ensure it is in your PATH.
    exit /b 1
)
echo Python version %ACTUAL_VERSION% verified. Proceeding...

:: 3. Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    %PYTHON_EXE% -m venv %VENV_DIR%
) else (
    echo Virtual environment '%VENV_DIR%' already exists.
)

:: 4. Activate the virtual environment
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    echo Error: Activation script not found.
    exit /b 1
)

:: 5. Install libraries from requirements.txt
if exist "%REQ_FILE%" (
    echo Installing dependencies from %REQ_FILE%...
    %PYTHON_EXE% -m pip install --upgrade pip
    pip install -r "%REQ_FILE%"
) else (
    echo Warning: %REQ_FILE% not found. Skipping library installation.
)

:: 6. Deactivate the virtual environment
echo Deactivating virtual environment...
call deactivate

echo Setup complete!