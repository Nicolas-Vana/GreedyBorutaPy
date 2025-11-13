@echo off
REM build_and_upload_pypi.bat
REM Automates building and uploading Python package to PRODUCTION PyPI

echo ========================================
echo Greedy Boruta - Build and Upload to PyPI
echo ========================================
echo.
echo WARNING: This will upload to PRODUCTION PyPI!
echo.
set /p confirm="Are you sure? (YES to continue): "
if /i not "%confirm%"=="YES" (
    echo Upload cancelled.
    pause
    exit /b 0
)
echo.

REM Clean old builds
echo [Step 1/4] Cleaning old builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"
echo Done!
echo.

REM Build the package
echo [Step 2/4] Building package...
python setup.py sdist bdist_wheel
if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)
echo Done!
echo.

REM Check the package
echo [Step 3/4] Checking package with twine...
twine check dist/*
if errorlevel 1 (
    echo ERROR: Package check failed!
    pause
    exit /b 1
)
echo Done!
echo.

REM Upload to PyPI
echo [Step 4/4] Uploading to PRODUCTION PyPI...
twine upload dist/*
if errorlevel 1 (
    echo ERROR: Upload failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Package uploaded to PyPI
echo ========================================
echo.
echo Users can now install with:
echo pip install greedyboruta
echo.
pause