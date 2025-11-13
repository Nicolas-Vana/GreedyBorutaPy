@echo off
REM build_and_upload_testpypi.bat
REM Automates building and uploading Python package to TestPyPI

echo ========================================
echo Greedy Boruta - Build and Upload to TestPyPI
echo ========================================
echo.

REM Step 1: Clean old builds
echo [Step 1/4] Cleaning old builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"
echo Done!
echo.

REM Step 2: Build the package
echo [Step 2/4] Building package...
python setup.py sdist bdist_wheel
if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)
echo Done!
echo.

REM Step 3: Check the package
echo [Step 3/4] Checking package with twine...
twine check dist/*
if errorlevel 1 (
    echo ERROR: Package check failed!
    pause
    exit /b 1
)
echo Done!
echo.

REM Step 4: Show files and confirm upload
echo [Step 4/4] Files to upload:
dir /b dist
echo.
echo Ready to upload to TestPyPI!
set /p confirm="Continue with upload? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Upload cancelled.
    pause
    exit /b 0
)
echo.

REM Upload to TestPyPI
echo Uploading to TestPyPI...
twine upload --repository testpypi dist/*
if errorlevel 1 (
    echo ERROR: Upload failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Package uploaded to TestPyPI
echo ========================================
echo.
echo Test installation with:
echo pip install --index-url https://test.pypi.org/simple/ --no-deps greedyboruta
echo.
pause