@echo off
setlocal
cd /d "%~dp0"
echo === PINNeAPPle Setup ===
echo.

:: ── Python / Backend ──────────────────────────────────────────────────────────
echo [1/4] Creating Python virtual environment...
python -m venv backend\venv
if errorlevel 1 (echo ERROR: python not found & pause & exit /b 1)

echo [2/4] Installing Python dependencies...
backend\venv\Scripts\pip install --upgrade pip
backend\venv\Scripts\pip install -r backend\requirements.txt
if errorlevel 1 (echo ERROR: pip install failed & pause & exit /b 1)

echo [3/4] Running Django migrations (includes JWT blacklist + auth)...
cd backend
..\backend\venv\Scripts\python manage.py migrate
cd ..

:: ── Node / Frontend ───────────────────────────────────────────────────────────
echo [4/4] Installing Node.js dependencies...
cd frontend
npm install
if errorlevel 1 (echo ERROR: npm install failed & pause & exit /b 1)
cd ..

echo.
echo Setup complete!
echo.
echo To start the app:
echo   run.bat
echo.
echo If you have an Anthropic API key, create backend\.env with:
echo   ANTHROPIC_API_KEY=sk-ant-...
echo.
pause
