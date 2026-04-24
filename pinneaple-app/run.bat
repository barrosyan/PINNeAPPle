@echo off
setlocal
cd /d "%~dp0"

echo === PINNeAPPle App ===
echo Starting Django backend + React frontend...
echo.

if not exist "backend\venv\Scripts\daphne.exe" (
    echo ERROR: Virtual environment not found.
    echo Run setup.bat first.
    pause & exit /b 1
)

:: Start backend in a new terminal window
start "PINNeAPPle Backend" cmd /k "cd /d "%~dp0backend" && ..\backend\venv\Scripts\activate && daphne -b 0.0.0.0 -p 8000 pinneaple_backend.asgi:application"

:: Give backend a moment to boot
timeout /t 2 /nobreak > nul

:: Start frontend in a new terminal window
start "PINNeAPPle Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo  Backend  → http://localhost:8000/api/
echo  Frontend → http://localhost:5173
echo.
echo Close the two terminal windows to stop the servers.
pause
