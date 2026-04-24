#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/backend/venv"

if [ ! -f "$VENV/bin/daphne" ]; then
    echo "ERROR: virtual environment not found."
    echo "Run: bash setup.sh"
    exit 1
fi

echo "=== PINNeAPPle App ==="
echo "Starting Django backend + React frontend..."
echo

# Backend (Daphne ASGI)
cd "$SCRIPT_DIR/backend"
"$VENV/bin/daphne" -b 0.0.0.0 -p 8000 pinneaple_backend.asgi:application &
BACKEND_PID=$!
echo "  Backend  PID $BACKEND_PID  →  http://localhost:8000/api/"

# Frontend (Vite dev server)
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID $FRONTEND_PID  →  http://localhost:5173"

echo
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
