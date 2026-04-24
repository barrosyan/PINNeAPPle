#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== PINNeAPPle Setup ==="
echo

# ── Python / Backend ──────────────────────────────────────────────────────────
echo "[1/4] Creating Python virtual environment..."
python3 -m venv "$SCRIPT_DIR/backend/venv"

echo "[2/4] Installing Python dependencies..."
"$SCRIPT_DIR/backend/venv/bin/pip" install --upgrade pip
"$SCRIPT_DIR/backend/venv/bin/pip" install -r "$SCRIPT_DIR/backend/requirements.txt"

echo "[3/4] Running Django migrations (includes JWT blacklist + auth)..."
cd "$SCRIPT_DIR/backend"
../backend/venv/bin/python manage.py migrate

# ── Node / Frontend ───────────────────────────────────────────────────────────
echo "[4/4] Installing Node.js dependencies..."
cd "$SCRIPT_DIR/frontend"
npm install

echo
echo "Setup complete!"
echo
echo "To start the app:"
echo "  bash run.sh"
echo
echo "If you have an Anthropic API key, create backend/.env:"
echo "  echo ANTHROPIC_API_KEY=sk-ant-... > backend/.env"
echo
