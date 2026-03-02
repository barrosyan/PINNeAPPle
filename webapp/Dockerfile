# Multi-stage: build frontend, then serve via FastAPI backend
############################
# 1) Frontend build
############################
FROM node:20-alpine AS frontend
WORKDIR /app/frontend
COPY webapp/frontend/package.json webapp/frontend/package-lock.json* ./
RUN npm install
COPY webapp/frontend ./
RUN npm run build

############################
# 2) Backend runtime
############################
FROM python:3.11-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (matplotlib + basic build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Pinneaple (repo) for runtime imports.
COPY pyproject.toml ./
COPY README.md ./
COPY pinneaple_* ./
COPY __init__.py ./
RUN pip install --upgrade pip && pip install -e .

# Install python deps for the web backend
COPY webapp/backend/requirements.txt /app/requirements-web.txt
RUN pip install -r /app/requirements-web.txt

# Copy backend code into an importable package
COPY webapp/backend/app /app/web_backend/app
RUN python - <<'PY'\nimport pathlib\npathlib.Path('/app/web_backend').mkdir(parents=True, exist_ok=True)\n(pathlib.Path('/app/web_backend/__init__.py')).write_text('')\n(pathlib.Path('/app/web_backend/app/__init__.py')).write_text('')\nPY

# Copy built frontend dist into image
COPY --from=frontend /app/frontend/dist /app/web_frontend/dist
ENV FRONTEND_DIST=/app/web_frontend/dist

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "web_backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
