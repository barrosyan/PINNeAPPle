from __future__ import annotations
import os
from .db import init_db
from .storage import ensure_bucket

def main() -> None:
    init_db()
    ensure_bucket()
    q = os.environ.get("QUEUE_NAME", "cpu").strip()
    os.execvp("celery", ["celery", "-A", "web_backend.app.celery_app:celery_app", "worker", "-Q", q, "--loglevel=INFO"])

if __name__ == "__main__":
    main()
