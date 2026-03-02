from __future__ import annotations
from celery import Celery
from .settings import settings

celery_app = Celery(
    "pinneaple_webapp",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_routes={
        "web_backend.app.tasks.run_job": {"queue": "cpu"},
        "web_backend.app.tasks.run_job_gpu": {"queue": "gpu"},
    },
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
