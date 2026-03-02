from __future__ import annotations
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://pinneaple:pinneaple@localhost:5432/pinneaple"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "pinneaple-artifacts"
    s3_region: str = "us-east-1"

    pinneaple_code_version: str = "dev"
    jobs_local_dir: str = "runs/webapp_jobs"

    class Config:
        env_prefix = ""
        case_sensitive = False

settings = Settings()
