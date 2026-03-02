from __future__ import annotations

from typing import Optional
import boto3
from botocore.client import Config

from .settings import settings

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4"),
    )

def ensure_bucket() -> None:
    c = s3_client()
    buckets = [b["Name"] for b in c.list_buckets().get("Buckets", [])]
    if settings.s3_bucket not in buckets:
        c.create_bucket(Bucket=settings.s3_bucket)

def upload_file(local_path: str, s3_key: str, content_type: Optional[str] = None) -> None:
    c = s3_client()
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    c.upload_file(local_path, settings.s3_bucket, s3_key, ExtraArgs=extra or None)

def presign_get(s3_key: str, expires_sec: int = 3600) -> str:
    c = s3_client()
    return c.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": s3_key},
        ExpiresIn=int(expires_sec),
    )
