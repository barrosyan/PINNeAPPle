from __future__ import annotations
import uuid
from django.conf import settings
from django.db import models
from django.utils import timezone


class Project(models.Model):
    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner        = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="projects", null=True, blank=True,
    )
    name         = models.CharField(max_length=256)
    problem_spec = models.JSONField(default=dict)
    model_config = models.JSONField(null=True, blank=True)
    status       = models.CharField(max_length=64, default="created")
    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class TrainingRun(models.Model):
    class Status(models.TextChoices):
        PENDING  = "pending"
        RUNNING  = "running"
        DONE     = "done"
        ERROR    = "error"
        STOPPED  = "stopped"

    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project      = models.ForeignKey(Project, on_delete=models.CASCADE,
                                     related_name="training_runs")
    model_type   = models.CharField(max_length=64)
    config       = models.JSONField(default=dict)
    history      = models.JSONField(default=list)
    final_loss   = models.FloatField(null=True, blank=True)
    status       = models.CharField(max_length=32, choices=Status.choices,
                                    default=Status.PENDING)
    error_msg    = models.TextField(blank=True)
    result_data  = models.JSONField(default=dict)
    ws_run_id    = models.CharField(max_length=32, blank=True)
    created_at   = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.model_type} run ({self.status})"


class BenchmarkResult(models.Model):
    id            = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner         = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="benchmark_results", null=True, blank=True,
    )
    benchmark_key = models.CharField(max_length=128)
    config        = models.JSONField(default=dict)
    metrics       = models.JSONField(default=dict)
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.benchmark_key}"


class UploadedFile(models.Model):
    id        = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project   = models.ForeignKey(Project, on_delete=models.CASCADE,
                                  related_name="files", null=True, blank=True)
    name      = models.CharField(max_length=256)
    file      = models.FileField(upload_to="uploads/")
    file_type = models.CharField(max_length=32)   # stl | csv | npy
    meta      = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name
