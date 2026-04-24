from rest_framework import serializers
from .models import Project, TrainingRun, BenchmarkResult, UploadedFile


class ProjectSerializer(serializers.ModelSerializer):
    run_count = serializers.SerializerMethodField()
    latest_run_status = serializers.SerializerMethodField()

    class Meta:
        model  = Project
        fields = ["id", "name", "problem_spec", "model_config", "status",
                  "created_at", "updated_at", "run_count", "latest_run_status"]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_run_count(self, obj):
        return obj.training_runs.count()

    def get_latest_run_status(self, obj):
        run = obj.training_runs.order_by("-created_at").first()
        return run.status if run else None


class TrainingRunSerializer(serializers.ModelSerializer):
    class Meta:
        model  = TrainingRun
        fields = ["id", "project", "model_type", "config", "history",
                  "final_loss", "status", "error_msg", "result_data",
                  "ws_run_id", "created_at", "completed_at"]
        read_only_fields = ["id", "created_at", "completed_at"]


class BenchmarkResultSerializer(serializers.ModelSerializer):
    class Meta:
        model  = BenchmarkResult
        fields = ["id", "benchmark_key", "config", "metrics", "created_at"]
        read_only_fields = ["id", "created_at"]


class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model  = UploadedFile
        fields = ["id", "project", "name", "file", "file_type", "meta", "created_at"]
        read_only_fields = ["id", "created_at"]
