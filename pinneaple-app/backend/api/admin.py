from django.contrib import admin
from .models import Project, TrainingRun, BenchmarkResult, UploadedFile

admin.site.register(Project)
admin.site.register(TrainingRun)
admin.site.register(BenchmarkResult)
admin.site.register(UploadedFile)
