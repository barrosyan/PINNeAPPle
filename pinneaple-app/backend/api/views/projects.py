from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from api.models import Project, TrainingRun
from api.serializers import ProjectSerializer, TrainingRunSerializer


class ProjectViewSet(viewsets.ModelViewSet):
    serializer_class  = ProjectSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Project.objects.filter(owner=self.request.user).order_by("-created_at")

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

    @action(detail=True, methods=["get"])
    def runs(self, request, pk=None):
        project = self.get_object()
        runs    = project.training_runs.all().order_by("-created_at")
        return Response(TrainingRunSerializer(runs, many=True).data)

    @action(detail=True, methods=["get"])
    def latest_run(self, request, pk=None):
        project = self.get_object()
        run     = project.training_runs.order_by("-created_at").first()
        if run:
            return Response(TrainingRunSerializer(run).data)
        return Response({}, status=status.HTTP_404_NOT_FOUND)
