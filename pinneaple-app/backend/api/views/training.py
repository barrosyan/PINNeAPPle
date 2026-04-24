import uuid
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from api.models import TrainingRun, Project
from api.serializers import TrainingRunSerializer
from api.training_worker import start_training, stop_training, get_status


class StartTrainingView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, project_id):
        try:
            project = Project.objects.get(id=project_id, owner=request.user)
        except Project.DoesNotExist:
            return Response({"error": "project not found"}, status=404)

        model_config = request.data.get("model_config", {})
        ts_data      = request.data.get("ts_data")
        model_type   = model_config.get("type", "pinn_mlp")
        ws_run_id    = str(uuid.uuid4())[:12]

        run = TrainingRun.objects.create(
            project    = project,
            model_type = model_type,
            config     = model_config,
            status     = "pending",
            ws_run_id  = ws_run_id,
        )

        problem = dict(project.problem_spec)
        if "obstacle" in model_config:
            params = dict(problem.get("params", {}))
            params["obstacle"] = model_config["obstacle"]
            problem["params"]  = params

        start_training(ws_run_id, str(run.id), model_type,
                       model_config, problem, ts_data)

        return Response({
            "ws_run_id": ws_run_id,
            "db_run_id": str(run.id),
            "ws_url":    f"/ws/training/{ws_run_id}/",
        }, status=status.HTTP_201_CREATED)


class TrainingRunDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, run_id):
        try:
            run = TrainingRun.objects.get(id=run_id, project__owner=request.user)
        except TrainingRun.DoesNotExist:
            return Response({"error": "not found"}, status=404)
        return Response(TrainingRunSerializer(run).data)


class TrainingRunListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        runs = (TrainingRun.objects
                .filter(project__owner=request.user)
                .order_by("-created_at")[:100])
        return Response(TrainingRunSerializer(runs, many=True).data)


class StopTrainingView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, ws_run_id):
        stop_training(ws_run_id)
        return Response({"status": "stop_requested"})


class TrainingStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, ws_run_id):
        data = get_status(ws_run_id)
        if data:
            return Response(data)
        try:
            run = TrainingRun.objects.get(
                ws_run_id=ws_run_id, project__owner=request.user
            )
            return Response({
                "status":  run.status,
                "history": run.history,
                "result":  run.result_data,
            })
        except TrainingRun.DoesNotExist:
            return Response({"error": "not found"}, status=404)
