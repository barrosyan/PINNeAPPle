import numpy as np
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from api.models import TrainingRun


class InferenceView(APIView):
    permission_classes = [IsAuthenticated]
    """
    POST /api/inference/<run_id>/
    Returns the result_data with additional computed fields (vorticity, Q-criterion).
    """
    def post(self, request, run_id):
        try:
            run = TrainingRun.objects.get(id=run_id, project__owner=request.user)
        except TrainingRun.DoesNotExist:
            return Response({"error": "not found"}, status=404)

        if run.status != "done":
            return Response({"error": f"run not done (status={run.status})"}, status=400)

        rd   = run.result_data
        kind = run.model_type

        # ── PINN ─────────────────────────────────────────────────────────────
        if kind == "pinn_mlp":
            return Response(rd.get("inference_grid", {}))

        # ── LBM ──────────────────────────────────────────────────────────────
        elif kind == "lbm":
            ux  = np.array(rd.get("ux", []))
            uy  = np.array(rd.get("uy", []))
            nx  = int(rd.get("nx", 1))
            ny  = int(rd.get("ny", 1))
            dx  = 1.0 / max(nx - 1, 1)
            dy  = 1.0 / max(ny - 1, 1)

            if ux.size > 0 and uy.size > 0:
                try:
                    dux_dx = np.gradient(ux, dx, axis=0)
                    duy_dy = np.gradient(uy, dy, axis=1)
                    dux_dy = np.gradient(ux, dy, axis=1)
                    duy_dx = np.gradient(uy, dx, axis=0)
                    vorticity = (duy_dx - dux_dy).tolist()
                    Q         = (-(dux_dx * duy_dy) - (duy_dx * dux_dy)).tolist()
                except Exception:
                    vorticity = []
                    Q         = []

                return Response({
                    "vel_mag":       rd.get("vel_mag"),
                    "ux":            rd.get("ux"),
                    "uy":            rd.get("uy"),
                    "rho":           rd.get("rho"),
                    "vorticity":     vorticity,
                    "Q":             Q,
                    "obstacle":      rd.get("obstacle"),
                    "nx":            nx,
                    "ny":            ny,
                    "trajectory_ux": rd.get("trajectory_ux", []),
                    "trajectory_uy": rd.get("trajectory_uy", []),
                })
            return Response(rd)

        # ── Timeseries ────────────────────────────────────────────────────────
        elif kind in ("tcn", "lstm", "tft", "fft"):
            ts_data = request.data.get("ts_data", [])
            return Response({
                "forecast":  rd.get("forecast", []),
                "horizon":   rd.get("horizon", len(rd.get("forecast", []))),
                "input_len": rd.get("input_len"),
                "type":      kind,
                "history":   rd.get("history", []),
            })

        # ── FDM / FEM ────────────────────────────────────────────────────────
        else:
            return Response(rd)
