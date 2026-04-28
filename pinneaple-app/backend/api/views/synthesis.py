"""Synthetic data generation views."""
from __future__ import annotations
import os
import sys

from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

# ── Static catalogue (drives both the API and the frontend form) ──────────────

CATALOGUE: dict = {
    "pde": {
        "label": "PDE / ODE Trajectories",
        "desc": (
            "Simulate 1-D heat equation, advection, or logistic ODE using "
            "explicit Euler time-stepping with configurable ICs and BCs."
        ),
        "output_type": "trajectory",
        "params": [
            {
                "name": "kind", "type": "select",
                "options": ["heat1d", "advection1d", "logistic"],
                "default": "heat1d", "label": "Equation kind",
            },
            {
                "name": "n_samples", "type": "int",
                "default": 8, "min": 1, "max": 32,
                "label": "Trajectories",
            },
            {
                "name": "steps", "type": "int",
                "default": 200, "min": 10, "max": 1000,
                "label": "Time steps",
            },
            {
                "name": "dt", "type": "float",
                "default": 0.001, "min": 0.0001, "max": 0.05,
                "label": "Δt",
            },
            {
                "name": "nx", "type": "int",
                "default": 64, "min": 16, "max": 256,
                "label": "Grid points (PDE)",
            },
            {
                "name": "alpha", "type": "float",
                "default": 0.01, "min": 0.001, "max": 1.0,
                "label": "Diffusivity α (heat)",
            },
            {
                "name": "c", "type": "float",
                "default": 1.0, "min": -5.0, "max": 5.0,
                "label": "Wave speed c (advection)",
            },
            {
                "name": "bc", "type": "select",
                "options": ["periodic", "dirichlet0"],
                "default": "periodic", "label": "Boundary conditions",
            },
            {
                "name": "seed", "type": "int",
                "default": 42, "min": 0, "max": 9999,
                "label": "Random seed",
            },
        ],
    },
    "distributions": {
        "label": "Distribution Sampling",
        "desc": (
            "Sample inputs from Gaussian, uniform, or mixture-of-Gaussians "
            "distributions and compute nonlinear supervised targets y = Σxᵢ²."
        ),
        "output_type": "scatter",
        "params": [
            {
                "name": "kind", "type": "select",
                "options": ["gaussian", "uniform", "mog"],
                "default": "gaussian", "label": "Distribution",
            },
            {
                "name": "n_samples", "type": "int",
                "default": 512, "min": 16, "max": 4096,
                "label": "Samples",
            },
            {
                "name": "dim", "type": "int",
                "default": 2, "min": 1, "max": 8,
                "label": "Dimensions",
            },
            {
                "name": "mean", "type": "float",
                "default": 0.0, "min": -5.0, "max": 5.0,
                "label": "Mean (Gaussian)",
            },
            {
                "name": "std", "type": "float",
                "default": 1.0, "min": 0.01, "max": 5.0,
                "label": "Std dev (Gaussian)",
            },
            {
                "name": "seed", "type": "int",
                "default": 42, "min": 0, "max": 9999,
                "label": "Random seed",
            },
        ],
    },
}


# ── Serialization helpers ─────────────────────────────────────────────────────

def _to_json_safe(obj):
    """Recursively convert tensors / numpy arrays to Python lists."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj


def _ensure_pinneaple_on_path():
    # synthesis.py → views/ → api/ → backend/ → pinneaple-app/ → repo root
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    if root not in sys.path:
        sys.path.insert(0, root)


# ── Views ─────────────────────────────────────────────────────────────────────

class SynthCatalogueView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(CATALOGUE)


class SynthGenerateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        generator_key = request.data.get("generator", "")
        params = dict(request.data.get("params", {}))

        if generator_key not in CATALOGUE:
            return Response(
                {"error": f"Unknown generator '{generator_key}'. "
                          f"Available: {list(CATALOGUE.keys())}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        meta = CATALOGUE[generator_key]
        seed = int(params.pop("seed", 42))

        # cast numeric params to correct Python types
        for p in meta["params"]:
            pname = p["name"]
            if pname == "seed" or pname not in params:
                continue
            if p["type"] == "int":
                params[pname] = int(params[pname])
            elif p["type"] == "float":
                params[pname] = float(params[pname])

        try:
            _ensure_pinneaple_on_path()
            from pinneaple_data.synth.base import SynthConfig
            from pinneaple_data.synth.registry import SynthCatalog

            cfg = SynthConfig(seed=seed)
            catalog = SynthCatalog()
            gen = catalog.build(generator_key, cfg=cfg)
            output = gen.generate(**params)

        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        # Serialize — cap at 8 samples to keep response manageable
        samples_data = []
        for s in output.samples[:8]:
            sd = {}
            if hasattr(s, "fields"):
                sd["fields"] = _to_json_safe(s.fields)
            if hasattr(s, "coords"):
                sd["coords"] = _to_json_safe(s.coords)
            if hasattr(s, "meta"):
                sd["meta"] = dict(s.meta)
            samples_data.append(sd)

        return Response({
            "generator":   generator_key,
            "output_type": meta["output_type"],
            "n_samples":   len(output.samples),
            "samples":     samples_data,
            "extras":      _to_json_safe(output.extras),
            "params":      {**params, "seed": seed},
        })
