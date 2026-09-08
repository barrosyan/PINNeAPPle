"""pinneapple_registry.registry — ``ArtifactRegistry``: unifies
``ModelStore``/``DatasetStore``/``ExperimentStore``/``ProblemStore`` under
one root, plus a Triton Inference Server model-repository generator.

Triton export is explicitly called out as unsupported in this repo's own
``examples/vs_physicsnemo`` comparison notes — this closes that gap for
any model exported to ONNX via ``ModelStore.export_onnx``.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Optional

from .dataset_store import DatasetStore
from .experiment_store import ExperimentStore
from .model_store import ModelStore
from .problem_store import ProblemStore

_TRITON_CONFIG_TEMPLATE = """\
name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ {in_dim} ]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ {out_dim} ]
  }}
]
"""


class ArtifactRegistry:
    """One root directory, four sub-stores, one experiment-tracking DB::

        <root>/models/<problem_id>/v.../model.pt
        <root>/datasets/<problem_id>/<scenario_id>/simulation.npz
        <root>/experiments.db
        <root>/problems/<problem_id>/v.../preset.json
        <root>/triton_models/<name>/{config.pbtxt, 1/model.onnx}
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.models = ModelStore(os.path.join(root, "models"))
        self.datasets = DatasetStore(os.path.join(root, "datasets"))
        self.experiments = ExperimentStore(os.path.join(root, "experiments.db"))
        self.problems = ProblemStore(os.path.join(root, "problems"))
        self.triton_root = os.path.join(root, "triton_models")

    # -- Triton export -----------------------------------------------------

    def export_triton_repository(
        self,
        problem_id: str,
        sample_input: Any,
        *,
        version: Optional[str] = None,
        model_cls: Optional[Any] = None,
        triton_model_name: Optional[str] = None,
        max_batch_size: int = 0,
        input_name: str = "input",
        output_name: str = "output",
        **init_kwargs: Any,
    ) -> str:
        """Exports one stored model version into a Triton-ready model
        directory (``config.pbtxt`` + ``1/model.onnx``). Returns the
        model directory path. Does not itself require a running Triton
        server — it produces the on-disk layout Triton's
        ``--model-repository`` flag expects.
        """
        name = triton_model_name or problem_id
        onnx_path = self.models.export_onnx(
            problem_id, sample_input, version=version, model_cls=model_cls, **init_kwargs
        )

        model_dir = os.path.join(self.triton_root, name)
        version_dir = os.path.join(model_dir, "1")
        os.makedirs(version_dir, exist_ok=True)
        shutil.copyfile(onnx_path, os.path.join(version_dir, "model.onnx"))

        in_dim = int(sample_input.shape[-1])
        # sample_input.shape[0] is the batch dim; run the model once (already
        # exported) to learn out_dim rather than guessing it.
        import torch
        model = self.models.load(problem_id, version, model_cls=model_cls, **init_kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_input[:1])
            out = out.y if hasattr(out, "y") else out
        out_dim = int(out.shape[-1])

        config = _TRITON_CONFIG_TEMPLATE.format(
            name=name, max_batch_size=max_batch_size,
            input_name=input_name, in_dim=in_dim,
            output_name=output_name, out_dim=out_dim,
        )
        with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
            f.write(config)

        return model_dir

    # -- summary -----------------------------------------------------------

    def summary(self) -> dict:
        """A dashboard-ready snapshot: every known problem_id, its latest
        model version + stage, and its dataset scenarios."""
        model_problems = sorted(
            d for d in os.listdir(self.models.root) if os.path.isdir(os.path.join(self.models.root, d))
        ) if os.path.isdir(self.models.root) else []

        rows = []
        for problem_id in model_problems:
            latest = self.models.latest(problem_id)
            row = {"problem_id": problem_id, "latest_model_version": latest}
            if latest:
                row["stage"] = self.models.metadata(problem_id, latest).get("stage")
            row["dataset_scenarios"] = self.datasets.list_scenarios(problem_id)
            rows.append(row)
        return {"root": self.root, "problems": rows}
