"""pinneapple_systems.component_modeling.edge — package a trained model as a
self-contained edge-deployment bundle (ONNX graph + manifest.json zipped
together), and a minimal runtime that can execute that bundle with
onnxruntime + numpy alone (no torch/pinneapple import at all).

Builds on ``pinneapple_tools.model_export.export_onnx`` (the existing ONNX
tracing logic) rather than duplicating it — this module only adds the
packaging (manifest + zip) and the torch-free runtime around that export.

``EdgeRuntime`` is genuinely new: the same code path a real edge device
would run, runnable locally for testing/validation before actual
deployment.
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile
from typing import Any, Dict, Optional

import torch

from pinneapple_tools.model_export import export_onnx as _export_onnx


def export_edge_package(
    model: Any,
    output_dir: str,
    *,
    name: str,
    sample_input: torch.Tensor,
    opset_version: int = 17,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Exports `model` to ONNX and zips it with a manifest.json into a
    self-contained edge deployment package. Returns
    {"manifest": {...}, "onnx_path": ..., "zip_path": ...}.

    `model` is used exactly as given (whatever weights it currently holds —
    the caller decides whether that's freshly initialized or trained).
    """
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"{name}.onnx")
    model.eval()
    _export_onnx(model, onnx_path, sample_input, opset_version=opset_version)

    with torch.no_grad():
        out = model(sample_input[:1])
        out = out.y if hasattr(out, "y") else out

    manifest: Dict[str, Any] = {
        "name": name,
        "in_dim": int(sample_input.shape[1]),
        "out_dim": int(out.shape[1]) if out.ndim > 1 else 1,
        "onnx_file": os.path.basename(onnx_path),
        "runtime_requirements": ["onnxruntime", "numpy"],
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    zip_path = os.path.join(output_dir, f"{name}_edge_package.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_path, arcname=os.path.basename(onnx_path))
        zf.write(manifest_path, arcname="manifest.json")

    return {"manifest": manifest, "onnx_path": onnx_path, "zip_path": zip_path}


class EdgeRuntime:
    """Minimal edge-device inference runtime — onnxruntime + numpy only, no
    torch/pinneapple import at all. Loads an ``export_edge_package()`` zip
    and runs real ONNX inference."""

    def __init__(self, zip_path: str):
        import onnxruntime as ort

        self._tmpdir = tempfile.mkdtemp(prefix="edge_runtime_")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self._tmpdir)
        with open(os.path.join(self._tmpdir, "manifest.json")) as f:
            self.manifest = json.load(f)
        onnx_path = os.path.join(self._tmpdir, self.manifest["onnx_file"])
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x: Any) -> Any:
        import numpy as np

        x = np.asarray(x, dtype=np.float32)
        return self.session.run(None, {self.input_name: x})[0]
