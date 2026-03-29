from __future__ import annotations
"""Model export utilities for pinneaple.

Enables deployment of trained models to:
- TorchScript (.pt) for PyTorch C++ runtime
- ONNX (.onnx) for cross-platform deployment (C++, MATLAB, ONNX Runtime)
- Pinneaple checkpoint (.pt) for reloading in Python

The export methods are also available directly on any PINNBase subclass:
    model.export_torchscript("model.pt", example_input)
    model.export_onnx("model.onnx", example_input)
    model.save_checkpoint("model_ckpt.pt")
"""

from pinneaple_models.pinns.base import PINNBase


def export_torchscript(model, path: str, example_input=None) -> str:
    """Export any nn.Module to TorchScript."""
    import os, torch
    model.eval()
    if example_input is not None:
        try:
            scripted = torch.jit.trace(model, example_input)
        except Exception:
            scripted = torch.jit.script(model)
    else:
        scripted = torch.jit.script(model)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    scripted.save(path)
    return path


def export_onnx(model, path: str, example_input, input_names=None, output_names=None, opset_version=17) -> str:
    """Export any nn.Module to ONNX."""
    import os, torch.onnx
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _in = input_names or ["input"]
    _out = output_names or ["output"]
    torch.onnx.export(
        model, example_input, path,
        input_names=_in, output_names=_out,
        opset_version=opset_version,
        dynamic_axes={n: {0: "batch"} for n in _in + _out},
    )
    return path


__all__ = ["export_torchscript", "export_onnx"]
