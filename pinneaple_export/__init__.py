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


def export_csv(
    coords,
    fields,
    path: str,
    *,
    coord_names=None,
    field_names=None,
) -> str:
    """Export coordinate + field arrays to a CSV file.

    Parameters
    ----------
    coords : array-like, shape (N, D) or list of 1-D arrays
        Spatial (and optionally temporal) coordinates.  If a list of 1-D
        arrays is given they are stacked column-wise.
    fields : array-like, shape (N,) or (N, F)
        Field values corresponding to each coordinate row.
    path : str
        Destination file path (``*.csv``).
    coord_names : list[str], optional
        Column names for coordinates.  Defaults to ``["x", "y", "z", "t"]``
        truncated to the number of coordinate dimensions.
    field_names : list[str], optional
        Column names for fields.  Defaults to ``["f0", "f1", …]``.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    import os
    import numpy as np

    # Normalise coords
    if isinstance(coords, (list, tuple)) and isinstance(coords[0], (list, tuple, type(np.array([])))):
        coords = np.column_stack([np.asarray(c).ravel() for c in coords])
    else:
        coords = np.asarray(coords)
    if coords.ndim == 1:
        coords = coords[:, None]

    # Normalise fields
    fields = np.asarray(fields)
    if fields.ndim == 1:
        fields = fields[:, None]

    N, D = coords.shape
    F = fields.shape[1]

    default_coord_names = ["x", "y", "z", "t"]
    cnames = list(coord_names or default_coord_names[:D])
    fnames = list(field_names or [f"f{i}" for i in range(F)])

    header = ",".join(cnames + fnames)
    data = np.concatenate([coords, fields], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path


def export_npz(
    coords,
    fields,
    path: str,
    *,
    coord_names=None,
    field_names=None,
) -> str:
    """Export coordinate + field arrays to a compressed NumPy ``.npz`` archive.

    Parameters
    ----------
    coords : array-like, shape (N, D) or list of 1-D arrays
    fields : array-like, shape (N,) or (N, F), or dict of arrays
        When a ``dict`` is passed the keys become archive keys and
        ``field_names`` is ignored.
    path : str
        Destination file path (``*.npz``).
    coord_names : list[str], optional
        Keys used for coordinate arrays in the archive.
    field_names : list[str], optional
        Keys used for field arrays.  Defaults to ``["f0", "f1", …]``.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    import os
    import numpy as np

    if isinstance(coords, (list, tuple)) and not isinstance(coords[0], (int, float)):
        coords = np.column_stack([np.asarray(c).ravel() for c in coords])
    else:
        coords = np.asarray(coords)
    if coords.ndim == 1:
        coords = coords[:, None]

    D = coords.shape[1]
    default_coord_names = ["x", "y", "z", "t"]
    cnames = list(coord_names or default_coord_names[:D])

    save_dict = {}
    for i, name in enumerate(cnames):
        save_dict[name] = coords[:, i]

    if isinstance(fields, dict):
        save_dict.update({k: np.asarray(v) for k, v in fields.items()})
    else:
        fields = np.asarray(fields)
        if fields.ndim == 1:
            fields = fields[:, None]
        F = fields.shape[1]
        fnames = list(field_names or [f"f{i}" for i in range(F)])
        for i, name in enumerate(fnames):
            save_dict[name] = fields[:, i]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **save_dict)
    return path


__all__ = ["export_torchscript", "export_onnx", "export_csv", "export_npz"]
