"""23_model_export.py — ONNX and TorchScript export with pinneaple_export.

Demonstrates:
- ONNXExporter: export a trained PINN to ONNX with shape verification
- TorchScriptExporter: trace/script a model to TorchScript (.pt)
- ExportValidator: run an inference sanity-check post-export
- Latency comparison: Python torch vs. ONNX Runtime vs. TorchScript
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np

from pinneaple_export.onnx_exporter import ONNXExporter, ONNXExportConfig
from pinneaple_export.torchscript import TorchScriptExporter, TorchScriptConfig
from pinneaple_export.validator import ExportValidator


# ---------------------------------------------------------------------------
# Evaluation model: trained Poisson PINN
# (same architecture as template 01, weights trained here for standalone use)
# ---------------------------------------------------------------------------

class PoissonPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        phi = xy[:, 0:1] * (1 - xy[:, 0:1]) * xy[:, 1:2] * (1 - xy[:, 1:2])
        return phi * self.net(xy)


def quick_train(model: nn.Module, device, n_epochs: int = 2000) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        opt.zero_grad()
        xy = torch.rand(1024, 2, device=device, requires_grad=True)
        u  = model(xy)
        g  = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_xx = torch.autograd.grad(g[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(g[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        f    = -2 * math.pi**2 * torch.sin(math.pi * xy[:, 0:1]) * \
                torch.sin(math.pi * xy[:, 1:2])
        (u_xx + u_yy - f).pow(2).mean().backward()
        opt.step()


def benchmark_inference(fn, x: torch.Tensor, n: int = 200) -> float:
    """Return mean latency in ms over n runs."""
    for _ in range(10):      # warmup
        fn(x)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(x)
    return (time.perf_counter() - t0) / n * 1000


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")     # ONNX Runtime typically on CPU for comparison
    print(f"Device: {device}")

    # --- Train ---------------------------------------------------------------
    print("Training Poisson PINN ...")
    model = PoissonPINN().to(device)
    quick_train(model, device, n_epochs=2000)
    model.eval()
    print("Training complete.")

    x_dummy = torch.rand(64, 2, device=device)

    # --- ONNX export ---------------------------------------------------------
    onnx_config = ONNXExportConfig(
        output_path="23_poisson.onnx",
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        input_names=["input"],
        output_names=["output"],
        simplify=True,
    )
    print("\nExporting to ONNX ...")
    onnx_exporter = ONNXExporter(model=model, config=onnx_config)
    onnx_exporter.export(example_input=x_dummy)
    print(f"  Saved: {onnx_config.output_path}")

    # --- TorchScript export --------------------------------------------------
    ts_config = TorchScriptConfig(
        output_path="23_poisson_scripted.pt",
        method="trace",     # "trace" | "script"
        strict=True,
    )
    print("Exporting to TorchScript ...")
    ts_exporter = TorchScriptExporter(model=model, config=ts_config)
    ts_exporter.export(example_input=x_dummy)
    print(f"  Saved: {ts_config.output_path}")

    # --- Validation ----------------------------------------------------------
    print("\nValidating exports ...")
    validator = ExportValidator(original_model=model, atol=1e-5)

    onnx_ok = validator.validate_onnx(
        onnx_path=onnx_config.output_path,
        test_input=x_dummy,
    )
    ts_ok = validator.validate_torchscript(
        ts_path=ts_config.output_path,
        test_input=x_dummy,
    )
    print(f"  ONNX valid:        {onnx_ok}")
    print(f"  TorchScript valid: {ts_ok}")

    # --- Latency benchmark ---------------------------------------------------
    print("\nBenchmarking inference latency (batch=64) ...")
    x_bench = torch.rand(64, 2, device=device)

    # PyTorch
    lat_torch = benchmark_inference(lambda x: model(x), x_bench)

    # TorchScript
    ts_model = torch.jit.load(ts_config.output_path)
    ts_model.eval()
    lat_ts = benchmark_inference(lambda x: ts_model(x), x_bench)

    # ONNX Runtime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_config.output_path,
                                    providers=["CPUExecutionProvider"])
        x_np = x_bench.numpy()
        lat_onnx = benchmark_inference(
            lambda x: sess.run(None, {"input": x.numpy()}),
            x_bench,
        )
    except ImportError:
        lat_onnx = float("nan")
        print("  onnxruntime not installed — skipping ONNX latency.")

    print(f"\n  PyTorch:      {lat_torch:.3f} ms")
    print(f"  TorchScript:  {lat_ts:.3f} ms")
    print(f"  ONNX Runtime: {lat_onnx:.3f} ms")

    speedup_ts   = lat_torch / lat_ts   if lat_ts   > 0 else float("nan")
    speedup_onnx = lat_torch / lat_onnx if lat_onnx > 0 else float("nan")
    print(f"\n  TorchScript speedup:  {speedup_ts:.2f}x")
    print(f"  ONNX Runtime speedup: {speedup_onnx:.2f}x")


if __name__ == "__main__":
    main()
