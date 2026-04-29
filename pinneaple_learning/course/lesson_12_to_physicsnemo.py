"""Lesson 12 — From PINNeAPPle to NVIDIA PhysicsNeMo.

This is the final lesson. You have built up:
  ✓ Forward PDE solves (lessons 1–5)
  ✓ Advanced architectures (lesson 6)
  ✓ Inverse problems (lesson 7)
  ✓ Uncertainty quantification (lesson 8)
  ✓ Time-marching for stiff problems (lesson 9)
  ✓ Physics validation (lesson 10)
  ✓ Operator learning (lesson 11)

Now: what does the production journey look like?

What this lesson does
---------------------
  Step 1 — Train and validate a heat PINN (review the full workflow)
  Step 2 — Run the production validation checklist
  Step 3 — Export to ONNX with PINNeAPPle's export_onnx
  Step 4 — Benchmark: PyTorch eager vs ONNX Runtime latency
  Step 5 — Print the PhysicsNeMo migration overview

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_12_to_physicsnemo
"""

from __future__ import annotations
import math
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PINNeAPPle imports — the full production stack
from pinneaple_validate import compare_to_analytical, PhysicsValidator
from pinneaple_export   import export_onnx

ALPHA   = 0.05
T_MAX   = 1.0
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ── Network ────────────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


# ── Train heat PINN (same as lesson 3) ────────────────────────────────────
def train(n_epochs: int = 8_000) -> nn.Module:
    torch.manual_seed(0)
    net = make_net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for epoch in range(n_epochs):
        opt.zero_grad()
        x  = torch.rand(2000, 1, device=DEVICE)
        t  = torch.rand(2000, 1, device=DEVICE) * T_MAX
        xt = torch.cat([x, t], dim=1).requires_grad_(True)
        u  = net(xt)
        g  = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_t = g[:, 1:2]
        u_x = g[:, 0:1]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        l_pde = (u_t - ALPHA * u_xx).pow(2).mean()

        x_bc = torch.cat([torch.zeros(200, 1), torch.ones(200, 1)]).to(DEVICE)
        t_bc = torch.rand(400, 1, device=DEVICE) * T_MAX
        l_bc = net(torch.cat([x_bc, t_bc], dim=1)).pow(2).mean()

        x_ic = torch.rand(200, 1, device=DEVICE)
        xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
        l_ic  = (net(xt_ic) - torch.sin(math.pi * x_ic)).pow(2).mean()

        (l_pde + 10.0 * l_bc + 100.0 * l_ic).backward()
        opt.step(); sch.step()

    return net


# ── Production validation checklist ──────────────────────────────────────
def run_production_checklist(net: nn.Module) -> dict:
    print("\n  ── Production Validation Checklist ──────────────────────────")

    # 1. Accuracy
    def exact(xy):
        return np.exp(-ALPHA * math.pi**2 * xy[:, 1]) * np.sin(math.pi * xy[:, 0])

    metrics = compare_to_analytical(
        model=net, analytical_fn=exact,
        coord_names=["x", "t"],
        domain_bounds={"x": (0.0, 1.0), "t": (0.0, T_MAX)},
        n_points=20_000, device=DEVICE,
    )
    l2_ok = metrics["rel_l2"] < 1e-2
    print(f"  [{'✓' if l2_ok else '✗'}] Relative L2 < 1e-2       : {metrics['rel_l2']:.4e}")

    # 2. BC residual
    x_bc = torch.cat([torch.zeros(1000, 1), torch.ones(1000, 1)]).to(DEVICE)
    t_bc = torch.rand(2000, 1, device=DEVICE) * T_MAX
    with torch.no_grad():
        u_bc = net(torch.cat([x_bc, t_bc], dim=1)).abs().mean().item()
    bc_ok = u_bc < 1e-3
    print(f"  [{'✓' if bc_ok else '✗'}] BC residual < 1e-3        : {u_bc:.4e}")

    # 3. IC residual
    x_ic = torch.rand(1000, 1, device=DEVICE)
    xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
    with torch.no_grad():
        ic_err = (net(xt_ic) - torch.sin(math.pi * x_ic)).abs().mean().item()
    ic_ok = ic_err < 1e-3
    print(f"  [{'✓' if ic_ok else '✗'}] IC residual < 1e-3        : {ic_err:.4e}")

    # 4. PDE residual at held-out points
    x_ho = torch.rand(5000, 1, device=DEVICE)
    t_ho = torch.rand(5000, 1, device=DEVICE) * T_MAX
    xt_ho = torch.cat([x_ho, t_ho], dim=1).requires_grad_(True)
    u_ho  = net(xt_ho)
    g     = torch.autograd.grad(u_ho.sum(), xt_ho, create_graph=False)[0]
    u_t   = g[:, 1:2]
    u_x   = g[:, 0:1]
    u_xx  = torch.autograd.grad(u_x.sum(), xt_ho, create_graph=False)[0][:, 0:1]
    pde_res = (u_t - ALPHA * u_xx).abs().mean().item()
    pde_ok  = pde_res < 1e-3
    print(f"  [{'✓' if pde_ok else '✗'}] PDE residual (held-out) < 1e-3: {pde_res:.4e}")

    n_pass = sum([l2_ok, bc_ok, ic_ok, pde_ok])
    print(f"\n  Result: {n_pass}/4 checks passed")
    return {"l2": metrics["rel_l2"], "bc_err": u_bc, "ic_err": ic_err, "pde_res": pde_res}


# ── ONNX export + latency benchmark ──────────────────────────────────────
def export_and_benchmark(net: nn.Module) -> None:
    print("\n  ── Export to ONNX ───────────────────────────────────────────")
    net_cpu = net.cpu()
    onnx_path = "lesson_12_heat_pinn.onnx"

    try:
        export_onnx(
            model       = net_cpu,
            path        = onnx_path,
            input_shape = (1, 2),           # (batch, features)
            input_names = ["xt"],
            output_names = ["u"],
        )
        print(f"  Exported: {onnx_path}")

        # Latency benchmark
        x_bench = torch.rand(1000, 2)

        # PyTorch eager
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad():
                _ = net_cpu(x_bench)
        t_torch = (time.perf_counter() - t0) / 200 * 1000  # ms

        # ONNX Runtime
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path,
                                         providers=["CPUExecutionProvider"])
            x_np = x_bench.numpy()
            t0   = time.perf_counter()
            for _ in range(200):
                _ = sess.run(None, {"xt": x_np})
            t_onnx = (time.perf_counter() - t0) / 200 * 1000  # ms
            speedup = t_torch / t_onnx
            print(f"\n  Latency benchmark (1000 points, 200 runs):")
            print(f"    PyTorch eager : {t_torch:.3f} ms / batch")
            print(f"    ONNX Runtime  : {t_onnx:.3f} ms / batch")
            print(f"    Speedup       : {speedup:.1f}×")
        except ImportError:
            print("  (onnxruntime not installed — skip latency comparison)")
    except Exception as e:
        print(f"  (ONNX export skipped: {e})")


# ── PhysicsNeMo overview ──────────────────────────────────────────────────
def print_physicsnemo_bridge() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║         You are ready for NVIDIA PhysicsNeMo.               ║
╚══════════════════════════════════════════════════════════════╝

You have completed all 12 lessons. You know how to:
  ✓ Define PDEs symbolically and build residual losses
  ✓ Enforce boundary conditions hard or soft
  ✓ Choose architectures (Tanh, SIREN, ModifiedMLP, Fourier)
  ✓ Solve inverse problems (gradient-based and EKI)
  ✓ Quantify uncertainty (MC Dropout, Ensemble, Conformal)
  ✓ Handle stiff problems with time-marching
  ✓ Validate models against analytics and conservation laws
  ✓ Learn operators across parameter families
  ✓ Export to ONNX for deployment

─── The PINNeAPPle → PhysicsNeMo bridge ───────────────────

  PINNeAPPle helps you:                PhysicsNeMo gives you:
  ─────────────────────────────────    ─────────────────────────────────
  Prototype in hours                   Scale to 512+ GPUs
  Understand every line of code        Battle-tested NVIDIA code
  Iterate quickly on new ideas         Enterprise support + SLA
  Run on CPU / single GPU              TensorRT inference (10-20× faster)
  Flexible, open-source                Triton Inference Server
                                       NVIDIA Omniverse integration

─── Install PhysicsNeMo ───────────────────────────────────

    pip install nvidia-physicsnemo
    python -c "import physicsnemo; print(physicsnemo.__version__)"

─── Your first PhysicsNeMo PINN ───────────────────────────

    from modulus.sym import Trainer, Domain
    from modulus.sym.geometry import Rectangle
    from modulus.sym.eq.pdes import HeatEquation

    geo   = Rectangle((-1, 0), (1, 1))
    heat  = HeatEquation(T="u", time=True, dim=1, diffusivity=0.05)
    # ... add constraints, validators, run Trainer

─── Full migration guide ─────────────────────────────────

    pinneaple_learning/physicsnemo_roadmap/migration_guide.md

─── Side-by-side comparisons ────────────────────────────

    examples/vs_physicsnemo/

Congratulations — you have completed the PINNeAPPle course.
""")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 12 — From PINNeAPPle to NVIDIA PhysicsNeMo")
    print("─" * 60)
    print(f"\n  Training heat PINN: u_t = {ALPHA} u_xx  (review of full workflow)\n")

    # Step 1: Train
    print("  [Step 1]  Training...")
    net = train(n_epochs=8_000)
    print("  Training complete.")

    # Step 2: Production checklist
    print("\n  [Step 2]  Running production validation checklist...")
    checklist = run_production_checklist(net)

    # Step 3+4: Export and benchmark
    print("\n  [Step 3+4]  Exporting to ONNX and benchmarking...")
    export_and_benchmark(net)

    # Step 5: PhysicsNeMo bridge
    print_physicsnemo_bridge()

    # ── Plot: validation summary ──────────────────────────────────────────
    x_np  = np.linspace(0, 1, 80, dtype=np.float32)
    t_slc = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_slc)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    net.to(DEVICE)
    for col, tval in zip(colors, t_slc):
        t_np = np.full_like(x_np, tval)
        xt   = torch.tensor(np.stack([x_np, t_np], axis=1), device=DEVICE)
        with torch.no_grad():
            u_p = net(xt).cpu().numpy().ravel()
        u_e = np.exp(-ALPHA * math.pi**2 * tval) * np.sin(math.pi * x_np)
        ax.plot(x_np, u_e, "-",  color=col, lw=2,   alpha=0.9)
        ax.plot(x_np, u_p, "--", color=col, lw=1.5, alpha=0.7)
    ax.set_title("Heat equation — exact (solid) vs PINN (dashed)")
    ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels   = ["L2 error", "BC error", "IC error", "PDE residual"]
    values   = [checklist["l2"], checklist["bc_err"],
                checklist["ic_err"], checklist["pde_res"]]
    targets  = [1e-2, 1e-3, 1e-3, 1e-3]
    colors_b = ["green" if v < t else "crimson" for v, t in zip(values, targets)]
    bars = ax.bar(labels, values, color=colors_b, alpha=0.8, edgecolor="k", lw=0.5)
    for lbl, tgt, bar in zip(labels, targets, bars):
        ax.axhline(tgt, color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title("Production checklist — green = PASS, red = FAIL")
    ax.set_ylabel("Error (log)")
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Lesson 12 — Production Validation Summary", fontsize=12)
    plt.tight_layout()
    out = "lesson_12_to_physicsnemo.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")
    print("\n  ── Course complete. Run the full course: ─────────────────────")
    print("  import pinneaple_learning.course as c; c.list_lessons()\n")


if __name__ == "__main__":
    main()
