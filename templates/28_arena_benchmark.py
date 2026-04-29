"""28_arena_benchmark.py — Arena benchmark runner with YAML configuration.

Demonstrates:
- ArenaRunner: load and execute a benchmark suite from a YAML config
- NativeBenchmarkTask: define a custom task programmatically and register it
- BenchmarkResult: parse, compare and export results
- Leaderboard: rank-order models on a shared task family
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
import tempfile
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_arena.runner import ArenaRunner
from pinneaple_arena.task import NativeBenchmarkTask, TaskSpec
from pinneaple_arena.results import BenchmarkResult
from pinneaple_arena.leaderboard import Leaderboard


# ---------------------------------------------------------------------------
# Task definition: Solve 1D Poisson  u'' = -π²sin(πx), u(0)=u(1)=0
# Metric: relative L2 error against analytic solution u=sin(πx)
# ---------------------------------------------------------------------------

def poisson_loss(model: nn.Module, device) -> torch.Tensor:
    x  = torch.linspace(0, 1, 300, device=device).unsqueeze(1).requires_grad_(True)
    u  = model(x)
    if hasattr(u, "y"):
        u = u.y
    u_x  = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    f    = -math.pi**2 * torch.sin(math.pi * x)
    res  = u_xx - f
    # BCs
    x_bc = torch.tensor([[0.0], [1.0]], device=device)
    u_bc = torch.zeros(2, 1, device=device)
    bc_out = model(x_bc)
    if hasattr(bc_out, "y"):
        bc_out = bc_out.y
    return res.pow(2).mean() + 50 * (bc_out - u_bc).pow(2).mean()


def eval_l2(model: nn.Module, device) -> float:
    x_t = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(x_t).cpu().numpy().ravel()
    u_ex = np.sin(math.pi * np.linspace(0, 1, 200))
    return float(np.sqrt(((u_pred - u_ex)**2).mean()) / np.sqrt((u_ex**2).mean()))


# ---------------------------------------------------------------------------
# Competitor models
# ---------------------------------------------------------------------------

def make_mlp(hidden: int = 64, layers: int = 4) -> nn.Module:
    mods = [nn.Linear(1, hidden), nn.Tanh()]
    for _ in range(layers - 2):
        mods += [nn.Linear(hidden, hidden), nn.Tanh()]
    mods += [nn.Linear(hidden, 1)]
    return nn.Sequential(*mods)


def train_model(model: nn.Module, device, n_epochs: int, lr: float = 1e-3) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        opt.zero_grad()
        poisson_loss(model, device).backward()
        opt.step()
    return eval_l2(model, device)


# ---------------------------------------------------------------------------
# Inline YAML benchmark config
# ---------------------------------------------------------------------------

ARENA_YAML = """
name: poisson_1d_benchmark
version: "1.0"
description: Benchmark suite for 1D Poisson PINN solvers

tasks:
  - id: poisson_1d_small
    description: "1D Poisson, 1000 epochs"
    n_epochs: 1000
    lr: 0.001

  - id: poisson_1d_medium
    description: "1D Poisson, 3000 epochs"
    n_epochs: 3000
    lr: 0.001

metrics:
  - name: relative_l2
    lower_is_better: true

models:
  - name: MLP_small
    hidden: 32
    layers: 3

  - name: MLP_medium
    hidden: 64
    layers: 4

  - name: MLP_deep
    hidden: 64
    layers: 6
"""


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Write YAML to temp file --------------------------------------------
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(ARENA_YAML)
        yaml_path = f.name

    # --- Build task specs ---------------------------------------------------
    yaml_cfg = yaml.safe_load(ARENA_YAML)
    task_specs = [
        TaskSpec(
            task_id=t["id"],
            description=t["description"],
            hyperparams={"n_epochs": t["n_epochs"], "lr": t["lr"]},
        )
        for t in yaml_cfg["tasks"]
    ]

    model_cfgs = yaml_cfg["models"]

    # --- Register tasks with Arena ------------------------------------------
    def make_native_task(spec: TaskSpec) -> NativeBenchmarkTask:
        def run_fn(model: nn.Module) -> dict:
            ep = spec.hyperparams["n_epochs"]
            lr = spec.hyperparams["lr"]
            l2 = train_model(model, device, n_epochs=ep, lr=lr)
            return {"relative_l2": l2}
        return NativeBenchmarkTask(spec=spec, run_fn=run_fn)

    tasks = [make_native_task(s) for s in task_specs]

    # --- ArenaRunner --------------------------------------------------------
    runner = ArenaRunner(tasks=tasks, device=str(device))

    all_results: list[BenchmarkResult] = []

    print("\nRunning benchmark ...")
    for m_cfg in model_cfgs:
        model = make_mlp(hidden=m_cfg["hidden"], layers=m_cfg["layers"]).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        for task in tasks:
            print(f"  model={m_cfg['name']:12s}  task={task.spec.task_id:22s} ...",
                  end=" ", flush=True)
            result = runner.run_task(model=model, task=task)
            result.model_name  = m_cfg["name"]
            result.n_params    = n_params
            all_results.append(result)
            print(f"L2={result.metrics['relative_l2']:.4e}")
            # reset weights between tasks
            for layer in model.modules():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    # --- Leaderboard --------------------------------------------------------
    board = Leaderboard(results=all_results, primary_metric="relative_l2")
    df = board.to_dataframe()
    print("\nLeaderboard:")
    print(df.to_string(index=False))

    # --- Export results -----------------------------------------------------
    board.to_csv("28_arena_results.csv")
    print("\nResults saved to 28_arena_results.csv")

    # --- Visualisation -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grouped bar chart per task
    tasks_ids  = [t.spec.task_id for t in tasks]
    model_names = [m["name"] for m in model_cfgs]
    x = np.arange(len(tasks_ids))
    width = 0.25

    for k, m_name in enumerate(model_names):
        vals = []
        for t_id in tasks_ids:
            match = [r for r in all_results
                     if r.model_name == m_name and r.task_id == t_id]
            vals.append(match[0].metrics["relative_l2"] if match else float("nan"))
        axes[0].bar(x + k * width, vals, width, label=m_name)

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(tasks_ids, rotation=10, fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative L2 error")
    axes[0].set_title("Arena benchmark: model × task matrix")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3, axis="y")

    # Pareto scatter: n_params vs. best L2
    best_l2 = {}
    best_np = {}
    for r in all_results:
        if r.model_name not in best_l2 or r.metrics["relative_l2"] < best_l2[r.model_name]:
            best_l2[r.model_name] = r.metrics["relative_l2"]
            best_np[r.model_name] = r.n_params

    for name in model_names:
        axes[1].scatter(best_np[name], best_l2[name], s=80, label=name, zorder=5)
        axes[1].annotate(name, (best_np[name], best_l2[name]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of parameters")
    axes[1].set_ylabel("Best relative L2 error")
    axes[1].set_title("Pareto front: accuracy vs. model size")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("28_arena_benchmark_result.png", dpi=120)
    print("Saved 28_arena_benchmark_result.png")

    os.unlink(yaml_path)


if __name__ == "__main__":
    main()
