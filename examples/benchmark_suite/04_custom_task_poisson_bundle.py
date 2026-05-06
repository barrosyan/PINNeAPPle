"""04 — Custom task + custom backend + synthetic bundle (Poisson2D).

This is the "maximum flexibility" example:
- Create a new task *without touching* pinneaple_arena core
- Create a new backend for that task
- Define a custom bundle schema and generate a minimal on-disk bundle

It uses a classic Poisson problem on [0,1]^2:
    u(x,y) = sin(pi x) sin(pi y)
    -Δu = f
    u=0 on the boundary

Run from repo root:
    python examples/arena/04_custom_task_poisson_bundle.py

Note: this example needs a parquet engine:
    pip install pyarrow
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pinneaple_arena.bundle.loader import BundleData, load_bundle
from pinneaple_arena.bundle.schema import load_bundle_schema
from pinneaple_arena.registry import register_backend, register_task
from pinneaple_arena.runner.run_benchmark import run_benchmark


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "This example needs parquet writing support. Install: pip install pyarrow\n"
            f"Original import error: {e}"
        )


def _u_true(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x) * np.sin(math.pi * y)


def _f_rhs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # -Δ sin(pi x) sin(pi y) = 2*pi^2*sin(pi x) sin(pi y)
    return 2.0 * (math.pi**2) * _u_true(x, y)


def _make_poisson_bundle(root: Path) -> Path:
    """Create a minimal bundle on disk."""
    if root.exists():
        shutil.rmtree(root)
    (root / "bundle").mkdir(parents=True, exist_ok=True)
    (root / "derived").mkdir(parents=True, exist_ok=True)

    # geometry placeholder (arena schema wants it, but this task doesn't)
    (root / "bundle" / "geometry.usd").write_bytes(b"")

    manifest = {
        "problem_id": "poisson_2d_sine",
        "pde": "-laplacian(u)=f",
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "fields": ["u"],
        "weights": {"pde": 1.0, "bc": 50.0},
    }
    (root / "bundle" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    conditions = {
        "boundary": {"type": "dirichlet", "u": 0.0},
    }
    (root / "bundle" / "conditions.json").write_text(json.dumps(conditions, indent=2), encoding="utf-8")

    rng = np.random.default_rng(0)

    # Collocation points (interior)
    n_col = 20000
    xy = rng.random((n_col, 2), dtype=np.float32)
    col = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})

    # Boundary points
    n_b = 8000
    t = rng.random(n_b, dtype=np.float32)
    edges = rng.integers(0, 4, size=n_b)
    xb = np.zeros(n_b, dtype=np.float32)
    yb = np.zeros(n_b, dtype=np.float32)
    xb[edges == 0] = 0.0
    yb[edges == 0] = t[edges == 0]
    xb[edges == 1] = 1.0
    yb[edges == 1] = t[edges == 1]
    xb[edges == 2] = t[edges == 2]
    yb[edges == 2] = 0.0
    xb[edges == 3] = t[edges == 3]
    yb[edges == 3] = 1.0

    bnd = pd.DataFrame({"x": xb, "y": yb, "region": "boundary"})

    col.to_parquet(root / "derived" / "points_collocation.parquet", index=False)
    bnd.to_parquet(root / "derived" / "points_boundary.parquet", index=False)

    return root


def _write_poisson_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                'bundle_schema_version: "1.0"',
                "",
                "required_files:",
                '  - "bundle/geometry.usd"',
                '  - "bundle/conditions.json"',
                '  - "bundle/manifest.json"',
                '  - "derived/points_boundary.parquet"',
                '  - "derived/points_collocation.parquet"',
                "",
                "conditions_json:",
                "  required_regions: [\"boundary\"]",
                "",
                "manifest_json:",
                "  required_keys: [problem_id, pde, domain, fields, weights]",
                "",
            ]
        ),
        encoding="utf-8",
    )


@register_task
@dataclass(frozen=True)
class Poisson2DTask:
    task_id: str = "poisson_2d"

    n_eval_col: int = 4096
    n_eval_bnd: int = 2048

    @staticmethod
    def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

    def compute_metrics(self, bundle: BundleData, backend_outputs: Dict[str, Any]) -> Dict[str, float]:
        model = backend_outputs.get("model")
        device = str(backend_outputs.get("device", "cpu"))
        if model is None:
            return {"test_pde_rms": float("nan"), "test_div_rms": float("nan"), "bc_mse": float("nan"), "test_l2_uv": float("nan")}

        model.eval()

        col = bundle.points_collocation.sample(n=min(self.n_eval_col, len(bundle.points_collocation)), random_state=0)
        bnd = bundle.points_boundary.sample(n=min(self.n_eval_bnd, len(bundle.points_boundary)), random_state=0)

        xy_col = torch.tensor(col[["x", "y"]].to_numpy(), dtype=torch.float32, device=device)
        xy_col.requires_grad_(True)
        u = model(xy_col)

        gu = self._grad(u, xy_col)
        u_x = gu[:, 0:1]
        u_y = gu[:, 1:2]
        u_xx = self._grad(u_x, xy_col)[:, 0:1]
        u_yy = self._grad(u_y, xy_col)[:, 1:2]

        x = xy_col[:, 0:1].detach().cpu().numpy()
        y = xy_col[:, 1:2].detach().cpu().numpy()
        f = torch.tensor(_f_rhs(x, y), dtype=torch.float32, device=device)

        res = -(u_xx + u_yy) - f
        test_pde_rms = torch.sqrt(res.pow(2).mean()).detach().cpu().item()

        with torch.inference_mode():
            xy_b = torch.tensor(bnd[["x", "y"]].to_numpy(), dtype=torch.float32, device=device)
            u_b = model(xy_b)
        bc_mse = (u_b.pow(2).mean()).detach().cpu().item()

        # L2 against analytic on random points
        rng = np.random.default_rng(1)
        pts = rng.random((4096, 2), dtype=np.float32)
        with torch.inference_mode():
            u_hat = model(torch.tensor(pts, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
        u_gt = _u_true(pts[:, 0], pts[:, 1]).reshape(-1)
        test_l2_u = float(np.sqrt(np.mean((u_hat - u_gt) ** 2)))

        return {"test_pde_rms": float(test_pde_rms), "test_div_rms": float("nan"), "bc_mse": float(bc_mse), "test_l2_uv": float(test_l2_u)}


class _MLP(nn.Module):
    def __init__(self, width: int = 128, depth: int = 4):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Tanh())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_backend
class PoissonPINNBackend:
    name = "poisson_pinn"

    def train(self, bundle: BundleData, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
        train = dict(run_cfg.get("train", {}))
        model_cfg = dict(run_cfg.get("model", {}))

        device = str(train.get("device", "cpu"))
        epochs = int(train.get("epochs", 3000))
        lr = float(train.get("lr", 1e-3))
        seed = int(train.get("seed", 0))
        log_every = int(train.get("log_every", 200))
        w_pde = float(train.get("w_pde", 1.0))
        w_bc = float(train.get("w_bc", 50.0))

        n_col = int(train.get("n_collocation", 2048))
        n_bnd = int(train.get("n_boundary", 1024))

        width = int(model_cfg.get("width", 128))
        depth = int(model_cfg.get("depth", 4))

        torch.manual_seed(seed)
        dev = torch.device(device)

        model = _MLP(width=width, depth=depth).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        col_df = bundle.points_collocation
        bnd_df = bundle.points_boundary

        def sample_col(n: int) -> np.ndarray:
            df = col_df.sample(n=min(n, len(col_df)), replace=(len(col_df) < n), random_state=None)
            return df[["x", "y"]].to_numpy(dtype=np.float32)

        def sample_bnd(n: int) -> np.ndarray:
            df = bnd_df.sample(n=min(n, len(bnd_df)), replace=(len(bnd_df) < n), random_state=None)
            return df[["x", "y"]].to_numpy(dtype=np.float32)

        def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

        last = {"total": float("nan"), "pde": float("nan"), "bc": float("nan")}

        for step in range(epochs):
            opt.zero_grad(set_to_none=True)

            # PDE loss
            xy = torch.tensor(sample_col(n_col), dtype=torch.float32, device=dev)
            xy.requires_grad_(True)
            u = model(xy)
            gu = grad(u, xy)
            u_x = gu[:, 0:1]
            u_y = gu[:, 1:2]
            u_xx = grad(u_x, xy)[:, 0:1]
            u_yy = grad(u_y, xy)[:, 1:2]

            x = xy[:, 0:1].detach().cpu().numpy()
            y = xy[:, 1:2].detach().cpu().numpy()
            f = torch.tensor(_f_rhs(x, y), dtype=torch.float32, device=dev)
            res = -(u_xx + u_yy) - f
            pde_loss = res.pow(2).mean()

            # Boundary loss
            xy_b = torch.tensor(sample_bnd(n_bnd), dtype=torch.float32, device=dev)
            u_b = model(xy_b)
            bc_loss = u_b.pow(2).mean()

            loss = w_pde * pde_loss + w_bc * bc_loss
            loss.backward()
            opt.step()

            if step % log_every == 0 or step == epochs - 1:
                last = {"total": float(loss.detach().cpu()), "pde": float(pde_loss.detach().cpu()), "bc": float(bc_loss.detach().cpu())}
                print(f"[poisson_pinn] step={step:05d} total={last['total']:.3e} pde={last['pde']:.3e} bc={last['bc']:.3e}")

        return {"device": device, "model": model, "metrics": {"train_total": last["total"], "train_pde": last["pde"], "train_bc": last["bc"]}}


def main() -> None:
    _require_pyarrow()

    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = repo_root / "artifacts" / "examples" / "toy_poisson_bundle"

    bundle_root = _make_poisson_bundle(artifacts_dir / "bundle_poisson2d")

    schema_path = artifacts_dir / "bundle_schema_poisson.yaml"
    _write_poisson_schema(schema_path)

    # Tiny task config / run config saved into artifacts for traceability
    task_cfg = artifacts_dir / "task_poisson2d.yaml"
    task_cfg.write_text(
        "\n".join(
            [
                "task_id: poisson_2d",
                f"bundle_root: {bundle_root.as_posix()}",
                "require_sensors: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_cfg = artifacts_dir / "run_poisson_pinn.yaml"
    run_cfg.write_text(
        "\n".join(
            [
                "run_name: poisson_pinn_demo",
                "",
                "backend:",
                "  name: poisson_pinn",
                "",
                "train:",
                "  device: cpu",
                "  epochs: 2500",
                "  lr: 0.001",
                "  seed: 0",
                "  log_every: 250",
                "  w_pde: 1.0",
                "  w_bc: 50.0",
                "  n_collocation: 2048",
                "  n_boundary: 1024",
                "",
                "model:",
                "  width: 128",
                "  depth: 4",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out = run_benchmark(
        artifacts_dir=artifacts_dir,
        task_cfg_path=task_cfg,
        run_cfg_path=run_cfg,
        bundle_schema_path=schema_path,
    )

    print("\n=== DONE ===")
    print("run_dir:", out["run_dir"])
    print("key metrics:", out["summary"]["key_metrics"])


if __name__ == "__main__":
    main()
