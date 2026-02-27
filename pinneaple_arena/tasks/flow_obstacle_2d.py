from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import torch

from pinneaple_arena.bundle.loader import BundleData
from pinneaple_arena.registry import register_task


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


def _ensure_float_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(k, str) and _is_number(v):
            out[k] = float(v)
    return out


@register_task
@dataclass(frozen=True)
class FlowObstacle2DTask:
    """
    Task definition for the 2D steady flow around obstacle benchmark.

    Avalia em 3 modos:

    1) Torch model (backend_outputs["model"]):
       - PDE residual (Navier–Stokes) via autograd
       - BC MSE
       - L2 em sensores (opcional)

    2) Predict function (backend_outputs["predict_fn"]):
       - BC MSE (sem gradiente)
       - L2 em sensores (opcional)
       - PDE metrics = NaN (a menos que o backend forneça)

    3) Backend metrics:
       - Se backend_outputs["metrics"] já vier com as chaves do benchmark,
         fazemos passthrough.
    """

    task_id: str = "flow_obstacle_2d"

    # evaluation subsampling
    n_collocation_eval: int = 8192
    n_boundary_eval: int = 4096
    n_sensors_eval: int = 4096

    @staticmethod
    def _to_torch_xy(df: pd.DataFrame, device: str) -> torch.Tensor:
        xy = torch.tensor(df[["x", "y"]].to_numpy(), dtype=torch.float32, device=device)
        xy.requires_grad_(True)
        return xy

    @staticmethod
    def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True, allow_unused=False
        )[0]

    def _compute_bc_mse_from_arrays(self, uvp: np.ndarray, regions: np.ndarray) -> float:
        # uvp: (N,3)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]

        bc_terms = []

        m_in = regions == "inlet"
        if m_in.any():
            bc_terms.append(np.mean((u[m_in] - 1.0) ** 2))
            bc_terms.append(np.mean((v[m_in] - 0.0) ** 2))

        for reg in ("walls", "obstacle"):
            m = regions == reg
            if m.any():
                bc_terms.append(np.mean((u[m] - 0.0) ** 2))
                bc_terms.append(np.mean((v[m] - 0.0) ** 2))

        m_out = regions == "outlet"
        if m_out.any():
            bc_terms.append(np.mean((p[m_out] - 0.0) ** 2))

        if not bc_terms:
            return float("nan")
        return float(np.mean(bc_terms))

    def _compute_l2_uv_from_predict_fn(self, predict_fn: Callable[[np.ndarray], np.ndarray], sensors: pd.DataFrame) -> float:
        if len(sensors) == 0:
            return float("nan")
        sen = sensors.sample(n=min(self.n_sensors_eval, len(sensors)), replace=False, random_state=0)
        xy = sen[["x", "y"]].to_numpy(dtype=np.float32)
        uvp = np.asarray(predict_fn(xy), dtype=np.float32)
        if uvp.ndim == 1:
            uvp = uvp.reshape(-1, 1)
        if uvp.shape[1] != 3:
            out = np.zeros((uvp.shape[0], 3), dtype=np.float32)
            out[:, : min(3, uvp.shape[1])] = uvp[:, : min(3, uvp.shape[1])]
            uvp = out

        u_hat = uvp[:, 0]
        v_hat = uvp[:, 1]
        u_true = sen["u"].to_numpy(dtype=np.float32)
        v_true = sen["v"].to_numpy(dtype=np.float32)
        return float(np.sqrt(np.mean((u_hat - u_true) ** 2 + (v_hat - v_true) ** 2)))

    def _compute_metrics_predict_fn(self, bundle: BundleData, predict_fn: Callable[[np.ndarray], np.ndarray]) -> Dict[str, float]:
        bnd_df = bundle.points_boundary.sample(
            n=min(self.n_boundary_eval, len(bundle.points_boundary)), replace=False, random_state=0
        )
        xy_b = bnd_df[["x", "y"]].to_numpy(dtype=np.float32)
        uvp_b = np.asarray(predict_fn(xy_b), dtype=np.float32)
        if uvp_b.ndim == 1:
            uvp_b = uvp_b.reshape(-1, 1)
        if uvp_b.shape[1] != 3:
            out = np.zeros((uvp_b.shape[0], 3), dtype=np.float32)
            out[:, : min(3, uvp_b.shape[1])] = uvp_b[:, : min(3, uvp_b.shape[1])]
            uvp_b = out

        regions = bnd_df["region"].astype(str).to_numpy()
        bc_mse = self._compute_bc_mse_from_arrays(uvp_b, regions)

        l2_uv = float("nan")
        if bundle.sensors is not None and {"x", "y", "u", "v"}.issubset(bundle.sensors.columns):
            sen = bundle.sensors
            sen = sen[sen["split"].astype(str) == "test"] if "split" in sen.columns else sen
            if len(sen) > 0:
                l2_uv = self._compute_l2_uv_from_predict_fn(predict_fn, sen)

        return {
            "test_pde_rms": float("nan"),
            "test_div_rms": float("nan"),
            "bc_mse": float(bc_mse),
            "test_l2_uv": float(l2_uv),
        }

    def compute_metrics(self, bundle: BundleData, backend_outputs: Dict[str, Any]) -> Dict[str, float]:
        backend_metrics = _ensure_float_dict(backend_outputs.get("metrics", {}) if isinstance(backend_outputs, dict) else {})
        required = {"test_pde_rms", "test_div_rms", "bc_mse", "test_l2_uv"}

        # passthrough se backend já entregou tudo do benchmark
        if required.issubset(backend_metrics.keys()):
            return backend_metrics

        model = backend_outputs.get("model", None) if isinstance(backend_outputs, dict) else None
        predict_fn = backend_outputs.get("predict_fn", None) if isinstance(backend_outputs, dict) else None
        device = str(backend_outputs.get("device", "cpu")) if isinstance(backend_outputs, dict) else "cpu"

        # Predict_fn mode (DeepXDE, etc.)
        if model is None and callable(predict_fn):
            out = backend_metrics.copy()
            out.update(self._compute_metrics_predict_fn(bundle, predict_fn))
            return out

        # Torch mode
        if model is None:
            out = backend_metrics.copy()
            for k in required:
                out.setdefault(k, float("nan"))
            return out

        model.eval()
        nu = float(bundle.manifest["nu"])

        col_df = bundle.points_collocation.sample(
            n=min(self.n_collocation_eval, len(bundle.points_collocation)), replace=False, random_state=0
        )
        bnd_df = bundle.points_boundary.sample(
            n=min(self.n_boundary_eval, len(bundle.points_boundary)), replace=False, random_state=0
        )

        xy_col = self._to_torch_xy(col_df, device=device)
        uvp_col = model(xy_col)
        u = uvp_col[:, 0:1]
        v = uvp_col[:, 1:2]
        p = uvp_col[:, 2:3]

        gu = self._grad(u, xy_col)
        gv = self._grad(v, xy_col)
        gp = self._grad(p, xy_col)

        u_x, u_y = gu[:, 0:1], gu[:, 1:2]
        v_x, v_y = gv[:, 0:1], gv[:, 1:2]
        p_x, p_y = gp[:, 0:1], gp[:, 1:2]

        u_xx = self._grad(u_x, xy_col)[:, 0:1]
        u_yy = self._grad(u_y, xy_col)[:, 1:2]
        v_xx = self._grad(v_x, xy_col)[:, 0:1]
        v_yy = self._grad(v_y, xy_col)[:, 1:2]

        mom_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        mom_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        cont = u_x + v_y

        test_pde_rms = torch.sqrt((mom_u.pow(2).mean() + mom_v.pow(2).mean()) / 2.0).detach().cpu().item()
        test_div_rms = torch.sqrt(cont.pow(2).mean()).detach().cpu().item()

        with torch.inference_mode():
            xy_b = torch.tensor(bnd_df[["x", "y"]].to_numpy(), dtype=torch.float32, device=device)
            uvp_b = model(xy_b)
            u_b = uvp_b[:, 0:1]
            v_b = uvp_b[:, 1:2]
            p_b = uvp_b[:, 2:3]

        regions = bnd_df["region"].astype(str).to_numpy()
        bc_terms = []

        m_in = torch.tensor(regions == "inlet", device=device)
        if bool(m_in.any()):
            bc_terms.append((u_b[m_in] - 1.0).pow(2).mean())
            bc_terms.append((v_b[m_in] - 0.0).pow(2).mean())

        for reg in ("walls", "obstacle"):
            m = torch.tensor(regions == reg, device=device)
            if bool(m.any()):
                bc_terms.append((u_b[m] - 0.0).pow(2).mean())
                bc_terms.append((v_b[m] - 0.0).pow(2).mean())

        m_out = torch.tensor(regions == "outlet", device=device)
        if bool(m_out.any()):
            bc_terms.append((p_b[m_out] - 0.0).pow(2).mean())

        bc_mse = torch.stack(bc_terms).mean().detach().cpu().item() if bc_terms else float("nan")

        l2_uv = float("nan")
        if bundle.sensors is not None and {"x", "y", "u", "v"}.issubset(bundle.sensors.columns):
            sen = bundle.sensors
            sen = sen[sen["split"].astype(str) == "test"] if "split" in sen.columns else sen
            if len(sen) > 0:
                sen = sen.sample(n=min(self.n_sensors_eval, len(sen)), replace=False, random_state=0)
                with torch.inference_mode():
                    xy_s = torch.tensor(sen[["x", "y"]].to_numpy(), dtype=torch.float32, device=device)
                    uvp_s = model(xy_s)
                    u_hat = uvp_s[:, 0].detach().cpu().numpy()
                    v_hat = uvp_s[:, 1].detach().cpu().numpy()
                u_true = sen["u"].to_numpy()
                v_true = sen["v"].to_numpy()
                l2_uv = float(np.sqrt(np.mean((u_hat - u_true) ** 2 + (v_hat - v_true) ** 2)))

        out = backend_metrics.copy()
        out.update(
            {
                "test_pde_rms": float(test_pde_rms),
                "test_div_rms": float(test_div_rms),
                "bc_mse": float(bc_mse),
                "test_l2_uv": float(l2_uv),
            }
        )
        return out