from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from pinneaple_arena.bundle.loader import BundleData


@dataclass(frozen=True)
class FlowObstacle2DTask:
    task_id: str = "flow_obstacle_2d"

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

    def compute_metrics(self, bundle: BundleData, backend_outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        backend_outputs may contain:
          - "model": a torch.nn.Module mapping (N,2)->(N,3) (u,v,p)
          - "device": string
          - optionally precomputed metrics
        If not available, returns whatever backend provided.
        """
        # If backend already produced full metrics, pass-through
        if "metrics" in backend_outputs and isinstance(backend_outputs["metrics"], dict):
            m = backend_outputs["metrics"]
            # ensure float values
            return {k: float(v) for k, v in m.items() if _is_number(v)}

        model = backend_outputs.get("model", None)
        device = str(backend_outputs.get("device", "cpu"))

        # If no model, return empty (backend must provide metrics)
        if model is None:
            return {}

        model.eval()
        nu = float(bundle.manifest["nu"])

        # Evaluate on a fixed subset for speed
        col_df = bundle.points_collocation.sample(n=min(8192, len(bundle.points_collocation)), replace=False, random_state=0)
        bnd_df = bundle.points_boundary.sample(n=min(4096, len(bundle.points_boundary)), replace=False, random_state=0)

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

        # BC MSE
        xy_b = torch.tensor(bnd_df[["x", "y"]].to_numpy(), dtype=torch.float32, device=device)
        uvp_b = model(xy_b)
        u_b = uvp_b[:, 0:1]
        v_b = uvp_b[:, 1:2]
        p_b = uvp_b[:, 2:3]

        regions = bnd_df["region"].astype(str).to_numpy()

        bc_terms = []

        # inlet u=1 v=0
        m_in = torch.tensor(regions == "inlet", device=device)
        if bool(m_in.any()):
            bc_terms.append((u_b[m_in] - 1.0).pow(2).mean())
            bc_terms.append((v_b[m_in] - 0.0).pow(2).mean())

        # walls, obstacle no-slip
        for reg in ("walls", "obstacle"):
            m = torch.tensor(regions == reg, device=device)
            if bool(m.any()):
                bc_terms.append((u_b[m] - 0.0).pow(2).mean())
                bc_terms.append((v_b[m] - 0.0).pow(2).mean())

        # outlet p=0
        m_out = torch.tensor(regions == "outlet", device=device)
        if bool(m_out.any()):
            bc_terms.append((p_b[m_out] - 0.0).pow(2).mean())

        bc_mse = torch.stack(bc_terms).mean().detach().cpu().item() if bc_terms else float("nan")

        # Optional supervised L2 if sensors exist and have u,v
        l2_uv = float("nan")
        if bundle.sensors is not None and {"x", "y", "u", "v"}.issubset(bundle.sensors.columns):
            sen = bundle.sensors
            sen = sen[sen["split"].astype(str) == "test"] if "split" in sen.columns else sen
            if len(sen) > 0:
                sen = sen.sample(n=min(4096, len(sen)), replace=False, random_state=0)
                xy_s = torch.tensor(sen[["x", "y"]].to_numpy(), dtype=torch.float32, device=device)
                uvp_s = model(xy_s)
                u_hat = uvp_s[:, 0].detach().cpu().numpy()
                v_hat = uvp_s[:, 1].detach().cpu().numpy()
                u_true = sen["u"].to_numpy()
                v_true = sen["v"].to_numpy()
                l2_uv = float(np.sqrt(np.mean((u_hat - u_true) ** 2 + (v_hat - v_true) ** 2)))

        return {
            "test_pde_rms": float(test_pde_rms),
            "test_div_rms": float(test_div_rms),
            "bc_mse": float(bc_mse),
            "test_l2_uv": float(l2_uv),
        }


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False
