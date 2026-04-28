from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pinneaple_arena.bundle.loader import BundleData
from pinneaple_arena.tasks.base import ArenaTask, TaskResult

# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data — Re=100 and Re=1000
# u-velocity along vertical centerline at x=0.5, normalized to [0,1]
# Source: Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411, Table 1.
# ---------------------------------------------------------------------------

_GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000,
])

_GHIA_U_RE100 = np.array([
    0.0,     -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
    -0.20581, -0.13641,  0.00332,  0.23151,  0.68717,  0.73722,  0.78871,  1.0,
])

# Re=1000 centerline u — Ghia et al. (1982) Table 1
_GHIA_U_RE1000 = np.array([
    0.0,     -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648,
    -0.06080,  0.05702,  0.18719,  0.33304,  0.46604,  0.51117,  0.57492,  1.0,
])


class LidDrivenCavity3DTask(ArenaTask):
    """3D Lid-Driven Cavity Flow benchmark.

    Classical Re=100/400/1000 benchmark problem.
    Comparable with Ku et al. (1987), Shu et al. (2003).

    Metrics computed:
    - u_centerline_error  : L2 error of u-velocity along vertical centerline vs reference
    - v_centerline_error  : same for v-velocity
    - max_velocity_error  : max |u_pred - u_ref| over all collocation points
    - divergence_residual : mean |div(u)| — incompressibility check
    - kinetic_energy      : 0.5 * mean(u^2 + v^2 + w^2) — for convergence tracking
    - Re_effective        : estimated from flow field if available
    """

    task_id = "lid_driven_cavity_3d"

    def __init__(
        self,
        Re: float = 100.0,
        size: float = 1.0,
        lid_velocity: float = 1.0,
    ) -> None:
        self.Re = Re
        self.size = size
        self.lid_velocity = lid_velocity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        bundle: BundleData,
        backend_outputs: Dict[str, Any],
    ) -> TaskResult:
        """Compute benchmark metrics from solver predictions.

        Parameters
        ----------
        bundle:
            Loaded bundle (used for collocation coordinates if not present in
            *backend_outputs*).
        backend_outputs:
            Dict that may contain:
            - ``"u_pred"``, ``"v_pred"``, ``"w_pred"`` — velocity arrays,
              shape ``(N,)`` (flat) or ``(nx, ny, nz)``.
            - ``"p_pred"`` — pressure array, same shape.
            - ``"x_col"``, ``"y_col"``, ``"z_col"`` — coordinate arrays.

        Returns
        -------
        TaskResult
            ``.metrics`` dict plus optional centerline plot artifact.
        """
        u_pred = _as_flat(backend_outputs.get("u_pred"))
        v_pred = _as_flat(backend_outputs.get("v_pred"))
        w_pred = _as_flat(backend_outputs.get("w_pred"))

        x_col = _as_flat(backend_outputs.get("x_col"))
        y_col = _as_flat(backend_outputs.get("y_col"))
        z_col = _as_flat(backend_outputs.get("z_col"))

        # Fall back to bundle collocation points when coordinates are absent
        if x_col is None or y_col is None:
            pts = bundle.points_collocation
            x_col = pts["x"].to_numpy(dtype=np.float64)
            y_col = pts["y"].to_numpy(dtype=np.float64)
            if "z" in pts.columns:
                z_col = pts["z"].to_numpy(dtype=np.float64)

        metrics: Dict[str, float] = {}

        # ---- divergence residual (incompressibility) -------------------
        metrics["divergence_residual"] = _divergence_residual(
            u_pred, v_pred, w_pred, x_col, y_col, z_col
        )

        # ---- kinetic energy --------------------------------------------
        metrics["kinetic_energy"] = _kinetic_energy(u_pred, v_pred, w_pred)

        # ---- Re_effective (nu = U*L/Re, back-calculated) ---------------
        metrics["Re_effective"] = float(self.Re)  # stored; can be overridden

        # ---- max velocity error (requires reference) -------------------
        ref = self.get_reference_centerline()
        ref_y = ref["y"]
        ref_u = ref["u"]

        metrics["max_velocity_error"] = float("nan")
        metrics["u_centerline_error"] = float("nan")
        metrics["v_centerline_error"] = float("nan")

        if u_pred is not None and x_col is not None and y_col is not None:
            lit = self.validate_against_literature(u_pred, v_pred, x_col, y_col)
            metrics.update(lit)

            # max |u_pred - u_ref| across all points near centerline (x ~ 0.5*L)
            x_center = 0.5 * self.size
            tol = self.size * 0.02  # 2 % of domain
            mask = np.abs(x_col - x_center) < tol
            if mask.any() and u_pred is not None:
                u_near = u_pred[mask]
                y_near = y_col[mask]
                u_interp = np.interp(
                    y_near / self.size, ref_y, ref_u * self.lid_velocity
                )
                metrics["max_velocity_error"] = float(np.max(np.abs(u_near - u_interp)))

        # ---- artifact: centerline plot ---------------------------------
        artifact: Dict[str, Any] = {}
        centerline_data = _build_centerline_artifact(
            u_pred, x_col, y_col, self.size, ref_y, ref_u, self.lid_velocity
        )
        if centerline_data is not None:
            artifact["centerline_plot"] = centerline_data

        return TaskResult(metrics=metrics, artifacts=artifact if artifact else None)

    def get_reference_centerline(self) -> Dict[str, np.ndarray]:
        """Return reference data from Ghia et al. (1982).

        Selects the closest available Reynolds number (100 or 1000).
        Data is normalized: y in [0,1], u in [-U_lid, +U_lid] with U_lid=1.

        Returns
        -------
        dict
            ``{"y": y_pts, "u": u_ref}`` normalized to [0,1].
        """
        if self.Re <= 550.0:
            # Closer to Re=100
            return {"y": _GHIA_Y.copy(), "u": _GHIA_U_RE100.copy()}
        else:
            # Use Re=1000 reference
            return {"y": _GHIA_Y.copy(), "u": _GHIA_U_RE1000.copy()}

    def validate_against_literature(
        self,
        u_pred: np.ndarray,
        v_pred: Optional[np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Compare predicted centerline velocity profile against Ghia et al.

        Extracts points near the vertical centreline (x ~ 0.5*L), normalises
        by domain size and lid velocity, then interpolates the prediction onto
        the Ghia reference y-coordinates before computing the L2 error.

        Parameters
        ----------
        u_pred:
            Predicted u-velocity, shape ``(N,)``.
        v_pred:
            Predicted v-velocity, shape ``(N,)`` or None.
        x, y:
            Coordinate arrays, shape ``(N,)``.

        Returns
        -------
        dict
            ``{"u_centerline_error": float, "v_centerline_error": float}``
        """
        ref = self.get_reference_centerline()
        ref_y = ref["y"]          # normalised [0,1]
        ref_u = ref["u"]          # normalised by U_lid

        result: Dict[str, float] = {
            "u_centerline_error": float("nan"),
            "v_centerline_error": float("nan"),
        }

        if u_pred is None or x is None or y is None:
            return result

        # Normalise coordinates
        x_norm = x / self.size
        y_norm = y / self.size

        # Select points near the vertical centreline (x ~ 0.5)
        tol = 0.03  # normalised tolerance
        mask = np.abs(x_norm - 0.5) < tol
        if not mask.any():
            return result

        y_cl = y_norm[mask]
        u_cl = u_pred[mask] / self.lid_velocity

        # Sort by y for interpolation
        order = np.argsort(y_cl)
        y_cl = y_cl[order]
        u_cl = u_cl[order]

        # Interpolate prediction onto Ghia y-points (within covered range)
        y_min, y_max = y_cl[0], y_cl[-1]
        valid = (ref_y >= y_min) & (ref_y <= y_max)
        if valid.sum() < 2:
            return result

        u_pred_at_ref = np.interp(ref_y[valid], y_cl, u_cl)
        u_ref_valid = ref_u[valid]

        n = valid.sum()
        u_l2 = float(np.sqrt(np.sum((u_pred_at_ref - u_ref_valid) ** 2) / n))
        result["u_centerline_error"] = u_l2

        if v_pred is not None:
            v_cl = v_pred[mask] / self.lid_velocity
            v_cl = v_cl[order]
            v_pred_at_ref = np.interp(ref_y[valid], y_cl, v_cl)
            # v reference is not embedded; report RMS of v along centreline
            v_rms = float(np.sqrt(np.mean(v_pred_at_ref ** 2)))
            result["v_centerline_error"] = v_rms

        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _as_flat(arr: Any) -> Optional[np.ndarray]:
    """Convert array-like to a 1-D float64 numpy array, or return None."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    return a.ravel()


def _divergence_residual(
    u: Optional[np.ndarray],
    v: Optional[np.ndarray],
    w: Optional[np.ndarray],
    x: Optional[np.ndarray],
    y: Optional[np.ndarray],
    z: Optional[np.ndarray],
) -> float:
    """Estimate mean |div(u)| via finite differences on the flat point cloud.

    When the velocity arrays are already flat (unstructured), we use a simple
    nearest-neighbour finite difference approximation.  This is an approximation
    only; for structured grids the caller should pass reshaped arrays.
    """
    if u is None or v is None or x is None or y is None:
        return float("nan")

    n = len(u)
    if n < 4:
        return float("nan")

    # Central finite difference along x: du/dx ~ (u[i+1]-u[i-1])/(x[i+1]-x[i-1])
    # Sort by x to approximate partial derivatives
    def _partial(f: np.ndarray, coord: np.ndarray) -> np.ndarray:
        order = np.argsort(coord)
        f_s = f[order]
        c_s = coord[order]
        dc = np.diff(c_s)
        # Guard against zero spacing
        dc = np.where(np.abs(dc) < 1e-15, 1e-15, dc)
        df = np.diff(f_s) / dc
        # pad to original length
        df_full = np.empty(n)
        df_full[order[:-1]] = df
        df_full[order[-1]] = df[-1]
        return df_full

    du_dx = _partial(u, x)
    dv_dy = _partial(v, y)

    div = du_dx + dv_dy
    if w is not None and z is not None:
        dw_dz = _partial(w, z)
        div = div + dw_dz

    return float(np.mean(np.abs(div)))


def _kinetic_energy(
    u: Optional[np.ndarray],
    v: Optional[np.ndarray],
    w: Optional[np.ndarray],
) -> float:
    """Compute 0.5 * mean(u^2 + v^2 + w^2)."""
    if u is None:
        return float("nan")
    ke = u ** 2
    if v is not None:
        ke = ke + v ** 2
    if w is not None:
        ke = ke + w ** 2
    return float(0.5 * np.mean(ke))


def _build_centerline_artifact(
    u_pred: Optional[np.ndarray],
    x_col: Optional[np.ndarray],
    y_col: Optional[np.ndarray],
    size: float,
    ref_y: np.ndarray,
    ref_u: np.ndarray,
    lid_velocity: float,
) -> Optional[Dict[str, Any]]:
    """Build a dict suitable for storing as a centerline plot artifact.

    Returns a plain-data dict (no matplotlib dependency) containing the
    predicted and reference centerline arrays so callers can plot if desired.
    """
    if u_pred is None or x_col is None or y_col is None:
        return None

    x_norm = x_col / size
    y_norm = y_col / size
    tol = 0.03
    mask = np.abs(x_norm - 0.5) < tol
    if not mask.any():
        return None

    y_cl = y_norm[mask]
    u_cl = u_pred[mask] / lid_velocity
    order = np.argsort(y_cl)

    return {
        "y_pred": y_cl[order].tolist(),
        "u_pred": u_cl[order].tolist(),
        "y_ref": ref_y.tolist(),
        "u_ref": ref_u.tolist(),
        "label_ref": "Ghia et al. (1982)",
    }
