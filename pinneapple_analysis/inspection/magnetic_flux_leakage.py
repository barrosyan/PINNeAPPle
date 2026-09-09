"""pinneapple_analysis.inspection.magnetic_flux_leakage -- a lightweight
2D nonlinear-permeability magnetostatic FDM ground truth for Magnetic
Flux Leakage (MFL) inspection, plus a physics-informed (PIML) training
path.

Physical picture (simplified, illustrative -- see caveats below)
------------------------------------------------------------------
A steel plate is magnetized by a yoke: this module models that as a
Dirichlet boundary condition on a magnetic SCALAR potential phi(x, y)
(``phi = +phi0`` at the left edge, ``phi = -phi0`` at the right edge,
representing the yoke's two poles), with the plate occupying the
remaining domain. Away from any free current, magnetostatics reduces to

    div(mu(x, y) * grad(phi)) = 0,          B = -mu * grad(phi)

which is solved by a standard variable-coefficient 5-point finite
DIFFERENCE / finite-VOLUME stencil (harmonic-mean face permeability,
the standard treatment for a discontinuous coefficient elliptic PDE --
see e.g. LeVeque, "Finite Difference Methods for ODEs and PDEs", 2007,
Sec. 3.7), with a subsurface DEFECT modeled as a small region of very
low permeability (mu ~ mu0, i.e. as if it were air/void -- a corrosion
pit or crack) embedded in the steel.

Steel is NONLINEAR: this module uses a simple Frohlich-Kennelly-type
saturation law

    mu(H) = mu0 + mu0*(mu_r_max - 1) / (1 + (|H| / H_sat)**2)

(a standard simple single-valued -- i.e. non-hysteretic -- saturation
curve shape used for illustrative nonlinear-magnetics demonstrations;
NOT a fit to any specific certified steel grade's measured B-H curve)
solved self-consistently via PICARD ITERATION: solve the linear problem
with the current mu-field guess, recompute H = |grad(phi)| and hence a
new mu(H), and repeat.

Leakage signal: the vertical flux density Bz just beneath the top
surface (a finite-difference derivative of phi at the last interior
row) is used as the synthetic "probe scan" signal -- a defect locally
diverts flux around/through its low-permeability region, producing a
local Bz anomaly directly above it, exactly the MFL detection mechanism
(see e.g. ASNT NDT Handbook, "Magnetic Flux Leakage Testing").

Caveats (explicit, not to be treated as a validated instrument model):
this is a 2D cross-section, not 3D; hysteresis is ignored; the leakage
readout uses the last interior grid row as a stand-in for a lifted
sensor rather than solving the field in the air gap above the plate.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .base import InspectionPINN, PhysicsLossFn, train_generic_inspection_pinn

MU0 = 4.0 * np.pi * 1e-7


def _saturation_mu(H_mag: np.ndarray, mu_r_max: float, H_sat: float) -> np.ndarray:
    """Frohlich-Kennelly-type single-valued saturation curve mu(H)."""
    return MU0 + MU0 * (mu_r_max - 1.0) / (1.0 + (H_mag / H_sat) ** 2)


def _assemble_variable_coeff_laplacian(mu: np.ndarray, dx: float, dy: float):
    """Assemble the sparse system for div(mu * grad(phi)) = 0 on a
    uniform (nx, ny) grid via a finite-volume 5-point stencil with
    harmonic-mean face permeability. Dirichlet BC applied by the caller
    (rows for boundary nodes are overwritten with identity rows).
    Node ordering n = i*ny + j (i: x-index, j: y-index)."""
    from scipy.sparse import lil_matrix

    nx, ny = mu.shape
    N = nx * ny
    M = lil_matrix((N, N))

    def hmean(a, b):
        return 2.0 * a * b / (a + b) if (a + b) > 0 else 0.0

    for i in range(nx):
        for j in range(ny):
            n = i * ny + j
            if i == 0 or i == nx - 1:
                M[n, n] = 1.0
                continue
            mu_c = mu[i, j]
            coeffs: Dict[Tuple[int, int], float] = {}
            diag = 0.0

            # x-direction faces (always present: i in [1, nx-2])
            mu_e = hmean(mu_c, mu[i + 1, j])
            mu_w = hmean(mu_c, mu[i - 1, j])
            coeffs[(i + 1, j)] = coeffs.get((i + 1, j), 0.0) + mu_e / dx ** 2
            coeffs[(i - 1, j)] = coeffs.get((i - 1, j), 0.0) + mu_w / dx ** 2
            diag -= (mu_e + mu_w) / dx ** 2

            # y-direction faces: Neumann/no-flux at j==0 / j==ny-1 (simply
            # omit the missing face -- equivalent to a zero-flux ghost node)
            if j < ny - 1:
                mu_n = hmean(mu_c, mu[i, j + 1])
                coeffs[(i, j + 1)] = coeffs.get((i, j + 1), 0.0) + mu_n / dy ** 2
                diag -= mu_n / dy ** 2
            if j > 0:
                mu_s = hmean(mu_c, mu[i, j - 1])
                coeffs[(i, j - 1)] = coeffs.get((i, j - 1), 0.0) + mu_s / dy ** 2
                diag -= mu_s / dy ** 2

            M[n, n] = diag
            for (ii, jj), c in coeffs.items():
                M[n, ii * ny + jj] = c

    return M.tocsr()


def solve_mfl_nonlinear(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    phi0: float,
    mu_r_max: float,
    H_sat: float,
    defect_mask: np.ndarray,
    n_picard: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the nonlinear MFL magnetostatic scalar-potential problem via
    Picard iteration. `defect_mask` (nx, ny) bool marks the low-permeability
    flaw region (held fixed at mu0, never updated by the saturation law).
    Returns (x, y, phi, mu) -- actually (x, y, phi) plus mu is returned
    separately by the caller; see `generate_magnetic_flux_leakage_synthetic`.
    """
    from scipy.sparse.linalg import spsolve

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]

    mu = MU0 * 200.0 * np.ones((nx, ny))  # initial linear guess (typical unsaturated steel mu_r)
    mu[defect_mask] = MU0

    phi = np.zeros((nx, ny))
    for _ in range(n_picard):
        M = _assemble_variable_coeff_laplacian(mu, dx, dy)
        b = np.zeros(nx * ny)
        for j in range(ny):
            b[0 * ny + j] = phi0
            b[(nx - 1) * ny + j] = -phi0
        phi_flat = spsolve(M, b)
        phi = phi_flat.reshape(nx, ny)

        Hx = -np.gradient(phi, dx, axis=0)
        Hy = -np.gradient(phi, dy, axis=1)
        H_mag = np.sqrt(Hx ** 2 + Hy ** 2) + 1e-6
        mu_new = _saturation_mu(H_mag, mu_r_max, H_sat)
        mu_new[defect_mask] = MU0
        mu = mu_new

    return x, y, phi, mu


def generate_magnetic_flux_leakage_synthetic(
    *,
    nx: int = 36,
    ny: int = 14,
    Lx: float = 0.2,
    Ly: float = 0.02,
    phi0: float = 500.0,
    mu_r_max: float = 500.0,
    H_sat: float = 2000.0,
    defect_center: float = 0.5,   # fraction of Lx
    defect_width: float = 0.02,
    defect_depth_frac: float = 0.5,  # fraction of Ly, measured from top surface
    n_picard: int = 6,
    n_supervised: int = 300,
    n_collocation: int = 150,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Generate a synthetic MFL scan: solves the nonlinear magnetostatic
    FDM ground truth above, then reads off the surface leakage field Bz
    as the "probe scan" signal along x.

    Returns a dict with:
      x, y            -- 1D grids (m)
      phi, mu         -- (nx, ny) solved scalar potential and converged
                          permeability field
      scan_positions  -- x-coordinates of the leakage readout (m)
      scan_signal     -- Bz(x) just beneath the top surface (leakage signal)
      x_train, y_train (as InspectionPINN "x"/"y") -- supervised (x,y)->phi pairs
      x_col           -- collocation points for the physics residual
      mu_grid_x, mu_grid_y -- the (x, y) grids `mu` is defined on, for
                          building the physics-residual permeability
                          interpolator
    """
    rng = np.random.default_rng(seed)

    x1 = np.linspace(0.0, Lx, nx)
    y1 = np.linspace(0.0, Ly, ny)
    XX, YY = np.meshgrid(x1, y1, indexing="ij")

    defect_x0 = defect_center * Lx
    defect_top = Ly * (1.0 - defect_depth_frac) - 0.15 * Ly
    defect_bot = Ly * (1.0 - defect_depth_frac) + 0.15 * Ly
    defect_mask = (np.abs(XX - defect_x0) <= defect_width / 2.0) & (YY >= defect_top) & (YY <= defect_bot)

    x, y, phi, mu = solve_mfl_nonlinear(nx, ny, Lx, Ly, phi0, mu_r_max, H_sat, defect_mask, n_picard=n_picard)

    # Leakage signal: Bz = -mu * dphi/dy, read at the row just beneath the
    # top surface (y = Ly), as a stand-in for a lightly-lifted-off sensor.
    dphi_dy_top = (phi[:, -1] - phi[:, -2]) / (y[-1] - y[-2])
    Bz_top = -mu[:, -1] * dphi_dy_top

    phi_scale = float(np.max(np.abs(phi)))
    if phi_scale <= 0:
        phi_scale = 1.0

    n_sup = min(n_supervised, nx * ny)
    flat_idx = rng.choice(nx * ny, size=n_sup, replace=False)
    xi, yi = flat_idx // ny, flat_idx % ny
    x_sup = np.stack([x[xi], y[yi]], axis=1).astype(np.float32)
    y_sup = (phi[xi, yi] / phi_scale).astype(np.float32)[:, None]

    margin_x = 0.1 * Lx
    x_col = np.stack(
        [rng.uniform(margin_x, Lx - margin_x, n_collocation), rng.uniform(0.02 * Ly, 0.98 * Ly, n_collocation)],
        axis=1,
    ).astype(np.float32)

    return {
        "x_grid": x,
        "y_grid": y,
        "phi": phi,
        "mu": mu,
        "scan_positions": x,
        "scan_signal": Bz_top,
        "x": x_sup,
        "y": y_sup,
        "x_col": x_col,
        "phi_scale": phi_scale,
        "defect_mask": defect_mask,
    }


def make_mfl_physics_loss(x_grid: np.ndarray, y_grid: np.ndarray, mu_field: np.ndarray) -> PhysicsLossFn:
    """Build a `(model, batch) -> (loss, comps)` physics loss enforcing a
    SIMPLIFIED version of div(mu*grad(phi))=0: `mu(x,y) * laplacian(phi)`,
    i.e. dropping the `grad(mu).grad(phi)` cross term. This is an honest
    simplification -- exact only where mu varies slowly, which is most of
    the domain except right at the sharp defect boundary -- adopted so the
    residual needs only a fixed, precomputed mu(x,y) lookup rather than a
    differentiable representation of mu's own spatial gradient."""
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (x_grid, y_grid), mu_field, bounds_error=False, fill_value=None,
    )

    def physics_loss_fn(model: torch.nn.Module, batch: Dict[str, Any]):
        x_col = batch["x_col"]
        if not x_col.requires_grad:
            x_col = x_col.clone().requires_grad_(True)

        phi = model.predict(x_col) if hasattr(model, "predict") else model(x_col)
        phi = phi[:, 0:1]

        g = torch.autograd.grad(phi, x_col, grad_outputs=torch.ones_like(phi), create_graph=True, retain_graph=True)[0]
        dphidx, dphidy = g[:, 0:1], g[:, 1:2]
        d2phidx2 = torch.autograd.grad(dphidx, x_col, grad_outputs=torch.ones_like(dphidx), create_graph=True, retain_graph=True)[0][:, 0:1]
        d2phidy2 = torch.autograd.grad(dphidy, x_col, grad_outputs=torch.ones_like(dphidy), create_graph=True, retain_graph=True)[0][:, 1:2]
        laplacian = d2phidx2 + d2phidy2

        mu_vals = interp(x_col.detach().cpu().numpy())
        mu_t = torch.as_tensor(mu_vals, dtype=phi.dtype, device=phi.device).reshape(-1, 1)

        residual = mu_t * laplacian
        loss = torch.mean(residual ** 2)
        comps = {"residual_rmse": float(torch.sqrt(torch.mean(residual.detach() ** 2)).item())}
        return loss, comps

    return physics_loss_fn


def train_magnetic_flux_leakage(
    synthetic_data: Optional[Dict[str, Any]] = None,
    *,
    hidden=(64, 64, 64),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    seed: Optional[int] = 0,
    log_dir: str = "runs/inspection_mfl",
    run_name: str = "mfl_pinn",
    w_supervised: float = 1.0,
    w_physics: float = 1.0,
    **synthetic_kwargs: Any,
) -> Dict[str, Any]:
    if synthetic_data is None:
        synthetic_data = generate_magnetic_flux_leakage_synthetic(seed=seed, **synthetic_kwargs)

    model = InspectionPINN(in_dim=2, out_dim=1, hidden=list(hidden), modality="magnetic_flux_leakage")
    physics_fn = make_mfl_physics_loss(synthetic_data["x_grid"], synthetic_data["y_grid"], synthetic_data["mu"])
    train_out = train_generic_inspection_pinn(
        model, synthetic_data, physics_fn,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device, seed=seed,
        log_dir=log_dir, run_name=run_name, w_supervised=w_supervised, w_physics=w_physics,
    )
    return {"model": model, "train_out": train_out, "synthetic_data": synthetic_data}
