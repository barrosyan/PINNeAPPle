"""Meshfree geometry utilities.

Provides geometry-focused meshfree operations that complement
pinneapple_geom's mesh and point-cloud tooling:

  RBFInterpolator
      Scattered field interpolation / transfer between point clouds.
      Useful for mapping CFD results to PINN collocation grids, upsampling
      scalar/vector fields, or generating smooth reference data.

  ImplicitSurfaceRBF
      Point cloud + normals -> continuous SDF approximation.
      Reconstructs a signed distance function from scattered boundary samples
      without requiring an explicit mesh, enabling direct use as a PINN domain.

For *PDE solving* via meshfree methods (Kansa RBF collocation, MLS collocation)
see ``pinneapple_solvers.meshfree``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# RBF kernels — numpy, returned value = phi(r)
# ---------------------------------------------------------------------------

def _rbf_gaussian(r: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-(eps * r) ** 2)


def _rbf_multiquadric(r: np.ndarray, eps: float) -> np.ndarray:
    return np.sqrt(1.0 + (eps * r) ** 2)


def _rbf_imq(r: np.ndarray, eps: float) -> np.ndarray:
    return 1.0 / np.sqrt(1.0 + (eps * r) ** 2)


def _rbf_thin_plate(r: np.ndarray, eps: float = 1.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0.0, r ** 2 * np.log(r), 0.0)


_KERNELS = {
    "gaussian":     _rbf_gaussian,
    "multiquadric": _rbf_multiquadric,
    "imq":          _rbf_imq,
    "thin_plate":   _rbf_thin_plate,
}


# ---------------------------------------------------------------------------
# RBF kernel gradients — returns dφ/dr / r  (multiply by displacement to get ∇φ)
#
# For any kernel: ∂φ/∂x_d = (dφ/dr / r) * (x_d - c_d)
# ---------------------------------------------------------------------------

def _grad_factor_gaussian(r: np.ndarray, eps: float) -> np.ndarray:
    # dφ/dr = -2ε²r φ  →  dφ/dr / r = -2ε² φ
    return -2.0 * eps ** 2 * np.exp(-(eps * r) ** 2)


def _grad_factor_multiquadric(r: np.ndarray, eps: float) -> np.ndarray:
    # dφ/dr = ε²r / φ  →  dφ/dr / r = ε² / φ
    phi = np.sqrt(1.0 + (eps * r) ** 2)
    return eps ** 2 / phi


def _grad_factor_imq(r: np.ndarray, eps: float) -> np.ndarray:
    # dφ/dr = -ε²r / φ³  →  dφ/dr / r = -ε² / φ³
    phi3 = (1.0 + (eps * r) ** 2) ** 1.5
    return -eps ** 2 / phi3


def _grad_factor_thin_plate(r: np.ndarray, eps: float = 1.0) -> np.ndarray:
    # φ = r² log r, dφ/dr = r(2 log r + 1)  →  dφ/dr / r = (2 log r + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0.0, 2.0 * np.log(r) + 1.0, 0.0)


_GRAD_FACTORS = {
    "gaussian":     _grad_factor_gaussian,
    "multiquadric": _grad_factor_multiquadric,
    "imq":          _grad_factor_imq,
    "thin_plate":   _grad_factor_thin_plate,
}


# ---------------------------------------------------------------------------
# Distance matrix helper
# ---------------------------------------------------------------------------

def _pairwise_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(N, d), (M, d) -> (N, M) Euclidean distances."""
    diff = X[:, None, :] - Y[None, :, :]    # (N, M, d)
    return np.sqrt((diff ** 2).sum(-1))      # (N, M)


# ---------------------------------------------------------------------------
# RBFInterpolator
# ---------------------------------------------------------------------------

class RBFInterpolator:
    """Scattered data interpolation via Radial Basis Functions.

    Fits a global RBF interpolant to paired (source_pts, source_vals) and
    evaluates it at arbitrary query points. Useful for:

    - Transferring field data (velocity, pressure, temperature) from a CFD
      mesh to a PINN collocation grid.
    - Upsampling scalar or vector fields on point clouds.
    - Mapping simulation results onto a denser evaluation set.

    This is a *pure interpolation* tool — it does not solve PDEs. For
    meshfree PDE solving (Kansa method, MLS collocation) see
    ``pinneapple_solvers.meshfree.RBFCollocationSolver``.

    Parameters
    ----------
    kernel : "gaussian" | "multiquadric" | "imq" | "thin_plate"
        Radial basis function kernel.
    eps : float
        Shape parameter controlling kernel width (not used for thin_plate).
    reg : float
        Tikhonov regularisation added to the interpolation matrix diagonal.

    Examples
    --------
    >>> interp = RBFInterpolator(kernel="gaussian", eps=2.0)
    >>> interp.fit(source_pts, source_vals)
    >>> u_query = interp(query_pts)      # (Q,)
    >>> grad_u  = interp.gradient(query_pts)   # (Q, d)
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        eps: float = 1.0,
        reg: float = 1e-10,
    ) -> None:
        if kernel not in _KERNELS:
            raise ValueError(
                f"Unknown kernel {kernel!r}. Choose from {list(_KERNELS)}"
            )
        self.kernel = kernel
        self.eps = float(eps)
        self.reg = float(reg)
        self._source_pts: Optional[np.ndarray] = None
        self._coeffs: Optional[np.ndarray] = None
        self._n_outputs: int = 1

    # ------------------------------------------------------------------

    def fit(
        self,
        source_pts: np.ndarray,
        source_vals: np.ndarray,
    ) -> "RBFInterpolator":
        """Fit the interpolant to scattered data.

        Parameters
        ----------
        source_pts : (N, d) — source point coordinates
        source_vals : (N,) or (N, F) — field values at source points

        Returns self for chaining.
        """
        X = np.asarray(source_pts, dtype=np.float64)
        v = np.asarray(source_vals, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]

        N = X.shape[0]
        R = _pairwise_dist(X, X)                          # (N, N)
        A = _KERNELS[self.kernel](R, self.eps)             # (N, N)
        A += self.reg * np.eye(N)

        self._coeffs = np.linalg.solve(A, v)              # (N, F)
        self._source_pts = X
        self._n_outputs = v.shape[1]
        return self

    def __call__(self, query_pts: np.ndarray) -> np.ndarray:
        """Evaluate the interpolant at query_pts.

        Parameters
        ----------
        query_pts : (Q, d)

        Returns
        -------
        (Q,) for scalar fields, (Q, F) for vector fields.
        """
        self._check_fitted()
        Q = np.asarray(query_pts, dtype=np.float64)
        R = _pairwise_dist(Q, self._source_pts)           # (Q, N)
        Phi = _KERNELS[self.kernel](R, self.eps)           # (Q, N)
        out = Phi @ self._coeffs                           # (Q, F)
        return out[:, 0] if self._n_outputs == 1 else out

    def gradient(self, query_pts: np.ndarray) -> np.ndarray:
        """Gradient of the interpolant at query_pts.

        Only valid for scalar fields (source_vals was 1-D or (N, 1)).

        Returns
        -------
        (Q, d) — gradient vectors at each query point.
        """
        self._check_fitted()
        if self._n_outputs != 1:
            raise ValueError("gradient() is only defined for scalar (1-D) fields.")

        Q = np.asarray(query_pts, dtype=np.float64)
        R = _pairwise_dist(Q, self._source_pts)                   # (Q, N)
        R_safe = np.where(R < 1e-14, 1e-14, R)

        # dφ/dr / r  — same formula for all kernels
        factor = _GRAD_FACTORS[self.kernel](R_safe, self.eps)     # (Q, N)

        # displacement: Q_i - src_j
        disp = Q[:, None, :] - self._source_pts[None, :, :]       # (Q, N, d)

        coeffs_1d = self._coeffs[:, 0]                            # (N,)
        # grad[i, d] = Σ_j factor[i,j] * disp[i,j,d] * c_j
        return np.einsum("qn,qnd,n->qd", factor, disp, coeffs_1d)

    def _check_fitted(self) -> None:
        if self._source_pts is None:
            raise RuntimeError("Call fit() before evaluating.")


# ---------------------------------------------------------------------------
# ImplicitSurfaceRBF
# ---------------------------------------------------------------------------

class ImplicitSurfaceRBF:
    """Reconstruct a continuous SDF approximation from a point cloud with normals.

    Fits an RBF interpolant to three constraint sets:

    - On-surface samples (N pts):  SDF = 0
    - Interior samples  (N pts):   pt - normal * offset  →  SDF = -offset
    - Exterior samples  (N pts):   pt + normal * offset  →  SDF = +offset

    The resulting function approximates the signed distance function (negative
    inside, positive outside) of the surface. This enables using a raw point
    cloud as a PINN geometry domain without building an explicit mesh.

    The reconstructed SDF can be passed directly to functions expecting a
    ``SDF2D = Callable[[ndarray], ndarray]`` signature (e.g. ``mesh_sdf_2d``,
    ``sdf2d_to_pointcloud``, ``SDFDomain2D``).

    Parameters
    ----------
    kernel : RBF kernel for the interpolant
    eps : shape parameter
    offset : offset magnitude for off-surface constraints (world units).
        Tip: use ~1% of the characteristic domain size.
    reg : Tikhonov regularisation for the linear system

    Examples
    --------
    >>> surf = ImplicitSurfaceRBF(eps=3.0, offset=0.02)
    >>> surf.fit(boundary_pts, normals)
    >>> u_col = surf.sample_interior(5000, bounds_min, bounds_max)
    >>> u_bc  = surf.sample_boundary(2000)
    >>> d_vals = surf.sdf(query_pts)   # (Q,) SDF values
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        eps: float = 2.0,
        offset: float = 0.01,
        reg: float = 1e-8,
    ) -> None:
        if kernel not in _KERNELS:
            raise ValueError(
                f"Unknown kernel {kernel!r}. Choose from {list(_KERNELS)}"
            )
        self.kernel = kernel
        self.eps = float(eps)
        self.offset = float(offset)
        self.reg = float(reg)
        self._interp: Optional[RBFInterpolator] = None
        self._surface_pts: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def fit(
        self,
        surface_pts: np.ndarray,
        normals: np.ndarray,
    ) -> "ImplicitSurfaceRBF":
        """Fit the implicit surface from a point cloud with outward normals.

        Parameters
        ----------
        surface_pts : (N, d) — on-surface point coordinates
        normals : (N, d) — outward unit normals (will be re-normalised)
        """
        pts = np.asarray(surface_pts, dtype=np.float64)
        nrm = np.asarray(normals, dtype=np.float64)
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-14)

        interior = pts - nrm * self.offset
        exterior = pts + nrm * self.offset

        all_pts = np.vstack([pts, interior, exterior])          # (3N, d)
        all_vals = np.concatenate([
            np.zeros(pts.shape[0]),
            np.full(pts.shape[0], -self.offset),
            np.full(pts.shape[0],  self.offset),
        ])                                                       # (3N,)

        self._interp = RBFInterpolator(
            kernel=self.kernel, eps=self.eps, reg=self.reg
        ).fit(all_pts, all_vals)
        self._surface_pts = pts
        return self

    def sdf(self, query_pts: np.ndarray) -> np.ndarray:
        """Evaluate the reconstructed SDF at query_pts.

        Returns
        -------
        (Q,) — negative inside the surface, positive outside.
        """
        self._check_fitted()
        return self._interp(np.asarray(query_pts, dtype=np.float64))

    # Make the object callable as a SDF2D function for compatibility
    # with sdf2d_to_tri_mesh, SDFDomain2D, etc.
    def __call__(self, query_pts: np.ndarray) -> np.ndarray:
        return self.sdf(query_pts)

    def is_inside(self, query_pts: np.ndarray) -> np.ndarray:
        """Boolean mask — True where SDF < 0 (inside the surface)."""
        return self.sdf(query_pts) < 0.0

    def sample_interior(
        self,
        n: int,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample n interior points via rejection using the reconstructed SDF.

        Parameters
        ----------
        n : desired number of interior points
        bounds_min, bounds_max : bounding box for candidate generation
        seed : random seed

        Returns
        -------
        (M, d) — accepted interior points (at most n; may be fewer if the
        domain is very small relative to the bounding box)
        """
        rng = np.random.default_rng(seed)
        bmin = np.asarray(bounds_min, dtype=np.float64)
        bmax = np.asarray(bounds_max, dtype=np.float64)
        d = bmin.shape[0]
        collected: list = []
        batch = max(n * 10, 1000)
        while sum(c.shape[0] for c in collected) < n:
            cands = rng.uniform(bmin, bmax, size=(batch, d))
            mask = self.is_inside(cands)
            if mask.any():
                collected.append(cands[mask])
        return np.vstack(collected)[:n]

    def sample_boundary(self, n: int, seed: int = 0) -> np.ndarray:
        """Return n samples near the surface from the fitted point cloud.

        Returns
        -------
        (n, d)
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)
        N = self._surface_pts.shape[0]
        idx = rng.choice(N, size=min(n, N), replace=(n > N))
        return self._surface_pts[idx]

    def _check_fitted(self) -> None:
        if self._interp is None:
            raise RuntimeError("Call fit() before evaluating.")


__all__ = ["RBFInterpolator", "ImplicitSurfaceRBF"]
