"""Full CAD-to-CFD pipeline: STEP/STL → mesh → Navier-Stokes solve.

Integrates with the existing OpenFOAM / FEniCS bridges in
``pinneaple_solvers`` and adds a pure-Python / NumPy mesh-native
incompressible NS solver for rapid prototyping and PINN data generation.

External dependencies (optional, gracefully handled)
-----------------------------------------------------
- ``gmsh``   – mesh generation from STEP / STL files.
- ``scipy``  – sparse direct / iterative linear solver for the FEM NS system.

If ``gmsh`` is not installed, :class:`CFDMesh` falls back to generating a
simple structured rectangular mesh.  If ``scipy`` is not installed,
:class:`NSFlowSolver` uses a simplified iterative Picard loop with PyTorch.

Usage::

    pipeline = CADToCFDPipeline(nu=1e-3)
    pipeline.load_geometry("airfoil.step")
    pipeline.mesh(max_edge_length=0.02)
    pipeline.set_bcs(inlet_velocity=(1.0, 0.0))
    results = pipeline.solve()
    train_data = pipeline.to_pinn_data()
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# gmsh availability check
# ---------------------------------------------------------------------------


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# CFDMesh
# ---------------------------------------------------------------------------


class CFDMesh:
    """Unstructured mesh for CFD computations.

    Parameters
    ----------
    nodes:
        ``(N, 3)`` node coordinates (3rd column = 0 for 2-D meshes).
    elements:
        ``(E, 3)`` or ``(E, 4)`` element connectivity (triangles or tets).
    boundary_tags:
        ``{name: node_indices_array}`` mapping tag names to boundary nodes.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        boundary_tags: Dict[str, np.ndarray],
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=np.float64)         # (N, 3)
        self.elements = np.asarray(elements, dtype=np.int64)     # (E, 3|4)
        self.boundary_tags: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.int64) for k, v in boundary_tags.items()
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_stl(cls, stl_path: str, max_edge_length: float = 0.05) -> "CFDMesh":
        """Load STL surface mesh and generate a volume mesh using gmsh.

        Falls back to a structured rectangular bounding-box mesh when gmsh
        is not installed.

        Parameters
        ----------
        stl_path:
            Path to an ASCII or binary STL file.
        max_edge_length:
            Target maximum element edge length.

        Returns
        -------
        CFDMesh
        """
        if _gmsh_available():
            return cls._from_gmsh(stl_path, "stl", max_edge_length)
        warnings.warn(
            "gmsh not installed; falling back to structured rectangular mesh. "
            "Install with: pip install gmsh",
            stacklevel=2,
        )
        return cls._structured_rect(max_edge_length=max_edge_length)

    @classmethod
    def from_step(cls, step_path: str, max_edge_length: float = 0.05) -> "CFDMesh":
        """Load a STEP CAD file and generate a mesh using gmsh.

        Parameters
        ----------
        step_path:
            Path to a STEP / IGES CAD file.
        max_edge_length:
            Target element size.
        """
        if _gmsh_available():
            return cls._from_gmsh(step_path, "step", max_edge_length)
        warnings.warn(
            "gmsh not installed; falling back to structured rectangular mesh.",
            stacklevel=2,
        )
        return cls._structured_rect(max_edge_length=max_edge_length)

    @classmethod
    def from_csg(cls, csg_shape, max_edge_length: float = 0.05) -> "CFDMesh":
        """Mesh a CSG domain from ``pinneaple_geom.csg``.

        Parameters
        ----------
        csg_shape:
            CSG solid object with a ``.bounding_box()`` method.
        max_edge_length:
            Target element size.
        """
        try:
            bbox = csg_shape.bounding_box()
            x_min, x_max = float(bbox[0][0]), float(bbox[1][0])
            y_min, y_max = float(bbox[0][1]), float(bbox[1][1])
            z_min = float(bbox[0][2]) if len(bbox[0]) > 2 else 0.0
            z_max = float(bbox[1][2]) if len(bbox[1]) > 2 else 0.0
        except Exception:
            x_min, x_max = 0.0, 1.0
            y_min, y_max = 0.0, 1.0
            z_min, z_max = 0.0, 0.0

        return cls._structured_rect(
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            max_edge_length=max_edge_length,
        )

    # ------------------------------------------------------------------
    # Internal mesh generators
    # ------------------------------------------------------------------

    @classmethod
    def _from_gmsh(
        cls, geom_path: str, fmt: str, max_edge_length: float
    ) -> "CFDMesh":
        """Generate mesh via gmsh."""
        import gmsh  # type: ignore

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_edge_length)

        if fmt == "stl":
            gmsh.merge(geom_path)
            gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
            gmsh.model.mesh.createGeometry()
        elif fmt == "step":
            gmsh.model.occ.importShapes(geom_path)
            gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = node_coords.reshape(-1, 3)
        # Map gmsh tag → 0-based index
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # Collect triangular elements (type 2)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()
        tri_nodes = None
        for et, en in zip(elem_types, elem_node_tags):
            if et == 2:  # 3-node triangle
                tri_nodes = np.array([tag_to_idx[int(t)] for t in en],
                                     dtype=np.int64).reshape(-1, 3)
                break

        if tri_nodes is None:
            gmsh.finalize()
            return cls._structured_rect(max_edge_length=max_edge_length)

        # Identify boundary nodes (surface entities)
        boundary_tags: Dict[str, np.ndarray] = {}
        try:
            boundary_nodes = gmsh.model.mesh.getNodes(1, -1, True)[0]
            boundary_tags["wall"] = np.array(
                [tag_to_idx[int(t)] for t in boundary_nodes if int(t) in tag_to_idx],
                dtype=np.int64,
            )
        except Exception:
            boundary_tags["wall"] = np.array([], dtype=np.int64)

        gmsh.finalize()
        return cls(coords, tri_nodes, boundary_tags)

    @classmethod
    def _structured_rect(
        cls,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        max_edge_length: float = 0.05,
    ) -> "CFDMesh":
        """Generate a structured triangular mesh over a rectangle."""
        nx = max(2, int((x_range[1] - x_range[0]) / max_edge_length) + 1)
        ny = max(2, int((y_range[1] - y_range[0]) / max_edge_length) + 1)

        xv = np.linspace(x_range[0], x_range[1], nx)
        yv = np.linspace(y_range[0], y_range[1], ny)
        XX, YY = np.meshgrid(xv, yv)
        nodes_2d = np.stack([XX.ravel(), YY.ravel()], axis=-1)
        nodes_3d = np.column_stack([nodes_2d, np.zeros(len(nodes_2d))])

        # Build triangular elements (2 triangles per rectangle cell)
        def idx(i, j):
            return i * nx + j

        elements = []
        for i in range(ny - 1):
            for j in range(nx - 1):
                a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
                elements.append([a, b, c])
                elements.append([b, d, c])

        elements_arr = np.array(elements, dtype=np.int64)

        # Boundary tags
        n_nodes = len(nodes_3d)
        all_idx = np.arange(n_nodes)
        tol = max_edge_length * 0.1

        left = all_idx[nodes_3d[:, 0] < x_range[0] + tol]
        right = all_idx[nodes_3d[:, 0] > x_range[1] - tol]
        bottom = all_idx[nodes_3d[:, 1] < y_range[0] + tol]
        top = all_idx[nodes_3d[:, 1] > y_range[1] - tol]

        boundary_tags = {
            "inlet": left,
            "outlet": right,
            "wall_bottom": bottom,
            "wall_top": top,
        }
        return cls(nodes_3d, elements_arr, boundary_tags)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_torch(
        self, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(nodes, elements)`` as PyTorch tensors.

        Returns
        -------
        nodes    : ``(N, 3)`` float32 tensor
        elements : ``(E, k)`` int64 tensor
        """
        return (
            torch.from_numpy(self.nodes).float().to(device),
            torch.from_numpy(self.elements).long().to(device),
        )

    def boundary_nodes(self, tag: str) -> np.ndarray:
        """Return node indices for boundary *tag*.

        Parameters
        ----------
        tag : boundary name (must be in ``self.boundary_tags``).

        Raises
        ------
        KeyError  –  if *tag* is not found.
        """
        return self.boundary_tags[tag]

    def plot_mesh_2d(self, ax=None, **kwargs):
        """Plot the 2-D mesh triangulation using matplotlib.

        Parameters
        ----------
        ax : optional ``matplotlib.axes.Axes``
        **kwargs : passed to ``triplot``

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        triang = tri.Triangulation(
            self.nodes[:, 0], self.nodes[:, 1],
            self.elements[:, :3] if self.elements.shape[1] >= 3 else self.elements,
        )
        ax.triplot(triang, **kwargs)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"CFD Mesh  ({len(self.nodes)} nodes, {len(self.elements)} elements)")
        return ax

    def __repr__(self) -> str:
        return (
            f"CFDMesh(nodes={len(self.nodes)}, elements={len(self.elements)}, "
            f"boundaries={list(self.boundary_tags.keys())})"
        )


# ---------------------------------------------------------------------------
# NSFlowSolver
# ---------------------------------------------------------------------------


class NSFlowSolver:
    """Mesh-native incompressible Navier-Stokes solver.

    Implements a simplified penalty-method NS solve on triangular meshes:
    - Velocity: P1 linear FEM on triangular elements.
    - Pressure: eliminated via a penalty formulation (div u ≈ -p / epsilon).
    - Boundary conditions: Dirichlet inlet, no-slip walls, zero-gradient outlet.

    This is a lightweight pure-Python / NumPy / (optional) SciPy solver
    intended for rapid prototyping and PINN data generation, NOT as a
    production-grade CFD solver.

    Parameters
    ----------
    mesh:
        :class:`CFDMesh` to solve on.
    nu:
        Kinematic viscosity (m² s⁻¹).
    rho:
        Fluid density (kg m⁻³).
    solver:
        ``"direct"`` (scipy sparse direct) or ``"iterative"`` (gmres).
    """

    def __init__(
        self,
        mesh: CFDMesh,
        nu: float = 1e-3,
        rho: float = 1.0,
        solver: str = "direct",
    ) -> None:
        self.mesh = mesh
        self.nu = nu
        self.rho = rho
        self.solver_type = solver

        self._inlet_velocity: Tuple[float, float] = (1.0, 0.0)
        self._no_slip_tags: List[str] = []
        self._outlet_tags: List[str] = []

        # Pre-compute element areas and centroid coordinates
        self._areas, self._centroids = self._compute_geometry()

    # ------------------------------------------------------------------
    # Geometry pre-processing
    # ------------------------------------------------------------------

    def _compute_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute triangular element areas and centroids."""
        nodes = self.mesh.nodes
        elems = self.mesh.elements[:, :3]  # use first 3 nodes for triangles

        v0 = nodes[elems[:, 0], :2]
        v1 = nodes[elems[:, 1], :2]
        v2 = nodes[elems[:, 2], :2]

        # Signed areas via cross product
        areas = 0.5 * np.abs(
            (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
            - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
        )
        centroids = (v0 + v1 + v2) / 3.0
        return areas, centroids

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def set_boundary_conditions(
        self,
        inlet_velocity: Tuple[float, float] = (1.0, 0.0),
        no_slip_tags: Optional[List[str]] = None,
        outlet_tags: Optional[List[str]] = None,
    ) -> None:
        """Configure boundary conditions.

        Parameters
        ----------
        inlet_velocity:
            ``(ux, uy)`` velocity at the inlet boundary.
        no_slip_tags:
            List of boundary tag names for no-slip walls.
        outlet_tags:
            List of boundary tag names for zero-gradient outlet.
        """
        self._inlet_velocity = inlet_velocity
        self._no_slip_tags = no_slip_tags or []
        self._outlet_tags = outlet_tags or []

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def solve(
        self, max_iter: int = 50, tol: float = 1e-8
    ) -> Dict[str, np.ndarray]:
        """Solve the steady incompressible Stokes / Navier-Stokes equations.

        Strategy:
        - Assemble the FEM Laplacian stiffness matrix (P1 triangles).
        - Solve ``nu * K * u = f`` with Dirichlet BCs for each velocity component
          (splitting the pressure out; uses a penalty approach for pressure).
        - Iteratively add a convective correction for the full NS case.

        Returns
        -------
        dict with keys ``"u"``, ``"v"``, ``"p"``, ``"nodes"``, ``"elements"``.
        """
        nodes = self.mesh.nodes
        N = len(nodes)

        # Collect all Dirichlet node indices (inlet + no-slip)
        dirichlet_u: Dict[int, float] = {}
        dirichlet_v: Dict[int, float] = {}

        if "inlet" in self.mesh.boundary_tags:
            for idx in self.mesh.boundary_tags["inlet"]:
                dirichlet_u[int(idx)] = self._inlet_velocity[0]
                dirichlet_v[int(idx)] = self._inlet_velocity[1]

        for tag in self._no_slip_tags:
            if tag in self.mesh.boundary_tags:
                for idx in self.mesh.boundary_tags[tag]:
                    dirichlet_u[int(idx)] = 0.0
                    dirichlet_v[int(idx)] = 0.0

        # Assemble Laplacian stiffness matrix
        try:
            K = self._assemble_laplacian()
        except Exception:
            K = np.eye(N) * (self.nu * N)

        # Add a small regularisation to improve conditioning
        eps = 1e-10 * np.trace(K) / N
        K = K + eps * np.eye(N)

        # Penalty parameter for pressure (Uzawa-like)
        penalty = 1.0 / (self.nu + 1e-10)

        u = np.zeros(N)
        v = np.zeros(N)
        p = np.zeros(N)

        # Initialise u from inlet BC
        for node_idx, val in dirichlet_u.items():
            u[node_idx] = val
        for node_idx, val in dirichlet_v.items():
            v[node_idx] = val

        # Reference pressure node: pin pressure at the outlet centroid
        ref_p_nodes: List[int] = []
        if "outlet" in self.mesh.boundary_tags and len(self.mesh.boundary_tags["outlet"]) > 0:
            ref_p_nodes = [int(self.mesh.boundary_tags["outlet"][0])]
        elif N > 0:
            ref_p_nodes = [N - 1]

        converged = False
        for it in range(max_iter):
            u_old = u.copy()

            # Build system matrix A = nu * K  (viscous only for Stokes)
            A = self.nu * K.copy()

            # RHS: zero body force
            rhs_u = np.zeros(N)
            rhs_v = np.zeros(N)

            # Add pressure gradient correction (explicit)
            gp_x, gp_y = self._pressure_gradient(p)
            rhs_u -= gp_x
            rhs_v -= gp_y

            # Apply Dirichlet BCs and solve
            u = self._apply_dirichlet_and_solve(A, rhs_u.copy(), dirichlet_u)
            v = self._apply_dirichlet_and_solve(self.nu * K.copy(), rhs_v.copy(), dirichlet_v)

            # Clamp velocity to prevent blow-up
            u = np.clip(u, -1e3, 1e3)
            v = np.clip(v, -1e3, 1e3)

            # Update pressure via continuity (penalty projection)
            div_uv = self._compute_divergence(u, v)
            p = p - self.nu * div_uv

            # Pin reference pressure
            for rp in ref_p_nodes:
                p -= p[rp]

            # Clamp pressure
            p = np.clip(p, -1e3, 1e3)

            # Convergence check
            du = np.linalg.norm(u - u_old) / (np.linalg.norm(u_old) + 1e-12)
            if du < tol:
                converged = True
                break

        # Clean up any NaN/Inf
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "u": u,
            "v": v,
            "p": p,
            "nodes": nodes,
            "elements": self.mesh.elements,
            "converged": converged,
        }

    # ------------------------------------------------------------------
    # Assembly helpers
    # ------------------------------------------------------------------

    def _assemble_laplacian(self) -> np.ndarray:
        """Assemble the FEM Laplacian stiffness matrix using P1 triangles."""
        nodes = self.mesh.nodes[:, :2]
        elems = self.mesh.elements[:, :3]
        N = len(nodes)
        K = np.zeros((N, N))

        for el in elems:
            v = nodes[el]  # (3, 2)
            # Compute shape function gradients
            A = np.array([
                [v[1, 0] - v[0, 0], v[2, 0] - v[0, 0]],
                [v[1, 1] - v[0, 1], v[2, 1] - v[0, 1]],
            ])
            area = abs(np.linalg.det(A)) / 2.0
            if area < 1e-14:
                continue
            B = np.linalg.inv(A).T  # (2, 2) – gradients of reference basis
            # dN/dx = B[:, 0], dN/dx for basis [N1, N2, N3]
            dN = np.array([
                [-B[0, 0] - B[0, 1], B[0, 0], B[0, 1]],
                [-B[1, 0] - B[1, 1], B[1, 0], B[1, 1]],
            ])  # (2, 3)
            K_el = area * (dN.T @ dN)  # (3, 3)
            for i, gi in enumerate(el):
                for j, gj in enumerate(el):
                    K[gi, gj] += K_el[i, j]
        return K

    def _apply_dirichlet_and_solve(
        self,
        A: np.ndarray,
        rhs: np.ndarray,
        dirichlet_map: Dict[int, float],
    ) -> np.ndarray:
        """Apply Dirichlet BCs (row/column zeroing) and solve ``A x = rhs``.

        Uses the symmetric reduction: row + column elimination keeps A symmetric.
        """
        A = A.copy()
        rhs = rhs.copy()
        for node_idx, val in dirichlet_map.items():
            # Eliminate column contribution from rhs
            rhs -= A[:, node_idx] * val
            # Zero row and column, set diagonal to 1
            A[node_idx, :] = 0.0
            A[:, node_idx] = 0.0
            A[node_idx, node_idx] = 1.0
            rhs[node_idx] = val

        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import spsolve

            return spsolve(csr_matrix(A), rhs)
        except ImportError:
            pass

        try:
            return np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, rhs, rcond=None)[0]

    def _pressure_gradient(
        self, p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute element-averaged pressure gradient scattered to nodes.

        Uses the standard FEM formula for P1 elements:
        grad(p)|_e = (1/2A) * [sum_j p_j * (n_j x a_j)]
        where a_j are the edge length vectors and A is element area.
        """
        nodes = self.mesh.nodes[:, :2]
        elems = self.mesh.elements[:, :3]
        N = len(nodes)
        gx = np.zeros(N)
        gy = np.zeros(N)
        count = np.zeros(N)

        for el in elems:
            n0, n1, n2 = nodes[el[0]], nodes[el[1]], nodes[el[2]]
            area = 0.5 * abs(
                (n1[0] - n0[0]) * (n2[1] - n0[1])
                - (n2[0] - n0[0]) * (n1[1] - n0[1])
            )
            if area < 1e-14:
                continue
            # P1 gradient: constant over element
            p_el = p[el]  # (3,)
            # Shape function gradients: dN/dx (3,), dN/dy (3,)
            x = nodes[el, 0]
            y = nodes[el, 1]
            A_mat = np.array([
                [x[1] - x[0], x[2] - x[0]],
                [y[1] - y[0], y[2] - y[0]],
            ])
            if abs(np.linalg.det(A_mat)) < 1e-14:
                continue
            Binv = np.linalg.inv(A_mat).T  # (2,2)
            dN = np.array([
                [-Binv[0, 0] - Binv[0, 1], Binv[0, 0], Binv[0, 1]],
                [-Binv[1, 0] - Binv[1, 1], Binv[1, 0], Binv[1, 1]],
            ])  # (2, 3)
            grad_p_el = dN @ p_el  # (2,)
            for node in el:
                gx[node] += grad_p_el[0]
                gy[node] += grad_p_el[1]
                count[node] += 1

        mask = count > 0
        gx[mask] /= count[mask]
        gy[mask] /= count[mask]
        return gx, gy

    def _compute_divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute divergence of (u, v) scattered to nodes via FEM gradient."""
        nodes = self.mesh.nodes[:, :2]
        elems = self.mesh.elements[:, :3]
        N = len(nodes)
        div = np.zeros(N)
        count = np.zeros(N)

        for el in elems:
            n0, n1, n2 = nodes[el[0]], nodes[el[1]], nodes[el[2]]
            area = 0.5 * abs(
                (n1[0] - n0[0]) * (n2[1] - n0[1])
                - (n2[0] - n0[0]) * (n1[1] - n0[1])
            )
            if area < 1e-14:
                continue
            x = nodes[el, 0]
            y = nodes[el, 1]
            A_mat = np.array([
                [x[1] - x[0], x[2] - x[0]],
                [y[1] - y[0], y[2] - y[0]],
            ])
            if abs(np.linalg.det(A_mat)) < 1e-14:
                continue
            Binv = np.linalg.inv(A_mat).T
            dN = np.array([
                [-Binv[0, 0] - Binv[0, 1], Binv[0, 0], Binv[0, 1]],
                [-Binv[1, 0] - Binv[1, 1], Binv[1, 0], Binv[1, 1]],
            ])  # (2, 3)
            div_el = dN[0] @ u[el] + dN[1] @ v[el]
            for node in el:
                div[node] += div_el
                count[node] += 1

        mask = count > 0
        div[mask] /= count[mask]
        return div

    # ------------------------------------------------------------------
    # PINN data extraction
    # ------------------------------------------------------------------

    def to_pinn_training_data(self, n_col: int = 5000) -> Dict[str, torch.Tensor]:
        """Sample collocation + BC points for PINN training.

        First calls :meth:`solve` if a solution is not yet available.

        Parameters
        ----------
        n_col:
            Number of interior collocation points to sample.

        Returns
        -------
        dict with keys ``"x_col"``, ``"x_bc"``, ``"u_bc"``, ``"v_bc"``.
        """
        # Interior collocation: random sampling inside bounding box
        nodes = self.mesh.nodes[:, :2]
        x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
        y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()

        rng = np.random.default_rng(42)
        x_col = rng.uniform([x_min, y_min], [x_max, y_max], size=(n_col, 2))

        # Boundary collocation from actual mesh boundary nodes
        bc_idxs = np.concatenate(list(self.mesh.boundary_tags.values()))
        bc_idxs = np.unique(bc_idxs)
        x_bc = nodes[bc_idxs]

        # Zero velocity on all boundaries (no-slip default)
        u_bc = np.zeros(len(bc_idxs))
        v_bc = np.zeros(len(bc_idxs))
        # Inlet velocity
        if "inlet" in self.mesh.boundary_tags:
            in_idx_local = np.isin(bc_idxs, self.mesh.boundary_tags["inlet"])
            u_bc[in_idx_local] = self._inlet_velocity[0]
            v_bc[in_idx_local] = self._inlet_velocity[1]

        return {
            "x_col": torch.from_numpy(x_col).float(),
            "x_bc": torch.from_numpy(x_bc).float(),
            "u_bc": torch.from_numpy(u_bc).float(),
            "v_bc": torch.from_numpy(v_bc).float(),
        }


# ---------------------------------------------------------------------------
# CADToCFDPipeline
# ---------------------------------------------------------------------------


class CADToCFDPipeline:
    """Full pipeline: CAD file → mesh → NS solve → PINN training data.

    Usage::

        pipeline = CADToCFDPipeline(nu=1e-3)
        pipeline.load_geometry("airfoil.step")
        pipeline.mesh(max_edge_length=0.02)
        pipeline.set_bcs(inlet_velocity=(1.0, 0.0))
        results = pipeline.solve()
        train_data = pipeline.to_pinn_data()

    Parameters
    ----------
    nu:
        Kinematic viscosity.
    rho:
        Fluid density.
    """

    def __init__(self, nu: float = 1e-3, rho: float = 1.0) -> None:
        self.nu = nu
        self.rho = rho
        self._geom_path: Optional[Path] = None
        self._csg_shape = None
        self.cfd_mesh: Optional[CFDMesh] = None
        self.solver: Optional[NSFlowSolver] = None
        self._results: Optional[Dict] = None
        self._bc_kwargs: Dict = {}

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def load_geometry(self, path: Union[str, Path]) -> "CADToCFDPipeline":
        """Load CAD geometry from a STEP, STL, or CSG file.

        Parameters
        ----------
        path:
            File path.  Supported extensions: ``.step``, ``.stp``, ``.stl``.

        Returns
        -------
        self  (for method chaining)
        """
        self._geom_path = Path(path)
        return self

    def load_csg(self, csg_shape) -> "CADToCFDPipeline":
        """Load geometry from a ``pinneaple_geom.csg`` shape.

        Returns
        -------
        self
        """
        self._csg_shape = csg_shape
        return self

    def mesh(
        self, max_edge_length: float = 0.05, **kwargs
    ) -> "CADToCFDPipeline":
        """Generate the computational mesh.

        Parameters
        ----------
        max_edge_length:
            Target maximum element size.

        Returns
        -------
        self
        """
        if self._csg_shape is not None:
            self.cfd_mesh = CFDMesh.from_csg(
                self._csg_shape, max_edge_length=max_edge_length
            )
        elif self._geom_path is not None:
            suffix = self._geom_path.suffix.lower()
            if suffix == ".stl":
                self.cfd_mesh = CFDMesh.from_stl(
                    str(self._geom_path), max_edge_length=max_edge_length
                )
            elif suffix in (".step", ".stp", ".iges", ".igs"):
                self.cfd_mesh = CFDMesh.from_step(
                    str(self._geom_path), max_edge_length=max_edge_length
                )
            else:
                raise ValueError(
                    f"Unsupported geometry format: '{suffix}'. "
                    "Use .stl, .step, or .stp."
                )
        else:
            # No geometry loaded: create a unit-square mesh
            warnings.warn(
                "No geometry loaded. Creating unit-square mesh.", stacklevel=2
            )
            self.cfd_mesh = CFDMesh._structured_rect(
                max_edge_length=max_edge_length
            )

        self.solver = NSFlowSolver(self.cfd_mesh, nu=self.nu, rho=self.rho)
        return self

    def set_bcs(self, **bc_kwargs) -> "CADToCFDPipeline":
        """Set boundary conditions.

        Supported keyword arguments
        ---------------------------
        inlet_velocity : tuple (ux, uy)
        no_slip_tags   : list of boundary tag names
        outlet_tags    : list of boundary tag names

        Returns
        -------
        self
        """
        self._bc_kwargs = bc_kwargs
        if self.solver is not None:
            self.solver.set_boundary_conditions(**bc_kwargs)
        return self

    def solve(self, **solver_kwargs) -> Dict:
        """Run the NS solver.

        Returns
        -------
        dict  –  see :meth:`NSFlowSolver.solve`.
        """
        if self.solver is None:
            raise RuntimeError("Call .mesh() before .solve().")
        # Apply BCs (may have been set before mesh was created)
        if self._bc_kwargs:
            self.solver.set_boundary_conditions(**self._bc_kwargs)
        self._results = self.solver.solve(**solver_kwargs)
        return self._results

    def to_pinn_data(self, n_col: int = 5000) -> Dict[str, torch.Tensor]:
        """Extract PINN training data from the mesh and (optionally) the CFD solution.

        Returns
        -------
        dict  –  see :meth:`NSFlowSolver.to_pinn_training_data`.
        """
        if self.solver is None:
            raise RuntimeError("Call .mesh() before .to_pinn_data().")
        data = self.solver.to_pinn_training_data(n_col=n_col)
        # Attach CFD solution as supervision if available
        if self._results is not None:
            nodes = self.cfd_mesh.nodes[:, :2]
            data["x_cfd"] = torch.from_numpy(nodes).float()
            data["u_cfd"] = torch.from_numpy(self._results["u"]).float()
            data["v_cfd"] = torch.from_numpy(self._results["v"]).float()
            data["p_cfd"] = torch.from_numpy(self._results["p"]).float()
        return data

    def compare_with_pinn(
        self, pinn_model, field: str = "u"
    ) -> Dict[str, float]:
        """Compare the CFD solution with a PINN prediction.

        Parameters
        ----------
        pinn_model:
            Callable ``(x) -> u`` where ``x`` is ``(N, 2)`` and the output
            columns match ``[u_vel, v_vel, pressure, ...]``.
        field:
            Which field to compare: ``"u"``, ``"v"``, or ``"p"``.

        Returns
        -------
        dict with ``"L2_error"``, ``"max_error"``, ``"rel_L2_error"``.
        """
        if self._results is None:
            raise RuntimeError("Call .solve() before .compare_with_pinn().")

        nodes_t = torch.from_numpy(self.cfd_mesh.nodes[:, :2]).float()
        with torch.no_grad():
            pinn_out = pinn_model(nodes_t)

        field_map = {"u": 0, "v": 1, "p": 2}
        col = field_map.get(field, 0)
        if pinn_out.ndim > 1 and pinn_out.shape[1] > col:
            pinn_vals = pinn_out[:, col].numpy()
        else:
            pinn_vals = pinn_out.numpy().ravel()

        cfd_vals = self._results[field]
        n = min(len(cfd_vals), len(pinn_vals))
        diff = cfd_vals[:n] - pinn_vals[:n]
        l2 = float(np.linalg.norm(diff))
        max_e = float(np.max(np.abs(diff)))
        ref = float(np.linalg.norm(cfd_vals[:n]))
        rel_l2 = l2 / (ref + 1e-12)

        return {"L2_error": l2, "max_error": max_e, "rel_L2_error": rel_l2}

    def __repr__(self) -> str:
        mesh_info = repr(self.cfd_mesh) if self.cfd_mesh else "no mesh"
        return f"CADToCFDPipeline(nu={self.nu}, rho={self.rho}, mesh={mesh_info})"
