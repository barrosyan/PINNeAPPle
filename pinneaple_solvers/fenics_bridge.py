"""FEniCS/dolfinx bridge for pinneaple.

Solves structural mechanics and heat transfer problems using FEniCS
(dolfinx preferred, legacy FEniCS as fallback).

Key workflow
------------
1. Build mesh from problem_spec domain_bounds (rectangle or from gmsh)
2. Define function space and variational form from PDE kind
3. Apply boundary conditions from problem_spec.conditions
4. Solve and extract field arrays
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FEniCS availability probe
# ---------------------------------------------------------------------------

_DOLFINX_AVAILABLE = False
_LEGACY_FENICS_AVAILABLE = False

try:
    import dolfinx  # type: ignore[import]
    _DOLFINX_AVAILABLE = True
except ImportError:
    pass

if not _DOLFINX_AVAILABLE:
    try:
        import fenics  # type: ignore[import]  # noqa: F401
        _LEGACY_FENICS_AVAILABLE = True
    except ImportError:
        pass

_FENICS_AVAILABLE = _DOLFINX_AVAILABLE or _LEGACY_FENICS_AVAILABLE

# ---------------------------------------------------------------------------
# Supported PDE kinds
# ---------------------------------------------------------------------------

_SUPPORTED_KINDS = {
    "heat_equation_steady",
    "linear_elasticity_plane_stress",
    "linear_elasticity_plane_strain",
}

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@SolverRegistry.register(
    name="fenics",
    family="pde",
    description="FEniCS/dolfinx structural and thermal solver",
    tags=["fenics", "dolfinx", "fem", "structural"],
)
class FEnicsBridge(SolverBase):
    """Bridge that formulates and solves FEM problems using FEniCS/dolfinx.

    Parameters
    ----------
    mesh_nx:
        Number of cells in the x-direction for the rectangular mesh.
    mesh_ny:
        Number of cells in the y-direction.
    element_degree:
        Polynomial degree of the Lagrange elements (default 1 = linear).
    solver_backend:
        ``"dolfinx"`` (default) or ``"legacy"`` to select the FEniCS API.
    """

    def __init__(
        self,
        mesh_nx: int = 32,
        mesh_ny: int = 32,
        element_degree: int = 1,
        solver_backend: str = "dolfinx",
    ) -> None:
        super().__init__()
        self.mesh_nx = int(mesh_nx)
        self.mesh_ny = int(mesh_ny)
        self.element_degree = int(element_degree)
        self.solver_backend = solver_backend

    # ------------------------------------------------------------------
    # SolverBase interface
    # ------------------------------------------------------------------

    def forward(self, problem_spec: Any, **kwargs: Any) -> SolverOutput:  # type: ignore[override]
        """Solve the PDE described by *problem_spec* and return a SolverOutput.

        Parameters
        ----------
        problem_spec:
            Must expose:

            * ``kind`` (str) – one of the supported PDE identifiers.
            * ``domain_bounds`` (tuple/list of 4 floats) – ``(x0, y0, x1, y1)``.
            * ``conditions`` (dict) – boundary condition objects.
            * ``parameters`` (dict, optional) – PDE parameters (``k``, ``E``, ``nu``, ``f``).

        Returns
        -------
        SolverOutput
            ``result`` holds a 1-D float32 tensor of the flattened DOF vector.
            ``extras`` contains ``dofs``, ``kind``, and any solver metadata.
        """
        if not _FENICS_AVAILABLE:
            warnings.warn(
                "Neither dolfinx nor legacy FEniCS is installed. "
                "Returning empty SolverOutput. "
                "Install FEniCS (https://fenicsproject.org/) or dolfinx "
                "(https://github.com/FEniCS/dolfinx) to use this bridge.",
                RuntimeWarning,
                stacklevel=2,
            )
            return SolverOutput(
                result=torch.empty(0),
                losses={},
                extras={"fenics_available": False},
            )

        kind = getattr(problem_spec, "kind", "heat_equation_steady")
        if kind not in _SUPPORTED_KINDS:
            raise ValueError(
                f"Unsupported PDE kind '{kind}'. Supported: {sorted(_SUPPORTED_KINDS)}"
            )

        field_names = kwargs.get("field_names", None)

        try:
            if self.solver_backend == "dolfinx" and _DOLFINX_AVAILABLE:
                solution, meta = self._solve_dolfinx(problem_spec, kind)
            else:
                solution, meta = self._solve_legacy(problem_spec, kind)

            fields = self._extract_fields(solution, field_names)
        except Exception as exc:
            log.exception("FEniCS solve failed: %s", exc)
            return SolverOutput(
                result=torch.empty(0),
                losses={},
                extras={"fenics_available": True, "error": str(exc)},
            )

        try:
            import numpy as np
            tensors = [torch.as_tensor(arr.ravel(), dtype=torch.float32) for arr in fields.values()]
            result = torch.cat(tensors) if tensors else torch.empty(0)
        except Exception:
            result = torch.empty(0)

        return SolverOutput(
            result=result,
            losses={},
            extras={
                "fenics_available": True,
                "backend": self.solver_backend,
                "kind": kind,
                "fields": fields,
                **meta,
            },
        )

    # ------------------------------------------------------------------
    # Mesh construction
    # ------------------------------------------------------------------

    def _build_mesh(self, problem_spec: Any) -> Any:
        """Create a rectangular mesh from *problem_spec.domain_bounds*.

        Parameters
        ----------
        problem_spec:
            Object with ``domain_bounds`` = ``(x0, y0, x1, y1)``.  Defaults
            to ``(0, 0, 1, 1)`` when not present.

        Returns
        -------
        Mesh object appropriate for the active backend.
        """
        bounds = getattr(problem_spec, "domain_bounds", (0.0, 0.0, 1.0, 1.0))
        x0, y0, x1, y1 = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])

        if self.solver_backend == "dolfinx" and _DOLFINX_AVAILABLE:
            from dolfinx.mesh import create_rectangle, CellType  # type: ignore[import]
            from mpi4py import MPI  # type: ignore[import]
            import numpy as np
            mesh = create_rectangle(
                MPI.COMM_WORLD,
                [[x0, y0], [x1, y1]],
                [self.mesh_nx, self.mesh_ny],
                CellType.triangle,
            )
            return mesh
        else:
            from fenics import RectangleMesh, Point  # type: ignore[import]
            mesh = RectangleMesh(
                Point(x0, y0),
                Point(x1, y1),
                self.mesh_nx,
                self.mesh_ny,
            )
            return mesh

    # ------------------------------------------------------------------
    # Variational form dispatch
    # ------------------------------------------------------------------

    def _build_variational_form(
        self,
        problem_spec: Any,
        V: Any,
        mesh: Any,
    ) -> Tuple[Any, Any]:
        """Return ``(a, L)`` bilinear and linear form for the PDE kind.

        Dispatches to one of:
        * ``_form_heat_equation``
        * ``_form_linear_elasticity``
        * ``_form_navier_stokes``
        """
        kind = getattr(problem_spec, "kind", "heat_equation_steady")
        if kind == "heat_equation_steady":
            return self._form_heat_equation(problem_spec, V)
        if kind in ("linear_elasticity_plane_stress", "linear_elasticity_plane_strain"):
            return self._form_linear_elasticity(problem_spec, V, kind)
        if kind == "navier_stokes_stokes":
            return self._form_navier_stokes(problem_spec, V)
        raise ValueError(f"No variational form defined for kind='{kind}'.")

    def _form_heat_equation(self, problem_spec: Any, V: Any) -> Tuple[Any, Any]:
        """Steady heat equation: -∇·(k ∇T) = f.

        DirichletBC sets T on marked boundaries; NeumannBC contributes to L
        via a surface integral  ∫ g v ds.
        """
        params = getattr(problem_spec, "parameters", {}) or {}
        k_val = float(params.get("k", 1.0))
        f_val = float(params.get("f", 0.0))

        if _DOLFINX_AVAILABLE and self.solver_backend == "dolfinx":
            import ufl  # type: ignore[import]
            from dolfinx.fem import Constant  # type: ignore[import]

            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            k = Constant(V.mesh, k_val)
            f = Constant(V.mesh, f_val)
            a = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
            L = f * v * ufl.dx
        else:
            from fenics import (  # type: ignore[import]
                TrialFunction, TestFunction, Constant, dot, grad, dx,
            )
            u = TrialFunction(V)
            v = TestFunction(V)
            k = Constant(k_val)
            f = Constant(f_val)
            a = k * dot(grad(u), grad(v)) * dx
            L = f * v * dx

        # Neumann contributions
        conditions: Dict[str, Any] = getattr(problem_spec, "conditions", {}) or {}
        for cond in conditions.values():
            cond_type = getattr(cond, "type", "")
            if "neumann" in str(cond_type).lower() or "heat_flux" in str(cond_type).lower():
                g_val = float(getattr(cond, "value", 0.0))
                if _DOLFINX_AVAILABLE and self.solver_backend == "dolfinx":
                    import ufl
                    from dolfinx.fem import Constant
                    g = Constant(V.mesh, g_val)
                    L = L + g * v * ufl.ds
                else:
                    from fenics import Constant, ds  # type: ignore[import]
                    g = Constant(g_val)
                    L = L + g * v * ds

        return a, L

    def _form_linear_elasticity(
        self,
        problem_spec: Any,
        V: Any,
        kind: str,
    ) -> Tuple[Any, Any]:
        """Plane stress or plane strain linear elasticity: -∇·σ = f."""
        params = getattr(problem_spec, "parameters", {}) or {}
        E_val = float(params.get("E", 1e6))
        nu_val = float(params.get("nu", 0.3))
        f_vals = params.get("f", [0.0, 0.0])
        if not isinstance(f_vals, (list, tuple)):
            f_vals = [float(f_vals), 0.0]

        plane_stress = kind == "linear_elasticity_plane_stress"

        if _DOLFINX_AVAILABLE and self.solver_backend == "dolfinx":
            import ufl
            from dolfinx.fem import Constant  # type: ignore[import]
            import numpy as np

            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            f = Constant(V.mesh, np.array(f_vals[:2], dtype=float))

            def epsilon(w):
                return ufl.sym(ufl.grad(w))

            if plane_stress:
                lam = Constant(V.mesh, E_val * nu_val / ((1 + nu_val) * (1 - nu_val)))
                mu = Constant(V.mesh, E_val / (2 * (1 + nu_val)))
            else:  # plane strain
                lam = Constant(V.mesh, E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val)))
                mu = Constant(V.mesh, E_val / (2 * (1 + nu_val)))

            def sigma(w):
                return lam * ufl.nabla_div(w) * ufl.Identity(len(w)) + 2 * mu * epsilon(w)

            a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
            L = ufl.dot(f, v) * ufl.dx
        else:
            from fenics import (  # type: ignore[import]
                TrialFunction, TestFunction, Constant, inner, dx,
                sym, grad, nabla_div, Identity, dot,
            )
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Constant((float(f_vals[0]), float(f_vals[1])))

            def epsilon(w):
                return sym(grad(w))

            if plane_stress:
                lam = Constant(E_val * nu_val / ((1 + nu_val) * (1 - nu_val)))
                mu = Constant(E_val / (2 * (1 + nu_val)))
            else:
                lam = Constant(E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val)))
                mu = Constant(E_val / (2 * (1 + nu_val)))

            def sigma(w):
                return lam * nabla_div(w) * Identity(w.geometric_dimension()) + 2 * mu * epsilon(w)

            a = inner(sigma(u), epsilon(v)) * dx
            L = dot(f, v) * dx

        return a, L

    def _form_navier_stokes(self, problem_spec: Any, V: Any) -> Tuple[Any, Any]:
        """Simplified Stokes (linearised Navier-Stokes) equations."""
        params = getattr(problem_spec, "parameters", {}) or {}
        mu_val = float(params.get("mu", 1e-3))

        if _DOLFINX_AVAILABLE and self.solver_backend == "dolfinx":
            import ufl
            from dolfinx.fem import Constant  # type: ignore[import]

            u, p = ufl.TrialFunctions(V)
            v, q = ufl.TestFunctions(V)
            mu = Constant(V.mesh, mu_val)

            a = (
                mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                - p * ufl.div(v) * ufl.dx
                + ufl.div(u) * q * ufl.dx
            )
            L = ufl.inner(ufl.Constant(V.mesh, (0.0, 0.0)), v) * ufl.dx
        else:
            from fenics import (  # type: ignore[import]
                TrialFunctions, TestFunctions, Constant, inner, grad, div, dx,
            )
            u, p = TrialFunctions(V)
            v, q = TestFunctions(V)
            mu = Constant(mu_val)
            a = (
                mu * inner(grad(u), grad(v)) * dx
                - p * div(v) * dx
                + div(u) * q * dx
            )
            L = inner(Constant((0.0, 0.0)), v) * dx

        return a, L

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _apply_bcs(
        self,
        problem_spec: Any,
        V: Any,
        mesh: Any,
    ) -> List[Any]:
        """Convert problem_spec.conditions to FEniCS DirichletBC objects.

        NeumannBC and Robin conditions are handled through the linear form L
        in the variational form builders.

        Parameters
        ----------
        problem_spec:
            Object with a ``conditions`` dict.
        V:
            FEniCS function space.
        mesh:
            The mesh object (used for boundary marking).

        Returns
        -------
        list
            List of FEniCS BC objects ready to pass to ``solve()``.
        """
        bcs: List[Any] = []
        conditions: Dict[str, Any] = getattr(problem_spec, "conditions", {}) or {}
        kind = getattr(problem_spec, "kind", "heat_equation_steady")
        is_vector = kind in ("linear_elasticity_plane_stress", "linear_elasticity_plane_strain")

        for cond_name, cond in conditions.items():
            cond_type = str(getattr(cond, "type", ""))
            if "neumann" in cond_type.lower() or "heat_flux" in cond_type.lower():
                continue  # handled in variational form

            value = getattr(cond, "value", 0.0)

            if _DOLFINX_AVAILABLE and self.solver_backend == "dolfinx":
                bcs.extend(
                    self._make_dolfinx_bc(cond, cond_name, V, mesh, value, is_vector)
                )
            else:
                bc = self._make_legacy_bc(cond, cond_name, V, mesh, value, is_vector)
                if bc is not None:
                    bcs.append(bc)

        return bcs

    def _make_dolfinx_bc(
        self,
        cond: Any,
        cond_name: str,
        V: Any,
        mesh: Any,
        value: Any,
        is_vector: bool,
    ) -> List[Any]:
        """Build dolfinx DirichletBC for a single condition."""
        try:
            from dolfinx.fem import (  # type: ignore[import]
                dirichletbc, locate_dofs_geometrical,
            )
            import numpy as np

            location_fn = getattr(cond, "location_fn", None)
            if location_fn is None:
                side = str(getattr(cond, "side", cond_name)).lower()
                bounds = getattr(
                    getattr(cond, "_problem_spec", None), "domain_bounds", (0, 0, 1, 1)
                )
                x0, y0, x1, y1 = bounds

                def _make_boundary_fn(s: str, x0: float, y0: float, x1: float, y1: float):
                    tol = 1e-12
                    if s in ("left", "inlet"):
                        return lambda x: np.isclose(x[0], x0, atol=tol)
                    if s in ("right", "outlet"):
                        return lambda x: np.isclose(x[0], x1, atol=tol)
                    if s in ("bottom",):
                        return lambda x: np.isclose(x[1], y0, atol=tol)
                    if s in ("top",):
                        return lambda x: np.isclose(x[1], y1, atol=tol)
                    # Default: entire boundary
                    return lambda x: np.ones(x.shape[1], dtype=bool)

                location_fn = _make_boundary_fn(side, x0, y0, x1, y1)

            dofs = locate_dofs_geometrical(V, location_fn)
            if is_vector:
                val = np.array(value if isinstance(value, (list, tuple)) else [float(value), 0.0])
            else:
                val = float(value) if not isinstance(value, (list, tuple)) else float(value[0])
            return [dirichletbc(val, dofs, V)]
        except Exception as exc:
            log.debug("dolfinx BC creation failed for '%s': %s", cond_name, exc)
            return []

    def _make_legacy_bc(
        self,
        cond: Any,
        cond_name: str,
        V: Any,
        mesh: Any,
        value: Any,
        is_vector: bool,
    ) -> Optional[Any]:
        """Build legacy FEniCS DirichletBC for a single condition."""
        try:
            from fenics import DirichletBC, Constant, CompiledSubDomain  # type: ignore[import]

            location_expr = getattr(cond, "location_expr", None)
            if location_expr is not None:
                boundary = CompiledSubDomain(location_expr)
            else:
                side = str(getattr(cond, "side", cond_name)).lower()
                if side in ("left", "inlet"):
                    boundary = CompiledSubDomain("near(x[0], 0) && on_boundary")
                elif side in ("right", "outlet"):
                    boundary = CompiledSubDomain("near(x[0], 1) && on_boundary")
                elif side in ("bottom",):
                    boundary = CompiledSubDomain("near(x[1], 0) && on_boundary")
                elif side in ("top",):
                    boundary = CompiledSubDomain("near(x[1], 1) && on_boundary")
                else:
                    boundary = CompiledSubDomain("on_boundary")

            if is_vector:
                val = Constant(
                    tuple(value) if isinstance(value, (list, tuple)) else (float(value), 0.0)
                )
            else:
                val = Constant(float(value) if not isinstance(value, (list, tuple)) else float(value[0]))

            return DirichletBC(V, val, boundary)
        except Exception as exc:
            log.debug("Legacy FEniCS BC creation failed for '%s': %s", cond_name, exc)
            return None

    # ------------------------------------------------------------------
    # Field extraction
    # ------------------------------------------------------------------

    def _extract_fields(
        self,
        solution: Any,
        field_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Convert FEniCS solution to a dict of numpy arrays.

        Parameters
        ----------
        solution:
            FEniCS Function (scalar or vector).
        field_names:
            Optional subset of fields to return.  If ``None`` the whole
            DOF vector is returned as a single entry under ``"u"``.

        Returns
        -------
        dict[str, np.ndarray]
        """
        try:
            import numpy as np
        except ImportError:
            return {}

        fields: Dict[str, Any] = {}
        try:
            if _DOLFINX_AVAILABLE and self.solver_backend == "dolfinx":
                dofs = solution.x.array.copy()
            else:
                dofs = solution.vector().get_local()

            key = field_names[0] if (field_names and len(field_names) == 1) else "u"
            fields[key] = dofs

            # If vector-valued, also split components
            if dofs.ndim == 1:
                try:
                    gdim = solution.geometric_dimension() if hasattr(solution, "geometric_dimension") else 2
                    n_nodes = len(dofs) // gdim
                    if n_nodes * gdim == len(dofs) and gdim > 1:
                        arr2d = dofs.reshape(n_nodes, gdim)
                        comp_names = ["ux", "uy", "uz"]
                        for i in range(gdim):
                            fields[comp_names[i]] = arr2d[:, i]
                except Exception:
                    pass
        except Exception as exc:
            log.debug("Field extraction failed: %s", exc)

        return fields

    # ------------------------------------------------------------------
    # Backend-specific solve paths
    # ------------------------------------------------------------------

    def _solve_dolfinx(
        self, problem_spec: Any, kind: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Build and solve the problem using the dolfinx API."""
        from dolfinx.fem import (  # type: ignore[import]
            FunctionSpace, Function, form,
        )
        from dolfinx.fem.petsc import (  # type: ignore[import]
            LinearProblem,
        )

        mesh = self._build_mesh(problem_spec)

        if kind in ("linear_elasticity_plane_stress", "linear_elasticity_plane_strain"):
            from basix.ufl import element as ufl_element  # type: ignore[import]
            el = ufl_element("Lagrange", mesh.topology.cell_name(), self.element_degree, shape=(mesh.geometry.dim,))
            V = FunctionSpace(mesh, el)
        else:
            from basix.ufl import element as ufl_element  # type: ignore[import]
            el = ufl_element("Lagrange", mesh.topology.cell_name(), self.element_degree)
            V = FunctionSpace(mesh, el)

        a_form, L_form = self._build_variational_form(problem_spec, V, mesh)
        bcs = self._apply_bcs(problem_spec, V, mesh)

        problem = LinearProblem(a_form, L_form, bcs=bcs)
        uh = problem.solve()
        meta: Dict[str, Any] = {"dofs": uh.x.array.shape, "mesh_cells": mesh.topology.index_map(mesh.topology.dim).size_local}
        return uh, meta

    def _solve_legacy(
        self, problem_spec: Any, kind: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Build and solve the problem using the legacy FEniCS API."""
        from fenics import (  # type: ignore[import]
            FunctionSpace, VectorFunctionSpace, Function, solve,
        )

        mesh = self._build_mesh(problem_spec)

        if kind in ("linear_elasticity_plane_stress", "linear_elasticity_plane_strain"):
            V = VectorFunctionSpace(mesh, "Lagrange", self.element_degree)
        else:
            V = FunctionSpace(mesh, "Lagrange", self.element_degree)

        a_form, L_form = self._build_variational_form(problem_spec, V, mesh)
        bcs = self._apply_bcs(problem_spec, V, mesh)

        u = Function(V)
        solve(a_form == L_form, u, bcs)
        meta: Dict[str, Any] = {"dofs": u.vector().size()}
        return u, meta
