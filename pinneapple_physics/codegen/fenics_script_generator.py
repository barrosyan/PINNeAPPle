"""Standalone UFL/dolfinx *script* generator for PINNeAPPle.

Like ``fdm_script_generator.py``, this renders complete, human-readable
Python source files — not in-memory closures — for a curated set of finite
element physics problems:

  * "diffusion"  -- steady linear diffusion / Poisson-like scalar field:
                    -div(k * grad(u)) = f
  * "stokes"     -- steady Stokes flow (Taylor-Hood P2/P1 velocity-pressure)
  * "elasticity" -- linear elasticity, optionally with a penalty-method
                    rigid-obstacle contact term

Physics-shape selection
------------------------
``classify_physics_shape(spec)`` inspects a ``ProblemSpec``-like object's
``pde.kind`` string and ``fields`` tuple and picks one of the three shapes
above: a displacement-looking field set (or "elasticity" in the kind
string) -> elasticity; a velocity+pressure field pair (or "stokes"/
"navier_stokes" in the kind string) -> stokes; anything else (the default)
-> diffusion. This is a 3-way heuristic classifier, not a general
PDE-shape inference engine — anything that doesn't obviously look like one
of these three families is treated as "diffusion" by default, which is the
right conservative fallback (a scalar Poisson-like solve) rather than a
guess at something more exotic.

Relationship to pinneapple_simulation.external_solvers.fenics
----------------------------------------------------------------
``pinneapple_simulation.external_solvers.fenics.FEniCSWorkflow.solve()``
does not execute an arbitrary script at all — it takes a ``FEniCSConfig``
(``pde`` name string + ``domain``/``bcs``/``params`` dicts) and forwards it
to the hardcoded variational forms in
``pinneapple_simulation.numerical_solvers.fenics_bridge.FEnicsBridge``
(``_form_heat_equation``, ``_form_linear_elasticity``). Those hardcoded
forms cover exactly two of this generator's three shapes:

  * "diffusion"  -> ``FEnicsBridge``'s ``"heat_equation_steady"`` kind
  * "elasticity" -> ``FEnicsBridge``'s ``"linear_elasticity_plane_stress"``/
                    ``"linear_elasticity_plane_strain"`` kinds

so scripts generated for those two shapes ALSO get a
``build_fenics_config(params=None) -> FEniCSConfig`` function, letting the
exact same problem run either standalone (``python script.py``) or through
the existing bridge:

    from pinneapple_simulation.external_solvers.fenics import FEniCSWorkflow
    sample = FEniCSWorkflow(build_fenics_config()).solve({"k": 2.0})

Two honest caveats about that integration, found while wiring it up (both
pre-existing, in files this module does not touch):

1. ``solve_and_package()`` in
   ``pinneapple_simulation/external_solvers/fenics/solver.py`` does
   ``from pinneapple_simulation.numerical_solvers.fenics_bridge import
   FEniCSBridge`` — but the class actually defined in that module is named
   ``FEnicsBridge`` (lowercase "nics"). That import raises ``ImportError``
   today whenever dolfinx or legacy FEniCS is actually installed, entirely
   independent of anything in this codegen package.
2. ``FEnicsBridge._SUPPORTED_KINDS`` is
   ``{"heat_equation_steady", "linear_elasticity_plane_stress",
   "linear_elasticity_plane_strain"}`` — it does not include
   ``"navier_stokes_stokes"``, even though
   ``FEnicsBridge._build_variational_form`` has a dispatch branch for it
   (``_form_navier_stokes``); that branch is unreachable through
   ``FEnicsBridge.forward()`` today because the kind whitelist rejects it
   first, and no contact-term hook exists on any kind.

Because of (2), "stokes" scripts and contact-enabled "elasticity" scripts
get NO ``build_fenics_config()`` — there is nothing in the existing bridge
for them to plug into yet. They are standalone-only: run them with
``python script.py`` / call their own ``solve(params)``.

dolfinx availability in this environment
------------------------------------------
dolfinx/ufl are NOT importable in the environment this generator was
written and tested in (``import dolfinx`` raises ``ModuleNotFoundError``).
The generated scripts are therefore validated here only via
``ast.parse()`` (syntactically valid Python) and a string/AST check that
the expected UFL/dolfinx API surface is present (see
tests/test_codegen.py) — NOT by actually running a real finite-element
solve end to end. If dolfinx becomes available, that same test file runs
the diffusion case for real and checks it against an analytic solution;
until then, treat the FEM physics here as "written to match the exact API
already used successfully elsewhere in this repo
(pinneapple_simulation/numerical_solvers/fenics_bridge.py)", not as
independently numerically verified.
"""
from __future__ import annotations

import datetime as _dt
from string import Template
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Physics-shape classifier
# ---------------------------------------------------------------------------

_DISPLACEMENT_FIELD_NAMES = {"ux", "uy", "uz", "disp", "displacement"}


def classify_physics_shape(spec: Any) -> str:
    """Classify a ProblemSpec-like object into "diffusion" | "stokes" |
    "elasticity" — see module docstring for the exact heuristic."""
    kind = str(getattr(getattr(spec, "pde", None), "kind", "") or "").lower()
    fields = tuple(getattr(spec, "fields", ()) or ())
    fields_l = tuple(str(f).lower() for f in fields)

    if "elasticity" in kind or any(f in _DISPLACEMENT_FIELD_NAMES for f in fields_l):
        return "elasticity"
    if "stokes" in kind or "navier_stokes" in kind or ("p" in fields_l and len(fields_l) >= 2):
        return "stokes"
    return "diffusion"


# ---------------------------------------------------------------------------
# Shared header / footer
# ---------------------------------------------------------------------------

_HEADER = Template('''\
"""AUTO-GENERATED by pinneapple_physics.codegen.fenics_script_generator.FEniCSScriptGenerator
-- do not hand-edit; regenerate instead.

Physics shape : $shape
PDE spec      : kind='$kind', fields=$fields
Generated     : $timestamp

This file is a STANDALONE UFL/dolfinx script: it imports nothing from
pinneapple_physics (the package that generated it), only dolfinx/ufl/mpi4py
and NumPy, so it can be run on its own (given a working dolfinx install):

    python $module_name.py

It exposes ``solve(params: dict) -> dict``, the same convention used by
every pinneapple_physics.codegen-generated solver script and by
pinneapple_tools.sandbox.custom_solver_runner.run_custom_solver.
$config_note
NOTE: this script was generated in an environment where dolfinx is not
installed. It has been checked for valid Python syntax (ast.parse) and for
the presence of the UFL/dolfinx API calls it depends on, but has NOT been
executed end-to-end against a real dolfinx install. Review before relying
on it for anything beyond that.
"""
from __future__ import annotations

''')

_NO_CONFIG_NOTE = (
    "\nThis shape has no matching hardcoded form in "
    "pinneapple_simulation.numerical_solvers.fenics_bridge.FEnicsBridge, so "
    "there is no build_fenics_config() helper here -- run this script "
    "standalone (see pinneapple_physics.codegen.fenics_script_generator's "
    "module docstring for exactly why)."
)

_CONFIG_NOTE = (
    "\nA matching pinneapple_simulation.external_solvers.fenics.FEniCSConfig "
    "is also available via build_fenics_config(params), so this same "
    "problem can alternatively be run through:\n\n"
    "    from pinneapple_simulation.external_solvers.fenics import FEniCSWorkflow\n"
    "    sample = FEniCSWorkflow(build_fenics_config()).solve()\n"
)


# ---------------------------------------------------------------------------
# Diffusion template (-> FEnicsBridge "heat_equation_steady")
# ---------------------------------------------------------------------------

_DIFFUSION_BODY = Template('''\
def solve(params: dict) -> dict:
    """Steady linear diffusion / Poisson-like problem:

        -div(k * grad(u)) = f   in (x0,x1) x (y0,y1)
        u = g_left  on the left edge  (x = x0)
        u = g_right on the right edge (x = x1)

    Mirrors pinneapple_simulation.numerical_solvers.fenics_bridge
    .FEnicsBridge._form_heat_equation / ._solve_dolfinx exactly (same
    dolfinx API calls), so this problem can also be run through
    pinneapple_simulation.external_solvers.fenics.FEniCSWorkflow via
    build_fenics_config() below.

    params: nx, ny (mesh resolution), domain=(x0,y0,x1,y1), k, f,
    g_left, g_right.
    """
    import numpy as np
    import ufl
    from mpi4py import MPI
    from dolfinx.mesh import create_rectangle, CellType
    from dolfinx.fem import FunctionSpace, Constant, dirichletbc, locate_dofs_geometrical
    from dolfinx.fem.petsc import LinearProblem
    from basix.ufl import element as ufl_element

    nx = int(params.get("nx", $nx))
    ny = int(params.get("ny", $ny))
    x0, y0, x1, y1 = params.get("domain", $domain)
    k_val = float(params.get("k", $k))
    f_val = float(params.get("f", $f))
    g_left = float(params.get("g_left", $g_left))
    g_right = float(params.get("g_right", $g_right))

    mesh = create_rectangle(MPI.COMM_WORLD, [[x0, y0], [x1, y1]], [nx, ny], CellType.triangle)
    el = ufl_element("Lagrange", mesh.topology.cell_name(), 1)
    V = FunctionSpace(mesh, el)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = Constant(mesh, k_val)
    f = Constant(mesh, f_val)
    a = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    tol = 1e-12
    left_dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x0, atol=tol))
    right_dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x1, atol=tol))
    bcs = [dirichletbc(g_left, left_dofs, V), dirichletbc(g_right, right_dofs, V)]

    problem = LinearProblem(a, L, bcs=bcs)
    uh = problem.solve()

    return {
        "coords": mesh.geometry.x.copy(),
        "$field": uh.x.array.copy(),
        "method": "dolfinx_linear_diffusion",
        "params": {"k": k_val, "f": f_val, "g_left": g_left, "g_right": g_right},
    }


def build_fenics_config(params: dict = None):
    """Build the pinneapple_simulation.external_solvers.fenics.FEniCSConfig
    for this same diffusion problem (kind="heat_equation_steady", the one
    FEnicsBridge._form_heat_equation already implements)."""
    from pinneapple_simulation.external_solvers.fenics import FEniCSConfig

    params = dict(params or {})
    nx = int(params.get("nx", $nx))
    ny = int(params.get("ny", $ny))
    x0, y0, x1, y1 = params.get("domain", $domain)
    return FEniCSConfig(
        pde="heat_equation_steady",
        domain={"type": "rectangle", "x": [x0, x1], "y": [y0, y1], "nx": nx, "ny": ny},
        bcs=[
            {"type": "dirichlet", "boundary": "left", "value": float(params.get("g_left", $g_left))},
            {"type": "dirichlet", "boundary": "right", "value": float(params.get("g_right", $g_right))},
        ],
        params={"k": float(params.get("k", $k)), "f": float(params.get("f", $f))},
    )
''')


# ---------------------------------------------------------------------------
# Stokes template (standalone only -- see module docstring)
# ---------------------------------------------------------------------------

_STOKES_BODY = Template('''\
def solve(params: dict) -> dict:
    """Steady Stokes flow, Taylor-Hood (P2 velocity / P1 pressure) mixed
    element, lid-driven-cavity-style boundary conditions:

        -mu * div(grad(u)) + grad(p) = 0
        div(u) = 0
    with u = (lid_velocity, 0) on the top edge and u = (0, 0) on the other
    three edges.

    params: nx, ny (mesh resolution), domain=(x0,y0,x1,y1), mu (dynamic
    viscosity), lid_velocity.
    """
    import numpy as np
    import ufl
    from mpi4py import MPI
    from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary
    from dolfinx.fem import FunctionSpace, Function, Constant, dirichletbc, locate_dofs_topological
    from dolfinx.fem.petsc import LinearProblem
    from basix.ufl import element as ufl_element, mixed_element

    nx = int(params.get("nx", $nx))
    ny = int(params.get("ny", $ny))
    x0, y0, x1, y1 = params.get("domain", $domain)
    mu_val = float(params.get("mu", $mu))
    lid_velocity = float(params.get("lid_velocity", $lid_velocity))

    mesh = create_rectangle(MPI.COMM_WORLD, [[x0, y0], [x1, y1]], [nx, ny], CellType.triangle)
    gdim = mesh.geometry.dim

    P2 = ufl_element("Lagrange", mesh.topology.cell_name(), 2, shape=(gdim,))
    P1 = ufl_element("Lagrange", mesh.topology.cell_name(), 1)
    W = FunctionSpace(mesh, mixed_element([P2, P1]))
    W0, _ = W.sub(0).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    mu = Constant(mesh, mu_val)
    f = Constant(mesh, (0.0, 0.0))

    a = (
        mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    tol = 1e-12

    def _wall(edge):
        if edge == "top":
            return lambda x: np.isclose(x[1], y1, atol=tol)
        if edge == "bottom":
            return lambda x: np.isclose(x[1], y0, atol=tol)
        if edge == "left":
            return lambda x: np.isclose(x[0], x0, atol=tol)
        return lambda x: np.isclose(x[0], x1, atol=tol)

    facet_dim = mesh.topology.dim - 1
    bcs = []
    for edge, value in (
        ("top", (lid_velocity, 0.0)),
        ("bottom", (0.0, 0.0)),
        ("left", (0.0, 0.0)),
        ("right", (0.0, 0.0)),
    ):
        facets = locate_entities_boundary(mesh, facet_dim, _wall(edge))
        dofs = locate_dofs_topological((W.sub(0), W0), facet_dim, facets)
        g = Function(W0)
        g.x.array[:] = 0.0
        g.interpolate(lambda x, value=value: np.full((gdim, x.shape[1]), np.array(value).reshape(-1, 1)))
        bcs.append(dirichletbc(g, dofs, W.sub(0)))

    problem = LinearProblem(a, L, bcs=bcs)
    wh = problem.solve()
    u_h = wh.sub(0).collapse()
    p_h = wh.sub(1).collapse()

    return {
        "coords": mesh.geometry.x.copy(),
        "u": u_h.x.array.copy(),
        "p": p_h.x.array.copy(),
        "method": "dolfinx_stokes_taylor_hood",
        "params": {"mu": mu_val, "lid_velocity": lid_velocity},
    }
''')


# ---------------------------------------------------------------------------
# Elasticity template (-> FEnicsBridge "linear_elasticity_plane_stress")
# ---------------------------------------------------------------------------

_ELASTICITY_BODY = Template('''\
def solve(params: dict) -> dict:
    """Linear elasticity (plane stress), body force f, mirroring
    pinneapple_simulation.numerical_solvers.fenics_bridge.FEnicsBridge
    ._form_linear_elasticity exactly (same lame-parameter/UFL API calls),
    so this problem can also be run through
    pinneapple_simulation.external_solvers.fenics.FEniCSWorkflow via
    build_fenics_config() below. The left edge is clamped (u=0); a body
    force f (default gravity-like, pointing in -y) is applied everywhere
    else.

    params: nx, ny (mesh resolution), domain=(x0,y0,x1,y1), E (Young's
    modulus), nu (Poisson ratio), fx, fy (body force components).
    """
    import ufl
    import numpy as np
    from mpi4py import MPI
    from dolfinx.mesh import create_rectangle, CellType
    from dolfinx.fem import FunctionSpace, Constant, dirichletbc, locate_dofs_geometrical
    from dolfinx.fem.petsc import LinearProblem
    from basix.ufl import element as ufl_element

    nx = int(params.get("nx", $nx))
    ny = int(params.get("ny", $ny))
    x0, y0, x1, y1 = params.get("domain", $domain)
    E_val = float(params.get("E", $E))
    nu_val = float(params.get("nu", $nu))
    fx = float(params.get("fx", $fx))
    fy = float(params.get("fy", $fy))

    mesh = create_rectangle(MPI.COMM_WORLD, [[x0, y0], [x1, y1]], [nx, ny], CellType.triangle)
    el = ufl_element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    V = FunctionSpace(mesh, el)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = Constant(mesh, np.array([fx, fy], dtype=float))
    lam = Constant(mesh, E_val * nu_val / ((1 + nu_val) * (1 - nu_val)))
    mu = Constant(mesh, E_val / (2 * (1 + nu_val)))

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return lam * ufl.nabla_div(w) * ufl.Identity(len(w)) + 2 * mu * epsilon(w)

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx

    tol = 1e-12
    clamped_dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x0, atol=tol))
    zero = np.zeros(mesh.geometry.dim, dtype=float)
    bcs = [dirichletbc(zero, clamped_dofs, V)]

    problem = LinearProblem(a, L, bcs=bcs)
    uh = problem.solve()

    return {
        "coords": mesh.geometry.x.copy(),
        "$field": uh.x.array.copy(),
        "method": "dolfinx_linear_elasticity_plane_stress",
        "params": {"E": E_val, "nu": nu_val, "fx": fx, "fy": fy},
    }


def build_fenics_config(params: dict = None):
    """Build the pinneapple_simulation.external_solvers.fenics.FEniCSConfig
    for this same problem (kind="linear_elasticity_plane_stress", the one
    FEnicsBridge._form_linear_elasticity already implements)."""
    from pinneapple_simulation.external_solvers.fenics import FEniCSConfig

    params = dict(params or {})
    nx = int(params.get("nx", $nx))
    ny = int(params.get("ny", $ny))
    x0, y0, x1, y1 = params.get("domain", $domain)
    return FEniCSConfig(
        pde="linear_elasticity_plane_stress",
        domain={"type": "rectangle", "x": [x0, x1], "y": [y0, y1], "nx": nx, "ny": ny},
        bcs=[{"type": "dirichlet", "boundary": "left", "value": [0.0, 0.0]}],
        params={
            "E": float(params.get("E", $E)),
            "nu": float(params.get("nu", $nu)),
            "f": [float(params.get("fx", $fx)), float(params.get("fy", $fy))],
        },
    )
''')


# ---------------------------------------------------------------------------
# Elasticity + rigid-obstacle contact template (standalone only)
# ---------------------------------------------------------------------------

_ELASTICITY_CONTACT_BODY = Template('''\
def solve(params: dict) -> dict:
    """Linear elasticity with a penalty-method unilateral rigid-obstacle
    contact term on the bottom edge -- a SIMPLIFIED demonstration of
    contact handling, not a full active-set/Lagrange-multiplier or
    friction-aware contact solver.

    The body occupies (x0,x1) x (y0,y1) and sits above a rigid flat
    obstacle at height ``obstacle_y <= y0``. Left edge is clamped (u=0); a
    body force f (default gravity-like, in -y) pushes the body toward the
    obstacle. Contact is enforced weakly by adding a penalty term to the
    residual wherever the deformed bottom edge would penetrate the
    obstacle (Wriggers-style penalty regularisation of the Signorini
    unilateral contact condition: a stiff spring activates only in the
    penetrating region, via ufl.conditional / the negative part of the
    gap function). Because that term is nonzero only where the argument to
    ufl.conditional says so, the overall residual is a genuinely nonlinear
    function of u even though the elasticity term itself is linear, so
    this is solved with dolfinx's NonlinearProblem + Newton solver rather
    than LinearProblem.

    params: nx, ny (mesh resolution), domain=(x0,y0,x1,y1), E, nu, fx, fy
    (body force), obstacle_y (rigid floor height), penalty (contact
    stiffness), newton_rtol, max_newton_iterations.
    """
    import ufl
    import numpy as np
    from mpi4py import MPI
    from dolfinx.mesh import create_rectangle, CellType
    from dolfinx.fem import FunctionSpace, Function, Constant, dirichletbc, locate_dofs_geometrical
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver
    from basix.ufl import element as ufl_element

    nx = int(params.get("nx", $nx))
    ny = int(params.get("ny", $ny))
    x0, y0, x1, y1 = params.get("domain", $domain)
    E_val = float(params.get("E", $E))
    nu_val = float(params.get("nu", $nu))
    fx = float(params.get("fx", $fx))
    fy = float(params.get("fy", $fy))
    obstacle_y = float(params.get("obstacle_y", $obstacle_y))
    penalty = float(params.get("penalty", $penalty))
    newton_rtol = float(params.get("newton_rtol", $newton_rtol))
    max_newton_iterations = int(params.get("max_newton_iterations", $max_newton_iterations))

    mesh = create_rectangle(MPI.COMM_WORLD, [[x0, y0], [x1, y1]], [nx, ny], CellType.triangle)
    el = ufl_element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    V = FunctionSpace(mesh, el)

    u = Function(V)
    v = ufl.TestFunction(V)
    f = Constant(mesh, np.array([fx, fy], dtype=float))
    lam = Constant(mesh, E_val * nu_val / ((1 + nu_val) * (1 - nu_val)))
    mu = Constant(mesh, E_val / (2 * (1 + nu_val)))

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return lam * ufl.nabla_div(w) * ufl.Identity(len(w)) + 2 * mu * epsilon(w)

    x = ufl.SpatialCoordinate(mesh)
    gap = (x[1] + u[1]) - obstacle_y  # > 0: clear of the obstacle; < 0: penetrating
    penetration = ufl.conditional(ufl.lt(gap, 0.0), -gap, 0.0)
    contact_traction = penalty * penetration * v[1]

    F = (
        ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        - ufl.dot(f, v) * ufl.dx
        + contact_traction * ufl.dx
    )

    tol = 1e-12
    clamped_dofs = locate_dofs_geometrical(V, lambda pt: np.isclose(pt[0], x0, atol=tol))
    zero = np.zeros(mesh.geometry.dim, dtype=float)
    bcs = [dirichletbc(zero, clamped_dofs, V)]

    problem = NonlinearProblem(F, u, bcs=bcs)
    solver = NewtonSolver(mesh.comm, problem)
    solver.rtol = newton_rtol
    solver.max_it = max_newton_iterations
    n_iterations, converged = solver.solve(u)

    return {
        "coords": mesh.geometry.x.copy(),
        "$field": u.x.array.copy(),
        "method": "dolfinx_linear_elasticity_penalty_contact",
        "params": {
            "E": E_val, "nu": nu_val, "fx": fx, "fy": fy,
            "obstacle_y": obstacle_y, "penalty": penalty,
            "newton_iterations": n_iterations, "converged": bool(converged),
        },
    }
''')


_FOOTER = Template('''\

if __name__ == "__main__":
    _result = solve({})
    print(f"Solved (method={_result['method']!r}); "
          f"{_result['coords'].shape[0]} mesh nodes, "
          f"field '$field' has {len(_result['$field'])} DOFs.")
''')


_BODIES = {
    "diffusion": (_DIFFUSION_BODY, True),
    "stokes": (_STOKES_BODY, False),
    "elasticity": (_ELASTICITY_BODY, True),
    "elasticity_contact": (_ELASTICITY_CONTACT_BODY, False),
}

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "diffusion": dict(nx=32, ny=32, domain=(0.0, 0.0, 1.0, 1.0), k=1.0, f=0.0, g_left=0.0, g_right=1.0),
    "stokes": dict(nx=24, ny=24, domain=(0.0, 0.0, 1.0, 1.0), mu=1e-3, lid_velocity=1.0),
    "elasticity": dict(nx=32, ny=8, domain=(0.0, 0.0, 4.0, 1.0), E=2.1e5, nu=0.3, fx=0.0, fy=-1.0),
    "elasticity_contact": dict(
        nx=32, ny=8, domain=(0.0, 0.0, 4.0, 1.0), E=2.1e5, nu=0.3, fx=0.0, fy=-5.0,
        obstacle_y=-0.5, penalty=1e7, newton_rtol=1e-8, max_newton_iterations=50,
    ),
}


class FEniCSScriptGenerator:
    """Renders a curated set of UFL/dolfinx finite-element physics
    problems into standalone, runnable Python scripts, chosen by
    inspecting a ProblemSpec-like object's physics shape (see
    ``classify_physics_shape`` / the module docstring).
    """

    def generate(
        self,
        spec: Any,
        *,
        module_name: str = "generated_fenics_solver",
        contact: bool = False,
        **overrides: Any,
    ) -> str:
        shape = classify_physics_shape(spec)
        template_key = "elasticity_contact" if (shape == "elasticity" and contact) else shape
        body_template, has_config = _BODIES[template_key]

        fields = tuple(getattr(spec, "fields", ()) or ())
        field = fields[0] if fields else ("u" if shape != "elasticity" else "disp")
        kind = str(getattr(getattr(spec, "pde", None), "kind", "") or "")

        defaults = dict(_DEFAULTS[template_key])
        defaults.update(overrides)

        config_note = _CONFIG_NOTE if has_config else _NO_CONFIG_NOTE
        header = _HEADER.substitute(
            shape=template_key,
            kind=kind,
            fields=list(fields),
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            module_name=module_name,
            config_note=config_note,
        )
        subst = dict(defaults)
        if template_key in ("diffusion", "elasticity", "elasticity_contact"):
            subst["field"] = field
        body = body_template.substitute(**subst)
        footer = _FOOTER.substitute(field=field if template_key != "stokes" else "u")
        return header + body + footer

    def write(self, spec: Any, path: str, *, module_name: Optional[str] = None, **overrides: Any) -> str:
        import os

        if module_name is None:
            module_name = os.path.splitext(os.path.basename(path))[0]
        source = self.generate(spec, module_name=module_name, **overrides)
        with open(path, "w") as fh:
            fh.write(source)
        return path
