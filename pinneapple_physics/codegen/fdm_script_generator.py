"""Standalone finite-difference *script* generator for PINNeAPPle.

This is a different artifact from ``pinneapple_physics.symbolic_pde.SymbolicPDE``:
that module compiles a SymPy PDE residual into an in-memory autograd
*closure* (``residual_fn(coords) -> tensor``) for PINN training, evaluated
inside the same Python process. Nothing here reuses or wraps that closure.

``FDMScriptGenerator`` instead renders a **complete, self-contained Python
source file** — text you can ``open(path, "w").write(...)`` — that solves a
PDE with an explicit finite-difference scheme and writes its result to a
``.npz`` file. The generated file imports nothing from this package (or
from ``pinneapple_physics`` at all): it is meant to be handed to someone
else, run with a bare ``python generated_solver.py``, or executed inside
``pinneapple_tools.sandbox.custom_solver_runner.run_custom_solver`` in a
separate process with resource limits.

Scope (curated, not general)
-----------------------------
This covers exactly three PDE forms, chosen to match the ``ProblemSpec``
presets already shipped in ``pinneapple_physics.pde_environment.presets``:

  * 1D heat/diffusion equation   (``pde.kind`` contains "heat" or "diffusion")
  * 1D viscous Burgers equation  (``pde.kind`` contains "burgers")
  * 2D Poisson/Laplace equation  (``pde.kind`` contains "poisson" or "laplace")

There is no attempt at a general symbolic-PDE-to-FDM compiler here — that
is explicitly out of scope. Any other ``pde.kind``, or a matching kind with
an unsupported coordinate layout (e.g. 2D heat), raises ``NotImplementedError``
with a message naming exactly what is and isn't covered, rather than
silently guessing.

Numerical schemes mirror the hand-written ones already in
``pinneapple_simulation.numerical_solvers.fdm`` (FTCS/upwind for Burgers,
an explicit CFL-limited step for heat) so the generated code reads like
idiomatic PINNeAPPle FDM code, not a foreign style. It is independent
code, though: the generated file cannot import ``pinneapple_simulation``
(it must run with nothing but the stdlib + NumPy), so the stencils are
re-expressed standalone rather than imported. The Poisson solver here
deliberately does NOT copy
``pinneapple_simulation.numerical_solvers.fdm._sor_2d``'s vectorized
simultaneous-update scheme: that scheme is a weighted-Jacobi update
despite being called "SOR" (all neighbor reads come from the previous
sweep), and it was found to diverge here for omega>1 on the exact
manufactured problem this generator's own test suite checks against.
This module uses a real (checkerboard red-black) Gauss-Seidel SOR
instead — see ``_POISSON_2D_BODY`` for the details and the convergence
check performed while writing it.

Usage
-----
    import pinneapple_physics as pp
    from pinneapple_physics.codegen import FDMScriptGenerator

    spec = pp.burgers_1d_default(nu=0.01)
    path = FDMScriptGenerator().write(spec, "burgers_solver.py", nx=256, nt=4000)
    # `python burgers_solver.py --out solution.npz` now runs standalone.

Output convention
------------------
The generated script's ``solve(params: dict) -> dict`` returns a dict with
a ``"coords"`` key (mapping coordinate name -> 1D array) plus one entry per
PDE field (e.g. ``"u"``/``"T"``) holding the solution array — the same
``solve(params) -> dict`` shape used by
``pinneapple_tools.sandbox.custom_solver_runner.run_custom_solver`` and by
the generated FEniCS scripts in
``pinneapple_physics.codegen.fenics_script_generator``, so all
codegen-produced solvers share one calling convention. ``main()`` saves
that same dict, flattened, to a ``.npz`` file via ``numpy.savez`` — one
top-level array per coordinate/field name, following the flat-array-per-key
convention used elsewhere in this repo (e.g.
``pinneapple_simulation.numerical_solvers.irk_gauss_legendre``'s cached
``.npz`` files) rather than a single opaque blob.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from string import Template
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Parsing: ProblemSpec -> a small, template-ready descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FDMTemplateSpec:
    """Everything a template needs, extracted from a ProblemSpec (or built
    by hand for callers who don't have one)."""

    template: str          # "heat_1d" | "burgers_1d" | "poisson_2d"
    field: str              # primary field name, e.g. "u", "T"
    coord_names: Tuple[str, ...]
    domain_bounds: Dict[str, Tuple[float, float]]
    params: Dict[str, Any]
    kind: str
    name: str


class PDEParser:
    """Classify a ``pinneapple_physics.pde_environment.ProblemSpec`` (or any
    duck-typed object exposing ``.pde.kind``, ``.coords``, ``.fields``,
    ``.domain_bounds``, ``.pde.params``) into one of the three curated FDM
    templates ``FDMScriptGenerator`` knows how to render.

    Deliberately narrow: this is a classifier over a fixed, documented set
    of PDE kinds (see module docstring), not a general PDE-string parser.
    It reuses the ``ProblemSpec``/``PDETermSpec`` vocabulary already defined
    in ``pinneapple_physics.pde_environment`` rather than inventing a new
    PDE description syntax.
    """

    SUPPORTED = ("heat_1d", "burgers_1d", "poisson_2d")

    def parse(self, spec: Any) -> _FDMTemplateSpec:
        pde = spec.pde
        kind = str(getattr(pde, "kind", "")).lower()
        coords = tuple(getattr(spec, "coords", ()))
        fields = tuple(getattr(spec, "fields", ()))
        domain_bounds = dict(getattr(spec, "domain_bounds", {}) or {})
        params = dict(getattr(pde, "params", {}) or {})
        name = str(getattr(spec, "name", "") or kind or "pde")
        field = fields[0] if fields else "u"

        spatial = [c for c in coords if c != "t"]
        has_time = "t" in coords

        if "burgers" in kind:
            if len(spatial) != 1 or not has_time:
                raise NotImplementedError(
                    f"FDMScriptGenerator's Burgers template only covers 1D Burgers "
                    f"(coords like ('x','t')); got coords={coords} for kind='{kind}'."
                )
            return _FDMTemplateSpec("burgers_1d", field, coords, domain_bounds, params, kind, name)

        if "heat" in kind or "diffusion" in kind:
            if len(spatial) != 1 or not has_time:
                raise NotImplementedError(
                    "FDMScriptGenerator's heat template only covers the 1D transient "
                    f"heat equation (coords like ('x','t')); got coords={coords} for "
                    f"kind='{kind}'. (2D/3D heat conduction is out of scope for this "
                    "generator — see pinneapple_simulation.numerical_solvers.fdm for "
                    "the repo's general-purpose 2D ADI heat solver.)"
                )
            return _FDMTemplateSpec("heat_1d", field, coords, domain_bounds, params, kind, name)

        if "poisson" in kind or "laplace" in kind:
            if len(spatial) != 2 or has_time:
                raise NotImplementedError(
                    "FDMScriptGenerator's Poisson template only covers steady 2D "
                    f"Poisson/Laplace (coords like ('x','y')); got coords={coords} for "
                    f"kind='{kind}'."
                )
            return _FDMTemplateSpec("poisson_2d", field, coords, domain_bounds, params, kind, name)

        raise NotImplementedError(
            f"FDMScriptGenerator covers a curated set of PDE forms only: "
            f"{self.SUPPORTED} (matched by substring on pde.kind). Got "
            f"kind='{kind}', coords={coords} — not one of them. This is a "
            "deliberate scope limit, not a bug: extend PDEParser.parse if you "
            "need another curated form, or use "
            "pinneapple_physics.symbolic_pde.SymbolicPDE for a fully general "
            "(but in-memory, non-standalone) autograd residual instead."
        )


# ---------------------------------------------------------------------------
# Script templates
# ---------------------------------------------------------------------------

_HEADER = Template('''\
"""AUTO-GENERATED by pinneapple_physics.codegen.fdm_script_generator.FDMScriptGenerator
-- do not hand-edit; regenerate instead. Nobody wrote this stencil by hand.

PDE spec  : kind='$kind', template='$template', field='$field', coords=$coord_names
Generated : $timestamp

This file is a STANDALONE finite-difference solver: it imports nothing from
pinneapple_physics or pinneapple_simulation, only the Python standard
library and NumPy, so it can be run on its own:

    python $module_name.py --out solution.npz

It exposes ``solve(params: dict) -> dict`` (params override the defaults
baked in below); ``main()`` runs it with CLI-overridable defaults and saves
the result to an .npz file with one array per coordinate/field key -- the
same convention used by every other pinneapple_physics.codegen-generated
solver script, so a caller can run it directly or drop it into
pinneapple_tools.sandbox.custom_solver_runner.run_custom_solver.
"""
from __future__ import annotations

import argparse

import numpy as np

''')


_HEAT_1D_BODY = Template('''\
def solve(params: dict) -> dict:
    """Explicit (FTCS) solve of the 1D transient heat equation

        u_t = alpha * u_xx,   x in [x0, x1],  t in [t0, t1]

    with homogeneous Dirichlet BCs (u=0 at both ends) and a sine initial
    condition u(x,0) = sin(mode*pi*(x-x0)/L). With those BCs and IC this
    PDE has the exact closed-form solution

        u(x,t) = exp(-alpha * (mode*pi/L)**2 * t) * sin(mode*pi*(x-x0)/L)

    which is what makes this template checkable in a test (see
    tests/test_codegen.py) rather than "ran without crashing".

    params
    ------
    nx, nt   : grid resolution (defaults $nx, $nt)
    x_bounds : (x0, x1), default $x_bounds
    t_bounds : (t0, t1), default $t_bounds
    alpha    : thermal diffusivity, default $alpha
    mode     : sine mode number for the IC, default 1
    """
    nx = int(params.get("nx", $nx))
    nt_req = int(params.get("nt", $nt))
    x0, x1 = params.get("x_bounds", $x_bounds)
    t0, t1 = params.get("t_bounds", $t_bounds)
    alpha = float(params.get("alpha", $alpha))
    mode = int(params.get("mode", 1))

    x = np.linspace(x0, x1, nx, dtype=np.float64)
    dx = (x1 - x0) / (nx - 1)
    L = x1 - x0

    # CFL-safe explicit step: alpha*dt/dx**2 <= 0.5 for FTCS stability.
    # Safety factor 0.4 mirrors pinneapple_simulation.numerical_solvers.fdm's
    # own heat-equation time-step choice.
    dt_max = 0.4 * dx ** 2 / max(alpha, 1e-30)
    dt = min(dt_max, (t1 - t0) / max(nt_req, 1))
    nt = max(1, int(round((t1 - t0) / dt)))

    u = np.sin(mode * np.pi * (x - x0) / L)
    u[0] = 0.0
    u[-1] = 0.0

    r = alpha * dt / dx ** 2
    for _ in range(nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new

    t_final = t0 + nt * dt
    return {
        "coords": {"x": x.astype(np.float32), "t": np.array([t_final], dtype=np.float32)},
        "$field": u.astype(np.float32),
        "method": "ftcs_explicit",
        "params": {"alpha": alpha, "mode": mode, "dt": dt, "nt": nt, "t_final": t_final},
    }
''')


_BURGERS_1D_BODY = Template('''\
def solve(params: dict) -> dict:
    """FTCS (diffusion) + one-sided upwind (advection) solve of the 1D
    viscous Burgers equation

        u_t + u * u_x = nu * u_xx,   x in [x0, x1],  t in [t0, t1]

    Two initial-condition modes:

      * "neg_sin" (default) -- the classic u(x,0) = -sin(pi*x) turbulence
        test case (same IC as
        pinneapple_physics.pde_environment.presets.academics.burgers_1d_default),
        with u held fixed at its initial (boundary) value at both ends --
        no closed form, used for smoke-testing rather than accuracy.
      * "tanh_front" -- the exact traveling viscous-shock solution
        u(x,t) = c - c*tanh(c*(x - x_c - c*t) / (2*nu)), which solves the
        PDE exactly for any c, nu, x_c (verified symbolically with SymPy
        while this generator was written: u_t + u*u_x - nu*u_xx simplifies
        to 0 -- translating x by a constant x_c doesn't change that, since
        the PDE has no explicit x-dependence). This is what
        tests/test_codegen.py checks against, holding the domain
        boundaries at their (near-saturated, near-time-invariant) initial
        values -- a good approximation as long as the front stays well
        inside the domain over the run, which the default params ensure.

    params
    ------
    nx, nt      : grid resolution (defaults $nx, $nt)
    x_bounds    : (x0, x1), default $x_bounds
    t_bounds    : (t0, t1), default $t_bounds
    nu          : viscosity, default $nu
    ic_type     : "neg_sin" | "tanh_front", default "$ic_type"
    front_speed  : traveling-wave speed c, only used for "tanh_front", default $front_speed
    front_center : x_c, front position at t=0, only used for "tanh_front", default $front_center
    """
    nx = int(params.get("nx", $nx))
    nt_req = int(params.get("nt", $nt))
    x0, x1 = params.get("x_bounds", $x_bounds)
    t0, t1 = params.get("t_bounds", $t_bounds)
    nu = float(params.get("nu", $nu))
    ic_type = str(params.get("ic_type", "$ic_type"))
    c = float(params.get("front_speed", $front_speed))
    x_c = float(params.get("front_center", $front_center))

    x = np.linspace(x0, x1, nx, dtype=np.float64)
    dx = (x1 - x0) / (nx - 1)

    if ic_type == "tanh_front":
        u = c - c * np.tanh(c * (x - x_c) / (2.0 * nu))
    elif ic_type == "neg_sin":
        u = -np.sin(np.pi * (x - x0) / (x1 - x0))
    else:
        raise ValueError(f"Unknown ic_type={ic_type!r}; expected 'neg_sin' or 'tanh_front'.")

    left_bc, right_bc = u[0], u[-1]

    # Diffusive CFL, same 0.4 safety factor as
    # pinneapple_simulation.numerical_solvers.fdm._burgers.
    dt_max = 0.4 * dx ** 2 / max(nu, 1e-12)
    dt = min(dt_max, (t1 - t0) / max(nt_req, 1))
    nt = max(1, int(round((t1 - t0) / dt)))

    for _ in range(nt):
        adv = np.where(
            u >= 0,
            u * (u - np.roll(u, 1)) / dx,
            u * (np.roll(u, -1) - u) / dx,
        )
        diff = nu * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx ** 2
        u_new = u + dt * (-adv + diff)
        u_new[0] = left_bc
        u_new[-1] = right_bc
        u = u_new

    t_final = t0 + nt * dt
    return {
        "coords": {"x": x.astype(np.float32), "t": np.array([t_final], dtype=np.float32)},
        "$field": u.astype(np.float32),
        "method": "ftcs_upwind",
        "params": {"nu": nu, "ic_type": ic_type, "front_speed": c, "front_center": x_c, "dt": dt, "nt": nt, "t_final": t_final},
    }
''')


_POISSON_2D_BODY = Template('''\
def solve(params: dict) -> dict:
    """Red-black Gauss-Seidel SOR solve of the steady 2D Poisson/Laplace
    equation

        -(u_xx + u_yy) = f(x, y),   (x, y) in [x0,x1] x [y0,y1]

    with homogeneous Dirichlet BCs (u=0 on the boundary).

    The grid is split into a checkerboard "red"/"black" coloring (color of
    node (i,j) = (i+j) mod 2); each half-sweep updates one color using the
    other color's just-written values, vectorized via NumPy boolean
    masking. This is a real (sequential-in-effect) Gauss-Seidel update, not
    the simultaneous-update ("all cells at once from the previous sweep")
    scheme, which is really weighted Jacobi under the hood despite being a
    common way to informally write "SOR" -- that distinction matters here:
    plain simultaneous-update SOR with omega>1 diverges on this exact
    manufactured problem (checked while writing this generator), whereas
    red-black SOR converges in a few hundred iterations even at omega~1.8.

    source_mode
    -----------
    "manufactured_sine" (default) -- f = 2*pi**2*sin(pi*x)*sin(pi*y), for
        which the exact solution is u(x,y) = sin(pi*x)*sin(pi*y) (the same
        manufactured Poisson problem used in
        pinneapple_physics.symbolic_pde.compiler's own module docstring
        example). tests/test_codegen.py checks the result against this
        closed form.
    "zero" -- f = 0 (Laplace's equation) with u=0 on the boundary, i.e. the
        trivial solution u=0 everywhere -- included for completeness, not
        because it's an interesting test case.

    params
    ------
    nx, ny      : grid resolution (defaults $nx, $ny)
    x_bounds    : (x0, x1), default $x_bounds
    y_bounds    : (y0, y1), default $y_bounds
    iters       : max SOR sweeps, default $iters
    omega       : SOR over-relaxation factor, default $omega (1 <= omega < 2)
    tol         : convergence tolerance, default $tol
    source_mode : "manufactured_sine" | "zero", default "$source_mode"
    """
    nx = int(params.get("nx", $nx))
    ny = int(params.get("ny", $ny))
    x0, x1 = params.get("x_bounds", $x_bounds)
    y0, y1 = params.get("y_bounds", $y_bounds)
    iters = int(params.get("iters", $iters))
    omega = float(params.get("omega", $omega))
    tol = float(params.get("tol", $tol))
    source_mode = str(params.get("source_mode", "$source_mode"))

    x = np.linspace(x0, x1, nx, dtype=np.float64)
    y = np.linspace(y0, y1, ny, dtype=np.float64)
    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)
    XX, YY = np.meshgrid(x, y, indexing="ij")

    if source_mode == "manufactured_sine":
        f = 2.0 * np.pi ** 2 * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    elif source_mode == "zero":
        f = np.zeros_like(XX)
    else:
        raise ValueError(f"Unknown source_mode={source_mode!r}; expected 'manufactured_sine' or 'zero'.")

    u = np.zeros((nx, ny), dtype=np.float64)  # also enforces the u=0 boundary
    dx2, dy2 = dx * dx, dy * dy
    denom = 2.0 / dx2 + 2.0 / dy2

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    red_mask = ((ii + jj) % 2 == 0)[1:-1, 1:-1]
    black_mask = ~red_mask

    for _ in range(iters):
        u_prev = u[1:-1, 1:-1].copy()
        for color_mask in (red_mask, black_mask):
            neighbor_sum = (
                (u[1:-1, 2:] + u[1:-1, :-2]) / dx2
                + (u[2:, 1:-1] + u[:-2, 1:-1]) / dy2
            )
            updated = (1 - omega) * u[1:-1, 1:-1] + omega * (neighbor_sum + f[1:-1, 1:-1]) / denom
            u[1:-1, 1:-1] = np.where(color_mask, updated, u[1:-1, 1:-1])
        if np.max(np.abs(u[1:-1, 1:-1] - u_prev)) < tol:
            break

    return {
        "coords": {"x": x.astype(np.float32), "y": y.astype(np.float32)},
        "$field": u.astype(np.float32),
        "method": "red_black_sor",
        "params": {"omega": omega, "source_mode": source_mode, "iters": iters},
    }
''')


_MAIN_FOOTER = Template('''\

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="$default_out", help="Output .npz path.")
    parser.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE",
        help="Override a solve() parameter, e.g. --param nx=256. May be repeated.",
    )
    args = parser.parse_args()

    params = {}
    for kv in args.param:
        key, _, value = kv.partition("=")
        try:
            value = float(value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            pass
        params[key] = value

    result = solve(params)

    save_kwargs = dict(result["coords"])
    save_kwargs["$field"] = result["$field"]
    np.savez(args.out, **save_kwargs)
    method = result["method"]
    print(f"Saved solution to {args.out} (method={method!r})")


if __name__ == "__main__":
    main()
''')


_BODIES = {
    "heat_1d": _HEAT_1D_BODY,
    "burgers_1d": _BURGERS_1D_BODY,
    "poisson_2d": _POISSON_2D_BODY,
}

_DEFAULTS = {
    "heat_1d": dict(nx=128, nt=2000, x_bounds=(0.0, 1.0), t_bounds=(0.0, 0.1), alpha=0.01),
    "burgers_1d": dict(nx=256, nt=4000, x_bounds=(-1.0, 1.0), t_bounds=(0.0, 0.4), nu=0.05, ic_type="neg_sin", front_speed=1.0, front_center=0.0),
    "poisson_2d": dict(nx=64, ny=64, x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), iters=5000, omega=1.8, tol=1e-9, source_mode="manufactured_sine"),
}


class FDMScriptGenerator:
    """Renders a curated PDE description into a standalone, runnable FDM
    Python script (source text, not an in-memory callable).

    See the module docstring for exactly which ``pde.kind`` values are
    supported (1D heat/diffusion, 1D Burgers, 2D Poisson/Laplace) and why
    that set is deliberately curated rather than fully general.
    """

    def __init__(self, parser: Optional[PDEParser] = None) -> None:
        self.parser = parser or PDEParser()

    def generate(self, spec: Any, *, module_name: str = "generated_fdm_solver", **overrides: Any) -> str:
        """Return the full generated Python source (as a string).

        ``overrides`` bakes numerical-scheme defaults (nx, nt, iters, ...)
        into the generated script; every one of them is still overridable
        again at run time via ``solve(params)`` or ``--param key=value``.
        """
        tspec = self.parser.parse(spec)
        defaults = dict(_DEFAULTS[tspec.template])
        defaults.update(tspec.params)  # ProblemSpec.pde.params (e.g. nu, alpha)

        # Pull spatial/temporal bounds out of the ProblemSpec's domain_bounds
        # when present, so the generated script matches the spec it came from.
        coords = tspec.coord_names
        spatial = [c for c in coords if c != "t"]
        if tspec.template in ("heat_1d", "burgers_1d") and spatial:
            defaults["x_bounds"] = tuple(tspec.domain_bounds.get(spatial[0], defaults["x_bounds"]))
            if "t" in tspec.domain_bounds:
                defaults["t_bounds"] = tuple(tspec.domain_bounds["t"])
        elif tspec.template == "poisson_2d" and len(spatial) == 2:
            defaults["x_bounds"] = tuple(tspec.domain_bounds.get(spatial[0], defaults["x_bounds"]))
            defaults["y_bounds"] = tuple(tspec.domain_bounds.get(spatial[1], defaults["y_bounds"]))

        defaults.update(overrides)

        header = _HEADER.substitute(
            kind=tspec.kind,
            template=tspec.template,
            field=tspec.field,
            coord_names=list(tspec.coord_names),
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            module_name=module_name,
        )
        body = _BODIES[tspec.template].substitute(field=tspec.field, **defaults)
        footer = _MAIN_FOOTER.substitute(default_out=f"{module_name}_output.npz", field=tspec.field)
        return header + body + footer

    def write(self, spec: Any, path: str, *, module_name: Optional[str] = None, **overrides: Any) -> str:
        """Generate and write the script to ``path``; returns ``path``."""
        import os

        if module_name is None:
            module_name = os.path.splitext(os.path.basename(path))[0]
        source = self.generate(spec, module_name=module_name, **overrides)
        with open(path, "w") as fh:
            fh.write(source)
        return path
