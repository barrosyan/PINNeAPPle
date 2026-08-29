"""Spectral-element FEM for a geometrically nonlinear (Von Karman) transient
Euler-Bernoulli beam, time-integrated with the Newmark-beta method and a full
Newton-Raphson iteration at every timestep to resolve the nonlinear
residual.

Discretization
--------------
Axial displacement u(z,t) uses Lagrange shape functions on Gauss-Lobatto-
Legendre (GLL) node placement (the "spectral element" part -- the node
distribution and Lagrange basis generalize to any order). Transverse
displacement w(z,t) and its slope dw/dz use cubic Hermite shape functions
(C1-continuous, the standard two-node Euler-Bernoulli beam element -- this
part of the formulation is fixed at a two-node cubic Hermite element
regardless of mesh order; see `NODES_PER_ELEMENT` below). The two fields are
coupled through the Von Karman nonlinear axial-strain term
(1/2)(dw/dz)^2 -- the standard geometrically-nonlinear extension of the
linear Euler-Bernoulli beam capturing bending-stretching coupling at
moderately large transverse deflection. A distributed viscous damping term
and an optional linear thermal-expansion axial load are also included as
generic terms (both default to "no effect": zero damping, uniform
temperature field).

Time integration: Newmark-beta (average-acceleration family by default),
with the full nonlinear residual re-linearized and solved by Newton-Raphson
at every timestep (and, optionally, ramped up over multiple load steps for
a difficult initial nonlinear solve).

Everything is caller-supplied
------------------------------
Geometry (length, cross-section area, second moment of area), material
properties (Young's modulus, density, damping coefficient, optional thermal
terms), mesh resolution (element count), time-integration parameters (total
time, number of steps, Newmark beta/gamma, Newton-Raphson iteration budget
and convergence tolerance), and the boundary-condition specification (which
DOFs are Dirichlet-fixed/prescribed at which nodes, and what Neumann loads
-- point or distributed, constant or sinusoidally time-varying -- are
applied at which DOFs) are all parameters. No boundary topology or load
case is hardcoded: a clamped-free cantilever under a tip load is just one
`boundary` dict a caller can construct; a simply-supported beam under a
distributed load, or any other combination of prescribed/free DOFs, is
exactly as valid an input.

Boundary condition dict shape (Dirichlet "D" / Neumann "N"):

    boundary = {
        "D": {
            "globalNode#": [[node_a, node_b, ...], [count_a, count_b, ...]],
            "Values": [[dof, value], ...],   # sum(count_i) entries, in the
                                              # same order as the flattened
                                              # (node, dof) pairs above
        },
        "N": {
            "globalNode#": [[node_a, ...], [count_a, ...]],
            "Values": [[dof, [const, sin_amp, cos_amp]], ...],
        },
    }

DOF indices are 0=axial (u), 1=transverse (w), 2=slope (dw/dz). A Neumann
value is evaluated at time t as
`const + sin_amp*sin(2*pi*f_drive*t) + cos_amp*cos(2*pi*f_drive*t)` for a
single caller-supplied driving frequency `f_drive` shared by all
time-varying Neumann terms (pass sin_amp=cos_amp=0 for a plain constant
load).

Verification hook
------------------
`natural_frequencies_hz` extracts the natural frequencies of the assembled
mass/stiffness system at the undeformed state via a generalized eigenvalue
solve -- independent of the time-marching/nonlinear-iteration code
entirely, so it's a strong, cheap sanity check against closed-form beam
eigenvalue results (e.g. the standard cantilever formula
`omega_1 = (1.875104)^2 * sqrt(EI/(rho*A*L^4))`) for whatever boundary
topology the caller constructs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.optimize import root_scalar
from scipy.sparse import coo_array
from scipy.sparse.linalg import eigsh, spsolve
from scipy.special import legendre

from .base import SolverBase, SolverOutput
from .beam_bvp_fdm import second_derivative
from .registry import SolverRegistry

# DOF index convention (matches the "N"/"D" boundary dict above).
DOF_AXIAL = 0
DOF_TRANSVERSE = 1
DOF_SLOPE = 2
DOFPN = 3  # degrees of freedom per node -- fixed by the u/w/theta formulation

# The Hermite shape functions below are a fixed two-endpoint cubic Hermite
# element (the standard Euler-Bernoulli beam element): only the two "corner"
# nodes of an element carry meaningful transverse/slope shape functions.
# Using more than 2 nodes per element would leave the transverse/slope DOFs
# at interior nodes without any Hermite basis support (rows of zeros in the
# element mass/stiffness contribution), which silently produces a singular
# global system. The axial field's Lagrange/GLL basis is written generally
# for any node count, but the coupled formulation as a whole is only valid
# at NODES_PER_ELEMENT == 2.
NODES_PER_ELEMENT = 2


# --------------------------------------------------------------------------
# Mesh / shape functions / quadrature
# --------------------------------------------------------------------------

def gauss_lobatto_legendre_points(n_points: int) -> np.ndarray:
    """Gauss-Lobatto-Legendre points on [-1, 1] for `n_points` nodes
    (endpoints included, interior points at the roots of P'_{n-1})."""
    order = n_points - 1
    derivative = legendre(order).deriv()
    roots = np.sort(derivative.roots)
    return np.insert(roots, [0, order - 1], [-1, 1])


def generate_1d_mesh(num_elements: int, nodes_per_element: int, dofpn: int, length: float):
    """Uniform 1D mesh of `num_elements` elements, each with `nodes_per_element`
    nodes placed at Gauss-Lobatto-Legendre points (so nodes_per_element=2 is a
    plain linear mesh; higher values place spectral interior nodes).

    Returns (conn, coords, total_nodes, dof_conn, element_length).
    """
    total_nodes = num_elements * (nodes_per_element - 1) + 1
    h = length / num_elements
    dofpe = nodes_per_element * dofpn
    conn = np.zeros(shape=[num_elements, nodes_per_element], dtype=np.int64)
    dof_conn = np.zeros(shape=[num_elements, dofpe], dtype=np.int64)
    coords = np.zeros(shape=total_nodes, dtype=np.float64)

    spectral_nodes = gauss_lobatto_legendre_points(nodes_per_element)
    for i in range(num_elements):
        for j in range(nodes_per_element):
            node_number = i * (nodes_per_element - 1) + j
            coords[node_number] = i * h + (spectral_nodes[j] + 1) * h / 2
            conn[i, j] = node_number
        for k in range(dofpn):
            dof_conn[i, k:dofpe:dofpn] = k + dofpn * conn[i, :]

    return conn, coords, total_nodes, dof_conn, h


def gauss_legendre_quadrature(n_points: int):
    """Gauss-Legendre quadrature points and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n_points)


def lagrange_shape_functions_1d(nodes_per_element: int, n_quad_points: int):
    """Lagrange shape functions and their first derivative (built on GLL
    nodes), evaluated at `n_quad_points` Gauss-Legendre quadrature points."""
    shape = np.zeros(shape=[nodes_per_element, n_quad_points], dtype=np.float64)
    dshape = np.zeros(shape=[nodes_per_element, n_quad_points], dtype=np.float64)
    nodes = gauss_lobatto_legendre_points(nodes_per_element)
    quad_pts, _ = gauss_legendre_quadrature(n_quad_points)

    for k in range(n_quad_points):
        for j in range(nodes_per_element):
            xj = nodes[j]
            product = 1.0
            for m in range(nodes_per_element):
                if m != j:
                    xm = nodes[m]
                    product *= (quad_pts[k] - xm) / (xj - xm)
            shape[j, k] = product

    for k in range(n_quad_points):
        for j in range(nodes_per_element):
            s = 0.0
            xj = nodes[j]
            for i in range(nodes_per_element):
                product = 1.0
                for m in range(nodes_per_element):
                    if m != j and m != i:
                        xm = nodes[m]
                        product *= (quad_pts[k] - xm) / (xj - xm)
                if i != j:
                    xi = nodes[i]
                    s += product / (xj - xi)
            dshape[j, k] = s

    return shape, dshape


def hermite_shape_functions_1d(n_quad_points: int, h: float):
    """Cubic Hermite shape functions (value + slope DOF at each of the two
    element endpoints) and their first/second derivatives, evaluated at
    `n_quad_points` Gauss-Legendre quadrature points on the element of
    length `h`. Fixed at two endpoints regardless of mesh order -- see
    `NODES_PER_ELEMENT`."""
    shape = np.zeros(shape=[4, n_quad_points], dtype=np.float64)
    dshape = np.zeros(shape=[4, n_quad_points], dtype=np.float64)
    ddshape = np.zeros(shape=[4, n_quad_points], dtype=np.float64)
    quad_pts, _ = gauss_legendre_quadrature(n_quad_points)

    shape[0] = 0.25 * (2 - 3 * quad_pts + quad_pts ** 3)
    shape[1] = -0.125 * h * (1 - quad_pts) * (1 - quad_pts ** 2)
    shape[2] = 0.25 * (2 + 3 * quad_pts - quad_pts ** 3)
    shape[3] = 0.125 * h * (1 + quad_pts) * (1 - quad_pts ** 2)

    dshape[0] = -0.75 * (1 - quad_pts ** 2)
    dshape[1] = 0.125 * h * (1 + 2 * quad_pts - 3 * quad_pts ** 2)
    dshape[2] = 0.75 * (1 - quad_pts ** 2)
    dshape[3] = 0.125 * h * (1 - 2 * quad_pts - 3 * quad_pts ** 2)

    ddshape[0] = 1.5 * quad_pts
    ddshape[1] = 0.25 * h * (1 - 3 * quad_pts)
    ddshape[2] = -1.5 * quad_pts
    ddshape[3] = -0.25 * h * (1 + 3 * quad_pts)

    return shape, dshape, ddshape


# --------------------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------------------

def neumann_boundary_time_series(num_elements, nodes_per_element, dofpn, conn,
                                  boundary_n, timesteps, delta_t, f_drive):
    """Pre-evaluate the Neumann ("N") boundary loads at every timestep,
    scattered onto the (element, local-dof) layout used during assembly.
    A load value is const + sin_amp*sin(2*pi*f_drive*t) +
    cos_amp*cos(2*pi*f_drive*t)."""
    boundary = np.zeros(shape=[num_elements, nodes_per_element * dofpn, timesteps], dtype=np.float64)
    w_drive = 2 * np.pi * f_drive
    for t in range(timesteps):
        time = (t + 1) * delta_t
        total_count = 0
        for i, node in enumerate(boundary_n["globalNode#"][0]):
            count = boundary_n["globalNode#"][1][i]
            for j in range(count):
                index = total_count + j
                elem_ind = np.where(conn == node)
                dof = elem_ind[1] * dofpn + boundary_n["Values"][index][0]
                const, sin_amp, cos_amp = boundary_n["Values"][index][1]
                value = const + sin_amp * np.sin(w_drive * time) + cos_amp * np.cos(w_drive * time)
                boundary[elem_ind[0], dof, t] = value / len(elem_ind[0])
            total_count += count
    return boundary


def dirichlet_boundary_values(num_elements, nodes_per_element, dofpn, conn, boundary_d):
    """Pre-scatter the Dirichlet ("D") boundary values onto the (element,
    local-dof) layout; np.nan marks an unconstrained (element, local-dof)."""
    boundary = np.full(shape=[num_elements, nodes_per_element * dofpn], dtype=np.float64, fill_value=np.nan)
    total_count = 0
    for i, node in enumerate(boundary_d["globalNode#"][0]):
        count = boundary_d["globalNode#"][1][i]
        for j in range(count):
            index = total_count + j
            elem_ind = np.where(conn == node)
            dof = elem_ind[1] * dofpn + boundary_d["Values"][index][0]
            value = boundary_d["Values"][index][1]
            boundary[elem_ind[0], dof] = value
        total_count += count
    return boundary


def apply_neumann_bc(i, t, load_vec_elem, boundary_n_series, ratio):
    """Add the pre-evaluated Neumann load (scaled by the current load-step
    ratio) into an element's residual/load vector."""
    load_vec_elem[:] += boundary_n_series[i, :, t] * ratio
    return load_vec_elem


def apply_dirichlet_bc(i, newton_iter, stiff_mat_elem, load_vec_elem, boundary_d_values):
    """Enforce Dirichlet constraints on one element's tangent matrix and
    residual by row/column elimination (constrained DOFs decouple from the
    rest of the system; their prescribed value is only injected on the
    first Newton iteration of a load step, since later iterations solve for
    the *correction* to an already-satisfied constraint)."""
    mask = ~np.isnan(boundary_d_values[i, :])
    indices = np.where(mask != 0)[0]
    for index in indices:
        value_bc = boundary_d_values[i, index] if newton_iter == 0 else 0.0
        diag = stiff_mat_elem[index, index]
        stiff_mat_elem[index, :] = 0.0
        load_vec_elem[:] -= stiff_mat_elem[:, index] * value_bc
        stiff_mat_elem[:, index] = 0.0
        stiff_mat_elem[index, index] = diag
        load_vec_elem[index] = diag * value_bc
    return stiff_mat_elem, load_vec_elem


def cantilever_frequency_roots(n: int) -> np.ndarray:
    """First `n` positive roots of cos(x)cosh(x) = -1, the transcendental
    frequency equation for a clamped-free (cantilever) Euler-Bernoulli beam
    -- solved by bracketed root-finding since no closed form exists. Useful
    as an independent closed-form check for `natural_frequencies_hz` when
    the caller's boundary dict happens to describe a cantilever."""

    def f(x):
        return np.cos(x) + 1 / np.cosh(x)

    roots = np.zeros(shape=n, dtype=np.float64)
    for k in range(1, n + 1):
        if k == 1:
            lo, hi = 1e-3, np.pi - 1e-6
        else:
            guess = (2 * k - 1) * np.pi / 2
            lo, hi = guess - np.pi / 2 + 1e-6, guess + np.pi / 2 - 1e-6
        roots[k - 1] = root_scalar(f, bracket=[lo, hi], method="brentq").root
    return roots


# --------------------------------------------------------------------------
# Nonlinear transient FEM engine
# --------------------------------------------------------------------------

class NonlinearBeamNewmarkFEM:
    """Spectral-element / Newmark-beta engine for the transient geometrically
    nonlinear (Von Karman) Euler-Bernoulli beam described in the module
    docstring. Construct once per problem (mesh + BCs are fixed at
    construction), then call `run()` to march through time, or
    `natural_frequencies_hz()` for the linearized-at-undeformed-state
    eigenvalue check.
    """

    def __init__(self, num_elements: int, nodes_per_element: int, length: float,
                 area: float, moment_of_inertia: float, E: float, density: float,
                 damping_coeff: float, axial_dist_load: float, transverse_dist_load: float,
                 boundary: Dict[str, Any], total_time: float, timesteps: int,
                 newmark_beta: float, newmark_gamma: float, newton_relaxation: float,
                 newton_iterations: int, newton_tolerance: float, load_steps: int,
                 initial_disp: Optional[np.ndarray] = None, initial_vel: Optional[np.ndarray] = None,
                 reference_temperature: float = 0.0, temperature_field: Optional[np.ndarray] = None,
                 thermal_expansion_coeff: float = 0.0, driving_freq_hz: float = 0.0):
        if nodes_per_element != NODES_PER_ELEMENT:
            raise ValueError(
                f"nodes_per_element must be {NODES_PER_ELEMENT}: the Hermite "
                f"transverse/slope shape functions are a fixed two-endpoint "
                f"cubic element, so any other node count leaves interior "
                f"transverse/slope DOFs without shape-function support "
                f"(singular system), got {nodes_per_element}"
            )
        if num_elements < 1:
            raise ValueError(f"num_elements must be >= 1, got {num_elements}")
        if not (length > 0 and area > 0 and moment_of_inertia > 0 and E > 0 and density > 0):
            raise ValueError("length, area, moment_of_inertia, E, and density must all be > 0")
        if timesteps < 1 or load_steps < 1 or newton_iterations < 1:
            raise ValueError("timesteps, load_steps, and newton_iterations must all be >= 1")
        if newton_tolerance <= 0:
            raise ValueError(f"newton_tolerance must be > 0, got {newton_tolerance}")

        self.NEL = num_elements
        self.NNPEL = nodes_per_element
        self.DOFPN = DOFPN
        self.iterations = newton_iterations
        self.convergence = newton_tolerance
        self.loadsteps = load_steps
        # Internal Newmark bookkeeping uses the original derivation's
        # parametrization: "gamma" (internal) = 2 * newmark_beta (standard),
        # "alpha" (internal) = newmark_gamma (standard). Verified against the
        # standard average-acceleration method (beta=0.25, gamma=0.5, the
        # common unconditionally-stable default) reducing to this engine's
        # historical default (alpha=0.5, gamma=0.5).
        self.beta = newton_relaxation
        self.alpha = newmark_gamma
        self.gamma = 2.0 * newmark_beta
        self.time = total_time
        self.timesteps = timesteps
        self.L = length
        self.I = moment_of_inertia
        self.A = area
        self.E = E
        self.rhof = density
        self.damp = damping_coeff
        self.f = axial_dist_load
        self.q = transverse_dist_load
        self.To = reference_temperature
        self.fDriv = driving_freq_hz
        self.boundaryDataN = boundary["N"]
        self.boundaryDataD = boundary["D"]
        self.NGQP = self.NNPEL
        self.redNGQP = self.NNPEL - 1

        self.deltaT = self.time / self.timesteps
        self.conn, self.globCoord, self.totalNodes, self.dofconn, self.h = generate_1d_mesh(
            self.NEL, self.NNPEL, self.DOFPN, self.L)
        self.Temp = (np.asarray(temperature_field, dtype=np.float64)
                     if temperature_field is not None
                     else np.full(self.totalNodes, self.To, dtype=np.float64))
        self.aTh = thermal_expansion_coeff
        self.DBC = dirichlet_boundary_values(self.NEL, self.NNPEL, self.DOFPN, self.conn, self.boundaryDataD)
        self.NBC = neumann_boundary_time_series(self.NEL, self.NNPEL, self.DOFPN, self.conn,
                                                 self.boundaryDataN, self.timesteps, self.deltaT, self.fDriv)
        self.lagSF, self.lagDSF = lagrange_shape_functions_1d(self.NNPEL, self.NGQP)
        self.redLagSF, self.redLagDSF = lagrange_shape_functions_1d(self.NNPEL, self.redNGQP)
        self.herSF, self.herDSF, self.herDDSF = hermite_shape_functions_1d(self.NGQP, self.h)
        self.redHerSF, self.redHerDSF, self.redHerDDSF = hermite_shape_functions_1d(self.redNGQP, self.h)
        self.points, self.weights = gauss_legendre_quadrature(self.NGQP)
        self.redPoints, self.redWeights = gauss_legendre_quadrature(self.redNGQP)

        self.eqnsElem = self.NNPEL * self.DOFPN
        self.eqnsGl = self.totalNodes * self.DOFPN
        self.a0 = self.E * self.A
        self.a1 = self.E * self.I
        self.a2 = self.rhof * self.A
        self.a3 = self.damp * self.A

        self.t1 = self.alpha * self.deltaT
        self.t2 = (1 - self.alpha) * self.deltaT
        self.t3 = 2 / self.gamma / self.deltaT ** 2
        self.t4 = self.t3 * self.deltaT
        self.t5 = (1 / self.gamma) - 1
        self.t6 = 2 * self.alpha / self.gamma / self.deltaT
        self.t7 = (2 * self.alpha / self.gamma) - 1
        self.t8 = self.deltaT * (self.alpha / self.gamma - 1)

        self.coeffStiffMatElem = {f"K{i}{j}": np.zeros([self.NNPEL, self.NNPEL]) for i in range(self.DOFPN) for j in range(self.DOFPN)}
        self.coeffMassMatElem = {f"K{i}{j}": np.zeros([self.NNPEL, self.NNPEL]) for i in range(self.DOFPN) for j in range(self.DOFPN)}
        self.coeffDampMatElem = {f"K{i}{j}": np.zeros([self.NNPEL, self.NNPEL]) for i in range(self.DOFPN) for j in range(self.DOFPN)}
        self.addTanMatElem = {f"K{i}{j}": np.zeros([self.NNPEL, self.NNPEL]) for i in range(self.DOFPN) for j in range(self.DOFPN)}
        self.coeffForceElem = {f"F{i}": np.zeros(self.NNPEL) for i in range(self.DOFPN)}
        self.coeffThermForceElem = {f"F{i}": np.zeros(self.NNPEL) for i in range(self.DOFPN)}

        self.massMatElem = np.zeros([self.eqnsElem, self.eqnsElem])
        self.dampMatElem = np.zeros([self.eqnsElem, self.eqnsElem])
        self.stiffMatElem = np.zeros([self.eqnsElem, self.eqnsElem])
        self.addtanMatElem = np.zeros([self.eqnsElem, self.eqnsElem])
        self.forceElem = np.zeros(self.eqnsElem)
        self.thermForceElem = np.zeros(self.eqnsElem)

        self.stiffMatGl = coo_array((self.eqnsGl, self.eqnsGl), dtype=np.float64)
        self.forceGl = np.zeros(self.eqnsGl)

        self.deltaSoln = np.zeros(self.eqnsGl)
        self.prevIterSoln = np.zeros([2, self.eqnsGl])
        self.newIterSoln = np.zeros(self.eqnsGl)

        self.timeSoln = np.zeros([self.timesteps + 1, self.eqnsGl])
        self.timeVel = np.zeros([self.timesteps + 1, self.eqnsGl])
        self.timeAcc = np.zeros([self.timesteps + 1, self.eqnsGl])

        init_disp = np.zeros(self.eqnsGl) if initial_disp is None else np.asarray(initial_disp, dtype=np.float64)
        init_vel = np.zeros(self.eqnsGl) if initial_vel is None else np.asarray(initial_vel, dtype=np.float64)
        self.prevIterSoln[0, :] = init_disp
        self.timeSoln[0, :] = init_disp
        self.timeVel[0, :] = init_vel

    def _solve_elem_level(self, i, ratio, t):
        """Element-level tangent (Newton) matrix and residual for element
        `i`, including the Von Karman nonlinear stiffness/tangent
        contributions and the Newmark mass/damping terms."""
        tan_mat_elem = np.zeros([self.eqnsElem, self.eqnsElem])
        stiffness = np.zeros([self.eqnsElem, self.eqnsElem])
        force = np.zeros(self.eqnsElem)
        residual_elem = np.zeros(self.eqnsElem)
        self.stiffMatElem[:, :] = 0.0
        self.forceElem[:] = 0.0
        self.thermForceElem[:] = 0.0

        for key in self.coeffStiffMatElem:
            self.coeffMassMatElem[key][:, :] = 0.0
            self.coeffDampMatElem[key][:, :] = 0.0
            self.coeffStiffMatElem[key][:, :] = 0.0
            self.addTanMatElem[key][:, :] = 0.0
        for key in self.coeffForceElem:
            self.coeffForceElem[key][:] = 0.0
            self.coeffThermForceElem[key][:] = 0.0

        elem_prev_soln = np.zeros([2, self.DOFPN * self.NNPEL])
        elem_prev_soln[0, :] = self.prevIterSoln[0, self.dofconn[i]]
        elem_prev_soln[1, :] = self.prevIterSoln[1, self.dofconn[i]]
        elem_vel = self.timeVel[t, self.dofconn[i]]
        elem_acc = self.timeAcc[t, self.dofconn[i]]
        elem_temp = self.Temp[self.conn[i]]
        diff_temp = elem_temp - self.To

        As = self.t3 * elem_prev_soln[0] + self.t4 * elem_vel + self.t5 * elem_acc
        Bs = self.t6 * elem_prev_soln[0] + self.t7 * elem_vel + self.t8 * elem_acc

        for GP in range(self.NGQP):
            jac = self.globCoord[self.conn[i]] @ self.lagDSF[:, GP]
            jac2 = jac ** 2
            SL0 = self.lagSF[:, GP]
            SL1 = self.lagDSF[:, GP] / jac
            SH10 = self.herSF[0::2, GP]
            SH11 = self.herDSF[0::2, GP] / jac
            SH12 = self.herDDSF[0::2, GP] / jac2
            SH20 = self.herSF[1::2, GP]
            SH21 = self.herDSF[1::2, GP] / jac
            SH22 = self.herDDSF[1::2, GP] / jac2
            outerSL = np.outer(SL1, SL1)
            outerSH11_2 = np.outer(SH12, SH12)
            outerSH12_2 = np.outer(SH12, SH22)
            outerSH21_2 = outerSH12_2.T
            outerSH22_2 = np.outer(SH22, SH22)
            outerSH11_0 = np.outer(SH10, SH10)
            outerSH22_0 = np.outer(SH20, SH20)
            constant = jac * self.weights[GP]

            self.coeffMassMatElem["K00"][:, :] += self.a2 * constant * outerSL
            self.coeffMassMatElem["K11"][:, :] += self.a2 * constant * outerSH11_0
            self.coeffMassMatElem["K22"][:, :] += self.rhof * self.I * constant * outerSH22_0

            self.coeffDampMatElem["K00"][:, :] += self.a3 * constant * outerSL
            self.coeffDampMatElem["K11"][:, :] += self.a3 * constant * outerSH11_0
            self.coeffDampMatElem["K22"][:, :] += self.a3 * constant * outerSH22_0

            self.coeffStiffMatElem["K00"][:, :] += self.a0 * constant * outerSL
            self.coeffStiffMatElem["K11"][:, :] += self.a1 * constant * outerSH11_2
            self.coeffStiffMatElem["K12"][:, :] += self.a1 * constant * outerSH12_2
            self.coeffStiffMatElem["K21"][:, :] += self.a1 * constant * outerSH21_2
            self.coeffStiffMatElem["K22"][:, :] += self.a1 * constant * outerSH22_2

            self.coeffForceElem["F0"][:] += self.f * ratio * constant * SL0
            self.coeffForceElem["F1"][:] += self.q * ratio * constant * SH10
            self.coeffForceElem["F2"][:] += self.q * ratio * constant * SH20
            self.coeffThermForceElem["F0"][:] += self.a0 * self.aTh * diff_temp * constant * SL0
            self.coeffThermForceElem["F1"][:] += self.a0 * self.aTh * diff_temp * constant * SH10
            self.coeffThermForceElem["F2"][:] += self.a0 * self.aTh * diff_temp * constant * SH20

        for GP in range(self.redNGQP):
            jac = self.globCoord[self.conn[i]] @ self.redLagDSF[:, GP]
            jac2 = jac ** 2
            SL1 = self.redLagDSF[:, GP] / jac
            SH11 = self.redHerDSF[0::2, GP] / jac
            SH12 = self.redHerDDSF[0::2, GP] / jac2
            SH21 = self.redHerDSF[1::2, GP] / jac
            SH22 = self.redHerDDSF[1::2, GP] / jac2

            prev_sol_du1 = elem_prev_soln[1, 0::self.DOFPN] @ SL1
            prev_sol_dw1 = elem_prev_soln[1, 1::self.DOFPN] @ SH11 + elem_prev_soln[1, 2::self.DOFPN] @ SH21

            outerSH11_1 = np.outer(SH11, SH11)
            outerSH12_1 = np.outer(SH11, SH21)
            outerSH21_1 = outerSH12_1.T
            outerSH22_1 = np.outer(SH21, SH21)
            outerSLH1_1 = np.outer(SL1, SH11)
            outerSLH2_1 = np.outer(SL1, SH21)
            outerSH1L_1 = outerSLH1_1.T
            outerSH2L_1 = outerSLH2_1.T
            constant = jac * self.redWeights[GP]

            self.coeffStiffMatElem["K01"][:, :] += 0.5 * self.a0 * constant * prev_sol_dw1 * outerSLH1_1
            self.coeffStiffMatElem["K02"][:, :] += 0.5 * self.a0 * constant * prev_sol_dw1 * outerSLH2_1
            self.coeffStiffMatElem["K10"][:, :] += self.a0 * constant * prev_sol_dw1 * outerSH1L_1
            self.coeffStiffMatElem["K20"][:, :] += self.a0 * constant * prev_sol_dw1 * outerSH2L_1
            self.coeffStiffMatElem["K11"][:, :] += 0.5 * self.a0 * constant * prev_sol_dw1 ** 2 * outerSH11_1
            self.coeffStiffMatElem["K12"][:, :] += 0.5 * self.a0 * constant * prev_sol_dw1 ** 2 * outerSH12_1
            self.coeffStiffMatElem["K21"][:, :] += 0.5 * self.a0 * constant * prev_sol_dw1 ** 2 * outerSH21_1
            self.coeffStiffMatElem["K22"][:, :] += 0.5 * self.a0 * constant * prev_sol_dw1 ** 2 * outerSH22_1

            self.addTanMatElem["K01"][:, :] += 0.5 * self.a0 * constant * prev_sol_dw1 * outerSLH1_1
            self.addTanMatElem["K02"][:, :] += 0.5 * self.a0 * constant * prev_sol_dw1 * outerSLH2_1
            self.addTanMatElem["K11"][:, :] += self.a0 * constant * (prev_sol_du1 + prev_sol_dw1 ** 2) * outerSH11_1
            self.addTanMatElem["K12"][:, :] += self.a0 * constant * (prev_sol_du1 + prev_sol_dw1 ** 2) * outerSH12_1
            self.addTanMatElem["K21"][:, :] += self.a0 * constant * (prev_sol_du1 + prev_sol_dw1 ** 2) * outerSH21_1
            self.addTanMatElem["K22"][:, :] += self.a0 * constant * (prev_sol_du1 + prev_sol_dw1 ** 2) * outerSH22_1

        for j in range(self.DOFPN):
            for k in range(self.DOFPN):
                self.massMatElem[j::self.DOFPN, k::self.DOFPN] = self.coeffMassMatElem[f"K{j}{k}"]
                self.dampMatElem[j::self.DOFPN, k::self.DOFPN] = self.coeffDampMatElem[f"K{j}{k}"]
                self.stiffMatElem[j::self.DOFPN, k::self.DOFPN] = self.coeffStiffMatElem[f"K{j}{k}"]
                self.addtanMatElem[j::self.DOFPN, k::self.DOFPN] = self.addTanMatElem[f"K{j}{k}"]
            self.forceElem[j::self.DOFPN] = self.coeffForceElem[f"F{j}"]
            self.thermForceElem[j::self.DOFPN] = self.coeffThermForceElem[f"F{j}"]

        stiffness[:, :] = self.stiffMatElem + self.t3 * self.massMatElem + self.t6 * self.dampMatElem
        tan_mat_elem[:, :] = stiffness + self.addtanMatElem
        force[:] = self.forceElem - self.thermForceElem + self.massMatElem @ As + self.dampMatElem @ Bs
        residual_elem[:] = force - stiffness @ elem_prev_soln[1]
        return tan_mat_elem, residual_elem

    def _solve_global(self, newton_iter, ratio, t):
        self.forceGl[:] = 0.0
        self.deltaSoln[:] = 0.0
        num_elem = self.eqnsElem * self.eqnsElem
        spar_stiff = np.zeros(self.NEL * num_elem)
        spar_row = np.zeros(self.NEL * num_elem, dtype=np.int64)
        spar_col = np.zeros(self.NEL * num_elem, dtype=np.int64)

        m = 0
        for i in range(self.NEL):
            tan_mat_elem, residual_elem = self._solve_elem_level(i, ratio, t)
            tan_mat_elem[:, :], residual_elem[:] = apply_dirichlet_bc(i, newton_iter, tan_mat_elem, residual_elem, self.DBC)
            residual_elem[:] = apply_neumann_bc(i, t, residual_elem, self.NBC, ratio)
            spar_stiff[m:m + num_elem] = tan_mat_elem.flatten()
            spar_row[m:m + num_elem] = np.repeat(self.dofconn[i], self.eqnsElem)
            spar_col[m:m + num_elem] = np.tile(self.dofconn[i], self.eqnsElem)
            m += num_elem
            self.forceGl[self.dofconn[i]] += residual_elem

        self.stiffMatGl = coo_array((spar_stiff, (spar_row, spar_col)), shape=[self.eqnsGl, self.eqnsGl]).tocsr()
        self.deltaSoln[:] = np.asarray(spsolve(self.stiffMatGl, self.forceGl))

    def _spatial_update(self):
        self.newIterSoln[:] = self.prevIterSoln[1] + self.deltaSoln
        self.prevIterSoln[1, :] = self.beta * self.prevIterSoln[1] + (1 - self.beta) * self.newIterSoln

    def _temporal_update(self, prev_time, curr_time):
        self.timeSoln[curr_time, :] = self.newIterSoln
        self.timeVel[curr_time, :] = (self.t6 * (self.timeSoln[curr_time, :] - self.timeSoln[prev_time, :])
                                       - self.t7 * self.timeVel[prev_time, :] - self.t8 * self.timeAcc[prev_time, :])
        self.timeAcc[curr_time, :] = (self.t3 * (self.timeSoln[curr_time, :] - self.timeSoln[prev_time, :])
                                       - self.t4 * self.timeVel[prev_time, :] - self.t5 * self.timeAcc[prev_time, :])
        self.prevIterSoln[0, :] = self.timeSoln[curr_time, :]
        self.prevIterSoln[1, :] = self.timeSoln[curr_time, :]

    def run(self):
        """March through all load steps / timesteps, Newton-iterating the
        nonlinear residual to `newton_tolerance` (or `newton_iterations`,
        whichever comes first) at every timestep. Populates
        `convergedSoln` ([DOFPN, load_steps, timesteps+1, totalNodes]),
        `timeVel`/`timeAcc` ([timesteps+1, eqnsGl]), `countIter`, and
        `nonConvergence`.
        """
        self.countIter = np.zeros([self.loadsteps, self.timesteps + 1])
        self.endTransDefl = np.zeros([self.loadsteps, self.timesteps + 1])
        self.convergedSoln = np.zeros([self.DOFPN, self.loadsteps, self.timesteps + 1, self.totalNodes])
        self.nonConvergence = 0

        for loadstep in range(self.loadsteps):
            ratio = (loadstep + 1) / self.loadsteps
            for t in range(self.timesteps):
                time = t + 1
                for newton_iter in range(self.iterations):
                    self._solve_global(newton_iter, ratio, t)
                    self._spatial_update()
                    # Relative update norm, with an absolute fallback when
                    # the solution itself is (numerically) exactly zero --
                    # e.g. an all-Dirichlet-satisfied, zero-load state --
                    # since dividing by a zero solution norm there is a 0/0
                    # that always compares False, spuriously reporting
                    # non-convergence for a trivially-correct answer.
                    soln_norm_sq = np.sum(self.newIterSoln ** 2)
                    update_norm = np.sqrt(np.sum(self.deltaSoln ** 2))
                    check = update_norm if soln_norm_sq < 1e-300 else update_norm / np.sqrt(soln_norm_sq)
                    if check < self.convergence:
                        self.countIter[loadstep, time] = newton_iter + 1
                        self.endTransDefl[loadstep, time] = self.newIterSoln[-2]
                        self.convergedSoln[0, loadstep, time] = self.newIterSoln[0::self.DOFPN]
                        self.convergedSoln[1, loadstep, time] = self.newIterSoln[1::self.DOFPN]
                        self.convergedSoln[2, loadstep, time] = self.newIterSoln[2::self.DOFPN]
                        break
                    elif newton_iter + 1 == self.iterations:
                        self.nonConvergence = 1
                self._temporal_update(t, time)
                if self.nonConvergence == 1:
                    break
            if self.nonConvergence == 1:
                break
        return self.newIterSoln

    def natural_frequencies_hz(self, t: int = 0, ratio: float = 1.0, num_modes: int = 6) -> np.ndarray:
        """Natural frequencies (Hz) of the assembled mass/stiffness system,
        linearized at whatever state `prevIterSoln`/`timeVel`/`timeAcc`
        currently hold (the undeformed state if called before `run()`),
        under the current Dirichlet constraints. A generalized eigenvalue
        problem (K, M) independent of the time-marching/Newton code."""
        num_modes = min(int(num_modes), self.eqnsGl - 2)
        num_elem = self.eqnsElem * self.eqnsElem
        spar_k = np.zeros(self.NEL * num_elem)
        spar_k_row = np.zeros(self.NEL * num_elem, dtype=np.int64)
        spar_k_col = np.zeros(self.NEL * num_elem, dtype=np.int64)
        spar_m = np.zeros(self.NEL * num_elem)
        spar_m_row = np.zeros(self.NEL * num_elem, dtype=np.int64)
        spar_m_col = np.zeros(self.NEL * num_elem, dtype=np.int64)

        m = 0
        for i in range(self.NEL):
            self._solve_elem_level(i, ratio, t)
            self.stiffMatElem[:, :], _ = apply_dirichlet_bc(i, 0, self.stiffMatElem, self.forceElem, self.DBC)
            self.massMatElem[:, :], _ = apply_dirichlet_bc(i, 0, self.massMatElem, self.forceElem, self.DBC)
            spar_k[m:m + num_elem] = self.stiffMatElem.flatten()
            spar_k_row[m:m + num_elem] = np.repeat(self.dofconn[i], self.eqnsElem)
            spar_k_col[m:m + num_elem] = np.tile(self.dofconn[i], self.eqnsElem)
            spar_m[m:m + num_elem] = self.massMatElem.flatten()
            spar_m_row[m:m + num_elem] = np.repeat(self.dofconn[i], self.eqnsElem)
            spar_m_col[m:m + num_elem] = np.tile(self.dofconn[i], self.eqnsElem)
            m += num_elem

        K = coo_array((spar_k, (spar_k_row, spar_k_col)), shape=[self.eqnsGl, self.eqnsGl])
        M = coo_array((spar_m, (spar_m_row, spar_m_col)), shape=[self.eqnsGl, self.eqnsGl])
        eigenvalues, _ = eigsh(K, k=num_modes, M=M, sigma=0, which="LM")
        omega = np.sqrt(eigenvalues)
        return omega / (2 * np.pi)


# --------------------------------------------------------------------------
# Functional entry point
# --------------------------------------------------------------------------

def solve_nonlinear_beam_transient(
    boundary: Dict[str, Any],
    L_m: float,
    A_m2: float,
    I_m4: float,
    E_Pa: float,
    rho_kg_m3: float,
    damping_coeff: float = 0.0,
    axial_dist_load_N_per_m: float = 0.0,
    transverse_dist_load_N_per_m: float = 0.0,
    num_elements: int = 50,
    nodes_per_element: int = NODES_PER_ELEMENT,
    time_total_s: float = 1.0,
    timesteps: int = 100,
    newmark_beta: float = 0.25,
    newmark_gamma: float = 0.5,
    newton_relaxation: float = 0.0,
    newton_iterations: int = 10,
    newton_tolerance: float = 1e-3,
    load_steps: int = 1,
    initial_disp: Optional[np.ndarray] = None,
    initial_vel: Optional[np.ndarray] = None,
    reference_temperature: float = 0.0,
    temperature_field: Optional[np.ndarray] = None,
    thermal_expansion_coeff: float = 0.0,
    driving_freq_hz: float = 0.0,
    compute_natural_frequencies: bool = True,
    num_modes: int = 6,
) -> dict:
    """Solve the transient geometrically nonlinear (Von Karman) Euler-
    Bernoulli beam FEM problem described in the module docstring, for a
    caller-supplied `boundary` dict (Dirichlet + Neumann DOF specification
    -- see module docstring for its shape). `newmark_beta`/`newmark_gamma`
    use the standard Newmark-beta naming (0.25/0.5 -- the unconditionally
    stable average-acceleration method -- by default).

    Returns a dict with time-history arrays (indexed [time, node]):
    z, time_s, u_m, w_m, theta_rad, w_vel_ms, w_acc_ms2, iterations,
    nonconverged, and (if requested) natural_frequencies_hz.
    """
    engine = NonlinearBeamNewmarkFEM(
        num_elements=num_elements, nodes_per_element=nodes_per_element, length=L_m,
        area=A_m2, moment_of_inertia=I_m4, E=E_Pa, density=rho_kg_m3,
        damping_coeff=damping_coeff, axial_dist_load=axial_dist_load_N_per_m,
        transverse_dist_load=transverse_dist_load_N_per_m, boundary=boundary,
        total_time=time_total_s, timesteps=timesteps, newmark_beta=newmark_beta,
        newmark_gamma=newmark_gamma, newton_relaxation=newton_relaxation,
        newton_iterations=newton_iterations, newton_tolerance=newton_tolerance,
        load_steps=load_steps, initial_disp=initial_disp, initial_vel=initial_vel,
        reference_temperature=reference_temperature, temperature_field=temperature_field,
        thermal_expansion_coeff=thermal_expansion_coeff, driving_freq_hz=driving_freq_hz,
    )

    natural_frequencies_hz = np.array([])
    if compute_natural_frequencies:
        try:
            natural_frequencies_hz = engine.natural_frequencies_hz(t=0, ratio=1.0, num_modes=num_modes)
        except Exception:
            natural_frequencies_hz = np.array([])

    engine.run()

    time_s = np.linspace(0.0, time_total_s, timesteps + 1)
    u = engine.convergedSoln[0, -1, :, :]
    w = engine.convergedSoln[1, -1, :, :]
    theta = engine.convergedSoln[2, -1, :, :]
    # Newmark internally tracks velocity/acceleration for every DOF; slice
    # out the transverse (w) component at every node.
    w_vel = engine.timeVel[:, DOF_TRANSVERSE::DOFPN]
    w_acc = engine.timeAcc[:, DOF_TRANSVERSE::DOFPN]

    return {
        "z": engine.globCoord,
        "time_s": time_s,
        "u_m": u,
        "w_m": w,
        "theta_rad": theta,
        "w_vel_ms": w_vel,
        "w_acc_ms2": w_acc,
        "iterations": engine.countIter[-1, :],
        "nonconverged": bool(engine.nonConvergence),
        "natural_frequencies_hz": natural_frequencies_hz,
        "L": L_m,
    }


def acceleration_from_displacement(w: np.ndarray, dt: float) -> np.ndarray:
    """Convenience/diagnostic: second time-derivative of a displacement
    time-history via a proper 3-point second-derivative stencil (reuses
    `beam_bvp_fdm.second_derivative`, which is exact for a quadratic
    displacement history and doesn't compound `np.gradient`'s first-order
    edge truncation the way differentiating twice would). Prefer the
    engine's own Newmark-tracked `w_acc_ms2` when available -- this is for
    when only a displacement history is on hand (e.g. cross-checking a
    probe location's acceleration against the internally-tracked one)."""
    return second_derivative(w, dt)


# --------------------------------------------------------------------------
# SolverBase / registry wrapper
# --------------------------------------------------------------------------

@SolverRegistry.register(
    name="nonlinear_beam_fem",
    family="pde",
    description="Spectral-element FEM for a geometrically nonlinear (Von Karman) transient "
                "Euler-Bernoulli beam, Newmark-beta time integration with per-timestep "
                "Newton-Raphson -- generic caller-supplied geometry/material/mesh/BC, no fixed topology.",
    tags=["fem", "spectral-element", "beam", "nonlinear", "von-karman", "newmark", "transient"],
)
class NonlinearBeamFEMSolver(SolverBase):
    """Thin `SolverBase`/registry wrapper. `solve_nonlinear_beam_transient`
    and `NonlinearBeamNewmarkFEM` are the primary entry points and can be
    used directly without this wrapper."""

    def __init__(self, num_elements: int = 50, nodes_per_element: int = NODES_PER_ELEMENT):
        super().__init__()
        self.num_elements = int(num_elements)
        self.nodes_per_element = int(nodes_per_element)

    def forward(
        self,
        boundary: Dict[str, Any],
        L_m: float,
        A_m2: float,
        I_m4: float,
        E_Pa: float,
        rho_kg_m3: float,
        damping_coeff: float = 0.0,
        time_total_s: float = 1.0,
        timesteps: int = 100,
        **kwargs,
    ) -> SolverOutput:
        out = solve_nonlinear_beam_transient(
            boundary=boundary, L_m=L_m, A_m2=A_m2, I_m4=I_m4, E_Pa=E_Pa,
            rho_kg_m3=rho_kg_m3, damping_coeff=damping_coeff,
            num_elements=self.num_elements, nodes_per_element=self.nodes_per_element,
            time_total_s=time_total_s, timesteps=timesteps, **kwargs,
        )
        field = np.stack([out["u_m"], out["w_m"], out["theta_rad"]], axis=0).astype(np.float32)
        return SolverOutput(
            result=torch.from_numpy(field),
            losses={"residual": torch.tensor(1.0 if out["nonconverged"] else 0.0)},
            extras={
                "z": out["z"].astype(np.float32),
                "time_s": out["time_s"].astype(np.float32),
                "w_vel_ms": out["w_vel_ms"].astype(np.float32),
                "w_acc_ms2": out["w_acc_ms2"].astype(np.float32),
                "natural_frequencies_hz": out["natural_frequencies_hz"].astype(np.float32),
                "iterations": out["iterations"],
                "method": "spectral_element_newmark_von_karman_beam",
            },
        )
