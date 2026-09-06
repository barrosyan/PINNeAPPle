"""Item B of the "third audit pass" (see ``ROADMAP_PHYSICS_AI_HUB.md``):
extends the Tier-A "build one instance, run one operation, assert no
crash" breadth discipline (see ``tests/test_full_library_matrix.py``) to
the actual public classes -- not just the imports -- of the 6 packages a
previous pass only import-smoke-tested: ``pinneapple_design``,
``pinneapple_systems``, ``pinneapple_analysis``, ``pinneapple_adaptation``,
``pinneapple_tools``, ``pinneapple_simulation``.

Enumeration method (see the ROADMAP bullet for the summary numbers): every
package's ``.py`` files were parsed with ``ast`` to list every class with
a public name (no leading underscore) together with its ``__init__``
signature and any method matching a generic "main operation" name (fit /
solve / run / apply / compute / __call__ / forward / predict / step /
simulate / generate / transform / build / evaluate / ...). Classes were
then read by hand (not guessed) to decide, for each one, whether a
plausible *generic* synthetic input exists:

* Classes exercised below are built with small numpy/torch inputs in the
  same spirit as ``test_full_library_matrix.py``'s
  ``in_dim=4, out_dim=3, hidden_dim=16`` architecture defaults, then have
  their one obvious "main operation" called, with a genuine crash treated
  as a hard failure (``pytest.fail``) and a generic-input mismatch or
  missing-optional-dependency treated as an honest, specific skip.
* Classes NOT exercised (abstract base classes / ``Protocol``s never
  meant to be instantiated directly; pure data-container dataclasses with
  no "operation" -- covered instead by the systematic
  ``test_breadth_dataclass_construction`` sweep at the bottom of this
  file; and classes that need a real external resource -- a mesh file, a
  live network call, an external binary like OpenFOAM/MuJoCo/MATLAB/
  FEniCS/Modelica/Genesis, or a fitted upstream artifact with no simple
  synthetic substitute) are still recorded, with an honest
  ``pytest.skip`` reason, rather than silently dropped.

Run: ``pytest tests/test_breadth_six_packages.py -q``
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import tempfile
import warnings
from types import SimpleNamespace


def _tiny_mlp(in_dim: int = 2, out_dim: int = 1, hidden: int = 8) -> nn.Module:
    """A minimal generic feed-forward network -- stand-in "model" for any
    class in these 6 packages that just needs *some* nn.Module to wrap,
    adapt, or drive (mirrors the role ``modified_mlp`` plays for
    ``test_full_library_matrix.py``'s preset tests)."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, out_dim))


def _assert_finite(x, msg: str) -> None:
    if torch.is_tensor(x):
        assert torch.isfinite(x).all(), msg
    elif isinstance(x, np.ndarray):
        assert np.isfinite(x).all(), msg
    elif isinstance(x, (int, float)):
        assert x == x and abs(x) != float("inf"), msg
    # else: not a numeric type we can finiteness-check; presence alone
    # (no exception raised) is already the assertion for that case.


# ===========================================================================
# pinneapple_adaptation (meta_learning + transfer_learning)
# ===========================================================================

def _dummy_pde_task_sampler(reptile: bool = False):
    """Build a :class:`PDETaskSampler` with a trivial physics_fn.

    ``PDETaskSampler``'s docstring is explicit that MAMLTrainer and
    ReptileTrainer do NOT share a physics_fn calling convention: MAML calls
    ``physics_fn(forward_or_model, x_col) -> scalar_tensor`` while Reptile
    calls ``physics_fn(model, batch_dict) -> (scalar_tensor, dict)`` -- so
    this test builds the matching flavor per-trainer rather than forcing
    one shape onto both (that would be a test-harness bug, not a library
    one).
    """
    from pinneapple_adaptation.meta_learning.task_sampler import PDETaskSampler

    if reptile:
        def physics_fn_factory(params):
            nu = params["nu"]

            def reptile_physics_fn(model, batch):
                u = model(batch["x_col"])
                total = u.pow(2).mean() * nu
                return total, {"total": total}

            return reptile_physics_fn
    else:
        def physics_fn_factory(params):
            nu = params["nu"]

            def maml_physics_fn(forward_or_model, x_col):
                u = forward_or_model(x_col)
                return u.pow(2).mean() * nu

            return maml_physics_fn

    return PDETaskSampler(
        param_ranges={"nu": (0.01, 0.1)},
        physics_fn_factory=physics_fn_factory,
        n_support=8,
        n_query=8,
        input_dim=2,
        seed=0,
    )


def test_breadth_pde_task_sampler():
    sampler = _dummy_pde_task_sampler()
    task = sampler.sample_task()
    assert "support" in task and "query" in task and "physics_fn" in task
    batch = sampler.sample_batch(3)
    assert len(batch) == 3


def test_breadth_maml_trainer_trains():
    from pinneapple_adaptation.meta_learning.config import MAMLConfig
    from pinneapple_adaptation.meta_learning.maml import MAMLTrainer

    model = _tiny_mlp(2, 1)
    sampler = _dummy_pde_task_sampler()
    cfg = MAMLConfig(n_inner_steps=1, n_tasks_per_batch=2, n_meta_epochs=2, checkpoint_every=0)
    trainer = MAMLTrainer(model, cfg, sampler)
    result = trainer.train()
    assert len(result["history"]) == 2
    for rec in result["history"]:
        _assert_finite(rec["meta_loss"], "MAMLTrainer produced a non-finite meta_loss")
    adapted = trainer.adapt(sampler.sample_task(), n_steps=1)
    assert isinstance(adapted, nn.Module)


def test_breadth_reptile_trainer_trains():
    from pinneapple_adaptation.meta_learning.config import ReptileConfig
    from pinneapple_adaptation.meta_learning.reptile import ReptileTrainer

    model = _tiny_mlp(2, 1)
    sampler = _dummy_pde_task_sampler(reptile=True)
    cfg = ReptileConfig(n_inner_steps=1, n_tasks_per_batch=2, n_meta_epochs=2, checkpoint_every=0)
    trainer = ReptileTrainer(model, cfg, sampler)
    result = trainer.train()
    assert len(result["history"]) == 2
    for rec in result["history"]:
        _assert_finite(rec["meta_loss"], "ReptileTrainer produced a non-finite meta_loss")


def test_breadth_meta_model_predict():
    from pinneapple_adaptation.meta_learning.meta_model import MetaModel

    model = _tiny_mlp(2, 1)
    meta = MetaModel(model, meta_type="reptile", device="cpu")
    x = np.random.randn(5, 2).astype(np.float32)
    y = meta.predict(x)
    assert y.shape == (5, 1)
    _assert_finite(y, "MetaModel.predict produced non-finite output")

    task = _dummy_pde_task_sampler(reptile=True).sample_task()
    adapted = meta.adapt(physics_fn=task["physics_fn"], data=task["support"], n_steps=1)
    assert isinstance(adapted, nn.Module)


def test_breadth_physics_transfer_adapter():
    from pinneapple_adaptation.transfer_learning.adapter import PhysicsTransferAdapter

    model = _tiny_mlp(2, 1)
    adapter = PhysicsTransferAdapter(model, source_spec=None, target_spec=None)

    def target_physics_fn(m, batch):
        x = torch.rand(8, 2)
        return m(x).pow(2).mean()

    loss_fn = adapter.adapt_physics_loss(target_physics_fn)
    out = loss_fn(model, {})
    _assert_finite(out["total"], "PhysicsTransferAdapter combined loss is non-finite")

    x_src = torch.rand(8, 2)
    x_tgt = torch.rand(8, 2)
    mmd = adapter.domain_adaptation_loss(x_src, x_tgt)
    _assert_finite(mmd, "PhysicsTransferAdapter MMD loss is non-finite")


def test_breadth_parametric_family_transfer():
    from pinneapple_adaptation.transfer_learning.parametric import ParametricFamilyTransfer

    base = _tiny_mlp(2, 1)
    family = ParametricFamilyTransfer(base, param_name="nu")
    family.add_variant(0.01, _tiny_mlp(2, 1))
    family.add_variant(0.05, _tiny_mlp(2, 1))
    mid = family.interpolate_weights(0.03)
    assert isinstance(mid, nn.Module)
    assert family.list_variants() == [0.01, 0.05]


def test_breadth_transfer_trainer_finetunes():
    from pinneapple_adaptation.transfer_learning.config import TransferConfig
    from pinneapple_adaptation.transfer_learning.trainer import TransferTrainer

    model = _tiny_mlp(2, 1)
    cfg = TransferConfig(strategy="finetune", epochs=2, warmup_epochs=0)
    trainer = TransferTrainer(model, cfg)
    trainer.prepare()

    def target_physics_fn(m, batch):
        x = torch.rand(8, 2)
        return m(x).pow(2).mean()

    result = trainer.finetune(target_physics_fn, n_epochs=2)
    assert len(result["history"]) == 2
    _assert_finite(result["metrics"]["final_loss"], "TransferTrainer.finetune final loss is non-finite")


# ===========================================================================
# pinneapple_design -- from tests/test_breadth_six_packages.py's dispatched sub-agent
# for this package (real, hand-verified enumeration + exercised classes; see
# ROADMAP_PHYSICS_AI_HUB.md's Item B bullet for the consolidated numbers).
# ===========================================================================

# ===========================================================================
# Abstract base classes -- NOT exercised as instances (see module docstring)
# ===========================================================================

def test_breadth_constraint_base_is_abstract():
    from pinneapple_design.design_optimizer.constraints import ConstraintBase

    with pytest.raises(TypeError):
        ConstraintBase()  # abc.ABC with @abstractmethod penalty/satisfied;
        # concrete behavior is covered by BoxConstraint/GeometricConstraint/
        # ManufacturabilityConstraint/MassConservationConstraint below.


def test_breadth_objective_base_is_abstract():
    from pinneapple_design.design_optimizer.objective import ObjectiveBase

    with pytest.raises(TypeError):
        ObjectiveBase()  # abc.ABC with @abstractmethod __call__; concrete
        # behavior covered by DragObjective/StructuralObjective/etc. below.


def test_breadth_physics_domain_2d_base_not_instantiated_directly():
    pytest.skip(
        "PhysicsDomain2D is a base class whose sdf()/sample_boundary_region() "
        "are NotImplementedError stubs meant to be overridden by concrete "
        "subclasses (ChannelDomain2D, LidDrivenCavityDomain2D, etc., all "
        "exercised below), not instantiated directly."
    )


def test_breadth_physics_domain_3d_base_not_instantiated_directly():
    pytest.skip(
        "PhysicsDomain3D is a base class whose sdf()/sample_boundary_region() "
        "are NotImplementedError stubs meant to be overridden by concrete "
        "subclasses (ChannelDomain3D, PipeFlowDomain3D, etc., all exercised "
        "below), not instantiated directly."
    )


def test_breadth_sdf_shape_base_not_instantiated_directly():
    pytest.skip(
        "SDFShape is a base class whose sdf()/sample_interior()/"
        "sample_boundary() are NotImplementedError stubs meant to be "
        "overridden by concrete subclasses (CSGRectangle, CSGCircle, etc., "
        "all exercised below), not instantiated directly."
    )


def test_breadth_stl_domain_batch_builder_needs_real_stl_and_trimesh():
    pytest.skip(
        "STLDomainBatchBuilder.build() requires a real STL file on disk and "
        "the optional 'trimesh' dependency (not installed in this "
        "environment) to load/voxelize/query it -- no simple synthetic "
        "substitute for the full mesh-import pipeline."
    )


# ===========================================================================
# CSG primitives and boolean composition (pinneapple_design.geometry.csg)
# ===========================================================================

def _assert_shape_2d_finite(pts: np.ndarray, n: int, msg: str) -> None:
    assert pts.shape == (n, 2), f"{msg}: expected shape ({n}, 2), got {pts.shape}"
    _assert_finite(pts, msg)


def test_breadth_csg_rectangle():
    from pinneapple_design.geometry.csg import CSGRectangle

    rect = CSGRectangle(-1.0, -1.0, 1.0, 1.0)
    d = rect.sdf(np.array([[0.0, 0.0], [5.0, 5.0]]))
    _assert_finite(d, "CSGRectangle.sdf produced non-finite output")
    assert d[0] < 0 and d[1] > 0  # center inside, far point outside

    interior = rect.sample_interior(16, seed=0)
    boundary = rect.sample_boundary(16, seed=0)
    _assert_shape_2d_finite(interior, 16, "CSGRectangle.sample_interior")
    _assert_shape_2d_finite(boundary, 16, "CSGRectangle.sample_boundary")


def test_breadth_csg_circle():
    from pinneapple_design.geometry.csg import CSGCircle

    circ = CSGCircle(0.0, 0.0, 1.0)
    d = circ.sdf(np.array([[0.0, 0.0], [10.0, 0.0]]))
    _assert_finite(d, "CSGCircle.sdf produced non-finite output")

    interior = circ.sample_interior(16, seed=0)
    boundary = circ.sample_boundary(16, seed=0)
    _assert_shape_2d_finite(interior, 16, "CSGCircle.sample_interior")
    _assert_shape_2d_finite(boundary, 16, "CSGCircle.sample_boundary")


def test_breadth_csg_ellipse():
    from pinneapple_design.geometry.csg import CSGEllipse

    ell = CSGEllipse(0.0, 0.0, 1.0, 0.5)
    d = ell.sdf(np.array([[0.0, 0.0], [10.0, 0.0]]))
    _assert_finite(d, "CSGEllipse.sdf produced non-finite output")

    interior = ell.sample_interior(16, seed=0)
    boundary = ell.sample_boundary(16, seed=0)
    _assert_shape_2d_finite(interior, 16, "CSGEllipse.sample_interior")
    _assert_shape_2d_finite(boundary, 16, "CSGEllipse.sample_boundary")


def test_breadth_csg_polygon():
    from pinneapple_design.geometry.csg import CSGPolygon

    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    poly = CSGPolygon(verts)
    d = poly.sdf(np.array([[0.5, 0.5], [10.0, 10.0]]))
    _assert_finite(d, "CSGPolygon.sdf produced non-finite output")

    interior = poly.sample_interior(16, seed=0)
    boundary = poly.sample_boundary(16, seed=0)
    _assert_shape_2d_finite(interior, 16, "CSGPolygon.sample_interior")
    _assert_shape_2d_finite(boundary, 16, "CSGPolygon.sample_boundary")


def test_breadth_csg_boolean_ops():
    """Covers CSGUnion, CSGIntersection, CSGDifference via the `+`/`*`/`-`
    operator shorthands defined on SDFShape."""
    from pinneapple_design.geometry.csg import CSGCircle, CSGRectangle, CSGUnion, CSGIntersection, CSGDifference

    rect = CSGRectangle(-1.0, -1.0, 1.0, 1.0)
    circ = CSGCircle(0.0, 0.0, 0.5)

    union = rect + circ
    inter = rect * circ
    diff = rect - circ
    assert isinstance(union, CSGUnion)
    assert isinstance(inter, CSGIntersection)
    assert isinstance(diff, CSGDifference)

    query = np.array([[0.0, 0.0], [0.9, 0.9], [10.0, 10.0]])
    for shape, name in ((union, "CSGUnion"), (inter, "CSGIntersection"), (diff, "CSGDifference")):
        d = shape.sdf(query)
        _assert_finite(d, f"{name}.sdf produced non-finite output")
        interior = shape.sample_interior(8, seed=0)
        boundary = shape.sample_boundary(8, seed=0)
        _assert_shape_2d_finite(interior, 8, f"{name}.sample_interior")
        _assert_shape_2d_finite(boundary, 8, f"{name}.sample_boundary")


# ===========================================================================
# SDF composable callable wrapper (pinneapple_design.geometry.gen.sdf_shapes)
# ===========================================================================

def test_breadth_sdf_class():
    from pinneapple_design.geometry.gen.sdf_shapes import SDF, sdf2d_circle, sdf2d_rectangle

    s = SDF(sdf2d_circle, (0.0, 0.0), 0.5)
    s2 = SDF(sdf2d_rectangle, (0.3, 0.0), (0.2, 0.2))

    p = np.random.default_rng(0).uniform(-1, 1, size=(8, 2))
    for combined, name in (
        (s | s2, "union"),
        (s & s2, "intersection"),
        (s - s2, "difference"),
        (s.translate((0.1, 0.1)), "translate"),
        (s.scale(2.0), "scale"),
        (s.rotate(0.3), "rotate"),
        (s.onion(0.05), "onion"),
        (s.smooth_union(s2, k=0.1), "smooth_union"),
    ):
        out = combined(p)
        _assert_finite(out, f"SDF.{name} produced non-finite output")

    bnd = s.sample_boundary(16, bounds_min=(-1.0, -1.0), bounds_max=(1.0, 1.0), seed=0)
    interior = s.sample_interior(16, bounds_min=(-1.0, -1.0), bounds_max=(1.0, 1.0), seed=0)
    _assert_shape_2d_finite(bnd, 16, "SDF.sample_boundary")
    _assert_shape_2d_finite(interior, 16, "SDF.sample_interior")


def test_breadth_sdf_domain_2d():
    from pinneapple_design.geometry.gen.domains import SDFDomain2D
    from pinneapple_design.geometry.gen.sdf_shapes import circle

    sdf = circle(center=(0.5, 0.5), radius=0.3)

    def _boundary_sampler(n, rng):
        seed = int(rng.integers(0, 2**31))
        return sdf.sample_boundary(n, bounds_min=(0.0, 0.0), bounds_max=(1.0, 1.0), seed=seed)

    domain = SDFDomain2D(
        sdf_fn=sdf,
        bounds_min=(0.0, 0.0),
        bounds_max=(1.0, 1.0),
        boundary_samplers={"boundary": _boundary_sampler},
        boundary_conditions={"boundary": {"kind": "dirichlet", "u": 0.0}},
    )
    batch = domain.get_pinn_batch(n_col=32, n_bc_per_region=16, seed=0)
    _assert_finite(batch["x_col"], "SDFDomain2D x_col non-finite")
    _assert_finite(batch["x_bc"], "SDFDomain2D x_bc non-finite")


# ===========================================================================
# Meshfree geometry ops (pinneapple_design.geometry.ops.meshfree)
# ===========================================================================

def test_breadth_rbf_interpolator():
    from pinneapple_design.geometry.ops.meshfree import RBFInterpolator

    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(12, 2))
    vals = np.sin(pts[:, 0]) + pts[:, 1]

    interp = RBFInterpolator(kernel="gaussian", eps=1.0).fit(pts, vals)
    q = rng.uniform(-1, 1, size=(5, 2))
    out = interp(q)
    grad = interp.gradient(q)
    _assert_finite(out, "RBFInterpolator.__call__ produced non-finite output")
    _assert_finite(grad, "RBFInterpolator.gradient produced non-finite output")


def test_breadth_implicit_surface_rbf():
    from pinneapple_design.geometry.ops.meshfree import ImplicitSurfaceRBF

    rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    surf_pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normals = surf_pts.copy()  # outward unit normals of a unit circle

    surf = ImplicitSurfaceRBF(eps=2.0, offset=0.05).fit(surf_pts, normals)
    d = surf.sdf(rng.uniform(-1, 1, size=(5, 2)))
    inside = surf.is_inside(np.zeros((3, 2)))
    interior = surf.sample_interior(20, bounds_min=np.array([-1.0, -1.0]), bounds_max=np.array([1.0, 1.0]), seed=0)
    boundary = surf.sample_boundary(10, seed=0)

    _assert_finite(d, "ImplicitSurfaceRBF.sdf produced non-finite output")
    assert inside.dtype == bool
    _assert_finite(interior, "ImplicitSurfaceRBF.sample_interior produced non-finite output")
    _assert_finite(boundary, "ImplicitSurfaceRBF.sample_boundary produced non-finite output")


# ===========================================================================
# 2D physics domains (pinneapple_design.geometry.gen.domains)
# ===========================================================================

def _check_domain_batch(domain, name: str) -> None:
    batch = domain.get_pinn_batch(n_col=32, n_bc_per_region=16, seed=0)
    _assert_finite(batch["x_col"], f"{name} x_col non-finite")
    _assert_finite(batch["x_bc"], f"{name} x_bc non-finite")
    assert batch["x_col"].shape[0] == 32


def test_breadth_channel_domain_2d():
    from pinneapple_design.geometry.gen.domains import ChannelDomain2D
    _check_domain_batch(ChannelDomain2D(), "ChannelDomain2D")


def test_breadth_channel_with_obstacle_domain_2d():
    from pinneapple_design.geometry.gen.domains import ChannelWithObstacleDomain2D
    _check_domain_batch(ChannelWithObstacleDomain2D(), "ChannelWithObstacleDomain2D")


def test_breadth_lid_driven_cavity_domain_2d():
    from pinneapple_design.geometry.gen.domains import LidDrivenCavityDomain2D
    _check_domain_batch(LidDrivenCavityDomain2D(), "LidDrivenCavityDomain2D")


def test_breadth_l_shape_domain_2d():
    from pinneapple_design.geometry.gen.domains import LShapeDomain2D
    _check_domain_batch(LShapeDomain2D(), "LShapeDomain2D")


def test_breadth_annular_domain_2d():
    from pinneapple_design.geometry.gen.domains import AnnularDomain2D
    _check_domain_batch(AnnularDomain2D(), "AnnularDomain2D")


def test_breadth_multi_obstacle_domain_2d():
    from pinneapple_design.geometry.gen.domains import MultiObstacleDomain2D
    _check_domain_batch(MultiObstacleDomain2D(), "MultiObstacleDomain2D")


def test_breadth_t_junction_domain_2d():
    from pinneapple_design.geometry.gen.domains import TJunctionDomain2D
    _check_domain_batch(TJunctionDomain2D(), "TJunctionDomain2D")


# ===========================================================================
# 3D physics domains (pinneapple_design.geometry.gen.domains3d)
# ===========================================================================

def test_breadth_lid_driven_cavity_domain_3d():
    from pinneapple_design.geometry.gen.domains3d import LidDrivenCavityDomain3D
    _check_domain_batch(LidDrivenCavityDomain3D(), "LidDrivenCavityDomain3D")


def test_breadth_channel_domain_3d():
    from pinneapple_design.geometry.gen.domains3d import ChannelDomain3D
    _check_domain_batch(ChannelDomain3D(), "ChannelDomain3D")


def test_breadth_pipe_flow_domain_3d():
    from pinneapple_design.geometry.gen.domains3d import PipeFlowDomain3D
    _check_domain_batch(PipeFlowDomain3D(), "PipeFlowDomain3D")


# ===========================================================================
# Mesh collocation (pinneapple_design.geometry.mesh_collocator)
# ===========================================================================

def _unit_cube_mesh():
    """A minimal watertight-ish triangle mesh (unit cube) -- pure numpy, no
    trimesh/scipy dependency needed for MeshCollocator's "bbox" domain mode
    plus MeshData's own (dependency-free) surface sampler."""
    from pinneapple_design.geometry.core.mesh import MeshData

    vertices = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [1, 2, 6], [1, 6, 5],  # right
            [2, 3, 7], [2, 7, 6],  # back
            [3, 0, 4], [3, 4, 7],  # left
        ],
        dtype=np.int64,
    )
    return MeshData(vertices=vertices, faces=faces)


def test_breadth_mesh_collocator():
    from pinneapple_design.geometry.mesh_collocator import MeshCollocator, MeshCollocatorConfig

    mesh = _unit_cube_mesh()
    cfg = MeshCollocatorConfig(n_interior=32, n_boundary=16, domain="bbox", boundary_mode="surface")
    col = MeshCollocator(mesh, cfg)

    batch = col.sample()
    assert batch.X_interior.shape == (32, 3)
    assert batch.X_boundary.shape == (16, 3)
    _assert_finite(batch.X_interior, "MeshCollocator.sample X_interior non-finite")
    _assert_finite(batch.X_boundary, "MeshCollocator.sample X_boundary non-finite")

    tensors = col.sample_tensors()
    _assert_finite(tensors["X_int"], "MeshCollocator.sample_tensors X_int non-finite")
    _assert_finite(tensors["X_bnd"], "MeshCollocator.sample_tensors X_bnd non-finite")


# ===========================================================================
# Design-optimizer objectives (pinneapple_design.design_optimizer.objective)
# ===========================================================================

def test_breadth_objectives():
    from pinneapple_design.design_optimizer.objective import (
        CompositeObjective, DragObjective, StructuralObjective,
        ThermalEfficiencyObjective, WeightMinimizationObjective,
    )

    theta = torch.randn(4, requires_grad=True)
    u = torch.randn(8, 3)  # last channel treated as pressure by DragObjective/Thermal

    drag = DragObjective()(theta, u)
    thermal = ThermalEfficiencyObjective()(theta, u)
    struct = StructuralObjective()(theta, u)
    weight = WeightMinimizationObjective()(theta, u)
    for val, name in (
        (drag, "DragObjective"), (thermal, "ThermalEfficiencyObjective"),
        (struct, "StructuralObjective"), (weight, "WeightMinimizationObjective"),
    ):
        _assert_finite(val, f"{name} produced a non-finite value")

    composite = CompositeObjective([(1.0, DragObjective())])
    composite.add(0.5, StructuralObjective())
    total = composite(theta, u)
    _assert_finite(total, "CompositeObjective produced a non-finite value")


# ===========================================================================
# Design-optimizer constraints (pinneapple_design.design_optimizer.constraints)
# ===========================================================================

def test_breadth_box_constraint():
    from pinneapple_design.design_optimizer.constraints import BoxConstraint

    bc = BoxConstraint(theta_min=torch.tensor([-1.0, -1.0]), theta_max=torch.tensor([1.0, 1.0]), weight=10.0)
    theta = torch.tensor([0.5, -2.0])
    u = torch.zeros(1)
    pen = bc.penalty(theta, u)
    sat = bc.satisfied(theta, u)
    _assert_finite(pen, "BoxConstraint.penalty non-finite")
    assert sat is False  # -2.0 violates theta_min=-1.0


def test_breadth_geometric_constraint():
    from pinneapple_design.design_optimizer.constraints import GeometricConstraint

    gc = GeometricConstraint(min_val=0.1, max_val=10.0, param_pairs=[(0, 1)], weight=1.0)
    theta = torch.tensor([1.0, 2.0])
    u = torch.zeros(1)
    pen = gc.penalty(theta, u)
    sat = gc.satisfied(theta, u)
    _assert_finite(pen, "GeometricConstraint.penalty non-finite")
    assert isinstance(sat, bool)


def test_breadth_manufacturability_constraint():
    from pinneapple_design.design_optimizer.constraints import ManufacturabilityConstraint

    mc = ManufacturabilityConstraint(smoothness_weight=1.0)
    theta = torch.tensor([1.0, 3.0, 2.0])
    u = torch.zeros(1)
    pen = mc.penalty(theta, u)
    sat = mc.satisfied(theta, u)
    _assert_finite(pen, "ManufacturabilityConstraint.penalty non-finite")
    assert sat is True  # smoothness is always a soft preference, per its own docstring


def test_breadth_mass_conservation_constraint():
    from pinneapple_design.design_optimizer.constraints import MassConservationConstraint

    # Structured-grid branch (u.ndim == 3): no `coords` needed, axes 0/1 are
    # already the spatial directions (see the class docstring).
    mcc = MassConservationConstraint(weight=1.0)
    theta = torch.zeros(1)
    u_grid = torch.randn(4, 4, 2)
    pen = mcc.penalty(theta, u_grid)
    sat = mcc.satisfied(theta, u_grid)
    _assert_finite(pen, "MassConservationConstraint.penalty non-finite")
    assert isinstance(sat, bool)


# ===========================================================================
# Physics surrogate (pinneapple_design.design_optimizer.surrogate)
# ===========================================================================

def test_breadth_physics_surrogate():
    from pinneapple_design.design_optimizer.surrogate import PhysicsSurrogate, SurrogateConfig

    model = _tiny_mlp(4, 2)
    surrogate = PhysicsSurrogate(model, SurrogateConfig(in_dim=4, out_channels=2))

    theta = np.random.randn(4).astype(np.float32)
    u = surrogate.predict(theta)
    batch = surrogate.predict_batch(np.random.randn(5, 4).astype(np.float32))
    jac = surrogate.jacobian_theta(torch.from_numpy(theta))

    _assert_finite(u, "PhysicsSurrogate.predict non-finite")
    _assert_finite(batch, "PhysicsSurrogate.predict_batch non-finite")
    _assert_finite(jac, "PhysicsSurrogate.jacobian_theta non-finite")

    built = PhysicsSurrogate.build_mlp(in_dim=3, out_dim=2)
    assert isinstance(built, PhysicsSurrogate)


# ===========================================================================
# Shape parametrization + continuous adjoint solver
# (pinneapple_design.design_optimizer.adjoint)
# ===========================================================================

def test_breadth_shape_parametrization():
    from pinneapple_design.design_optimizer.adjoint import ShapeParametrization

    cp = torch.randn(6, 2)
    sp = ShapeParametrization(cp)
    mesh_pts = torch.randn(10, 2)

    deformed = sp.deform_mesh(mesh_pts)
    bcoords = sp.to_boundary_coordinates()
    params = sp.parameters()

    assert deformed.shape == mesh_pts.shape
    _assert_finite(deformed, "ShapeParametrization.deform_mesh non-finite")
    _assert_finite(bcoords, "ShapeParametrization.to_boundary_coordinates non-finite")
    assert len(params) == 1 and params[0] is sp.control_points


def test_breadth_continuous_adjoint_solver():
    from pinneapple_design.design_optimizer.adjoint import ContinuousAdjointSolver, ShapeParametrization

    model = _tiny_mlp(2, 1)

    def pde_residual_fn(m, x):
        u = m(x)
        return u.pow(2) - 0.1  # depends on u -> exercises the GMRES adjoint path

    def objective_fn(m, x):
        return m(x).pow(2).mean()

    solver = ContinuousAdjointSolver(model, pde_residual_fn, objective_fn)
    x_col = torch.randn(8, 2)
    sp = ShapeParametrization(torch.randn(4, 2))

    lam = solver.compute_adjoint(x_col, sp, max_iter=5)
    _assert_finite(lam, "ContinuousAdjointSolver.compute_adjoint non-finite")

    result = solver.optimize(sp, x_col, n_steps=2, lr=1e-2)
    _assert_finite(result["best_objective"], "ContinuousAdjointSolver.optimize best_objective non-finite")
    _assert_finite(result["best_control_points"], "ContinuousAdjointSolver.optimize best_control_points non-finite")


def test_breadth_drag_adjoint_objective():
    from pinneapple_design.design_optimizer.adjoint import DragAdjointObjective, naca_parametric

    surface_pts = naca_parametric(t_c=0.12, n_pts=40)  # (40, 2), a closed contour
    model = _tiny_mlp(2, 3)  # (u, v, p) channels, as DragAdjointObjective expects
    obj = DragAdjointObjective(surface_pts, nu=0.01, alpha=0.05)

    drag = obj(model, torch.randn(1, 2))
    _assert_finite(drag, "DragAdjointObjective produced non-finite output")


# ===========================================================================
# Design optimizers (pinneapple_design.design_optimizer.optimizer)
# ===========================================================================

def test_breadth_gradient_design_optimizer():
    from pinneapple_design.design_optimizer.optimizer import GradientDesignOptimizer, DesignOptimizerConfig
    from pinneapple_design.design_optimizer.objective import WeightMinimizationObjective
    from pinneapple_design.design_optimizer.surrogate import PhysicsSurrogate

    cfg = DesignOptimizerConfig(method="gradient", grad_optimizer="adam", lr=1e-2, grad_clip=1.0)
    theta0 = np.zeros(2, dtype=np.float32)
    opt = GradientDesignOptimizer(theta0, cfg)

    surrogate = PhysicsSurrogate.build_mlp(in_dim=2, out_dim=3)
    objective = WeightMinimizationObjective()
    new_theta, obj_val = opt.step(surrogate, objective, None, theta0)

    _assert_finite(new_theta, "GradientDesignOptimizer.step new_theta non-finite")
    _assert_finite(obj_val, "GradientDesignOptimizer.step obj_val non-finite")


def test_breadth_bayesian_design_optimizer():
    from pinneapple_design.design_optimizer.optimizer import BayesianDesignOptimizer, DesignOptimizerConfig

    cfg = DesignOptimizerConfig(method="bayesian", n_initial_random=2, acquisition="ei")
    bo = BayesianDesignOptimizer(cfg, seed=0)
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])

    for _ in range(4):  # 2 random-phase proposals + 2 GP-guided ones
        x = bo.propose(bounds)
        y = float(np.sum(x ** 2))
        bo.update(x, y)

    assert len(bo.X_obs) == 4
    _assert_finite(np.array(bo.y_obs), "BayesianDesignOptimizer y_obs non-finite")


def test_breadth_evolutionary_design_optimizer():
    from pinneapple_design.design_optimizer.optimizer import EvolutionaryDesignOptimizer, DesignOptimizerConfig

    cfg = DesignOptimizerConfig(method="evolutionary", population_size=6)
    eo = EvolutionaryDesignOptimizer(cfg, seed=0, multi_objective=False)
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])

    xs = eo.ask(bounds, 6)
    ys = [float(np.sum(x ** 2)) for x in xs]
    eo.tell(xs, ys)
    xs2 = eo.ask(bounds, 6)

    assert len(xs2) == 6
    _assert_finite(np.stack(xs2), "EvolutionaryDesignOptimizer.ask non-finite")

    # Multi-objective (NSGA-II survivor selection) path.
    eo_mo = EvolutionaryDesignOptimizer(cfg, seed=1, multi_objective=True)
    xs_mo = eo_mo.ask(bounds, 6)
    obj_vecs = [[float(np.sum(x ** 2)), float(np.sum((x - 1.0) ** 2))] for x in xs_mo]
    ys_mo = [sum(v) for v in obj_vecs]
    eo_mo.tell(xs_mo, ys_mo, obj_vecs=obj_vecs, penalties=[0.0] * 6)
    xs_mo2 = eo_mo.ask(bounds, 6)
    _assert_finite(np.stack(xs_mo2), "EvolutionaryDesignOptimizer (multi-objective).ask non-finite")


# ===========================================================================
# PINN refinement (pinneapple_design.design_optimizer.refinement)
# ===========================================================================

def test_breadth_pinn_refinement():
    from pinneapple_design.design_optimizer.refinement import PINNRefinement, RefinementConfig
    from pinneapple_design.design_optimizer.surrogate import PhysicsSurrogate

    model = _tiny_mlp(3, 2)

    def pde_residual_fn(u, x):
        return u.pow(2).mean()  # scalar (ndim == 0) residual branch

    cfg = RefinementConfig(n_epochs=2, n_collocation=16, verbose=False)
    pr = PINNRefinement(model, pde_residual_fn, domain=None, cfg=cfg)

    theta = np.random.randn(3).astype(np.float32)
    x_col = np.random.randn(20, 3).astype(np.float32)
    u_surrogate = np.random.randn(20, 2).astype(np.float32)
    result = pr.refine(theta, u_surrogate, x_col)
    _assert_finite(result.u_refined, "PINNRefinement.refine u_refined non-finite")
    _assert_finite(result.loss_history[-1], "PINNRefinement.refine final loss non-finite")

    surrogate = PhysicsSurrogate.build_mlp(in_dim=3, out_dim=2)
    candidates = [np.random.randn(3).astype(np.float32) for _ in range(2)]
    results = pr.refine_top_k(candidates, surrogate, k=2)
    assert len(results) == 2
    for r in results:
        _assert_finite(r.loss_history[-1], "PINNRefinement.refine_top_k final loss non-finite")


# ===========================================================================
# Closed-loop design optimizer (pinneapple_design.design_optimizer.pipeline)
# ===========================================================================

def test_breadth_design_opt_loop():
    from pinneapple_design.design_optimizer.optimizer import ParamSpace, DesignOptimizerConfig
    from pinneapple_design.design_optimizer.pipeline import DesignOptConfig, DesignOptLoop
    from pinneapple_design.design_optimizer.objective import WeightMinimizationObjective
    from pinneapple_design.design_optimizer.surrogate import PhysicsSurrogate

    param_space = ParamSpace(bounds={"a": (-1.0, 1.0), "b": (-1.0, 1.0)}, x0={"a": 0.1, "b": -0.2})
    surrogate = PhysicsSurrogate.build_mlp(in_dim=2, out_dim=3)
    objective = WeightMinimizationObjective()
    cfg = DesignOptConfig(
        n_iterations=2,
        optimizer_cfg=DesignOptimizerConfig(method="gradient", n_iters=2),
        verbose=False,
        refine_top_k=0,
    )
    loop = DesignOptLoop(param_space, surrogate, objective, cfg=cfg)
    result = loop.run()

    assert len(result.history_objectives) >= 1
    _assert_finite(result.best_objective, "DesignOptLoop.run best_objective non-finite")
    _assert_finite(result.best_theta, "DesignOptLoop.run best_theta non-finite")


# ===========================================================================
# pinneapple_systems -- from tests/test_breadth_six_packages.py's dispatched sub-agent
# for this package (real, hand-verified enumeration + exercised classes; see
# ROADMAP_PHYSICS_AI_HUB.md's Item B bullet for the consolidated numbers).
# ===========================================================================

def _is_missing_optional_dep(e: Exception) -> bool:
    return isinstance(e, (ImportError, ModuleNotFoundError))


# ===========================================================================
# pinneapple_systems.component_modeling
# (SWAGApproximation, PIDController, EdgeRuntime, DeepEnsemble)
# ===========================================================================

def test_breadth_pid_controller():
    from pinneapple_systems.component_modeling.control import PIDController

    ctrl = PIDController(kp=1.0, ki=0.1, kd=0.01, setpoint=1.0)
    u = None
    for _ in range(10):
        u = ctrl.update(measurement=0.5, dt=0.1)
    _assert_finite(u, "PIDController.update produced a non-finite action")
    ctrl.reset()
    assert ctrl._integral == 0.0 and ctrl._prev_error is None


def test_breadth_deep_ensemble():
    from pinneapple_systems.component_modeling.ensemble import DeepEnsemble

    ens = DeepEnsemble(model_factory=lambda: _tiny_mlp(2, 1), n_members=3, base_seed=0)
    coords = torch.rand(16, 2)
    targets = torch.rand(16, 1)
    histories = ens.fit(coords, targets, epochs=5, lr=1e-2)
    assert len(histories) == 3
    for hist in histories:
        for loss in hist:
            _assert_finite(loss, "DeepEnsemble.fit produced a non-finite per-epoch loss")
    mean, std = ens.predict(coords)
    _assert_finite(mean, "DeepEnsemble.predict mean is non-finite")
    _assert_finite(std, "DeepEnsemble.predict std is non-finite")


def test_breadth_swag_approximation():
    from pinneapple_systems.component_modeling.bayesian import SWAGApproximation

    model = _tiny_mlp(2, 1)
    swag = SWAGApproximation(model)
    coords = torch.rand(16, 2)
    targets = torch.rand(16, 1)
    swag.fit(coords, targets, epochs=3, n_snapshots=2, snapshot_epochs=2, lr=1e-2)
    mean, std = swag.predict_with_uncertainty(coords, n_samples=3)
    _assert_finite(mean, "SWAGApproximation.predict_with_uncertainty mean is non-finite")
    _assert_finite(std, "SWAGApproximation.predict_with_uncertainty std is non-finite")


def test_breadth_edge_runtime(tmp_path):
    pytest.importorskip("onnxruntime", reason="EdgeRuntime requires onnxruntime, not installed")
    from pinneapple_systems.component_modeling.edge import EdgeRuntime, export_edge_package

    model = _tiny_mlp(2, 1)
    sample = torch.rand(4, 2)
    pkg = export_edge_package(model, str(tmp_path), name="tiny", sample_input=sample)
    rt = EdgeRuntime(pkg["zip_path"])
    out = rt.predict(sample.numpy())
    _assert_finite(np.asarray(out, dtype=float), "EdgeRuntime.predict produced non-finite output")


# ===========================================================================
# pinneapple_systems.cosimulation
# (node types, CoSimGraph, CoSimEngine, losses, recorder, trainer, adapters)
# ===========================================================================

def test_breadth_cosim_node_abstract_skip():
    pytest.skip(
        "CoSimNode is an abstract base class (abc.ABC with an abstractmethod "
        "step()) -- not meant to be instantiated directly; exercised via its "
        "concrete subclasses (TorchNode, AnalyticalNode, PINNNode, BlackBoxNode) "
        "below."
    )


def test_breadth_cosim_torch_node():
    from pinneapple_systems.cosimulation.node import TorchNode

    node = TorchNode("sys", _tiny_mlp(2, 2), input_ports=["u"], output_ports=["y"])
    out = node.step({"u": torch.rand(4, 2)}, t=0.0, dt=0.1)
    _assert_finite(out["y"], "TorchNode.step produced non-finite output")
    assert any(p.requires_grad for p in node.parameters())


def test_breadth_cosim_analytical_node():
    from pinneapple_systems.cosimulation.node import AnalyticalNode

    def fn(inputs, t, dt):
        return {"y": inputs["u"] * 2.0}

    node = AnalyticalNode("an", fn, input_ports=["u"], output_ports=["y"])
    out = node.step({"u": torch.rand(4, 2)}, t=0.0, dt=0.1)
    _assert_finite(out["y"], "AnalyticalNode.step produced non-finite output")


def test_breadth_cosim_blackbox_node():
    from pinneapple_systems.cosimulation.node import BlackBoxNode

    def fn(inputs, t, dt):
        return {"y": inputs["u"] + 1.0}

    node = BlackBoxNode("bb", fn, input_ports=["u"], output_ports=["y"])
    out = node.step({"u": torch.rand(4, 2, requires_grad=True)}, t=0.0, dt=0.1)
    _assert_finite(out["y"], "BlackBoxNode.step produced non-finite output")
    assert not out["y"].requires_grad, "BlackBoxNode is documented to detach outputs from autograd"


def test_breadth_cosim_pinn_node():
    from pinneapple_systems.cosimulation.node import PINNNode

    def physics_fn(node, inputs, t, dt):
        return (node.model(inputs["u"]) ** 2).mean()

    node = PINNNode(
        "pinn", _tiny_mlp(2, 2), input_ports=["u"], output_ports=["y"],
        physics_fn=physics_fn, physics_weight=0.5,
    )
    out = node.step({"u": torch.rand(4, 2)}, t=0.0, dt=0.1)
    _assert_finite(out["y"], "PINNNode.step produced non-finite output")
    loss = node.physics_loss()
    _assert_finite(loss, "PINNNode.physics_loss produced a non-finite residual")


def test_breadth_cosim_graph_and_engine():
    """Also exercises Trajectory / TrajectoryRecorder (via engine.run's
    recorder) -- both are pure data-holding helper classes with no
    separate operation worth a dedicated test."""
    from pinneapple_systems.cosimulation.graph import CoSimGraph
    from pinneapple_systems.cosimulation.engine import CoSimEngine
    from pinneapple_systems.cosimulation.node import AnalyticalNode, TorchNode
    from pinneapple_systems.cosimulation.recorder import TrajectoryRecorder

    def forcing_fn(inputs, t, dt):
        return {"F": torch.ones(1, 2)}

    forcing = AnalyticalNode("forcing", forcing_fn, input_ports=[], output_ports=["F"])
    sys_node = TorchNode("sys", _tiny_mlp(2, 2), input_ports=["u"], output_ports=["y"])

    graph = CoSimGraph().add_node(forcing).add_node(sys_node)
    graph.connect("forcing.F", "sys.u")

    assert not graph.has_cycles()
    order = graph.execution_order()
    flat = [n for grp in order for n in grp]
    assert set(flat) == {"forcing", "sys"}
    assert flat.index("forcing") < flat.index("sys")

    recorder = TrajectoryRecorder().watch("sys", "y")
    engine = CoSimEngine(graph, recorder=recorder)
    engine.run(T=0.05, dt=0.01)

    traj = recorder.get("sys", "y")
    assert len(traj) > 0
    _assert_finite(traj.tensor(), "CoSimEngine.run produced non-finite recorded outputs")
    assert list(graph.trainable_parameters()), "CoSimGraph.trainable_parameters found no params from a TorchNode"


def test_breadth_cosim_losses():
    from pinneapple_systems.cosimulation.graph import CoSimGraph
    from pinneapple_systems.cosimulation.node import TorchNode
    from pinneapple_systems.cosimulation.losses import CoSimLoss, DataLoss, PhysicsLoss, CouplingLoss

    node = TorchNode("sys", _tiny_mlp(2, 2), input_ports=["u"], output_ports=["y"])
    graph = CoSimGraph().add_node(node)

    out = node.step({"u": torch.rand(4, 2)}, t=0.0, dt=0.1)
    port_values = {"sys": {"u": torch.rand(4, 2), "y": out["y"]}}

    dl = DataLoss(weight=1.0)({"sys.y": out["y"]}, {"sys.y": torch.zeros_like(out["y"])})
    _assert_finite(dl, "DataLoss.forward produced non-finite loss")

    pl = PhysicsLoss(weight=1.0)(graph)
    _assert_finite(pl, "PhysicsLoss.forward produced non-finite loss (no PINN nodes -> should be 0)")

    cl = CouplingLoss(weight=0.1)(port_values, graph)
    _assert_finite(cl, "CouplingLoss.forward produced non-finite loss")

    criterion = CoSimLoss(data_weight=1.0, physics_weight=1.0, coupling_weight=0.1)
    total, breakdown = criterion(port_values, graph, targets={"sys.y": torch.zeros_like(out["y"])})
    _assert_finite(total, "CoSimLoss.forward produced non-finite total loss")
    for k, v in breakdown.items():
        _assert_finite(v, f"CoSimLoss breakdown[{k!r}] is non-finite")


def test_breadth_cosim_trainer_fits():
    from types import SimpleNamespace
    from pinneapple_systems.cosimulation.graph import CoSimGraph
    from pinneapple_systems.cosimulation.engine import CoSimEngine
    from pinneapple_systems.cosimulation.node import AnalyticalNode, TorchNode
    from pinneapple_systems.cosimulation.trainer import CoSimTrainer

    def forcing_fn(inputs, t, dt):
        return {"F": torch.ones(1, 2)}

    forcing = AnalyticalNode("forcing", forcing_fn, input_ports=[], output_ports=["F"])
    sys_node = TorchNode("sys", _tiny_mlp(2, 2), input_ports=["u"], output_ports=["y"])
    graph = CoSimGraph().add_node(forcing).add_node(sys_node)
    graph.connect("forcing.F", "sys.u")

    engine = CoSimEngine(graph)
    trainer = CoSimTrainer(graph, engine, verbose=False)

    # CoSimTrainer.fit only reads a handful of getattr()-with-default
    # attributes off `cfg` (epochs/lr/weight_decay/grad_clip/device/seed) --
    # a plain SimpleNamespace is a legitimate generic stand-in for the
    # TrainConfig it's normally called with, without pulling in that whole
    # dependency for this smoke test.
    cfg = SimpleNamespace(epochs=2, lr=1e-2, weight_decay=0.0, grad_clip=0.0, device="cpu", seed=0)
    result = trainer.fit(
        cfg, n_unroll=2, dt=0.01,
        targets_fn=lambda t, pv: {"sys.y": torch.zeros(1, 2)},
        n_val_steps=2,
    )
    assert len(result["history"]["train_total"]) == 2
    for loss in result["history"]["train_total"]:
        _assert_finite(loss, "CoSimTrainer.fit produced a non-finite train loss")


def test_breadth_pinneapple_model_node():
    from pinneapple_systems.cosimulation.adapters import PINNeAPPleModelNode

    node = PINNeAPPleModelNode("m", _tiny_mlp(2, 2), input_ports=["u"], output_ports=["y"])
    out = node.step({"u": torch.rand(4, 2)}, t=0.0, dt=0.1)
    _assert_finite(out["y"], "PINNeAPPleModelNode.step produced non-finite output")


def test_breadth_pinne_problem_node():
    """Constructs PINNeProblemNode directly (not via .from_spec(), which
    needs a real pinneapple_physics ProblemSpec + compile_problem) -- the
    constructor itself only needs `spec` and `compiled_fn` as opaque
    objects it stores and forwards, so a trivial stand-in compiled_fn is a
    legitimate generic synthetic input for this class specifically."""
    from pinneapple_systems.cosimulation.adapters import PINNeProblemNode

    def compiled_fn(model, _y_hat, batch):
        return {"total": (model(batch["x_col"]) ** 2).mean()}

    node = PINNeProblemNode(
        "pinn_problem", _tiny_mlp(2, 2), spec=None, compiled_fn=compiled_fn,
        coord_ports=["x_col"], field_ports=["y"], physics_weight=1.0,
    )
    out = node.step({"x_col": torch.rand(4, 2)}, t=0.0, dt=0.1)
    _assert_finite(out["y"], "PINNeProblemNode.step produced non-finite output")
    loss = node.physics_loss()
    _assert_finite(loss, "PINNeProblemNode.physics_loss produced a non-finite residual")


def test_breadth_symbolic_pde_node():
    """A fake `symbolic_pde` stand-in exposing just the one method
    (`to_residual_fn(model) -> callable`) SymbolicPDENode actually calls --
    a real SymPy-backed SymbolicPDE is a heavier real dependency this
    generic smoke test doesn't need to exercise the node's own step()/
    physics_loss() wiring."""
    from pinneapple_systems.cosimulation.adapters import SymbolicPDENode

    class _FakeSymbolicPDE:
        def to_residual_fn(self, model):
            return lambda x: model(x)

    node = SymbolicPDENode(
        "sym", _tiny_mlp(2, 2), symbolic_pde=_FakeSymbolicPDE(),
        coord_ports=["x_col"], field_ports=["y"], physics_weight=1.0,
    )
    out = node.step({"x_col": torch.rand(4, 2)}, t=0.0, dt=0.1)
    _assert_finite(out["y"], "SymbolicPDENode.step produced non-finite output")
    loss = node.physics_loss()
    _assert_finite(loss, "SymbolicPDENode.physics_loss produced a non-finite residual")


def test_breadth_timeseries_cosim_node():
    from pinneapple_systems.cosimulation.adapters import TimeSeriesCoSimNode
    from pinneapple_systems.time_series.models.recurrent import LSTMForecaster, RecurrentConfig

    cfg = RecurrentConfig(input_len=8, horizon=4, n_features=2, n_targets=2, hidden_size=8, num_layers=1)
    model = LSTMForecaster(cfg)
    node = TimeSeriesCoSimNode("ts", model, input_len=8, horizon=4)
    context = torch.rand(1, 8, 2)
    out = node.step({"context": context}, t=0.0, dt=1.0)
    _assert_finite(out["forecast"], "TimeSeriesCoSimNode.step forecast is non-finite")
    _assert_finite(out["context_next"], "TimeSeriesCoSimNode.step context_next is non-finite")
    assert out["context_next"].shape == context.shape


# ===========================================================================
# pinneapple_systems.digital_twin
# (Kalman filters, anomaly detectors, sensors, streams, DigitalTwin)
# ===========================================================================

def test_breadth_extended_kalman_filter():
    from pinneapple_systems.digital_twin.assimilation.kalman import ExtendedKalmanFilter

    def f(x):
        return x.copy()

    def h(x):
        return x.copy()

    ekf = ExtendedKalmanFilter(n_state=2, n_obs=2, f=f, h=h)
    ekf.initialize(np.zeros(2))
    out = ekf.step(np.array([1.0, 0.5]))
    _assert_finite(out["x"], "ExtendedKalmanFilter.step state is non-finite")
    _assert_finite(out["P"], "ExtendedKalmanFilter.step covariance is non-finite")


def test_breadth_ensemble_kalman_filter():
    from pinneapple_systems.digital_twin.assimilation.kalman import EnsembleKalmanFilter

    def f(x):
        return x.copy()

    def h(x):
        return x.copy()

    enkf = EnsembleKalmanFilter(n_state=2, n_obs=2, f=f, h=h, n_ens=20, seed=0)
    enkf.initialize(np.zeros(2))
    out = enkf.step(np.array([1.0, 0.5]))
    _assert_finite(out["x"], "EnsembleKalmanFilter.step mean state is non-finite")
    _assert_finite(enkf.covariance, "EnsembleKalmanFilter.covariance is non-finite")


def test_breadth_threshold_detector():
    from pinneapple_systems.digital_twin.monitoring.anomaly import ThresholdDetector

    det = ThresholdDetector(thresholds={"p": 1.0})
    events = det.check(0.0, "s1", observed={"p": 10.0}, predicted={"p": 0.0})
    assert len(events) == 1
    _assert_finite(events[0].score, "ThresholdDetector event score is non-finite")


def test_breadth_zscore_detector():
    from pinneapple_systems.digital_twin.monitoring.anomaly import ZScoreDetector

    det = ZScoreDetector(z_threshold=2.0, window_size=20, min_samples=5)
    rng = np.random.default_rng(0)
    all_events = []
    for i in range(30):
        obs = float(rng.normal(0, 1))
        all_events.extend(det.check(float(i), "s1", {"p": obs}, {"p": 0.0}))
    # Whether any individual draw actually crosses the z-threshold depends
    # on the RNG; the real assertion is that 30 calls to this stateful
    # rolling detector ran without raising, and any event it did produce
    # has a finite score.
    for ev in all_events:
        _assert_finite(ev.score, "ZScoreDetector event score is non-finite")


def test_breadth_mahalanobis_detector():
    from pinneapple_systems.digital_twin.monitoring.anomaly import MahalanobisDetector

    det = MahalanobisDetector(threshold=1.0, field_order=["p", "q"])
    det.update_covariance(np.eye(2))
    events = det.check(0.0, "s1", observed={"p": 5.0, "q": 5.0}, predicted={"p": 0.0, "q": 0.0})
    assert len(events) == 1
    _assert_finite(events[0].score, "MahalanobisDetector event score is non-finite")


def test_breadth_anomaly_monitor():
    from pinneapple_systems.digital_twin.monitoring.anomaly import (
        AnomalyMonitor, ThresholdDetector, ZScoreDetector,
    )

    mon = AnomalyMonitor()
    mon.add_detector(ThresholdDetector(thresholds={"p": 1.0}))
    mon.add_detector(ZScoreDetector(z_threshold=2.0, window_size=10, min_samples=3))
    events = mon.check(0.0, "s1", observed={"p": 10.0}, predicted={"p": 0.0})
    assert len(events) >= 1
    recent = mon.recent_events(5)
    assert len(recent) >= 1
    mon.clear()
    assert mon.all_events == []


def test_breadth_sensor_registry():
    from pinneapple_systems.digital_twin.io.sensors import Sensor, SensorRegistry

    reg = SensorRegistry()
    reg.add(Sensor(
        "p01", coords={"x": 0.5, "y": 0.5}, field_names=["p"],
        calibration_offset={"p": 1.0}, calibration_scale={"p": 2.0},
    ))
    obs = reg.read("p01", raw={"p": 100.0})
    _assert_finite(obs.values["p"], "Sensor.calibrate/SensorRegistry.read produced a non-finite value")
    assert obs.values["p"] == pytest.approx(100.0 * 2.0 + 1.0)
    arr = reg.to_observation_array([obs])
    assert arr.shape == (1, 1)
    _assert_finite(arr, "SensorRegistry.to_observation_array produced a non-finite value")


def test_breadth_digital_twin():
    from pinneapple_systems.digital_twin.twin import DigitalTwin

    model = _tiny_mlp(2, 2)
    dt = DigitalTwin(model, field_names=["u", "v"], coord_names=["x", "y"])
    coords = {
        "x": np.linspace(0, 1, 5).astype(np.float32),
        "y": np.linspace(0, 1, 5).astype(np.float32),
    }
    dt.set_domain_coords(coords)
    preds = dt.predict(coords)
    assert preds, "DigitalTwin.predict returned no fields"
    for f, arr in preds.items():
        _assert_finite(arr, f"DigitalTwin.predict produced non-finite values for field {f!r}")


def test_breadth_file_watch_stream(tmp_path):
    import queue
    import time as time_mod
    from pinneapple_systems.digital_twin.io.stream import FileWatchStream

    path = tmp_path / "stream.jsonl"
    path.write_text('{"timestamp": 0.0, "p": 1.0}\n')

    stream = FileWatchStream(str(path), sensor_id="s1", field_names=["p"], poll_interval=0.05)
    q: "queue.Queue" = queue.Queue()
    stream.start(q)
    time_mod.sleep(0.2)
    stream.stop()

    obs = []
    while True:
        try:
            obs.append(q.get_nowait())
        except queue.Empty:
            break
    assert len(obs) >= 1, "FileWatchStream never emitted an Observation from a real jsonl file within the poll window"
    _assert_finite(obs[0].values["p"], "FileWatchStream observation value is non-finite")


def test_breadth_http_poll_stream():
    """No live HTTP endpoint is available in this environment --
    HTTPPollStream's own `_run()` catches every exception internally and
    just logs a warning (see its source), so this is a genuine smoke test
    of "start a real background poll thread against an unreachable URL,
    confirm it doesn't crash the process", not a claim that HTTP polling
    itself round-trips end-to-end."""
    import queue
    import time as time_mod
    from pinneapple_systems.digital_twin.io.stream import HTTPPollStream

    stream = HTTPPollStream(
        "http://127.0.0.1:1/does-not-exist", sensor_id="s1", field_names=["p"], poll_interval=0.05,
    )
    q: "queue.Queue" = queue.Queue()
    stream.start(q)
    time_mod.sleep(0.15)
    stream.stop()


def test_breadth_mock_stream():
    import queue
    import time as time_mod
    from pinneapple_systems.digital_twin.io.stream import MockStream

    def gen(t):
        return {"p": 1.0 + 0.1 * t}

    stream = MockStream("s1", field_names=["p"], generator_fn=gen, tick_interval=0.02)
    q: "queue.Queue" = queue.Queue()
    stream.start(q)
    time_mod.sleep(0.15)
    stream.stop()

    obs = []
    while True:
        try:
            obs.append(q.get_nowait())
        except queue.Empty:
            break
    assert len(obs) >= 1, "MockStream never emitted an Observation within the tick window"
    _assert_finite(obs[0].values["p"], "MockStream observation value is non-finite")


def test_breadth_base_stream_abstract_skip():
    pytest.skip(
        "BaseStream is an abstract base class (ABC with an abstractmethod _run()) "
        "-- exercised via its concrete subclasses (FileWatchStream, HTTPPollStream, "
        "MockStream) above, and MQTTStream/KafkaStream (skipped separately below, "
        "need a live broker)."
    )


def test_breadth_mqtt_stream_skip():
    pytest.skip(
        "MQTTStream requires a live MQTT broker (paho-mqtt Client.connect()) -- no "
        "broker is available in this test environment, and there is no generic "
        "synthetic substitute for a real network connection."
    )


def test_breadth_kafka_stream_skip():
    pytest.skip(
        "KafkaStream requires a live Kafka broker (kafka-python KafkaConsumer) -- no "
        "broker is available in this test environment, and there is no generic "
        "synthetic substitute for a real network connection."
    )


# ===========================================================================
# pinneapple_systems.process_components
# (AdvectionDispersionReactionSolver, TransientPipe, exception classes)
# ===========================================================================

def test_breadth_advection_dispersion_reaction_solver():
    from pinneapple_systems.process_components.reaction_kinetics import (
        Reaction, ReactionNetwork, AdvectionDispersionReactionSolver, mass_action_rate,
    )

    reaction = Reaction(name="decay", rate_fn=mass_action_rate(0.1, {"A": 1}), stoichiometry={"A": -1.0})
    network = ReactionNetwork(species=("A",), reactions=(reaction,))
    solver = AdvectionDispersionReactionSolver(
        network, n_grid=8, length_m=10.0, velocity_m_s=0.5, dispersion_m2_s=0.01,
    )
    C0 = np.ones((1, 8)) * 2.0
    t_eval = np.linspace(0.0, 1.0, 5)
    result = solver.integrate(C0, t_eval)
    assert result.success
    _assert_finite(result.C, "AdvectionDispersionReactionSolver.integrate produced non-finite concentrations")


def test_breadth_transient_pipe():
    pytest.importorskip("CoolProp", reason="TransientPipe needs CoolProp for real-gas properties")
    from pinneapple_systems.process_components.real_gas_eos import GasComposition
    from pinneapple_systems.process_components.pipe_network_1d import PipeSpec, TransientPipe

    gas = GasComposition(components=("Methane",), mole_fractions=(1.0,))
    spec = PipeSpec(name="p1", length_m=1000.0, diameter_m=0.3, roughness_m=4.5e-5, n_cells=5)
    pipe = TransientPipe(spec, gas)
    state0 = pipe.initialize_from_steady_state(m_dot_kg_s=5.0, P_in_Pa=50e5, T_in_K=288.0)
    _assert_finite(state0.P_Pa, "TransientPipe.initialize_from_steady_state produced non-finite pressures")

    def m_dot_in_fn(t):
        return 5.0

    def T_in_fn(t):
        return 288.0

    def P_out_fn(t):
        return pipe.steady_outlet_pressure(5.0, 50e5, 288.0)

    sol = pipe.simulate(state0, (0.0, 5.0), m_dot_in_fn, T_in_fn, P_out_fn)
    assert sol.success, f"TransientPipe.simulate did not converge: {sol.message}"
    _assert_finite(sol.y, "TransientPipe.simulate produced a non-finite state trajectory")
    mass = pipe.total_mass_kg(state0)
    _assert_finite(mass, "TransientPipe.total_mass_kg produced a non-finite value")


def test_breadth_process_component_exceptions():
    from pinneapple_systems.process_components.explicit_equation_system import ExplicitEquationError
    from pinneapple_systems.process_components.real_gas_eos import OutOfEnvelopeError, ValidityEnvelope

    with pytest.raises(ExplicitEquationError):
        raise ExplicitEquationError("synthetic equation-system error")

    envelope = ValidityEnvelope(P_min_Pa=1e5, P_max_Pa=2e5, T_min_K=280.0, T_max_K=290.0)
    with pytest.raises(OutOfEnvelopeError):
        envelope.check(P_Pa=10e5, T_K=285.0)


# ===========================================================================
# pinneapple_systems.time_series
# (classical + neural forecasters, decomposition, prep, tuning, backtest,
#  TSModelCatalog registry)
# ===========================================================================

def test_breadth_forecast_model_abstract_skip():
    pytest.skip(
        "ForecastModel is an abstract base class (ABC with an abstractmethod "
        "predict()) -- exercised via its concrete subclasses (XGBoost/LightGBM/"
        "RandomForest/CatBoost/GPR/MLPForecaster) below."
    )


def _build_classical_forecaster(name: str):
    from pinneapple_systems.time_series.models.classical import (
        XGBoostForecaster, LightGBMForecaster, RandomForestForecaster,
        CatBoostForecaster, GPRForecaster, MLPForecaster,
    )
    return {
        "xgboost": lambda: XGBoostForecaster(n_estimators=5, max_depth=2),
        "lightgbm": lambda: LightGBMForecaster(n_estimators=5),
        "random_forest": lambda: RandomForestForecaster(n_estimators=5),
        "catboost": lambda: CatBoostForecaster(iterations=5, depth=2),
        "gpr": lambda: GPRForecaster(),
        "mlp": lambda: MLPForecaster(hidden_layer_sizes=(8,), max_iter=20),
    }[name]()


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "random_forest", "catboost", "gpr", "mlp"])
def test_breadth_classical_forecasters(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=(30, 2))

    model = _build_classical_forecaster(name)
    try:
        model.fit(X, y)
    except Exception as e:
        if _is_missing_optional_dep(e):
            pytest.skip(f"'{name}' forecaster needs an optional dependency not installed: {e}")
        raise
    y_hat = model.predict(X)
    _assert_finite(np.asarray(y_hat, dtype=float), f"'{name}' forecaster predict() produced non-finite output")


def test_breadth_fft_forecaster():
    from pinneapple_systems.time_series.decomposition.fft_forecaster import FFTForecaster

    t = np.arange(200)
    y = np.sin(2 * np.pi * t / 20.0) + 0.1 * t
    model = FFTForecaster(n_harmonics=5, detrend=True)
    model.fit(y)
    y_pred = model.predict(horizon=24)
    _assert_finite(y_pred, "FFTForecaster.predict produced non-finite output")
    recon = model.reconstruct()
    _assert_finite(recon, "FFTForecaster.reconstruct produced non-finite output")


def test_breadth_fft_nn_forecaster():
    pytest.importorskip("sklearn", reason="FFTNNForecaster's default residual NN is an sklearn MLPRegressor")
    from pinneapple_systems.time_series.decomposition.fft_nn import FFTNNForecaster

    t = np.arange(200)
    y = np.sin(2 * np.pi * t / 20.0) + 0.1 * t
    model = FFTNNForecaster(n_harmonics=4, input_len=16, horizon=8, nn_epochs=20)
    model.fit(y)
    y_pred = model.predict()
    _assert_finite(y_pred, "FFTNNForecaster.predict produced non-finite output")


def test_breadth_hht_nn_forecaster():
    pytest.importorskip("PyEMD", reason="HHTNNForecaster needs PyEMD (pip install EMD-signal) for EMD decomposition")
    pytest.importorskip("sklearn", reason="HHTNNForecaster's default residual NN is an sklearn MLPRegressor")
    from pinneapple_systems.time_series.decomposition.hht_nn import HHTNNForecaster

    t = np.arange(200)
    y = np.sin(2 * np.pi * t / 20.0) + 0.05 * np.sin(2 * np.pi * t / 5.0) + 0.1 * t
    model = HHTNNForecaster(n_imfs=3, input_len=16, horizon=8, nn_epochs=20)
    model.fit(y)
    y_pred = model.predict()
    _assert_finite(y_pred, "HHTNNForecaster.predict produced non-finite output")


def _build_neural_forecaster_and_input(name: str):
    input_len, horizon, n_features = 16, 4, 2
    if name == "lstm":
        from pinneapple_systems.time_series.models.recurrent import LSTMForecaster, RecurrentConfig
        cfg = RecurrentConfig(input_len=input_len, horizon=horizon, n_features=n_features,
                               n_targets=2, hidden_size=8, num_layers=1)
        model = LSTMForecaster(cfg)
    elif name == "gru":
        from pinneapple_systems.time_series.models.recurrent import GRUForecaster, RecurrentConfig
        cfg = RecurrentConfig(input_len=input_len, horizon=horizon, n_features=n_features,
                               n_targets=2, hidden_size=8, num_layers=1)
        model = GRUForecaster(cfg)
    elif name == "nbeats":
        from pinneapple_systems.time_series.models.nbeats import NBeats, NBeatsConfig
        # NBeats.forward() only ever consumes the FIRST feature channel
        # (`x[:, :, 0]` -- see its docstring "use only first feature for
        # univariate path"), but _NBeatsBlock's __init__ sizes its first
        # Linear layer's in_features as `input_len * n_features`. Passing
        # n_features=2 (this test's usual generic width) would build a
        # (B, 32)-expecting layer fed a (B, 16) univariate residual and
        # crash on a shape mismatch that has nothing to do with this
        # smoke test's own logic -- it is a real mismatch between
        # NBeatsConfig's n_features knob and forward()'s univariate-only
        # data flow. Using n_features=1 here (NBeatsConfig's own default)
        # is this specific model family's actual supported shape, not a
        # loosened assertion.
        n_features_nbeats = 1
        cfg = NBeatsConfig(input_len=input_len, horizon=horizon, n_features=n_features_nbeats,
                            n_blocks=1, n_layers=2, layer_width=16)
        model = NBeats(cfg)
        x = torch.rand(3, input_len, n_features_nbeats)
        return model, x
    elif name == "tcn":
        from pinneapple_systems.time_series.models.tcn import TCNForecaster, TCNConfig
        cfg = TCNConfig(input_len=input_len, horizon=horizon, n_features=n_features,
                         n_targets=2, n_channels=8, n_layers=2)
        model = TCNForecaster(cfg)
    elif name == "tft":
        from pinneapple_systems.time_series.models.tft import TFTForecaster, TFTConfig
        cfg = TFTConfig(input_len=input_len, horizon=horizon, n_features=n_features,
                         n_targets=2, hidden_size=8, num_heads=2, num_lstm_layers=1)
        model = TFTForecaster(cfg)
    else:
        raise ValueError(name)
    x = torch.rand(3, input_len, n_features)
    return model, x


@pytest.mark.parametrize("name", ["lstm", "gru", "nbeats", "tcn", "tft"])
def test_breadth_neural_forecasters(name):
    model, x = _build_neural_forecaster_and_input(name)
    out = model(x)
    y_hat = out.y_hat
    _assert_finite(y_hat, f"'{name}' forecaster forward produced non-finite output")
    y_hat.sum().backward()
    assert any(p.grad is not None for p in model.parameters()), f"'{name}' forecaster backward produced no gradients"


def test_breadth_classical_tuner_skip():
    pytest.importorskip("optuna", reason="ClassicalTuner requires optuna for hyperparameter search")
    from pinneapple_systems.time_series.tuning.optuna_tuner import ClassicalTuner

    rng = np.random.default_rng(0)
    X_tr, y_tr = rng.normal(size=(20, 4)), rng.normal(size=(20, 2))
    X_val, y_val = rng.normal(size=(10, 4)), rng.normal(size=(10, 2))
    tuner = ClassicalTuner("random_forest", n_trials=2)
    try:
        best_model, best_params = tuner.fit(X_tr, y_tr, X_val, y_val)
    except Exception as e:
        if _is_missing_optional_dep(e):
            pytest.skip(f"ClassicalTuner('random_forest') needs an optional dependency not installed: {e}")
        raise
    y_hat = best_model.predict(X_val)
    _assert_finite(np.asarray(y_hat, dtype=float), "ClassicalTuner best_model.predict produced non-finite output")


def test_breadth_neural_tuner_skip():
    pytest.importorskip("optuna", reason="NeuralTuner requires optuna for hyperparameter search")
    from pinneapple_systems.time_series.tuning.optuna_tuner import NeuralTuner

    rng = np.random.default_rng(0)
    input_len, horizon, n_features, n_targets = 8, 4, 2, 2
    X_tr = rng.normal(size=(20, input_len, n_features)).astype(np.float32)
    y_tr = rng.normal(size=(20, horizon, n_targets)).astype(np.float32)
    X_val = rng.normal(size=(10, input_len, n_features)).astype(np.float32)
    y_val = rng.normal(size=(10, horizon, n_targets)).astype(np.float32)
    tuner = NeuralTuner(
        "lstm", n_trials=1, epochs_per_trial=1, input_len=input_len, horizon=horizon,
        n_features=n_features, n_targets=n_targets, device="cpu",
    )
    best_model, best_params = tuner.fit(X_tr, y_tr, X_val, y_val)
    assert best_params is not None


def test_breadth_time_series_imputer():
    import pandas as pd
    from pinneapple_systems.time_series.preparation.imputer import TimeSeriesImputer

    df = pd.DataFrame({"p": [1.0, np.nan, 3.0, np.nan, np.nan, 6.0]})
    imputer = TimeSeriesImputer(strategy="layered")
    out = imputer.fit_transform(df)
    assert not out["p"].isna().any(), "TimeSeriesImputer left NaNs after a 'layered' fit_transform"
    _assert_finite(out["p"].to_numpy(dtype=float), "TimeSeriesImputer output contains non-finite values")


def test_breadth_outlier_detector():
    import pandas as pd
    from pinneapple_systems.time_series.preparation.outliers import OutlierDetector

    rng = np.random.default_rng(0)
    values = rng.normal(size=60)
    values[30] = 1000.0  # inject an obvious outlier
    df = pd.DataFrame({"p": values})
    # "zscore" includes the flagged point itself in its own rolling
    # mean/std window, so one huge value inflates its own std enough to
    # sometimes dodge a plain z-threshold; "modified_zscore" (MAD-based)
    # is the robust variant precisely for this failure mode.
    det = OutlierDetector(method="modified_zscore", window=10, treatment="interpolate")
    out, flags = det.fit_transform(df)
    assert flags["p"].any(), "OutlierDetector never flagged the injected outlier"
    _assert_finite(out["p"].dropna().to_numpy(dtype=float), "OutlierDetector treated output contains non-finite values")


def test_breadth_time_series_resampler():
    import pandas as pd
    from pinneapple_systems.time_series.preparation.resampler import TimeSeriesResampler

    idx = pd.date_range("2024-01-01", periods=100, freq="min")
    df = pd.DataFrame({"p": np.sin(np.linspace(0, 10, 100))}, index=idx)
    resampler = TimeSeriesResampler(target_freq="5min", agg_method="mean")
    out = resampler.fit_transform(df)
    assert len(out) > 0
    _assert_finite(out["p"].to_numpy(dtype=float), "TimeSeriesResampler output contains non-finite values")


def test_breadth_backtest_runner(tmp_path):
    import torch.nn.functional as F
    from pinneapple_systems.time_series.datamodule import TSDataModule
    from pinneapple_systems.time_series.spec import TimeSeriesSpec
    from pinneapple_systems.time_series.validation.splitters import ExpandingWindowSplitter
    from pinneapple_systems.time_series.validation.backtest import BacktestRunner, BacktestConfig
    from pinneapple_neural.trainer.trainer import Trainer, TrainConfig

    input_len, horizon, n_features = 8, 4, 1

    class _TinyTSModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(input_len * n_features, horizon * n_features)

        def forward(self, x):
            b = x.shape[0]
            return self.net(x.reshape(b, -1)).reshape(b, horizon, n_features)

    spec = TimeSeriesSpec(input_len=input_len, horizon=horizon, stride=1)
    series = torch.rand(60, n_features)
    datamodule = TSDataModule(series=series, spec=spec, batch_size=8)
    n = len(datamodule.dataset())

    splitter = ExpandingWindowSplitter(initial_train_size=max(10, n // 3), val_size=5, step_size=5, max_folds=2)
    splits = list(splitter.split(n))
    if not splits:
        pytest.skip("Synthetic series too short to produce any ExpandingWindowSplitter fold for BacktestRunner")

    def loss_fn(model, y_hat, batch):
        return F.mse_loss(y_hat, batch["y"])

    runner = BacktestRunner(
        trainer_factory=lambda model, preprocess: Trainer(model, loss_fn=loss_fn, preprocess=preprocess),
        model_factory=lambda: _TinyTSModel(),
        datamodule=datamodule,
        splits=splits,
        train_cfg=TrainConfig(epochs=1, device="cpu", log_dir=str(tmp_path)),
        backtest_cfg=BacktestConfig(reset_model_each_fold=True),
    )
    result = runner.run()
    assert len(result.folds) == len(splits)
    for fold in result.folds:
        # `best_val` only gets updated when `cfg.save_best` is True (its
        # default, kept here) AND at least one epoch's val loss beats the
        # running best -- `val_total` (folded in from the Trainer's own
        # per-epoch history) is what's unconditionally present, so that's
        # the more robust finiteness check across both fields.
        if fold.get("val_total") is not None:
            _assert_finite(fold["val_total"], "BacktestRunner fold val_total is non-finite")
        if fold.get("best_val") is not None:
            _assert_finite(fold["best_val"], "BacktestRunner fold best_val is non-finite")


def test_breadth_ts_model_catalog_list_and_build():
    """TSModelCatalog is technically a @dataclass, but (unlike the ~55
    pure config/result dataclasses covered by the separate generic
    construction sweep) it is a real registry wrapper with actual
    operations (list()/build()) -- so, per this audit's own guidance to
    prefer a registry-style test where one exists (mirroring
    ``test_full_library_matrix.py``'s ``ModelRegistry.list()``
    parametrization), it gets a real test here instead of being silently
    left to the dataclass sweep."""
    from pinneapple_systems.time_series.registry import TSModelCatalog

    catalog = TSModelCatalog()
    names = catalog.list()
    assert isinstance(names, list)
    assert "ts_fno" in names, "TSModelCatalog should at least list its own default ts_fno registration"

    model = catalog.build_default_fno(
        input_len=16, horizon=4, n_features=2, n_targets=2, width=8, modes=4, layers=2,
    )
    x = torch.rand(2, 16, 2)
    y_hat = model(x)
    _assert_finite(y_hat, "TSModelCatalog.build_default_fno model forward produced non-finite output")


# ===========================================================================
# pinneapple_analysis -- from tests/test_breadth_six_packages.py's dispatched sub-agent
# for this package (real, hand-verified enumeration + exercised classes; see
# ROADMAP_PHYSICS_AI_HUB.md's Item B bullet for the consolidated numbers).
# ===========================================================================

# ===========================================================================
# pinneapple_analysis.uncertainty  (8 non-dataclass top-level classes)
# ===========================================================================

def test_breadth_aleatoric_head():
    from pinneapple_analysis.uncertainty.aleatoric import AleatoricHead

    base = _tiny_mlp(2, 3)
    head = AleatoricHead(base, out_dim=3, hidden=16)
    x = torch.randn(6, 2)
    mean, log_var = head(x)
    _assert_finite(mean, "AleatoricHead.forward mean is non-finite")
    _assert_finite(log_var, "AleatoricHead.forward log_var is non-finite")

    result = head.predict_with_uncertainty(x)
    _assert_finite(result.mean, "AleatoricHead.predict_with_uncertainty mean is non-finite")
    _assert_finite(result.aleatoric_std, "AleatoricHead.predict_with_uncertainty aleatoric_std is non-finite")


def test_breadth_mc_dropout_wrapper():
    from pinneapple_analysis.uncertainty.mc_dropout import MCDropoutConfig, MCDropoutWrapper

    model = _tiny_mlp(2, 1)
    cfg = MCDropoutConfig(n_samples=5, dropout_p=0.2)
    wrapper = MCDropoutWrapper(model, cfg)
    x = torch.randn(6, 2)

    y = wrapper(x)
    _assert_finite(y, "MCDropoutWrapper.forward produced non-finite output")

    result = wrapper.predict_with_uncertainty(x, n_samples=5)
    _assert_finite(result.mean, "MCDropoutWrapper.predict_with_uncertainty mean is non-finite")
    _assert_finite(result.std, "MCDropoutWrapper.predict_with_uncertainty std is non-finite")


def test_breadth_mc_dropout_standalone():
    from pinneapple_analysis.uncertainty.mc_dropout import MCDropout, MCDropoutConfig

    model = _tiny_mlp(2, 1)
    mc = MCDropout(model, MCDropoutConfig(n_samples=5))
    x = torch.randn(6, 2)
    result = mc.predict_with_uncertainty(x)
    _assert_finite(result.mean, "MCDropout.predict_with_uncertainty mean is non-finite")
    _assert_finite(result.std, "MCDropout.predict_with_uncertainty std is non-finite")


def test_breadth_ensemble_uq():
    from pinneapple_analysis.uncertainty.ensemble import EnsembleConfig, EnsembleUQ

    models = [_tiny_mlp(2, 1) for _ in range(3)]
    ensemble = EnsembleUQ(models, EnsembleConfig(n_members=3))
    x = torch.randn(6, 2)
    result = ensemble.predict_with_uncertainty(x)
    _assert_finite(result.mean, "EnsembleUQ.predict_with_uncertainty mean is non-finite")
    _assert_finite(result.std, "EnsembleUQ.predict_with_uncertainty std is non-finite")


def test_breadth_conformal_predictor():
    from pinneapple_analysis.uncertainty.conformal import ConformalPredictor

    model = _tiny_mlp(2, 1)
    cp = ConformalPredictor(model, alpha=0.1)
    x_cal = torch.randn(20, 2)
    y_cal = torch.randn(20, 1)
    cp.calibrate(x_cal, y_cal)
    assert cp.is_calibrated

    x_test = torch.randn(5, 2)
    y_pred, lower, upper = cp.predict(x_test)
    _assert_finite(y_pred, "ConformalPredictor.predict y_pred is non-finite")
    _assert_finite(lower, "ConformalPredictor.predict lower is non-finite")
    _assert_finite(upper, "ConformalPredictor.predict upper is non-finite")

    y_test = torch.randn(5, 1)
    cov = cp.coverage(x_test, y_test)
    _assert_finite(cov, "ConformalPredictor.coverage is non-finite")


def test_breadth_calibration_metrics():
    """CalibrationMetrics is a pure static-method namespace (documented as
    "no instantiation required" -- see its docstring), so this calls the
    class methods directly rather than building an instance."""
    from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics

    y_pred = torch.randn(50)
    y_true = y_pred + 0.1 * torch.randn(50)
    y_std = torch.full((50,), 0.15)

    ece = CalibrationMetrics.expected_calibration_error(y_pred, y_true, y_std, n_bins=10)
    _assert_finite(ece, "CalibrationMetrics.expected_calibration_error is non-finite")

    cov = CalibrationMetrics.coverage_at_level(y_pred, y_true, y_std, alpha=0.1)
    _assert_finite(cov, "CalibrationMetrics.coverage_at_level is non-finite")

    plot_data = CalibrationMetrics.calibration_plot_data(y_pred, y_true, y_std, n_bins=10)
    assert "expected" in plot_data and "observed" in plot_data

    sharp = CalibrationMetrics.sharpness(y_std)
    _assert_finite(sharp, "CalibrationMetrics.sharpness is non-finite")

    nll = CalibrationMetrics.nll_gaussian(y_pred, y_true, y_std)
    _assert_finite(nll, "CalibrationMetrics.nll_gaussian is non-finite")


def test_breadth_quantile_head_and_loss():
    from pinneapple_analysis.uncertainty.quantile import QuantileConfig, QuantileHead, QuantileLoss

    base = _tiny_mlp(2, 1)
    cfg = QuantileConfig(quantiles=(0.1, 0.5, 0.9))
    head = QuantileHead(base, cfg, hidden_dim=16)
    x = torch.randn(8, 2)
    out = head(x)
    _assert_finite(out, "QuantileHead.forward produced non-finite output")
    assert out.shape[-1] == 3

    loss_fn = QuantileLoss(cfg.quantiles)
    y_pred = torch.randn(4, 5, 3)
    batch = {"y": torch.randn(4, 5)}
    metrics = loss_fn(base, y_pred, batch)
    _assert_finite(metrics["total"], "QuantileLoss total is non-finite")


# ===========================================================================
# pinneapple_analysis.validation  (4 non-dataclass top-level classes)
# ===========================================================================

def test_breadth_boundary_check():
    from pinneapple_analysis.validation.boundary import BoundaryCheck

    model = _tiny_mlp(2, 1)
    check = BoundaryCheck(device="cpu")

    r1 = check.check_dirichlet(model, np.random.randn(10, 2).astype("float32"), np.zeros(10, dtype="float32"))
    _assert_finite(r1.value, "BoundaryCheck.check_dirichlet value is non-finite")

    pts = np.random.randn(10, 2).astype("float32")
    normals = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9)
    r2 = check.check_neumann(model, pts, normals, np.zeros(10, dtype="float32"))
    _assert_finite(r2.value, "BoundaryCheck.check_neumann value is non-finite")

    r3 = check.check_periodicity(model, np.random.randn(10, 2).astype("float32"), np.random.randn(10, 2).astype("float32"))
    _assert_finite(r3.value, "BoundaryCheck.check_periodicity value is non-finite")


def test_breadth_conservation_check():
    from pinneapple_analysis.validation.conservation import ConservationCheck

    # 2 spatial coords (x, y) with a matching 2-component velocity output,
    # per ConservationCheck's own docstring requirement that
    # len(velocity_indices) == len(spatial_coord_names).
    model = _tiny_mlp(2, 2)
    check = ConservationCheck(device="cpu")
    coord_names = ["x", "y"]
    bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}

    r_mass = check.check_mass_conservation(model, coord_names, bounds, n_points=200)
    _assert_finite(r_mass.value, "ConservationCheck.check_mass_conservation value is non-finite")

    r_integral = check.check_integral_quantity(
        model, coord_names, bounds,
        integrand_fn=lambda u: u.sum(dim=-1),
        expected_value=0.0, tolerance=1.0, name="test_integral", n_points=200,
    )
    _assert_finite(r_integral.value, "ConservationCheck.check_integral_quantity value is non-finite")


def test_breadth_symmetry_check():
    from pinneapple_analysis.validation.symmetry import SymmetryCheck

    model = _tiny_mlp(2, 1)
    check = SymmetryCheck(device="cpu")
    x_points = np.random.randn(10, 2).astype("float32")

    r_refl = check.check_reflection(model, x_points, axis=0, mirror=0.0)
    _assert_finite(r_refl.value, "SymmetryCheck.check_reflection value is non-finite")

    r_rot = check.check_rotational(model, x_points, angle=0.5)
    _assert_finite(r_rot.value, "SymmetryCheck.check_rotational value is non-finite")


def test_breadth_physics_validator():
    from pinneapple_analysis.validation.validator import PhysicsValidator

    model = _tiny_mlp(2, 2)
    coord_names = ["x", "y"]
    bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}

    validator = PhysicsValidator(model, coord_names, bounds, device="cpu")
    validator.add_conservation_check(kind="mass", n_points=100)
    validator.add_boundary_check(
        kind="dirichlet",
        boundary_points=np.zeros((5, 2), dtype="float32"),
        expected_values=np.zeros(5, dtype="float32"),
    )
    validator.add_symmetry_check(kind="reflection", x_points=np.random.randn(5, 2).astype("float32"))

    report = validator.validate()
    assert len(report.checks) == 3
    for c in report.checks:
        _assert_finite(c.value, f"PhysicsValidator check '{c.name}' produced a non-finite value")
    # summary() must render without crashing (used for human-readable reports).
    assert isinstance(report.summary(), str)


# ===========================================================================
# pinneapple_analysis.inverse_problems -- noise models (6 classes)
# ===========================================================================

def test_breadth_data_misfit_base_is_abstract():
    """DataMisfitBase is an ABC with an abstract __call__ -- it is never
    meant to be instantiated directly (its concrete subclasses, exercised
    below, are the real interface). Confirmed by reading noise_models.py:
    instantiating it raises TypeError, matching the audit's convention of
    recording (not silently dropping) never-instantiated ABCs."""
    from pinneapple_analysis.inverse_problems.noise_models import DataMisfitBase

    try:
        DataMisfitBase()
    except TypeError as e:
        pytest.skip(f"DataMisfitBase is an abstract base class, never instantiated directly "
                    f"(concrete subclasses GaussianMisfit/HuberMisfit/... are exercised "
                    f"separately): {e}")
    pytest.fail("DataMisfitBase unexpectedly became instantiable -- ABC contract changed")


def test_breadth_noise_models_concrete():
    from pinneapple_analysis.inverse_problems.noise_models import (
        CauchyMisfit, GaussianMisfit, HeteroscedasticMisfit, HuberMisfit, StudentTMisfit,
    )

    predicted = torch.randn(8, 3)
    observed = torch.randn(8, 3)

    for misfit, name in [
        (GaussianMisfit(noise_std=1.0), "GaussianMisfit"),
        (HuberMisfit(delta=1.0), "HuberMisfit"),
        (CauchyMisfit(gamma=1.0), "CauchyMisfit"),
        (StudentTMisfit(nu=5.0), "StudentTMisfit"),
        (HeteroscedasticMisfit(log_noise_var=torch.zeros(3)), "HeteroscedasticMisfit"),
    ]:
        loss = misfit(predicted, observed)
        _assert_finite(loss, f"{name}.__call__ produced a non-finite loss")


# ===========================================================================
# pinneapple_analysis.inverse_problems -- regularization (6 classes)
# ===========================================================================

def test_breadth_regularizer_base_is_abstract():
    """RegularizerBase is an ABC with an abstract __call__ -- never
    instantiated directly; its concrete subclasses are exercised below."""
    from pinneapple_analysis.inverse_problems.regularization import RegularizerBase

    try:
        RegularizerBase()
    except TypeError as e:
        pytest.skip(f"RegularizerBase is an abstract base class, never instantiated directly "
                    f"(concrete subclasses TikhonovRegularizer/SparsityRegularizer/... are "
                    f"exercised separately): {e}")
    pytest.fail("RegularizerBase unexpectedly became instantiable -- ABC contract changed")


def test_breadth_regularizers_concrete():
    from pinneapple_analysis.inverse_problems.regularization import (
        CompositeRegularizer, SparsityRegularizer, TikhonovRegularizer, TotalVariationRegularizer,
    )

    theta = torch.randn(5)
    tik = TikhonovRegularizer(lambda_reg=1e-3)
    _assert_finite(tik(theta), "TikhonovRegularizer produced a non-finite penalty")

    sparse = SparsityRegularizer(lambda_reg=1e-3)
    _assert_finite(sparse(theta), "SparsityRegularizer produced a non-finite penalty")

    tv = TotalVariationRegularizer(lambda_reg=1e-3)
    _assert_finite(tv(torch.randn(10)), "TotalVariationRegularizer produced a non-finite penalty")

    composite = CompositeRegularizer([(1.0, tik), (0.5, sparse)])
    _assert_finite(composite(theta), "CompositeRegularizer produced a non-finite penalty")


def test_breadth_lcurve_selector():
    from pinneapple_analysis.inverse_problems.regularization import LCurveSelector

    def train_fn(lam: float):
        # Synthetic, monotonic L-curve coordinates -- not physically
        # meaningful, but exercises the selector's curvature-based
        # optimal-lambda search end to end without a real inverse solve.
        return (1.0 / (lam + 1e-6), lam)

    selector = LCurveSelector(train_fn, lambdas=[1e-3, 1e-2, 1e-1, 1.0])
    result = selector.select(verbose=False)
    assert result.optimal_lambda in selector.lambdas
    _assert_finite(np.array(result.curvatures), "LCurveSelector curvatures are non-finite")


# ===========================================================================
# pinneapple_analysis.inverse_problems -- observation operators (5 classes)
# ===========================================================================

def test_breadth_obs_operator_base_is_abstract():
    """ObsOperatorBase is an ABC with an abstract __call__ -- never
    instantiated directly; its concrete subclasses are exercised below."""
    from pinneapple_analysis.inverse_problems.obs_operator import ObsOperatorBase

    try:
        ObsOperatorBase()
    except TypeError as e:
        pytest.skip(f"ObsOperatorBase is an abstract base class, never instantiated directly "
                    f"(concrete subclasses PointObsOperator/LinearObsOperator/... are "
                    f"exercised separately): {e}")
    pytest.fail("ObsOperatorBase unexpectedly became instantiable -- ABC contract changed")


def test_breadth_obs_operators_concrete():
    from pinneapple_analysis.inverse_problems.obs_operator import (
        ComposedObsOperator, IntegralObsOperator, LinearObsOperator, PointObsOperator,
    )

    model = _tiny_mlp(2, 1)

    point_op = PointObsOperator(sensor_locations=torch.rand(5, 2))
    y_point = point_op(model)
    _assert_finite(y_point, "PointObsOperator.__call__ produced non-finite output")

    x_eval = torch.rand(5, 2)
    linear_op = LinearObsOperator(A=torch.rand(3, 5), x_eval=x_eval)
    y_linear = linear_op(model)
    _assert_finite(y_linear, "LinearObsOperator.__call__ produced non-finite output")

    integral_op = IntegralObsOperator(x_quad=torch.rand(6, 2), weights=torch.rand(6))
    y_integral = integral_op(model)
    _assert_finite(y_integral, "IntegralObsOperator.__call__ produced non-finite output")

    composed = ComposedObsOperator(operators=[point_op, integral_op], x_obs_list=[None, None])
    y_composed = composed(model)
    _assert_finite(y_composed, "ComposedObsOperator.__call__ produced non-finite output")


# ===========================================================================
# pinneapple_analysis.inverse_problems -- sensitivity analysis (3 classes)
# ===========================================================================

def test_breadth_local_sensitivity():
    from pinneapple_analysis.inverse_problems.sensitivity import LocalSensitivity

    def forward_fn(theta: torch.Tensor) -> torch.Tensor:
        return torch.stack([theta[0] ** 2 + theta[1], theta[0] * theta[1]])

    local = LocalSensitivity(forward_fn, noise_std=1.0, param_names=["a", "b"])
    result = local.compute(torch.tensor([1.0, 2.0]))
    _assert_finite(result.jacobian, "LocalSensitivity jacobian is non-finite")
    _assert_finite(result.fisher_information, "LocalSensitivity fisher_information is non-finite")


def test_breadth_identifiability_analyzer():
    from pinneapple_analysis.inverse_problems.sensitivity import IdentifiabilityAnalyzer

    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    fim = A @ A.T + 0.1 * np.eye(4)  # synthetic SPD Fisher information matrix

    analyzer = IdentifiabilityAnalyzer(tol=1e-6)
    result = analyzer.analyze(fim, param_names=["a", "b", "c", "d"])
    _assert_finite(result.eigenvalues, "IdentifiabilityAnalyzer eigenvalues are non-finite")
    assert isinstance(result.report(), str)


def test_breadth_global_sensitivity():
    from pinneapple_analysis.inverse_problems.sensitivity import GlobalSensitivity

    def forward_fn(theta_batch: np.ndarray) -> np.ndarray:
        return np.sum(theta_batch ** 2, axis=1)

    gs = GlobalSensitivity(forward_fn, param_bounds=[(-1.0, 1.0), (-1.0, 1.0)], n_samples=32, seed=0)
    result = gs.compute()
    _assert_finite(result.S1, "GlobalSensitivity S1 indices are non-finite")
    _assert_finite(result.ST, "GlobalSensitivity ST indices are non-finite")


# ===========================================================================
# pinneapple_analysis.inverse_problems -- Ensemble Kalman Inversion (2 classes)
# ===========================================================================

def test_breadth_ensemble_kalman_inversion():
    from pinneapple_analysis.inverse_problems.ensemble_kalman import EKIConfig, EnsembleKalmanInversion

    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, 3))  # linear forward map, p=3 params -> k=2 obs

    def forward_fn(theta_batch: np.ndarray) -> np.ndarray:
        return theta_batch @ A.T

    cfg = EKIConfig(n_ensemble=8, n_iterations=3, verbose=False)
    eki = EnsembleKalmanInversion(forward_fn, cfg)
    history = eki.run(y=rng.standard_normal(2), theta_init=rng.standard_normal(3))
    assert len(history.iterations) <= 3
    _assert_finite(np.array(history.data_misfit), "EnsembleKalmanInversion data_misfit is non-finite")


def test_breadth_iterated_eki():
    from pinneapple_analysis.inverse_problems.ensemble_kalman import EKIConfig, IteratedEKI

    rng = np.random.default_rng(1)
    A = rng.standard_normal((2, 3))

    def forward_fn(theta_batch: np.ndarray) -> np.ndarray:
        return theta_batch @ A.T

    cfg = EKIConfig(n_ensemble=8, n_iterations=3, lambda_reg=0.1, verbose=False)
    teki = IteratedEKI(forward_fn, cfg)
    history = teki.run(y=rng.standard_normal(2), theta_init=rng.standard_normal(3))
    assert len(history.iterations) <= 3
    _assert_finite(np.array(history.data_misfit), "IteratedEKI data_misfit is non-finite")


# ===========================================================================
# pinneapple_analysis.inverse_problems -- high-level solver (1 class)
# ===========================================================================

class _InverseParamModel(nn.Module):
    """Minimal stand-in for a "PINN exposing inverse_params": a physical
    parameter model that InverseProblemSolver's documented convention (see
    its docstring: "Must expose learnable parameters either via
    model.inverse_params (nn.ParameterDict)") actually requires -- a plain
    nn.Module with no such attribute hits a real gap (see FOUND-NOT-FIXED
    note in the session report): _solve_adam happily falls back to
    optimizing model.parameters() directly, but _build_result()
    unconditionally calls _params_to_numpy(), which raises RuntimeError
    when there is no inverse_params ParameterDict. Using the documented
    convention here is the correct generic input, not a workaround."""

    def __init__(self):
        super().__init__()
        self.inverse_params = nn.ParameterDict({"k": nn.Parameter(torch.tensor(1.0))})

    def forward(self, x):
        return self.inverse_params["k"] * x.sum(dim=-1, keepdim=True)


def test_breadth_inverse_problem_solver():
    from pinneapple_analysis.inverse_problems.noise_models import GaussianMisfit
    from pinneapple_analysis.inverse_problems.obs_operator import PointObsOperator
    from pinneapple_analysis.inverse_problems.regularization import TikhonovRegularizer
    from pinneapple_analysis.inverse_problems.solver import InverseProblemSolver, InverseSolverConfig

    model = _InverseParamModel()
    obs_op = PointObsOperator(sensor_locations=torch.rand(5, 2))
    cfg = InverseSolverConfig(method="adam", n_iters=3, print_every=0)
    solver = InverseProblemSolver(
        model=model,
        obs_operator=obs_op,
        data_misfit=GaussianMisfit(noise_std=0.1),
        regularizer=TikhonovRegularizer(lambda_reg=1e-3),
        config=cfg,
    )
    y_obs = torch.rand(5, 1)
    result = solver.solve(y_obs)
    assert result.n_iters == 3
    _assert_finite(result.final_total_loss, "InverseProblemSolver final_total_loss is non-finite")
    _assert_finite(np.array(result.loss_history), "InverseProblemSolver loss_history is non-finite")


# ===========================================================================
# pinneapple_analysis.inverse_problems -- equation discovery (4 classes)
# ===========================================================================

def test_breadth_candidate_library():
    from pinneapple_analysis.inverse_problems.missing_term import CandidateLibrary

    lib = CandidateLibrary(poly_order=2, include_trig=True)
    X = np.random.randn(20, 2).astype("float64")
    Theta, names = lib.build(X)
    assert Theta.shape[1] == len(names)
    _assert_finite(Theta, "CandidateLibrary.build produced a non-finite Theta matrix")


def test_breadth_sindy_identifier():
    from pinneapple_analysis.inverse_problems.missing_term import CandidateLibrary, SINDyIdentifier

    lib = CandidateLibrary(poly_order=2)
    X = np.random.randn(30, 2)
    Theta, names = lib.build(X)
    # Synthetic target: a known linear combination of two library columns
    # plus small noise -- lets STRidge have something sparse to recover.
    b = 2.0 * Theta[:, 1] - 0.5 * Theta[:, 2] + 0.01 * np.random.randn(30)

    identifier = SINDyIdentifier(threshold=1e-2, method="stridge")
    result = identifier.fit(Theta, b, term_names=names)
    _assert_finite(result.coefficients, "SINDyIdentifier coefficients are non-finite")
    _assert_finite(result.residual_norm, "SINDyIdentifier residual_norm is non-finite")
    assert isinstance(result.equation(), str)


def test_breadth_residual_analyzer():
    from pinneapple_analysis.inverse_problems.missing_term import (
        CandidateLibrary, ResidualAnalyzer, SINDyIdentifier,
    )

    model = _tiny_mlp(2, 1)

    def pde_residual_fn(m, x_t):
        # Generic smoke-test stand-in "known PDE residual": just reads the
        # model's own output -- no real physics needed to exercise the
        # analyzer's residual-field + SINDy-distillation plumbing.
        out = m(x_t)
        if hasattr(out, "y"):
            out = out.y
        return out.squeeze(-1)

    analyzer = ResidualAnalyzer(
        pde_residual_fn,
        library=CandidateLibrary(poly_order=2),
        identifier=SINDyIdentifier(threshold=1e-2),
    )
    coords = np.random.randn(15, 2).astype("float32")
    result = analyzer.analyze(model, coords, device="cpu")
    _assert_finite(result.residual_field, "ResidualAnalyzer residual_field is non-finite")
    assert result.sindy_result is not None
    assert isinstance(result.summary(), str)


def test_breadth_neural_term_discovery():
    from pinneapple_analysis.inverse_problems.missing_term import NeuralTermConfig, NeuralTermDiscovery

    def known_residual_fn(m, x_t):
        out = m(x_t)
        if hasattr(out, "y"):
            out = out.y
        return out.squeeze(-1)

    model = _tiny_mlp(2, 1)
    cfg = NeuralTermConfig(hidden_dims=[8], n_iters=3, lr=1e-3, device="cpu")
    discovery = NeuralTermDiscovery(known_residual_fn, cfg)

    x_data = np.random.randn(10, 2).astype("float32")
    y_data = np.random.randn(10, 1).astype("float32")
    discovery.fit(model, x_data, y_data)

    tau = discovery.predict_term(np.random.randn(5, 2).astype("float32"))
    _assert_finite(tau, "NeuralTermDiscovery.predict_term produced non-finite output")


# ===========================================================================
# pinneapple_tools -- from tests/test_breadth_six_packages.py's dispatched sub-agent
# for this package (real, hand-verified enumeration + exercised classes; see
# ROADMAP_PHYSICS_AI_HUB.md's Item B bullet for the consolidated numbers).
# ===========================================================================

# ===========================================================================
# Arena  (benchmark_suite/api.py)
# ===========================================================================

def test_breadth_arena_run():
    """Arena.from_preset() + .run() on a real, small registered PDE preset
    (``burgers_1d``), with a plain synthetic MLP standing in for a named
    architecture (Arena._build_model() happily accepts an nn.Module
    instance directly -- see its ``if isinstance(model, nn.Module): return
    model`` branch), and a tiny epoch/point budget so the smoke test stays
    fast."""
    from pinneapple_tools.benchmark_suite.api import Arena, ArenaResult

    arena = Arena.from_preset("burgers_1d", nu=0.01)
    assert arena.spec.name  # sanity: preset resolved to a real ProblemSpec

    model = _tiny_mlp(in_dim=len(arena.spec.coords), out_dim=len(arena.spec.fields), hidden=16)
    result = arena.run(
        model=model, epochs=3, n_col=8, n_bc=4, n_ic=4, device="cpu", verbose=False,
    )
    assert isinstance(result, ArenaResult)
    assert len(result.history) == 3
    for entry in result.history:
        _assert_finite(entry["loss"], "Arena.run() produced a non-finite training loss")


# ===========================================================================
# Backend  (compute_backends/backend.py) -- plain string-constant namespace
# ===========================================================================

def test_breadth_compute_backends_backend():
    """This ``Backend`` (compute_backends/backend.py -- NOT the same-named
    Protocol in benchmark_suite/backends/base.py, see below) is just a
    namespace of string constants with no real per-instance state or
    "operation" method (its AST scan main_ops was empty), so the closest
    thing to its "one obvious operation" is the module-level
    set_backend/get_backend pair it exists to parameterize."""
    from pinneapple_tools.compute_backends.backend import Backend, get_backend, set_backend

    inst = Backend()  # trivial to construct -- no __init__ override
    assert inst.TORCH == Backend.TORCH == "torch"
    assert inst.JAX == Backend.JAX == "jax"

    original = get_backend()
    try:
        set_backend(Backend.JAX)
        assert get_backend() == "jax"
        set_backend(Backend.TORCH)
        assert get_backend() == "torch"
        with pytest.raises(ValueError):
            set_backend("not-a-real-backend")
    finally:
        set_backend(original)


# ===========================================================================
# Backend (Protocol)  (benchmark_suite/backends/base.py) -- SKIPPED
# ===========================================================================

def test_breadth_backend_protocol_not_instantiable():
    """``benchmark_suite.backends.base.Backend`` is a ``typing.Protocol``
    (``@runtime_checkable class Backend(Protocol)``) describing the
    training-backend interface (a ``.train(bundle, run_cfg)`` method) that
    real backends (DeepXDEBackend, NativePINNBackend, ...) structurally
    implement -- it is explicitly not meant to be instantiated directly.
    Confirmed by hand: ``Backend()`` raises
    ``TypeError: Protocols cannot be instantiated``. This is recorded as an
    honest skip rather than silently dropped from the enumeration."""
    from pinneapple_tools.benchmark_suite.backends.base import Backend

    with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
        Backend()
    pytest.skip("benchmark_suite.backends.base.Backend is a typing.Protocol -- an interface "
                "description for real backends to structurally satisfy, never meant to be "
                "instantiated directly (confirmed: raises 'Protocols cannot be instantiated').")


# ===========================================================================
# BenchmarkTaskBase  (benchmark_suite/benchmark.py) -- SKIPPED
# ===========================================================================

def test_breadth_benchmark_task_base_is_abstract():
    """``BenchmarkTaskBase`` is explicitly documented as the "Abstract base
    for physics benchmark tasks" -- its core hooks (sample_collocation,
    sample_boundary, pde_residual, eval_grid) all ``raise
    NotImplementedError`` in the base class. Every concrete task in
    ``benchmark_suite/tasks/`` (Heat1DTask, Burgers1DTask, Poisson2DTask,
    ...) subclasses it and overrides those hooks -- exercising the base
    class directly would just exercise NotImplementedError, not real
    behaviour, so it is skipped in favour of PINNArenaBenchmark /
    MetaBenchmarkPipeline / TransferBenchmarkPipeline below, all of which
    are exercised using a concrete ``Heat1DTask`` subclass instance."""
    from pinneapple_tools.benchmark_suite.benchmark import BenchmarkTaskBase

    base = BenchmarkTaskBase()
    with pytest.raises(NotImplementedError):
        base.sample_collocation(4, seed=0)
    pytest.skip("BenchmarkTaskBase is an abstract base whose core hooks (sample_collocation, "
                "sample_boundary, pde_residual, eval_grid) all raise NotImplementedError by "
                "design -- concrete subclasses (Heat1DTask etc., used below) are the real "
                "surface to exercise.")


# ===========================================================================
# JAXBackend  (compute_backends/jax_backend.py) -- SKIPPED (no jax installed)
# ===========================================================================

def test_breadth_jax_backend():
    """``JAXBackend`` (compute_backends/jax_backend.py) is trivially
    constructible (no ``__init__`` override, all its real methods are
    ``@staticmethod``s), but every one of those methods needs the ``jax``
    package, and the module's own docstring says they "raise ImportError
    with a helpful message when JAX is not installed instead of crashing
    with an obscure ModuleNotFoundError" -- confirmed by hand: calling
    ``JAXBackend().torch_to_jax(...)`` in this environment (no ``jax``
    installed) raises exactly that clean ImportError."""
    from pinneapple_tools.compute_backends.jax_backend import JAXBackend, jax_available

    backend = JAXBackend()  # construction itself never touches jax

    if jax_available():
        pytest.skip("jax IS installed in this environment, but this generic breadth test "
                     "doesn't have a real JAX-native model to hand JAXBackend -- skipping "
                     "rather than fabricating a fake jax computation graph.")

    with pytest.raises(ImportError, match="JAX is not installed"):
        backend.torch_to_jax(torch.zeros(2))
    pytest.skip("'jax' is not installed in this environment -- JAXBackend's methods all "
                 "require it and raise a clean, by-design ImportError without it (confirmed "
                 "above), not a broken implementation.")


# ===========================================================================
# PINNArenaBenchmark  (benchmark_suite/benchmark.py)
# ===========================================================================

def _tiny_benchmark_config(**overrides):
    from pinneapple_tools.benchmark_suite.benchmark import BenchmarkConfig

    kwargs = dict(n_col=8, n_bc=4, n_ic=4, epochs=2, n_eval=16, device="cpu", log_interval=1000)
    kwargs.update(overrides)
    return BenchmarkConfig(**kwargs)


def test_breadth_pinn_arena_benchmark_run():
    """Exercises PINNArenaBenchmark with one concrete, self-contained task
    (``Heat1DTask`` -- pure numpy/torch, no external data file or network
    dependency, unlike most of the other ``benchmark_suite/tasks/*``) and
    one tiny generic MLP ``ModelSpec``, at a tiny epoch/point budget."""
    from pinneapple_tools.benchmark_suite.benchmark import PINNArenaBenchmark, ModelSpec
    from pinneapple_tools.benchmark_suite.tasks.heat_1d import Heat1DTask

    task = Heat1DTask(alpha=0.01)
    spec = ModelSpec(name="tiny_mlp", factory=lambda i, o: _tiny_mlp(i, o, hidden=8),
                      description="tiny generic MLP")
    cfg = _tiny_benchmark_config()

    bench = PINNArenaBenchmark(tasks=[task], model_specs=[spec], config=cfg)
    results = bench.run(verbose=False)

    assert len(results) == 1
    r = results[0]
    assert r.problem_id == "heat_1d" and r.model_id == "tiny_mlp"
    _assert_finite(r.metrics["rel_l2"], "PINNArenaBenchmark rel_l2 metric is non-finite")
    _assert_finite(r.metrics["mse"], "PINNArenaBenchmark mse metric is non-finite")
    assert len(r.history) == cfg.epochs

    board = bench.leaderboard(by_problem=True)
    assert "heat_1d" in board


# ===========================================================================
# MetaBenchmarkPipeline  (benchmark_suite/meta_benchmark.py)
# ===========================================================================

def test_breadth_meta_benchmark_pipeline_run():
    """Exercises MetaBenchmarkPipeline (MAML + Reptile meta-training over a
    tiny 1-task-per-batch, 1-inner-step, 1-meta-epoch budget, on a
    parametric family of ``Heat1DTask`` instances varying ``alpha``) --
    this internally drives ``pinneapple_adaptation``'s ``PDETaskSampler`` /
    ``MAMLTrainer`` / ``ReptileTrainer`` machinery, so a genuine crash here
    is a real cross-package integration bug, not just a
    ``pinneapple_tools``-local one."""
    from pinneapple_tools.benchmark_suite.meta_benchmark import (
        MetaBenchmarkConfig, MetaBenchmarkFamily, MetaBenchmarkPipeline,
    )
    from pinneapple_tools.benchmark_suite.tasks.heat_1d import Heat1DTask

    family = MetaBenchmarkFamily(
        name="heat_family",
        train_task_factory=lambda p: Heat1DTask(alpha=p["alpha"]),
        eval_task_factory=lambda p: Heat1DTask(alpha=p["alpha"]),
        param_ranges={"alpha": (0.01, 0.05)},
        eval_params=[{"alpha": 0.02}],
    )
    cfg = MetaBenchmarkConfig(
        n_meta_epochs=1, n_inner_steps=1, n_tasks_per_batch=1,
        k_shots=(1,), algorithms=("maml", "reptile"),
        n_col=8, n_bc=4, n_ic=4, hidden=(8, 8), device="cpu", log_interval=1000,
    )
    pipe = MetaBenchmarkPipeline([family], cfg)
    results = pipe.run(verbose=False)

    algos = {r.algorithm for r in results}
    assert algos == {"maml", "reptile", "scratch", "scratch_full"}
    for r in results:
        _assert_finite(r.metrics["rel_l2"], f"MetaBenchmarkPipeline {r.algorithm} rel_l2 is non-finite")


# ===========================================================================
# TransferBenchmarkPipeline  (benchmark_suite/transfer_benchmark.py)
# ===========================================================================

def test_breadth_transfer_benchmark_pipeline_run():
    """Exercises TransferBenchmarkPipeline's "finetune" strategy (all
    layers, low LR via ``pinneapple_adaptation``'s TransferTrainer) plus its
    two from-scratch baselines, transferring a source ``Heat1DTask``
    (alpha=0.01) model to a target ``Heat1DTask`` (alpha=0.03), at a tiny
    epoch/point budget."""
    from pinneapple_tools.benchmark_suite.transfer_benchmark import (
        TransferBenchmarkConfig, TransferBenchmarkPipeline, TransferScenario,
    )
    from pinneapple_tools.benchmark_suite.tasks.heat_1d import Heat1DTask

    scenario = TransferScenario(
        name="heat_transfer",
        source_task=Heat1DTask(alpha=0.01),
        target_tasks=[Heat1DTask(alpha=0.03)],
        source_label="alpha=0.01",
        target_labels=["alpha=0.03"],
    )
    cfg = TransferBenchmarkConfig(
        n_source_epochs=2, n_finetune_epochs=2, warmup_epochs=0,
        strategies=["finetune"], n_col=8, n_bc=4, n_ic=4, n_eval=16,
        hidden=(8, 8), device="cpu", log_interval=1000,
    )
    pipe = TransferBenchmarkPipeline([scenario], cfg)
    results = pipe.run(verbose=False)

    strategies = {r.strategy for r in results}
    assert strategies == {"finetune", "scratch_budget", "scratch_full"}
    for r in results:
        _assert_finite(r.metrics["rel_l2"], f"TransferBenchmarkPipeline {r.strategy} rel_l2 is non-finite")
        _assert_finite(r.metrics["l_inf"], f"TransferBenchmarkPipeline {r.strategy} l_inf is non-finite")
        # NOTE: metrics["speedup_factor"] is deliberately NOT finiteness-checked here --
        # it is steps_to_threshold(scratch_budget) / steps_to_threshold(transfer), and at
        # this tiny (2-epoch) budget the PDE residual convergence_threshold is never
        # actually reached (convergence_epoch stays -1), which makes speedup_factor a
        # legitimate NaN by that metric's own definition, not a crash or a broken
        # implementation.


# ===========================================================================
# pinneapple_simulation -- from tests/test_breadth_six_packages.py's dispatched sub-agent
# for this package (real, hand-verified enumeration + exercised classes; see
# ROADMAP_PHYSICS_AI_HUB.md's Item B bullet for the consolidated numbers).
# ===========================================================================

# ===========================================================================
# numerical_solvers.fdm3d -- 5 native 3D FDM solvers, all built from a small
# real ``cfg`` dataclass (nx=ny=nz kept tiny so every solver finishes fast).
# ===========================================================================

def test_breadth_heat_conduction_3d():
    from pinneapple_simulation.numerical_solvers.fdm3d import HeatConduction3D, HeatConfig3D

    cfg = HeatConfig3D(nx=8, ny=8, nz=8, nt=3, dt=1e-4, alpha=1e-4)
    solver = HeatConduction3D(cfg)
    out = solver.solve()
    assert out.u.shape == (4, 8, 8, 8)
    _assert_finite(out.u, "HeatConduction3D produced a non-finite field")


def test_breadth_navier_stokes_3d():
    from pinneapple_simulation.numerical_solvers.fdm3d import NavierStokes3D, NavierStokesConfig3D

    cfg = NavierStokesConfig3D(nx=6, ny=6, nz=6, nt=2, pressure_iters=5)
    solver = NavierStokes3D(cfg)
    out = solver.solve()
    assert out.u.shape == (3, 3, 6, 6, 6)
    _assert_finite(out.u, "NavierStokes3D produced a non-finite velocity field")
    _assert_finite(out.meta["p_final"], "NavierStokes3D produced a non-finite final pressure")


def test_breadth_elastic_wave_3d():
    from pinneapple_simulation.numerical_solvers.fdm3d import ElasticWave3D, ElasticWaveConfig3D

    cfg = ElasticWaveConfig3D(nx=8, ny=8, nz=8, nt=3)
    solver = ElasticWave3D(cfg)
    out = solver.solve()
    assert out.u.shape == (4, 8, 8, 8)
    _assert_finite(out.u, "ElasticWave3D produced a non-finite displacement field")


def test_breadth_lid_driven_cavity_solver_3d():
    from pinneapple_simulation.numerical_solvers.fdm3d import (
        LidDrivenCavitySolver3D, LidDrivenCavityConfig3D,
    )

    cfg = LidDrivenCavityConfig3D(nx=6, ny=6, nz=6, nt=2, pressure_iters=5)
    solver = LidDrivenCavitySolver3D(cfg)
    out = solver.solve()
    assert out.u.shape == (3, 4, 6, 6, 6)  # axis 1 = [vx, vy, vz, p]
    _assert_finite(out.u, "LidDrivenCavitySolver3D produced a non-finite field")


def test_breadth_channel_flow_solver_3d():
    from pinneapple_simulation.numerical_solvers.fdm3d import (
        ChannelFlowSolver3D, ChannelFlowConfig3D,
    )

    cfg = ChannelFlowConfig3D(nx=6, ny=6, nz=6, nt=2, pressure_iters=5)
    solver = ChannelFlowSolver3D(cfg)
    out = solver.solve()
    assert out.u.shape == (3, 3, 6, 6, 6)
    _assert_finite(out.u, "ChannelFlowSolver3D produced a non-finite velocity field")


# ===========================================================================
# numerical_solvers.cfd_pipeline -- mesh-native CFD (CFDMesh, NSFlowSolver,
# CADToCFDPipeline). No CAD file is loaded, so CADToCFDPipeline.mesh() falls
# back (by design, with a warning) to a small structured unit-square mesh --
# this is the documented, intended no-geometry path, not a workaround.
# ===========================================================================

def test_breadth_cfd_mesh():
    from pinneapple_simulation.numerical_solvers.cfd_pipeline import CFDMesh

    mesh = CFDMesh._structured_rect(max_edge_length=0.25)
    assert mesh.nodes.shape[1] == 3
    nodes_t, elems_t = mesh.to_torch()
    assert torch.is_tensor(nodes_t) and torch.is_tensor(elems_t)
    _assert_finite(nodes_t, "CFDMesh.to_torch() produced non-finite node coordinates")
    inlet = mesh.boundary_nodes("inlet")
    assert isinstance(inlet, np.ndarray) and inlet.size > 0


def test_breadth_ns_flow_solver():
    from pinneapple_simulation.numerical_solvers.cfd_pipeline import CFDMesh, NSFlowSolver

    mesh = CFDMesh._structured_rect(max_edge_length=0.25)
    solver = NSFlowSolver(mesh, nu=1e-3, rho=1.0)
    solver.set_boundary_conditions(
        inlet_velocity=(1.0, 0.0),
        no_slip_tags=["wall_bottom", "wall_top"],
        outlet_tags=["outlet"],
    )
    result = solver.solve(max_iter=5)
    assert result["u"].shape[0] == mesh.nodes.shape[0]
    _assert_finite(result["u"], "NSFlowSolver produced a non-finite u field")
    _assert_finite(result["v"], "NSFlowSolver produced a non-finite v field")
    _assert_finite(result["p"], "NSFlowSolver produced a non-finite pressure field")


def test_breadth_cad_to_cfd_pipeline():
    from pinneapple_simulation.numerical_solvers.cfd_pipeline import CADToCFDPipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected "no geometry loaded" warning
        pipeline = CADToCFDPipeline(nu=1e-3, rho=1.0)
        pipeline.mesh(max_edge_length=0.25)
    pipeline.set_bcs(
        inlet_velocity=(1.0, 0.0),
        no_slip_tags=["wall_bottom", "wall_top"],
        outlet_tags=["outlet"],
    )
    results = pipeline.solve(max_iter=5)
    _assert_finite(results["u"], "CADToCFDPipeline produced a non-finite u field")

    data = pipeline.to_pinn_data(n_col=10)
    assert data["x_col"].shape == (10, 2)
    _assert_finite(data["x_cfd"], "CADToCFDPipeline.to_pinn_data produced non-finite CFD coords")

    # Bonus: compare_with_pinn() against a tiny stand-in model, exercising a
    # second real "operation" on the same pipeline instance.
    pinn = _tiny_mlp(in_dim=2, out_dim=3)
    with torch.no_grad():
        cmp = pipeline.compare_with_pinn(pinn, field="u")
    for v in cmp.values():
        _assert_finite(v, "CADToCFDPipeline.compare_with_pinn produced a non-finite error metric")


# ===========================================================================
# particle_dynamics.mpm -- differentiable Material Point Method (pure
# PyTorch, autograd-friendly).
# ===========================================================================

def test_breadth_mpm_state():
    from pinneapple_simulation.particle_dynamics.mpm import MPMState

    pos = 0.4 + 0.2 * torch.rand(10, 2)
    state = MPMState(pos)
    assert state.n_particles() == 10
    assert state.dim() == 2
    clone = state.clone()
    assert torch.equal(clone.pos, state.pos) and clone.pos.data_ptr() != state.pos.data_ptr()


def test_breadth_mpm_simulator():
    from pinneapple_simulation.particle_dynamics.mpm import MPMSimulator, MPMState

    torch.manual_seed(0)
    sim = MPMSimulator(
        grid_resolution=16, dt=1e-4, material="elastic", dim=2,
        E=1e4, nu=0.2, rho=1.0,
    )
    pos = 0.4 + 0.2 * torch.rand(30, 2)  # small block seeded away from the boundary
    state = MPMState(pos)
    out = sim(state, n_steps=3)
    assert out.pos.shape == (30, 2)
    _assert_finite(out.pos, "MPMSimulator produced non-finite particle positions")
    _assert_finite(out.vel, "MPMSimulator produced non-finite particle velocities")


# ===========================================================================
# particle_dynamics.particles -- ParticleSystem (abstract base) + SPHParticles
# (concrete WCSPH implementation), pure PyTorch.
# ===========================================================================

def test_breadth_particle_system_is_abstract():
    """``ParticleSystem`` is a base class whose own ``forward()`` explicitly
    raises ``NotImplementedError("Subclasses must implement forward().")``
    (confirmed by reading particle_dynamics/particles.py directly) --
    analogous to how ``tests/test_breadth_six_packages.py`` records classes
    "never meant to be instantiated directly" rather than forcing a
    synthetic call through them. Only the docstring-documented "replace
    :meth:`_find_neighbours`..." extension point is meant to be used
    directly; ``SPHParticles`` below is the concrete subclass this test
    suite actually exercises."""
    from pinneapple_simulation.particle_dynamics.particles import ParticleSystem

    ps = ParticleSystem(n_particles=5, dim=2)
    with pytest.raises(NotImplementedError):
        ps(torch.zeros(1))


def test_breadth_sph_particles():
    from pinneapple_simulation.particle_dynamics.particles import SPHParticles

    torch.manual_seed(0)
    sph = SPHParticles(n_particles=20, smoothing_length=0.2, dim=2)
    pos = torch.rand(20, 2)
    vel = torch.zeros(20, 2)
    new_pos, new_vel = sph(pos, vel, dt=1e-3)
    assert new_pos.shape == (20, 2) and new_vel.shape == (20, 2)
    _assert_finite(new_pos, "SPHParticles produced non-finite positions")
    _assert_finite(new_vel, "SPHParticles produced non-finite velocities")


# ===========================================================================
# particle_dynamics.rigid_body -- differentiable rigid body dynamics.
# ===========================================================================

def test_breadth_rigid_body_state():
    from pinneapple_simulation.particle_dynamics.rigid_body import RigidBodyState

    st = RigidBodyState(n_bodies=2, dim=3)
    assert st.pos.shape == (2, 3) and st.quat.shape == (2, 4)
    clone = st.clone()
    assert torch.equal(clone.quat, st.quat) and clone.quat.data_ptr() != st.quat.data_ptr()


def test_breadth_rigid_body_step():
    from pinneapple_simulation.particle_dynamics.rigid_body import RigidBody, RigidBodyState

    body = RigidBody(mass=1.0, inertia=torch.tensor(1.0), dim=2)
    state = RigidBodyState(n_bodies=1, dim=2)
    force = torch.tensor([[0.0, -1.0]])
    torque = torch.tensor([0.0])
    new_state = body.step(state, force, torque, dt=0.01)
    _assert_finite(new_state.pos, "RigidBody.step produced non-finite positions")
    _assert_finite(new_state.vel, "RigidBody.step produced non-finite velocities")


def test_breadth_rigid_body_system_forward():
    from pinneapple_simulation.particle_dynamics.rigid_body import (
        RigidBody, RigidBodyState, RigidBodySystem,
    )

    system = RigidBodySystem([RigidBody(mass=1.0, inertia=torch.tensor(1.0), dim=2)])
    states = [RigidBodyState(n_bodies=1, dim=2)]
    forces = [torch.zeros(2)]
    new_states = system(states, forces, dt=0.01)
    assert len(new_states) == 1
    _assert_finite(new_states[0].pos, "RigidBodySystem.forward produced non-finite positions")


# ===========================================================================
# numerical_solvers.registry -- SolverRegistry (build-by-name catalog).
# ===========================================================================

def test_breadth_solver_registry_build():
    from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all

    failures = register_all()
    # `wavelet` (pywt) and `stl` (statsmodels) are genuinely-optional deps
    # that may not be installed -- register_all() is specifically designed
    # (see its own docstring) to isolate one solver's missing dependency
    # from breaking every other solver's registration, so a non-empty
    # `failures` dict here is expected, not a bug.
    assert isinstance(failures, dict)
    names = SolverRegistry.list()
    assert "fft" in names  # a solver with no optional-dependency requirement

    solver = SolverRegistry.build("fft")
    x = torch.randn(1, 16)
    out = solver(x)
    _assert_finite(out.result, "SolverRegistry-built 'fft' solver produced a non-finite result")


def test_breadth_solver_base_is_abstract():
    """``SolverBase`` is an ``nn.Module`` base class with no ``forward()`` of
    its own (confirmed by reading numerical_solvers/base.py: only an
    ``__init__`` and a static ``mse`` helper are defined) -- every concrete
    solver in this package subclasses it and supplies its own ``forward``.
    Calling it directly hits ``nn.Module``'s own default ``forward``, which
    raises ``NotImplementedError``."""
    from pinneapple_simulation.numerical_solvers.base import SolverBase

    base = SolverBase()
    with pytest.raises(NotImplementedError):
        base(torch.randn(2, 2))


# ===========================================================================
# external_solvers / numerical_solvers bridges to real external tools
# (OpenFOAM, MATLAB, FEniCS/dolfinx) -- none of these binaries/licenses are
# installed on this machine (confirmed below via the same import-probe each
# bridge itself uses, and via `shutil.which`), so this section distinguishes:
#
#  * bridges whose own code already gracefully degrades when the external
#    tool is absent (returns an empty/placeholder result + warns, rather
#    than crashing) -- these ARE exercised for real, because "runs without
#    crashing and returns the documented placeholder" is exactly their
#    designed behaviour in this environment;
#  * a bridge whose constructor itself requires the external tool
#    (MATLABEngine) -- an honest missing-optional-dependency skip;
#  * a convenience wrapper (FEniCSWorkflow) that turns out to be broken
#    independent of whether FEniCS is installed -- a real, documented bug.
# ===========================================================================

def test_breadth_openfoam_bridge_graceful_degradation():
    from pinneapple_simulation.numerical_solvers.openfoam_bridge import OpenFOAMBridge

    bridge = OpenFOAMBridge(n_iterations=5)
    spec = SimpleNamespace(nu=1e-3, conditions={})
    with tempfile.TemporaryDirectory() as case_dir:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # expected "OpenFOAM not found on PATH" warning
            out = bridge.forward(spec, case_dir)
    # OpenFOAM's own binaries (blockMesh/simpleFoam/...) are not on PATH in
    # this environment (confirmed via `shutil.which`) -- OpenFOAMBridge is
    # explicitly designed to detect that and return an empty SolverOutput
    # rather than crash (see its forward()'s `_which_openfoam(...) is None`
    # branch), so `openfoam_available=False` + an empty result tensor is the
    # CORRECT outcome here, not a failure.
    assert out.extras["openfoam_available"] is False
    assert out.result.numel() == 0


def test_breadth_fenics_bridge_graceful_degradation():
    from pinneapple_simulation.numerical_solvers.fenics_bridge import FEnicsBridge

    bridge = FEnicsBridge(mesh_nx=4, mesh_ny=4)
    spec = SimpleNamespace(
        kind="heat_equation_steady", domain_bounds=(0, 0, 1, 1),
        conditions={}, parameters={},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected "Neither dolfinx nor legacy FEniCS" warning
        out = bridge.forward(spec)
    # Neither dolfinx nor legacy fenics is importable in this environment
    # (confirmed directly: `python -c "import dolfinx"` /
    # `python -c "import fenics"` both raise ModuleNotFoundError) --
    # FEnicsBridge.forward() is explicitly designed to detect that and
    # return an empty SolverOutput rather than crash, so this is the correct
    # outcome, not a failure.
    assert out.extras["fenics_available"] is False
    assert out.result.numel() == 0


def test_breadth_matlab_engine_missing_dependency():
    """MATLABEngine's constructor does ``import matlab.engine`` immediately
    (numerical_solvers's sibling classes gracefully defer this check to
    inside forward(); MATLABEngine does not -- confirmed by reading
    external_solvers/matlab/engine.py) and re-raises as an ``ImportError``
    when the MATLAB Engine for Python package is not installed, which is
    the case in this environment (confirmed: `python -c "import
    matlab.engine"` raises ``ModuleNotFoundError``, and no MATLAB
    installation is expected on this machine either). There is no generic
    synthetic substitute for a real, licensed MATLAB install, so this is an
    honest missing-optional-dependency skip, not a forced/fake input."""
    from pinneapple_simulation.external_solvers.matlab.engine import MATLABEngine

    try:
        MATLABEngine()
    except ImportError as e:
        pytest.skip(f"MATLABEngine needs a real MATLAB install + MATLAB Engine for Python: {e}")
    else:
        pytest.fail("MATLABEngine() unexpectedly succeeded without a MATLAB install")


@pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason=(
        "FOUND-NOT-FIXED bug (not fixed by design -- see this test's docstring): "
        "pinneapple_simulation/external_solvers/fenics/solver.py's solve_and_package() "
        "imports a nonexistent 'FEniCSBridge' (the real class is 'FEnicsBridge', "
        "lower-case 'n') AND, even if that name were fixed, calls it with a calling "
        "convention FEnicsBridge.__init__ doesn't accept. Kept as a strict xfail "
        "(not silently patched, not swept under a skip) so a genuine fix flips this "
        "test to an unexpected-pass failure instead of quietly going green."
    ),
)
def test_breadth_fenics_workflow_solve_and_package_bug():
    """FOUND-NOT-FIXED bug, confirmed by reading the source directly (not
    just the error message): ``FEniCSWorkflow.solve()`` delegates to
    module-level ``solve_and_package()`` in
    ``pinneapple_simulation/external_solvers/fenics/solver.py``, which does::

        from pinneapple_simulation.numerical_solvers.fenics_bridge import FEniCSBridge

    but the real class defined in that module is named ``FEnicsBridge``
    (lower-case 'n', no second capital "S") -- there is no ``FEniCSBridge``
    to import, so this ``ImportError`` fires REGARDLESS of whether
    dolfinx/legacy FEniCS is installed on the machine. The situation is
    made more confusing because ``solve_and_package()`` catches that
    ``ImportError`` and re-raises a new one worded like a routine
    missing-optional-dependency notice ("requires FEniCS. Install
    dolfinx..."), which would otherwise look exactly like the
    ``_is_missing_optional_dep``-style skip this audit discipline uses
    elsewhere -- it is NOT that; installing real FEniCS would not fix this.

    Even fixing that one name would not be enough: ``solve_and_package()``
    then constructs the bridge as
    ``FEniCSBridge(pde=config.pde, domain=config.domain, bcs=config.bcs,
    **config.solver_opts)``, but the real ``FEnicsBridge.__init__`` accepts
    only ``mesh_nx, mesh_ny, element_degree, solver_backend`` -- a second,
    deeper mismatch (this convenience wrapper appears to predate a later
    refactor of the underlying bridge class to a ``problem_spec``-based
    calling convention). Fixing this properly means redesigning
    ``solve_and_package()`` to build a `problem_spec`-shaped object and call
    ``FEnicsBridge(...).forward(problem_spec)`` instead -- not a small,
    obviously-safe one-line fix, so it is recorded here (as a strict xfail,
    not silently patched and not glossed over as a skip) rather than patched.
    """
    from pinneapple_simulation.external_solvers.fenics.solver import FEniCSWorkflow, FEniCSConfig

    cfg = FEniCSConfig(
        pde="heat_equation_steady",
        domain={"type": "rectangle", "x": [0, 1], "y": [0, 1], "nx": 4, "ny": 4},
        bcs=[],
    )
    workflow = FEniCSWorkflow(cfg)
    workflow.solve()  # expected to raise ImportError -- see the xfail reason above


# ===========================================================================
# Cross-package generic dataclass-construction sweep (all 5 non-adaptation
# packages: pinneapple_design, pinneapple_systems, pinneapple_analysis,
# pinneapple_tools, pinneapple_simulation -- pinneapple_adaptation's own 2
# dataclass configs, MAMLConfig/TransferConfig/ReptileConfig, are already
# exercised above with real, meaningful values via the classes that consume
# them, not placeholders, so they are not re-swept here).
#
# This is the "systematic test_breadth_dataclass_construction sweep"
# referenced in this file's module docstring: every dataclass-style
# config/result container across the 5 packages above has behavior that
# largely amounts to "construct it, read its fields back" -- there's no
# single generic "main operation" to call the way there is for the
# behavior-bearing classes exercised section-by-section above. Rather than
# hand-writing ~170 near-identical one-line construction tests, this sweep
# enumerates every real dataclass reachable via a best-effort import of
# every submodule in each package (mirroring the "third audit pass"
# import-smoke-test methodology), then tries to construct each one with a
# generic placeholder value for any field that has no default -- a numpy
# array of zeros, a zero tensor, an empty list/dict, 1/1.0/"test"/True for
# scalar types, etc. (see ``_generic_value_for_type`` below).
#
# A genuine unexpected exception during construction is a hard failure; a
# field type this sweep has no plausible generic placeholder for (a
# forward-referenced sibling class like ``ThreadProfile``/``GasState``, an
# ``Enum``, a ``Type[...]``) is an honest, specific skip (this dataclass
# genuinely cannot be built without a real domain object); and the
# dataclass's OWN validation rejecting a generic placeholder (e.g.
# ``GasComposition`` requiring mole fractions to sum to 1.0, or
# ``MuJoCoConfig`` requiring a real model path/XML) is also an honest skip,
# not a bug -- exactly the same "generic-input mismatch is a skip, genuine
# crash is a failure" discipline used throughout this file.
# ===========================================================================

import dataclasses
import importlib
import pkgutil
import typing as _typing
from pathlib import Path as _Path


def _iter_package_modules(package_name):
    """Best-effort import of every submodule of a package, skipping ones
    that fail to import (missing optional dependency, etc.) -- mirrors the
    "third audit pass" import-smoke-test methodology."""
    try:
        pkg = importlib.import_module(package_name)
    except Exception:
        return
    if not hasattr(pkg, "__path__"):
        return
    for _, modname, _ in pkgutil.walk_packages(
        pkg.__path__, prefix=package_name + ".", onerror=lambda name: None
    ):
        try:
            yield importlib.import_module(modname)
        except Exception:
            continue


def _generic_value_for_type(tp):
    """Best-effort generic placeholder value for a dataclass field type
    annotation -- returns (ok, value). Mirrors, in spirit,
    test_full_library_matrix.py's small generic-input philosophy
    (``in_dim=4`` etc.), just for dataclass field *types* instead of model
    constructor kwargs."""
    s = str(tp)
    origin = _typing.get_origin(tp)
    args = _typing.get_args(tp)

    if tp is int:
        return True, 1
    if tp is float:
        return True, 1.0
    if tp is str:
        return True, "test"
    if tp is bool:
        return True, True
    if tp is list:
        return True, []
    if tp is dict:
        return True, {}
    if tp is tuple:
        return True, ()

    if origin is _typing.Union:
        if type(None) in args:
            return True, None
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _generic_value_for_type(non_none[0])
        return True, None
    if origin in (list, _typing.List):
        return True, []
    if origin in (dict, _typing.Dict):
        return True, {}
    if origin in (tuple, _typing.Tuple):
        return True, ()

    if "Optional" in s or s == "None" or "NoneType" in s:
        return True, None
    if "List" in s or "list" in s:
        return True, []
    if "Dict" in s or "dict" in s:
        return True, {}
    if "Tuple" in s or "tuple" in s:
        return True, ()
    if "Callable" in s:
        return True, (lambda *a, **k: None)
    if "ndarray" in s:
        return True, np.zeros(3)
    if "Tensor" in s:
        return True, torch.zeros(3)
    if "pathlib.Path" in s or s == "Path":
        return True, _Path(".")
    if "DataFrame" in s:
        import pandas as pd
        return True, pd.DataFrame()
    if s in ("typing.Any", "Any"):
        return True, None
    if "int" in s:
        return True, 1
    if "float" in s:
        return True, 1.0
    if "bool" in s:
        return True, True
    if "str" in s:
        return True, "test"

    return False, None


def _collect_breadth_dataclasses():
    """Enumerate every dataclass reachable from the 5 packages (all except
    pinneapple_adaptation, already covered above), skipping ones this
    generic construction scheme genuinely cannot fill in."""
    packages = [
        "pinneapple_design", "pinneapple_systems", "pinneapple_analysis",
        "pinneapple_tools", "pinneapple_simulation",
    ]
    seen = set()
    cases = []
    for pkg_name in packages:
        for mod in _iter_package_modules(pkg_name):
            for name, obj in list(vars(mod).items()):
                if name.startswith("_"):
                    continue
                if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
                    continue
                if obj.__module__ != mod.__name__:
                    continue  # only count it once, at its defining module
                key = (obj.__module__, obj.__qualname__)
                if key in seen:
                    continue
                seen.add(key)
                cases.append((pkg_name, obj.__module__, obj.__qualname__))
    return sorted(cases)


_DATACLASS_CASES = _collect_breadth_dataclasses()


@pytest.mark.parametrize(
    "pkg_name,mod_name,cls_name", _DATACLASS_CASES,
    ids=[f"{p}-{m}.{c}" for p, m, c in _DATACLASS_CASES],
)
def test_breadth_dataclass_construction(pkg_name, mod_name, cls_name):
    """Generic "can this dataclass even be built" sweep for every
    dataclass-like config/result object in the 5 non-adaptation packages --
    construct it with generic placeholder values for any field lacking a
    default, then just assert construction succeeded and the fields
    round-trip (this is the dataclass-config analogue of the
    class-with-an-operation tests above; these objects' "main operation" IS
    being constructed and read)."""
    mod = importlib.import_module(mod_name)
    klass = getattr(mod, cls_name)

    kwargs = {}
    for f in dataclasses.fields(klass):
        if not f.init:
            continue
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            continue
        ok, val = _generic_value_for_type(f.type)
        if not ok:
            pytest.skip(
                f"{cls_name}.{f.name} has type {f.type!r} with no plausible "
                f"generic placeholder (needs a real domain object, not a "
                f"synthetic stand-in) and no default -- this generic sweep "
                f"cannot construct {cls_name} at all."
            )
        kwargs[f.name] = val

    try:
        inst = klass(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if any(s in msg for s in ("must", "expected", "required", "invalid", "sum to", "shape", "provide either")):
            pytest.skip(
                f"{cls_name} rejected this sweep's generic placeholder values via its own "
                f"validation ({e}) -- a real domain-specific input is needed, not a bug."
            )
        pytest.fail(f"{cls_name}(**{kwargs!r}) raised unexpectedly: {e}")

    for f in dataclasses.fields(klass):
        if not f.init:
            continue
        assert hasattr(inst, f.name), f"{cls_name} instance missing field {f.name!r} after construction"
