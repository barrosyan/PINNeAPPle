"""``PhysicsCase`` -- a minimal bridge between the three disjoint "partial
canonical representations" of a physics problem already in this repo:

1. Geometry/mesh objects (``MeshData``, a trimesh mesh, a voxel grid, ...)
   produced by ``pinneapple_design.geometry`` and its ``io`` bridges.
2. ``pinneapple_physics.pde_environment.spec.ProblemSpec`` -- the PDE/BC/IC
   *definition* of a physics problem, built by ``ProblemBuilder``
   (``pinneapple_physics/pde_environment/builder.py``).
3. ``pinneapple_data`` UPD-family *results* objects (e.g. ``PhysicalSample``
   from ``pinneapple_data.physical_sample``, or whatever a format reader
   such as ``meshio_to_upd``/``cgns_to_upd`` hands back) once a case has
   actually been run.

Separately, ``pinneapple_pdb.benchmarks.BenchmarkEntry`` is a fourth, even
narrower thing: a named *reference* dataset used to sanity-check results.

None of the above know about each other. ``PhysicsCase`` does not redefine
or replace any of them -- it is a single dataclass that *references* the
real objects (geometry, a real ``ProblemSpec``, a solver name/config, a
real UPD-family results object, and a benchmark name) so a whole physics
case can be constructed, passed around, and (once results exist) checked
against a named benchmark as one unit, instead of the caller manually
keeping four disconnected variables in sync.

Deliberately NOT included: a registry, a fluent builder, or a plugin
system for any of the three underlying representations -- those already
exist (``ProblemBuilder``, ``SolverRegistry``, ``pinneapple_pdb.benchmarks``
itself) and this module only points at them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_physics.pde_environment.conditions import ConditionSpec
from pinneapple_physics.pde_environment.scales import ScaleSpec

__all__ = ["PhysicsCase", "BenchmarkComparison"]


def _to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion of a torch tensor / array-like / scalar list
    to a detached float64 numpy array. Mirrors the dtype-coercion style
    already used by ``pinneapple_pdb.benchmarks`` (float32 there; float64
    here since this is a comparison utility, not a training-data payload)."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float64)
    except Exception:
        pass
    return np.asarray(value, dtype=np.float64)


@dataclass
class BenchmarkComparison:
    """Plain numeric comparison produced by
    ``PhysicsCase.validate_against_benchmark()``.

    This intentionally stops at numbers, not a verdict: whether a given
    RMSE/relative error counts as "passing" is a policy decision that
    already lives in ``pinneapple_analysis/verification`` and
    ``pinneapple_llm.guardrail.PhysicsGuardrail`` (see
    ``PhysicsGuardrail._check_reference`` for the thresholded version of
    this same comparison) -- duplicating that judgment here would make
    this module the "new validation engine" it is explicitly scoped not
    to be.
    """

    benchmark: str
    x_vars: Tuple[str, ...]
    y_vars: Tuple[str, ...]
    n_reference_points: int
    n_compared_points: int
    rmse: float
    relative_l2_error: float
    per_variable_rmse: Dict[str, float]
    per_variable_relative_error: Dict[str, float]
    note: str = ""


@dataclass
class PhysicsCase:
    """One physics case: geometry + physics spec + solver choice + results,
    referenced together.

    Every field is optional -- a case may be defined before geometry
    exists, before it has been assigned a solver, or before it has been
    run at all. Nothing here is copied or redefined: ``physics`` is a
    direct reference to a real ``ProblemSpec``, ``results`` is a direct
    reference to a real UPD-family object (e.g. ``PhysicalSample``, or
    whatever ``meshio_to_upd``/``cgns_to_upd``/... produced), and
    ``geometry`` is a direct reference to whatever the geometry-io layer
    already produces (``MeshData``, a trimesh object, a voxel grid, ...).

    Attributes
    ----------
    name : short human-readable identifier for this case (not required to
        match ``physics.name``/``problem_id``, since a case may exist
        before ``physics`` is set).
    geometry : reference to a geometry/mesh object, or ``None``.
    geometry_kind : disambiguation tag for ``geometry``, e.g. "mesh",
        "voxel", "cad", "points" -- free-form, not a registry key.
    physics : a real ``pinneapple_physics.pde_environment.spec.ProblemSpec``,
        or ``None`` if the physics has not been specified yet.
    solver : name of the intended solver/backend, expected to match a
        ``pinneapple_simulation.numerical_solvers.registry.SolverRegistry``
        entry (e.g. "lbm", "fem", "fvm", "openfoam", "sph", ...) -- this
        module does not validate against the registry (that would add an
        import-time dependency on every solver backend); it is just a
        name.
    solver_config : kwargs for that solver, or ``None``.
    results : reference to a UPD-family object once this case has been
        run (e.g. a ``pinneapple_data.physical_sample.PhysicalSample``, or
        another UPD-shaped object), or ``None`` before it has been run.
    reference_benchmark : a name resolvable via
        ``pinneapple_pdb.benchmarks.get_benchmark()``, or ``None``.
    metadata : free-form provenance dict. Mirrors the
        ``provenance``/``schema`` dict convention already used by
        ``pinneapple_data.physical_sample.PhysicalSample`` (plain
        ``Dict[str, Any]``, no fixed schema beyond convention) rather than
        inventing a new provenance shape.
    """

    name: str = ""
    geometry: Optional[Any] = None
    geometry_kind: Optional[str] = None
    physics: Optional[ProblemSpec] = None
    solver: Optional[str] = None
    solver_config: Optional[Dict[str, Any]] = None
    results: Optional[Any] = None
    reference_benchmark: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Benchmark validation
    # ------------------------------------------------------------------
    def _extract_xy_from_results(
        self, x_vars: Tuple[str, ...], y_vars: Tuple[str, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pull ``(x, y)`` arrays shaped ``(N, len(x_vars))``/``(N,
        len(y_vars))`` out of ``self.results``, whatever concrete UPD-family
        type it happens to be. Supports, in order:

        1. An object with a ``to_train_dict(x_vars=, y_vars=)`` method
           (e.g. ``pinneapple_data.physical_sample.PhysicalSample``) --
           used directly, since it already implements exactly this
           extraction.
        2. A mapping (``dict``, ``xr.Dataset`` via ``__getitem__``, ...)
           keyed by variable name.
        3. An object exposing the variable names as attributes.
        """
        results = self.results
        if results is None:
            raise ValueError(
                "PhysicsCase.validate_against_benchmark(): 'results' is not set yet."
            )

        to_train_dict = getattr(results, "to_train_dict", None)
        if callable(to_train_dict):
            out = to_train_dict(x_vars=x_vars, y_vars=y_vars)
            if "y" not in out:
                raise ValueError(
                    "PhysicsCase.validate_against_benchmark(): results.to_train_dict() "
                    f"did not return a 'y' array for y_vars={y_vars!r}."
                )
            x = _to_numpy(out["x"]).reshape(-1, len(x_vars))
            y = _to_numpy(out["y"]).reshape(-1, len(y_vars))
            return x, y

        errors = []
        for accessor in ("__getitem__", "attributes"):
            try:
                if accessor == "__getitem__":
                    x_cols = [_to_numpy(results[v]).reshape(-1) for v in x_vars]
                    y_cols = [_to_numpy(results[v]).reshape(-1) for v in y_vars]
                else:
                    x_cols = [_to_numpy(getattr(results, v)).reshape(-1) for v in x_vars]
                    y_cols = [_to_numpy(getattr(results, v)).reshape(-1) for v in y_vars]
                return np.column_stack(x_cols), np.column_stack(y_cols)
            except Exception as exc:  # noqa: BLE001 - collecting for the final message
                errors.append(f"{accessor}: {exc}")

        raise TypeError(
            "PhysicsCase.validate_against_benchmark(): don't know how to extract "
            f"variables {tuple(x_vars) + tuple(y_vars)!r} from results of type "
            f"{type(results).__name__}. Supported: an object with "
            ".to_train_dict(x_vars=, y_vars=) (e.g. PhysicalSample), a mapping keyed "
            "by variable name, or an object exposing the variable names as "
            f"attributes. Tried: {'; '.join(errors)}"
        )

    def validate_against_benchmark(self) -> BenchmarkComparison:
        """Compare ``self.results`` against ``self.reference_benchmark``.

        Requires both ``results`` and ``reference_benchmark`` to be set.
        Resolves the benchmark via ``pinneapple_pdb.benchmarks.get_benchmark``,
        extracts the matching quantity from ``results`` (see
        ``_extract_xy_from_results``), and returns a ``BenchmarkComparison``
        of plain numbers -- RMSE and relative L2 error, overall and per
        output variable. No pass/fail judgment is made here.

        Matching on ``x``: when the benchmark has exactly one input
        coordinate (true for both ``lane_emden_*`` and
        ``concorde_high_aoa``), ``results`` is linearly interpolated onto
        the benchmark's reference grid via ``numpy.interp`` -- it does not
        need to be sampled on the same points. For benchmarks with more
        than one input coordinate this module does not implement N-D
        interpolation (that belongs to a real interpolation library, not
        this bridge); ``results`` must then already be sampled on exactly
        the benchmark's reference points, or a ``ValueError`` is raised.
        """
        if not self.reference_benchmark:
            raise ValueError(
                "PhysicsCase.validate_against_benchmark(): 'reference_benchmark' is not set."
            )

        from pinneapple_pdb.benchmarks import get_benchmark

        entry = get_benchmark(self.reference_benchmark)
        x_res, y_res = self._extract_xy_from_results(entry.x_vars, entry.y_vars)

        ref_x = np.asarray(entry.reference_x, dtype=np.float64)
        ref_y = np.asarray(entry.reference_y, dtype=np.float64)

        if len(entry.x_vars) == 1:
            order = np.argsort(x_res[:, 0])
            xs = x_res[order, 0]
            ys = y_res[order]
            y_interp = np.column_stack(
                [np.interp(ref_x[:, 0], xs, ys[:, j]) for j in range(ys.shape[1])]
            )
            n_compared = ref_x.shape[0]
            note = (
                f"results linearly interpolated onto the benchmark's "
                f"{entry.x_vars[0]!r} grid ({n_compared} points)."
            )
        else:
            if x_res.shape[0] != ref_x.shape[0]:
                raise ValueError(
                    f"PhysicsCase.validate_against_benchmark(): benchmark "
                    f"'{entry.name}' has {len(entry.x_vars)} input coordinates "
                    f"{entry.x_vars!r}; N-D interpolation is not implemented here, "
                    f"so results must already be sampled on exactly the benchmark's "
                    f"{ref_x.shape[0]} reference points (got {x_res.shape[0]})."
                )
            y_interp = y_res
            n_compared = x_res.shape[0]
            note = (
                "results assumed already sampled on the benchmark's exact reference "
                "points (no interpolation attempted: benchmark has >1 input coordinate)."
            )

        diff = y_interp - ref_y
        rmse = float(np.sqrt(np.mean(diff**2)))
        ref_norm = float(np.sqrt(np.mean(ref_y**2)))
        relative_l2 = float(rmse / ref_norm) if ref_norm > 0 else float("nan")

        per_var_rmse: Dict[str, float] = {}
        per_var_rel: Dict[str, float] = {}
        for j, yv in enumerate(entry.y_vars):
            col_rmse = float(np.sqrt(np.mean(diff[:, j] ** 2)))
            col_norm = float(np.sqrt(np.mean(ref_y[:, j] ** 2)))
            per_var_rmse[yv] = col_rmse
            per_var_rel[yv] = float(col_rmse / col_norm) if col_norm > 0 else float("nan")

        return BenchmarkComparison(
            benchmark=entry.name,
            x_vars=entry.x_vars,
            y_vars=entry.y_vars,
            n_reference_points=int(ref_x.shape[0]),
            n_compared_points=int(n_compared),
            rmse=rmse,
            relative_l2_error=relative_l2,
            per_variable_rmse=per_var_rmse,
            per_variable_relative_error=per_var_rel,
            note=note,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize this case to a plain ``dict``, following the same
        ``dataclasses.asdict``-based convention already used elsewhere in
        ``pinneapple_data`` (see ``pinneapple_data.zarr_store``).

        Honest limitation: ``geometry`` and ``results`` typically wrap
        large array-backed objects (mesh vertices/faces, tensors, an
        ``xr.Dataset``, ...) that this method does *not* attempt to inline
        -- doing so would either blow up memory/size or silently fail to
        round-trip. For those two fields only a lightweight, descriptive
        reference is recorded (type name, plus a ``summary()``/
        ``domain_type()`` call when the object provides one, mirroring
        ``PhysicalSample.summary()``). Persist the actual geometry/results
        objects separately with their own native save/load (e.g.
        ``pinneapple_data.zarr_store`` for UPD data, ``save_meshio`` for
        meshes) and re-attach them after ``from_dict`` if a true round
        trip is needed.

        ``physics`` (a real, frozen ``ProblemSpec``) *is* serialized in
        full via ``dataclasses.asdict`` -- note that ``ProblemSpec``'s own
        ``conditions`` may embed Python callables (``value_fn``/
        ``selector``); those survive ``asdict`` as live objects (functions
        are treated as atomic, not copied, by the underlying
        ``copy.deepcopy``) so an in-process ``to_dict()``/``from_dict()``
        round trip works, but the result is not literal JSON unless every
        condition is callable-free.
        """
        out: Dict[str, Any] = {
            "name": self.name,
            "geometry_kind": self.geometry_kind,
            "solver": self.solver,
            "solver_config": dict(self.solver_config) if self.solver_config else self.solver_config,
            "reference_benchmark": self.reference_benchmark,
            "metadata": dict(self.metadata),
            "physics": asdict(self.physics) if self.physics is not None else None,
            "geometry": self._describe_ref(self.geometry),
            "results": self._describe_ref(self.results),
        }
        return out

    @staticmethod
    def _describe_ref(obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        desc: Dict[str, Any] = {"type": type(obj).__name__}
        summary = getattr(obj, "summary", None)
        if callable(summary):
            try:
                desc["summary"] = summary()
            except Exception:
                pass
        domain_type = getattr(obj, "domain_type", None)
        if callable(domain_type):
            try:
                desc["domain_type"] = domain_type()
            except Exception:
                pass
        return desc

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PhysicsCase":
        """Reconstruct a ``PhysicsCase`` from a dict produced by
        ``to_dict()``.

        ``physics`` is rebuilt into a real ``ProblemSpec`` (via the nested
        ``PDETermSpec``/``ScaleSpec``/``ConditionSpec`` dataclasses) when
        present. ``geometry``/``results`` are NOT reconstructed -- per
        ``to_dict()``'s docstring, only a lightweight description was ever
        stored for them, so they come back as that plain description dict
        (or ``None``); the caller must re-attach the real objects (e.g.
        after reloading them from their own native storage).
        """
        physics_dict = d.get("physics")
        physics = _problem_spec_from_dict(physics_dict) if physics_dict else None
        return PhysicsCase(
            name=d.get("name", ""),
            geometry=d.get("geometry"),
            geometry_kind=d.get("geometry_kind"),
            physics=physics,
            solver=d.get("solver"),
            solver_config=d.get("solver_config"),
            results=d.get("results"),
            reference_benchmark=d.get("reference_benchmark"),
            metadata=dict(d.get("metadata") or {}),
        )


def _problem_spec_from_dict(d: Dict[str, Any]) -> ProblemSpec:
    """Rebuild a real ``ProblemSpec`` from the nested dict produced by
    ``dataclasses.asdict(problem_spec)``. Only used by ``PhysicsCase.from_dict``."""
    d = dict(d)
    pde_d = dict(d["pde"])
    pde = PDETermSpec(
        kind=pde_d["kind"],
        fields=tuple(pde_d["fields"]),
        coords=tuple(pde_d["coords"]),
        params=dict(pde_d.get("params") or {}),
        meta=dict(pde_d.get("meta") or {}),
    )
    conditions = tuple(ConditionSpec(**c) for c in d.get("conditions") or ())
    scales_d = dict(d.get("scales") or {})
    scales = ScaleSpec(**scales_d) if scales_d else ScaleSpec()
    return ProblemSpec(
        name=d["name"],
        dim=d["dim"],
        coords=tuple(d["coords"]),
        fields=tuple(d["fields"]),
        pde=pde,
        conditions=conditions,
        sample_defaults=dict(d.get("sample_defaults") or {}),
        scales=scales,
        field_ranges={k: tuple(v) for k, v in (d.get("field_ranges") or {}).items()},
        references=tuple(d.get("references") or ()),
        domain_bounds={k: tuple(v) for k, v in (d.get("domain_bounds") or {}).items()},
        solver_spec=dict(d.get("solver_spec") or {}),
        meta=dict(d.get("meta") or {}),
        problem_id=d.get("problem_id", ""),
    )
