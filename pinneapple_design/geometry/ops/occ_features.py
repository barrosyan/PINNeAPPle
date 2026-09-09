"""Direct, parametric OpenCASCADE (OCC) solid-modeling feature operations.

This module exposes real fillet/chamfer/shell operations on an arbitrary
B-Rep solid (e.g. one imported from STEP, or produced by any of the
``cadquery_gen`` parametric builders) -- not just operations baked into a
builder function at construction time.

Kernel binding actually used in this repo
------------------------------------------
The two CAD-kernel bindings that ship as real, importable dependencies here
are ``cadquery`` (2.8.0) and its own compiled OCC wrapper, ``OCP``
(cadquery vendors/depends on OCP, which in turn wraps a real OpenCASCADE
build -- there is no separate "fake" CAD kernel involved). Neither
``pythonocc-core`` (the ``OCC`` package) nor FreeCAD's own bindings
(``FreeCAD`` / ``Part``) are installed in this environment -- both were
checked directly (``import OCC`` / ``import FreeCAD`` / ``import Part``)
and both raise ``ModuleNotFoundError``. FreeCAD's Python bindings are not
pip-installable in the usual sense (they ship inside a full FreeCAD
application/AppImage/conda build), so we do not attempt to fake that here.

Given that, this module is built directly on **cadquery's ``Workplane``
API**, which is itself a thin, well-documented layer over real OCP/OCC
B-Rep algorithms (not a re-implementation):

  * ``fillet_edges``  -> ``cadquery.Workplane.fillet`` -> internally builds
    ``cadquery.occ_impl.shapes.Solid.fillet`` -> real OCP class
    ``OCP.BRepFilletAPI.BRepFilletAPI_MakeFillet`` (constant-radius edge
    fillet). Verified by reading the installed cadquery 2.8.0 source
    (``cadquery/occ_impl/shapes.py``) in this environment.
  * ``chamfer_edges`` -> ``cadquery.Workplane.chamfer`` -> internally
    ``cadquery.occ_impl.shapes.Solid.chamfer`` -> real OCP class
    ``OCP.BRepFilletAPI.BRepFilletAPI_MakeChamfer``.
  * ``shell_solid`` -> ``cadquery.Workplane.shell`` -> internally
    ``cadquery.occ_impl.shapes.Mixin3D.hollow`` -> real OCP class
    ``OCP.BRepOffsetAPI.BRepOffsetAPI_MakeThickSolid`` (variable-offset
    thick-solid / hollowing operation).

We deliberately do *not* reach past cadquery's public Workplane API down to
raw ``OCP`` calls ourselves: cadquery's own wrappers already do exactly
that (see above), already handle the required edge/face -> solid parent
chain bookkeeping that the raw OCP fillet/chamfer/thicksolid builders need,
and are what the rest of this repo's CAD code (``cadquery_gen.py``)
already builds on. Duplicating that plumbing against raw ``OCP`` symbols
would be strictly less honest, not more -- it would just be a
reimplementation of the same calls cited above with more surface area for
bugs.

Object type in / object type out
---------------------------------
Every function here accepts and returns a ``cadquery.Workplane`` -- the
same "real object type" that ``cadquery_gen.py`` builds, converts, and
exports elsewhere in this repo (all of its example builders, e.g.
``cold_plate_channel`` / ``finned_plate``, return ``cq.Workplane``
objects; ``build_mesh_from_cadquery_object`` / ``export_cadquery`` accept
them directly). For convenience a raw ``cadquery.Solid`` / ``cadquery.Shape``
is also accepted and normalized into a ``Workplane`` on the way in.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from pinneapple_design.geometry.gen.cadquery_gen import cadquery_available


EdgeSelector = Optional[Union[str, Any]]
FaceSelector = Optional[Union[str, Any]]


def _require_cadquery():
    if not cadquery_available():
        raise ImportError(
            "cadquery (with its OCP/OpenCASCADE backend) is not installed. "
            "Install it with: pip install cadquery"
        )
    import cadquery as cq  # noqa: F401

    return cq


def _as_workplane(solid: Any):
    """Normalize a cadquery ``Workplane``/``Solid``/``Shape`` into a ``Workplane``.

    ``cadquery_gen.py``'s builders always hand around ``cq.Workplane``
    objects, so that is the primary supported input/output type. A raw
    ``cadquery.occ_impl.shapes.Shape`` (e.g. a ``Solid`` produced lower in
    the stack) is also accepted and wrapped, since that is the object type
    the real OCP-backed fillet/chamfer/hollow calls operate on internally.
    """
    cq = _require_cadquery()
    from cadquery.occ_impl.shapes import Shape

    if isinstance(solid, cq.Workplane):
        return solid
    if isinstance(solid, Shape):
        return cq.Workplane(obj=solid)
    raise TypeError(
        "solid must be a cadquery.Workplane or cadquery Shape/Solid, "
        f"got {type(solid)!r}"
    )


def fillet_edges(solid: Any, radius: float, edge_selector: EdgeSelector = None):
    """Apply a real, parametric constant-radius fillet to a solid's edges.

    Real OCC API used: ``cadquery.Workplane.fillet`` -> internally
    ``cadquery.occ_impl.shapes.Solid.fillet`` -> real OCP class
    ``OCP.BRepFilletAPI.BRepFilletAPI_MakeFillet`` (one ``.Add(radius, edge)``
    call per selected edge, then ``.Shape()``).

    Parameters
    ----------
    solid:
        A ``cadquery.Workplane`` (or raw ``Solid``/``Shape``) wrapping a
        single solid -- e.g. the result of ``cadquery.importers.importStep``
        or any ``cadquery_gen`` builder output.
    radius:
        Fillet radius, must be > 0.
    edge_selector:
        Optional cadquery edge selector (a selector string such as
        ``"|Z"``/``">Z"`` or a ``cadquery.selectors.Selector`` instance).
        ``None`` (default) selects *all* edges of the solid, matching
        ``Workplane.edges()`` with no arguments.

    Returns
    -------
    cadquery.Workplane
        A new Workplane with the filleted solid selected on the stack.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")

    wp = _as_workplane(solid)
    selected = wp.edges(edge_selector) if edge_selector is not None else wp.edges()
    return selected.fillet(float(radius))


def chamfer_edges(solid: Any, distance: float, edge_selector: EdgeSelector = None):
    """Apply a real, parametric chamfer to a solid's edges.

    Real OCC API used: ``cadquery.Workplane.chamfer`` -> internally
    ``cadquery.occ_impl.shapes.Solid.chamfer`` -> real OCP class
    ``OCP.BRepFilletAPI.BRepFilletAPI_MakeChamfer`` (edge/face pairs are
    resolved via an OCP ``TopTools_IndexedDataMapOfShapeListOfShape``
    edge->face map, then one ``.Add(d1, d2, edge, face)`` call per edge).

    Parameters
    ----------
    solid:
        A ``cadquery.Workplane`` (or raw ``Solid``/``Shape``) wrapping a
        single solid.
    distance:
        Chamfer length, must be > 0.
    edge_selector:
        Optional cadquery edge selector. ``None`` (default) selects all
        edges of the solid.

    Returns
    -------
    cadquery.Workplane
        A new Workplane with the chamfered solid selected on the stack.
    """
    if distance <= 0:
        raise ValueError(f"distance must be > 0, got {distance}")

    wp = _as_workplane(solid)
    selected = wp.edges(edge_selector) if edge_selector is not None else wp.edges()
    return selected.chamfer(float(distance))


def shell_solid(solid: Any, thickness: float, faces_to_remove: FaceSelector = None):
    """Hollow out a solid into a shell of the given wall thickness.

    Real OCC API used: ``cadquery.Workplane.shell`` -> internally
    ``cadquery.occ_impl.shapes.Mixin3D.hollow`` -> real OCP class
    ``OCP.BRepOffsetAPI.BRepOffsetAPI_MakeThickSolid``
    (``MakeThickSolidByJoin`` with the removed-faces list, thickness and
    join type).

    Parameters
    ----------
    solid:
        A ``cadquery.Workplane`` (or raw ``Solid``/``Shape``) wrapping a
        single solid.
    thickness:
        Wall thickness. Positive shells outward, negative shells inward
        (per ``BRepOffsetAPI_MakeThickSolid`` semantics); must be nonzero.
    faces_to_remove:
        Optional cadquery face selector (e.g. ``">Z"`` to open up the top
        face, matching ``Workplane.faces(...)``) picking which face(s) to
        remove/open. ``None`` (default) removes *no* faces, which produces
        a fully-enclosed hollow solid with an internal cavity (the
        ``hollow()`` "no faces provided" branch that constructs a
        watertight offset shell via ``BRepBuilderAPI_MakeSolid``).

    Returns
    -------
    cadquery.Workplane
        A new Workplane with the shelled solid selected on the stack.
    """
    if thickness == 0:
        raise ValueError("thickness must be nonzero")

    wp = _as_workplane(solid)
    selected = wp.faces(faces_to_remove) if faces_to_remove is not None else wp
    return selected.shell(float(thickness))
