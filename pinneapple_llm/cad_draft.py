"""LLM-assisted CAD/geometry generation -- the same constrained pattern
as ``draft.py``'s ``draft_problem`` and ``geometry_draft.py``'s
``draft_geometry``, extended to genuinely NEW (not just named-catalog)
geometry: instead of picking one fixed preset name, the LLM composes a
recipe out of PINNeAPPle's own already-implemented, already-tested
building blocks -- primitive mesh builders
(``pinneapple_design.geometry.gen.primitives.build_mesh``, e.g. box,
sphere, cylinder) combined via real boolean CSG operations (union/cut/
intersect), optionally nested to arbitrary depth -- or, for a genuine
parametric CAD (STEP-exportable) part, one of
``pinneapple_design.geometry.gen.cadquery_gen``'s registered CadQuery
templates (e.g. a finned plate, a channeled cold-plate).

This is still never "the LLM writes CAD code": every recipe node is a
plain JSON dict of ``{"name": <real builder name>, <params...>,
"boolean": {"op": ..., "other": {<recursive node>}}}`` -- checked,
recursively, against the real registry before a single line of geometry
code runs. An LLM that names a builder or CSG op that doesn't exist gets
rejected exactly like ``draft_problem`` rejects a hallucinated preset
name, at any depth of nesting, not just the top level.

Two "kinds" of result, because they answer different questions:

- ``"mesh_recipe"``: a (possibly CSG-composed) triangle mesh, always
  available (no optional CAD-kernel dependency beyond ``trimesh``,
  already required by ``pinneapple_design.geometry``). Exported as a
  real ``.stl`` file. If ``cadquery`` is installed, can ALSO be wrapped
  into a ``.step`` file -- but see :func:`export_recipe`'s docstring for
  why that STEP is a triangulated shell, not a true parametric B-rep,
  and the honest distinction that implies.
- ``"cadquery_template"``: one of a small, curated set of genuinely
  parametric CAD templates (real B-rep solids from the ``cadquery``/OCC
  kernel), exported as a true STEP file -- but requires the optional
  ``cadquery`` dependency (``pip install "pinneapple[cad]"``).

Complexity scales with what the recipe composes: a single primitive
(low), a CSG-nested combination of several (medium), or a multi-feature
CadQuery template like a finned plate (high) -- all through the exact
same drafting-and-validation path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._dispatch import call_llm

_ALLOWED_BOOLEAN_OPS = {"union", "cut", "diff", "difference", "intersect", "intersection"}

# A deliberately curated subset of `primitives.py`'s full builder
# registry -- not every registered name (e.g. `woven_tube`/`braid_tube`
# take many interacting parameters and are much easier for a small local
# model to malform). Any name actually in `list_builders()` is still
# VALID and will be accepted/executed if the LLM names it (the
# validation check is against the real registry, not this catalog list),
# but only these get their parameters explained in the prompt, honestly
# scoping what this drafting module actively encourages versus merely
# tolerates.
_CATALOG_BUILDERS = [
    {"name": "box", "params": {"extents": "[x, y, z] side lengths"}},
    {"name": "sphere", "params": {"radius": "float"}},
    {"name": "cylinder", "params": {"radius": "float", "height": "float"}},
    {"name": "plane", "params": {"size": "[x, y]"}},
    {"name": "channel", "params": {"length": "float", "width": "float", "height": "float"}},
]

_SYSTEM_PROMPT_MESH = """You are composing a 3D geometry recipe for the \
PINNeAPPle library out of a FIXED set of real primitive builders, never \
writing CAD code. You MUST respond with a single JSON object and nothing \
else, of the exact form:

{"kind": "mesh_recipe", "recipe": <node>, "reasoning": "<one sentence>"}

where <node> is:
{"name": "<a builder name from AVAILABLE BUILDERS>", "params": {<builder's own params>}, \
"boolean": {"op": "<union|cut|intersect>", "other": <node>}}

"params" is a flat object of that builder's own parameters (e.g. \
{"radius": 0.5} for a sphere) -- always include it, even if empty ({}). \
"boolean" is OPTIONAL (omit it entirely for a single primitive with no CSG \
combination). When present, it combines this node with ANOTHER node \
("other", which is itself a full node with its own "name"/"params" and \
optionally its own nested "boolean" -- arbitrary nesting is allowed for \
more complex shapes). "op" must be one of: union (merge), cut (subtract \
"other" from this shape), intersect (keep only the overlap).

Rules:
- "name" (at every level of nesting) MUST be exactly one of the names in \
AVAILABLE BUILDERS. Never invent a name.
- "params" MUST only use the parameter names documented for that builder. \
Never invent a parameter name.
- If the user's request cannot be reasonably composed from these builders, \
respond with {"kind": "mesh_recipe", "recipe": null, "reasoning": "<why nothing fits>"} \
instead of guessing.
"""

_SYSTEM_PROMPT_CADQUERY = """You are selecting a parametric CAD template \
for the PINNeAPPle library, not writing CAD code. You MUST respond with a \
single JSON object and nothing else, of the exact form:

{"kind": "cadquery_template", "name": "<one name from AVAILABLE TEMPLATES>", \
"params": {"<param>": <value>, ...}, "reasoning": "<one sentence>"}

Rules:
- "name" MUST be exactly one of the names in AVAILABLE TEMPLATES. Never \
invent a name.
- "params" MUST only contain parameter names that template accepts (you \
are told each template's accepted params and their min/max/default). \
Never invent a parameter name.
- If nothing in the list is a reasonable match, respond with \
{"kind": "cadquery_template", "name": null, "params": {}, "reasoning": "<why nothing fits>"} \
instead of guessing.
"""


@dataclass
class CadRecipeResult:
    kind: str  # "mesh_recipe" | "cadquery_template"
    recipe: Optional[Dict[str, Any]]  # for "mesh_recipe": the validated node tree (or None)
    name: Optional[str] = None  # for "cadquery_template": the chosen template name (or None)
    params: Dict[str, Any] = field(default_factory=dict)  # for "cadquery_template"
    reasoning: str = ""
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Recursive validation of a mesh-recipe node against the real registry
# ---------------------------------------------------------------------------

def _normalize_param_value(v: Any) -> Any:
    """Undo a real, observed small-local-model quirk: llama3.2:3b's JSON-
    mode output sometimes stringifies an array/number value that should
    be a plain JSON array (e.g. ``"extents": "[1, 1, 1]"`` instead of
    ``"extents": [1, 1, 1]``), even though the surrounding JSON is
    otherwise well-formed and the CSG structure it describes is
    completely correct. This is a lossless, unambiguous re-decode (not a
    guess at a missing value -- the actual numbers are right there,
    just double-encoded), so it is fixed here rather than left to crash
    deep inside a builder function with a confusing ``TypeError`` for
    what is, semantically, a perfectly clear response. A string that
    does NOT parse as JSON is left untouched -- it will still fail
    naturally downstream (as it should; this only recovers a known
    formatting quirk, it does not swallow a genuinely wrong response)."""
    if isinstance(v, str) and v.strip()[:1] in "[{-0123456789":
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    return v


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _normalize_param_value(v) for k, v in (params or {}).items()}


def _validate_mesh_node(node: Any, *, builder_names: set, path: str = "recipe") -> None:
    """Recursively validate one recipe node against the real builder
    registry AND the exact ``{"name", "params", "boolean"?}`` shape --
    checking shape explicitly (rather than tolerating a differently-
    shaped-but-plausible node, e.g. params merged flat into the node, or
    nested one level too deep) matters here specifically because a
    silently-misshapen node does not raise inside ``build_mesh`` (its
    ``**params`` just won't contain the key a builder function looks
    for, so it silently falls back to THAT PARAMETER'S OWN DEFAULT --
    e.g. a requested ``radius=0.5`` quietly becoming radius=1.0 with no
    error at all). Confirmed against a real local-model response before
    this check existed: llama3.2:3b naturally nested params under
    ``"params"`` even when asked for a flatter shape, which silently
    produced the wrong-sized geometry until this exact-shape check and
    the matching ``params`` field in the prompt/schema were added."""
    if not isinstance(node, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(node).__name__}")
    name = node.get("name")
    if name not in builder_names:
        raise ValueError(
            f"{path}.name: LLM named builder '{name}', not in the real registry "
            f"({sorted(builder_names)}) -- refusing to use a hallucinated builder name."
        )
    params = node.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{path}.params: expected a JSON object, got {type(params).__name__}: {params!r}")
    # "reasoning" is tolerated-but-ignored at any node level: a real
    # local-model response was observed redundantly repeating its
    # top-level "reasoning" string inside the recipe node too. Unlike a
    # builder name or CSG op, free-text reasoning cannot change what
    # geometry gets built -- there is nothing to hallucinate about it in
    # a way that matters here -- so silently dropping it is a real
    # reliability improvement, not a weakening of the actual safety
    # check (which is about "name"/"boolean.op", never about prose).
    extra_keys = set(node.keys()) - {"name", "params", "boolean", "reasoning"}
    if extra_keys:
        raise ValueError(
            f"{path}: unexpected key(s) {sorted(extra_keys)} at the node's top level -- builder "
            f"parameters belong inside \"params\", not merged into the node itself."
        )
    boolean = node.get("boolean")
    if boolean is not None:
        if not isinstance(boolean, dict) or "op" not in boolean or "other" not in boolean:
            raise ValueError(f"{path}.boolean: must be an object with 'op' and 'other' keys, got: {boolean!r}")
        op = str(boolean["op"]).lower().strip()
        if op not in _ALLOWED_BOOLEAN_OPS:
            raise ValueError(
                f"{path}.boolean.op: '{op}' is not a real CSG op ({sorted(_ALLOWED_BOOLEAN_OPS)}) "
                "-- refusing to use a hallucinated operation."
            )
        _validate_mesh_node(boolean["other"], builder_names=builder_names, path=f"{path}.boolean.other")


def _node_to_build_mesh_kwargs(node: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a validated ``{"name", "params", "boolean"?}`` node into
    the flat ``**params`` shape ``primitives.build_mesh`` itself expects
    (name + its params merged at one level, with an optional "boolean"
    key whose "other" is the same flat shape, recursively) -- this
    conversion is what lets the LLM-facing schema stay recursively
    self-similar and explicit about which keys are real builder params
    (nested under "params") without that ambiguity leaking into
    ``build_mesh``'s own, differently-shaped calling convention."""
    flat = _normalize_params(node.get("params", {}))
    flat["name"] = node["name"]
    boolean = node.get("boolean")
    if boolean is not None:
        flat["boolean"] = {"op": boolean["op"], "other": _node_to_build_mesh_kwargs(boolean["other"])}
    return flat


def draft_mesh_recipe(
    description: str,
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    conversation_store=None,
) -> CadRecipeResult:
    """Ask an LLM to compose a (possibly CSG-combined) geometry recipe
    for a natural-language description, e.g. "a box with a cylindrical
    hole through it" or "two overlapping spheres fused together".

    Does not build the mesh itself -- call :func:`build_recipe` on the
    result (after looking at ``.reasoning``, same human-in-the-loop
    pause as ``draft_problem``/``draft_geometry``: an LLM's choice of
    WHICH shape to build is not something a mechanical check can
    validate the way a hallucinated name can be caught).

    Raises
    ------
    ValueError
        If the LLM's response isn't valid JSON, or names a builder or
        CSG operation not in the real registry, at ANY depth of nesting.
    """
    from pinneapple_design.geometry.gen.primitives import list_builders

    builder_names = set(list_builders())
    prompt = (
        f"USER REQUEST:\n{description}\n\n"
        f"AVAILABLE BUILDERS (name and accepted params):\n{json.dumps(_CATALOG_BUILDERS, indent=2)}\n"
    )

    raw = call_llm(
        prompt, provider=provider, model=model, api_key=api_key, system=_SYSTEM_PROMPT_MESH,
        json_mode=True, module="draft_mesh_recipe", conversation_store=conversation_store,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e

    recipe = parsed.get("recipe")
    reasoning = parsed.get("reasoning", "")
    if recipe is not None:
        _validate_mesh_node(recipe, builder_names=builder_names)

    return CadRecipeResult(kind="mesh_recipe", recipe=recipe, reasoning=reasoning, raw_response=raw)


def _cadquery_catalog() -> List[Dict[str, Any]]:
    from pinneapple_design.geometry.gen.cadquery_gen import CadQueryRegistry, register_default_templates

    reg = register_default_templates(CadQueryRegistry())
    return [
        {"name": name, "accepted_params": reg.get_schema(name)}
        for name in reg.builders
    ]


def draft_cadquery_template(
    description: str,
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    conversation_store=None,
) -> CadRecipeResult:
    """Ask an LLM to pick + parametrise a genuinely parametric CAD
    template (real B-rep solid, true STEP-exportable) for a
    natural-language description, e.g. "a cold plate with an internal
    cooling channel" or "a heat sink plate with 12 fins".

    Requires the optional ``cadquery`` dependency to actually BUILD the
    result (this drafting step itself does not need it, only the
    template catalog/schema, which is available even without cadquery
    installed -- see ``cadquery_gen.register_default_templates``).
    """
    catalog = _cadquery_catalog()
    catalog_names = {c["name"] for c in catalog}

    prompt = (
        f"USER REQUEST:\n{description}\n\n"
        f"AVAILABLE TEMPLATES (name and accepted params with min/max/default):\n{json.dumps(catalog, indent=2)}\n"
    )

    raw = call_llm(
        prompt, provider=provider, model=model, api_key=api_key, system=_SYSTEM_PROMPT_CADQUERY,
        json_mode=True, module="draft_cadquery_template", conversation_store=conversation_store,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e

    name = parsed.get("name")
    params = parsed.get("params", {}) or {}
    reasoning = parsed.get("reasoning", "")

    if name is not None:
        if name not in catalog_names:
            raise ValueError(
                f"LLM named template '{name}', not in the real registered catalog ({sorted(catalog_names)}) "
                "-- refusing to use a hallucinated template name."
            )
        accepted = next(c["accepted_params"] for c in catalog if c["name"] == name)
        unknown = set(params) - set(accepted)
        if unknown:
            raise ValueError(
                f"LLM proposed params {sorted(unknown)} that template '{name}' does not accept "
                f"(accepted: {sorted(accepted)}) -- refusing to use hallucinated parameters."
            )

    return CadRecipeResult(kind="cadquery_template", recipe=None, name=name, params=params, reasoning=reasoning, raw_response=raw)


# ---------------------------------------------------------------------------
# Execution + export (never LLM-driven -- plain, deterministic code)
# ---------------------------------------------------------------------------

def build_recipe(result: CadRecipeResult):
    """Actually build the geometry a :class:`CadRecipeResult` describes.

    Returns a ``pinneapple_design.geometry.core.mesh.MeshData`` for
    ``"mesh_recipe"``, or an in-memory CadQuery object (Workplane/Solid)
    for ``"cadquery_template"`` (requires ``cadquery`` installed).

    Re-validates the recipe here too, not just inside
    ``draft_mesh_recipe``/``draft_cadquery_template`` -- a
    :class:`CadRecipeResult` can also be constructed by hand or reloaded
    from a saved JSON file, and skipping validation at THIS boundary
    would silently let a malformed or hallucinated recipe through
    (confirmed while building this module: a node with builder params
    merged flat instead of nested under "params" builds without error
    and quietly produces the wrong-sized geometry -- exactly the
    "runs but is wrong" failure mode this repo's audit methodology
    exists to catch -- so this is checked at the point of execution,
    not trusted to have already been checked upstream).
    """
    if result.kind == "mesh_recipe":
        if result.recipe is None:
            raise ValueError("result.recipe is None -- the LLM judged nothing fit; nothing to build.")
        from pinneapple_design.geometry.gen.primitives import build_mesh, list_builders

        _validate_mesh_node(result.recipe, builder_names=set(list_builders()))
        return build_mesh(**_node_to_build_mesh_kwargs(result.recipe))

    if result.kind == "cadquery_template":
        if result.name is None:
            raise ValueError("result.name is None -- the LLM judged nothing fit; nothing to build.")
        from pinneapple_design.geometry.gen.cadquery_gen import (
            CadQueryRegistry, ParametricCadSpec, register_default_templates,
        )

        reg = register_default_templates(CadQueryRegistry())
        spec = ParametricCadSpec(
            builder_id=result.name, params=_normalize_params(result.params), schema=reg.get_schema(result.name),
        )
        return reg.build(spec)

    raise ValueError(f"unknown CadRecipeResult.kind: {result.kind!r}")


def export_recipe(result: CadRecipeResult, out_path: str, *, fmt: Optional[str] = None) -> str:
    """Export a built recipe to a real CAD/mesh file on disk.

    For ``"mesh_recipe"``: writes a genuine triangle-mesh file (``.stl``
    natively via ``trimesh``; ``.step``/``.stp`` ALSO works if
    ``cadquery`` is installed, by importing the STL into CadQuery and
    re-exporting -- but be precise about what that STEP actually is: a
    triangulated shell wrapped in the STEP container format, not a true
    parametric B-rep solid with real faces/edges. That distinction
    matters for anyone opening it in a CAD tool expecting to edit
    features -- it will look and measure correctly, but has no
    parametric history.

    For ``"cadquery_template"``: writes a genuine parametric B-rep STEP
    (or STL/other cadquery-supported format), via the real OCC kernel --
    this is the "true CAD file" tier.
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    built = build_recipe(result)
    fmt = (fmt or out_path.rsplit(".", 1)[-1]).lower()

    if result.kind == "mesh_recipe":
        import trimesh

        tm = trimesh.Trimesh(vertices=built.vertices, faces=built.faces, process=False)
        if fmt == "stl":
            tm.export(out_path)
            return out_path
        if fmt in ("step", "stp"):
            from pinneapple_design.geometry.gen.cadquery_gen import cadquery_available

            if not cadquery_available():
                raise ImportError(
                    "cadquery is not installed -- mesh_recipe STL export works without it "
                    "(pass fmt='stl'), but wrapping a mesh into STEP needs cadquery."
                )
            import tempfile
            from pathlib import Path

            import cadquery as cq

            with tempfile.TemporaryDirectory() as td:
                stl_path = Path(td) / "recipe.stl"
                tm.export(str(stl_path))
                cq_obj = cq.importers.importShape(cq.exporters.ExportTypes.STL, str(stl_path)) \
                    if hasattr(cq.importers, "importShape") else None
                if cq_obj is None:
                    # Older/other cadquery versions: build a Shape directly from the STL mesh.
                    shape = cq.Shape.importStl(str(stl_path)) if hasattr(cq.Shape, "importStl") else None
                    if shape is None:
                        raise RuntimeError("this cadquery version has no STL->STEP import path available")
                    cq_obj = cq.Workplane(obj=shape)
                cq.exporters.export(cq_obj, out_path)
            return out_path
        raise ValueError(f"unsupported export format for a mesh_recipe: {fmt!r} (use 'stl' or 'step')")

    if result.kind == "cadquery_template":
        from pinneapple_design.geometry.gen.cadquery_gen import export_cadquery

        return str(export_cadquery(built, out_path, fmt=fmt))

    raise ValueError(f"unknown CadRecipeResult.kind: {result.kind!r}")
