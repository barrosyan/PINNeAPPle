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
_CUT_OPS = {"cut", "diff", "difference"}

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

A <node> is EITHER:
(a) a plain primitive, optionally unioned/intersected with another node: \
{"name": "<a builder name from AVAILABLE BUILDERS>", "params": {<builder's own params>}, \
"boolean": {"op": "union"|"intersect", "other": <node>}}
(b) ONLY for cutting a hole, a compound cut node -- see "cut" below -- \
which has NO "name"/"params" of its own: \
{"boolean": {"op": "cut", "base": <node>, "tool": <node>}}

"params" is a flat object of that builder's own parameters (e.g. \
{"radius": 0.5} for a sphere) -- always include it, even if empty ({}), on \
every node that has a "name".

"union" (merge) and "intersect" (keep only the overlap) are SYMMETRIC: the \
result is identical no matter which shape is "this node" and which is \
"other", so operand order never matters for them.

"cut" is DIFFERENT and DIRECTIONAL: material is actually removed, so \
swapping the two shapes produces a DIFFERENT, WRONG piece of geometry even \
though it still builds without error (this is a real, previously-observed \
failure mode -- getting this backwards is silent, not a crash). For this \
reason "cut" is NEVER written using form (a)'s implicit-"self" shape --  \
always use the explicit compound form (b) instead: \
{"boolean": {"op": "cut", "base": <node>, "tool": <node>}}, where:
- "base" is the shape that KEEPS its material -- the solid block that ends \
up with a hole in it (e.g. the box).
- "tool" is the shape used to remove material FROM "base", like a drill \
bit -- it does not appear in the final result itself, only the negative \
imprint it leaves behind (e.g. the cylinder that becomes the hole).
For "a box with a cylindrical hole cut through it": "base" is the box, \
"tool" is the cylinder. NEVER the other way around -- double-check which \
of the two shapes is the container being drilled into (base) versus the \
drill/hole shape (tool) before writing the JSON.
Either "base" or "tool" can itself be an arbitrarily nested node (its own \
"boolean" of any op, including another cut), for more complex shapes.

"boolean" is OPTIONAL on a plain node (form (a)) -- omit it entirely for a \
single primitive with no CSG combination at all.

Rules:
- "name" (on every node that has one) MUST be exactly one of the names in \
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


def _normalize_recipe_node(node: Any) -> Any:
    """Undo another real, observed local-model quirk with the new
    explicit-cut ``{"boolean": {"op": "cut", "base": <node>, "tool":
    <node>}}`` schema: llama3.2:3b sometimes closes the "boolean" object
    one bracket too early (right after "base") and writes "tool" as a
    SIBLING of "boolean" inside the enclosing recipe node instead of
    nested inside it -- i.e. ``{"boolean": {"op": "cut", "base": ...}, \
    "tool": ...}`` instead of ``{"boolean": {"op": "cut", "base": ..., \
    "tool": ...}}``. This is a pure bracket-placement slip, not a
    semantic one: "base"/"tool" are still individually present and still
    the semantically-correct pair (confirmed across repeated real runs --
    the model is not confused about WHICH shape is which, only about
    where the closing brace belongs), so this is a lossless,
    unambiguous structural re-nesting -- exactly the same kind of fix as
    ``_normalize_param_value``'s stringified-array recovery, not a guess
    at any missing value. A response that is ACTUALLY missing "tool"
    entirely (not just misplaced) is left untouched and still fails
    naturally in ``_validate_mesh_node``, as it should."""
    if not isinstance(node, dict):
        return node
    boolean = node.get("boolean")
    if (
        isinstance(boolean, dict)
        and str(boolean.get("op", "")).lower().strip() in _CUT_OPS
        and "base" in boolean and "tool" not in boolean
        and "tool" in node
    ):
        node = dict(node)
        boolean = dict(boolean)
        boolean["tool"] = node.pop("tool")
        node["boolean"] = boolean
    # Recurse into every nested node position, whichever shape this node
    # turned out to be.
    boolean = node.get("boolean")
    if isinstance(boolean, dict):
        boolean = dict(boolean)
        for key in ("base", "tool", "other"):
            if key in boolean:
                boolean[key] = _normalize_recipe_node(boolean[key])
        node = dict(node)
        node["boolean"] = boolean
    return node


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
    the matching ``params`` field in the prompt/schema were added.

    A "cut" node may be shaped two ways: the legacy implicit-"self" form
    (``{"name", "params", "boolean": {"op": "cut", "other": <node>}}``,
    where THIS node's own name/params are the base and "other" is the
    tool being removed), still accepted here for backward compatibility
    with hand-built/previously-saved recipes; or the newer, explicit
    compound form (``{"boolean": {"op": "cut", "base": <node>, "tool":
    <node>}}``, with no "name"/"params" of its own), which the prompt now
    exclusively teaches the LLM to use instead -- see the module
    docstring and ``_SYSTEM_PROMPT_MESH`` for why the implicit-"self"
    shape was a real, observed source of the model swapping which shape
    keeps its material versus which one is drilled out."""
    if not isinstance(node, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(node).__name__}")
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
    is_explicit_cut = (
        isinstance(boolean, dict)
        and "base" in boolean and "tool" in boolean
        and str(boolean.get("op", "")).lower().strip() in _CUT_OPS
    )
    if is_explicit_cut:
        # Explicit compound cut node: "name"/"params" of the ENCLOSING
        # node are not required (a cut node's identity comes entirely
        # from "base"/"tool") -- but if a model includes a "name" anyway,
        # still reject it if it's a hallucinated one rather than
        # silently ignoring a signal that something is off.
        if "name" in node and node["name"] not in builder_names:
            raise ValueError(
                f"{path}.name: LLM named builder '{node['name']}', not in the real registry "
                f"({sorted(builder_names)}) -- refusing to use a hallucinated builder name."
            )
        _validate_mesh_node(boolean["base"], builder_names=builder_names, path=f"{path}.boolean.base")
        _validate_mesh_node(boolean["tool"], builder_names=builder_names, path=f"{path}.boolean.tool")
        return
    name = node.get("name")
    if name not in builder_names:
        raise ValueError(
            f"{path}.name: LLM named builder '{name}', not in the real registry "
            f"({sorted(builder_names)}) -- refusing to use a hallucinated builder name."
        )
    params = node.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{path}.params: expected a JSON object, got {type(params).__name__}: {params!r}")
    if boolean is not None:
        if not isinstance(boolean, dict) or "op" not in boolean or "other" not in boolean:
            raise ValueError(
                f"{path}.boolean: must be an object with 'op' and 'other' keys (or, for 'cut', "
                f"'base'+'tool' instead of 'other'), got: {boolean!r}"
            )
        op = str(boolean["op"]).lower().strip()
        if op not in _ALLOWED_BOOLEAN_OPS:
            raise ValueError(
                f"{path}.boolean.op: '{op}' is not a real CSG op ({sorted(_ALLOWED_BOOLEAN_OPS)}) "
                "-- refusing to use a hallucinated operation."
            )
        _validate_mesh_node(boolean["other"], builder_names=builder_names, path=f"{path}.boolean.other")


def _node_to_build_mesh_kwargs(node: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a validated node into the flat ``**params`` shape
    ``primitives.build_mesh`` itself expects (name + its params merged at
    one level, with an optional "boolean" key whose "other" is the same
    flat shape, recursively) -- this conversion is what lets the
    LLM-facing schema stay recursively self-similar and explicit about
    which keys are real builder params (nested under "params") without
    that ambiguity leaking into ``build_mesh``'s own, differently-shaped
    calling convention.

    Handles both accepted "cut" shapes: the explicit compound form
    (``{"boolean": {"op": "cut", "base": <node>, "tool": <node>}}``) is
    flattened by converting "base" first (its own name/params/nested
    booleans become THIS flat dict, exactly as if "base" had been the
    node all along) and then attaching "tool" as build_mesh's own
    "other" -- i.e. "base"+"tool" desugars directly to the same flat
    ``{"op": "cut", "other": <tool-flat>}`` shape build_mesh has always
    taken, so no change was needed downstream in build_mesh/primitives
    itself. The legacy implicit-"self" form (name/params at this level,
    "boolean": {"op": ..., "other": <node>}) is flattened as before."""
    boolean = node.get("boolean")
    is_explicit_cut = (
        isinstance(boolean, dict)
        and "base" in boolean and "tool" in boolean
        and str(boolean.get("op", "")).lower().strip() in _CUT_OPS
    )
    if is_explicit_cut:
        flat = _node_to_build_mesh_kwargs(boolean["base"])
        flat["boolean"] = {"op": boolean["op"], "other": _node_to_build_mesh_kwargs(boolean["tool"])}
        return flat
    flat = _normalize_params(node.get("params", {}))
    flat["name"] = node["name"]
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
        recipe = _normalize_recipe_node(recipe)
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

        recipe = _normalize_recipe_node(result.recipe)
        _validate_mesh_node(recipe, builder_names=set(list_builders()))
        return build_mesh(**_node_to_build_mesh_kwargs(recipe))

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
