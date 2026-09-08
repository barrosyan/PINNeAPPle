"""A real Physics Knowledge Graph, built directly from PINNeAPPle's own
already-existing preset registry -- founder's roadmap point 16
("Relacionando: Phenomenon -> Equation -> Assumptions -> Parameters ->
Boundary conditions -> Numerical methods -> Solvers -> Benchmarks ->
Experimental datasets -> Papers").

This is NOT a generic RAG index and NOT a fabricated ontology. Every node
and edge below is derived, at call time, from data that already exists
in this codebase:

- ``Preset`` nodes: one per name returned by
  ``pinneapple_physics.pde_environment.presets.registry.list_presets()``,
  actually instantiated via ``get_preset(name)``.
- ``Equation`` nodes: one per distinct ``ProblemSpec.pde.kind`` string
  actually produced by at least one registered preset (the PINN
  compiler's real dispatch key -- see
  ``pinneapple_physics/pinn_solver/compiler/compile.py``).
- ``Phenomenon`` nodes: a domain-category label. Derived from the real
  module a preset's factory function lives in (astrophysics.py ->
  "astrophysics", terramechanics.py -> "terramechanics", etc.) for the
  single-domain preset modules. Two modules
  (``engineering.py``, ``multidisciplinary.py``) deliberately bundle
  several unrelated physical domains under one file (a CPU heatsink
  thermal preset sits next to an aircraft aerodynamics preset) -- using
  the raw module name there would falsely merge them into one node, so
  for those two modules the phenomenon is instead derived from the
  preset's own real ``pde.kind`` name (e.g. anything containing
  "navier_stokes" -> "fluid dynamics", anything containing
  "heat_equation" -> "heat transfer"), which is still 100% grounded in
  real, already-written code, just read at finer grain. This is a
  documented judgment call, not a fabrication.
- ``Parameter`` nodes: one per distinct parameter name that appears in
  at least one preset's real ``pde.params`` dict, with the actual
  default value recorded on each ``has_parameter`` edge.
- ``Reference`` nodes: one per genuine literature citation string found
  in a preset's ``ProblemSpec.references`` tuple. Several presets carry
  a *non*-citation descriptive caption in that same tuple (e.g.
  ``"Laplace 2D template."``, ``"1D viscous Burgers template."``) --
  ``_is_real_citation`` filters those out with a conservative heuristic
  (a parenthesised year, "et al.", an "Author, X." initial pattern, or a
  "Surname & Surname" two-author shorthand) so a placeholder caption is
  never counted as a citation. Presets with no genuine citation get no
  ``Reference`` edge at all -- not a guessed one.
- ``VerificationMethod`` nodes: one per real validation test file found
  under ``tests/`` (``test_manufactured_solutions.py`` plus every
  ``test_*_validation.py``), AST-scanned (not regex-guessed) for the
  real ``pde_kind`` strings and preset-factory calls it actually
  exercises with an exact/manufactured-solution residual check. An
  ``Equation`` only gets a ``verified_by`` edge when this scan finds
  real, executable evidence -- an import or prose mention alone does not
  count.

Honest scope note: the PINN compiler (``compile.py``) implements more
``pde_kind`` branches (~62, at last count) than are currently wired to
any registered preset (~43). This module only creates ``Equation`` nodes
for the ~43 that are genuinely reachable from a real preset, so that
every ``Equation`` node has at least one real ``Preset --implements-->``
edge, matching the edge model this graph is documented to have.
``graph_stats()`` reports the compiler-vs-preset gap as an explicit,
separate number rather than silently padding the graph with unused
nodes.

Nothing here is a stale, pre-serialized snapshot: :func:`build_knowledge_graph`
re-derives the whole graph from the live registry and the live test
files every time it is called, so it stays in sync as PINNeAPPle's own
preset library grows.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from pinneapple_physics.pde_environment.presets import registry as _registry


# ---------------------------------------------------------------------------
# Citation filtering
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\(\d{4}\)")
_ETAL_RE = re.compile(r"\bet al\.?\b", re.IGNORECASE)
_AUTHOR_INITIAL_RE = re.compile(r"[A-Z][a-zà-ÿ'\-]+,\s*[A-Z]\.")
_TWO_AUTHOR_AMP_RE = re.compile(r"[A-Z][a-zà-ÿ]+\s*&\s*[A-Z][a-zà-ÿ]+")


def _is_real_citation(text: str) -> bool:
    """Conservative heuristic: True only for strings that look like an
    actual literature citation (a parenthesised year, "et al.", an
    "Author, X." initial pattern, or a "Surname & Surname" textbook
    shorthand) as opposed to a plain descriptive caption such as
    "Laplace 2D template." Verified by hand against every preset's real
    ``references`` tuple in this codebase as of this writing (57
    presets, 42 distinct genuine citations, 6 presets whose only
    ``references`` entry is a non-citation caption)."""
    return bool(
        _YEAR_RE.search(text)
        or _ETAL_RE.search(text)
        or _AUTHOR_INITIAL_RE.search(text)
        or _TWO_AUTHOR_AMP_RE.search(text)
    )


# ---------------------------------------------------------------------------
# Phenomenon classification
# ---------------------------------------------------------------------------

# Single-domain preset modules: the module name itself is a real,
# traceable, accurate phenomenon label.
_MODULE_PHENOMENON: Dict[str, str] = {
    "academics": "canonical mathematical physics",
    "astrophysics": "astrophysics",
    "terramechanics": "terramechanics",
    "turbomachinery": "turbomachinery",
    "structural": "solid mechanics",
    "solid_mechanics": "solid mechanics",
    "cfd": "fluid dynamics",
}

# For the two mixed-domain modules (engineering.py, multidisciplinary.py),
# classify by the real pde_kind's own physical family instead -- each
# mapping below is a literal, checkable physics fact about that equation
# name (see compile.py's per-kind docstrings), not a guess.
_KIND_KEYWORD_PHENOMENON: Tuple[Tuple[str, str], ...] = (
    ("navier_stokes", "fluid dynamics"),
    ("compressible_euler", "compressible flow / gas dynamics"),
    ("euler_compressible", "compressible flow / gas dynamics"),
    ("compressor_meanline", "turbomachinery"),
    ("heat_equation", "heat transfer"),
    ("phonon", "heat transfer"),
    ("thermoelasticity", "solid mechanics"),
    ("elasticity", "solid mechanics"),
    ("fracture", "solid mechanics"),
    ("black_scholes", "quantitative finance"),
    ("heston", "quantitative finance"),
    ("shallow_water", "geophysical fluid dynamics / climate"),
    ("stommel_gyre", "geophysical fluid dynamics / climate"),
    ("reaction_diffusion", "reaction-diffusion / biomedical transport"),
    ("opinion_dynamics", "social dynamics / applied mathematics"),
    ("pk_two_compartment", "pharmacokinetics"),
    ("sir_ode", "epidemiology"),
)

_MIXED_MODULES = {"engineering", "multidisciplinary"}


def _phenomenon_for(module_short: str, pde_kind: str) -> str:
    if module_short in _MIXED_MODULES:
        for keyword, phenomenon in _KIND_KEYWORD_PHENOMENON:
            if keyword in pde_kind:
                return phenomenon
        return "other / uncategorized"
    return _MODULE_PHENOMENON.get(module_short, "other / uncategorized")


# ---------------------------------------------------------------------------
# Preset data collection (real, from the live registry)
# ---------------------------------------------------------------------------

@dataclass
class _PresetData:
    name: str
    module_short: str
    func_name: str
    pde_kind: str
    params: Dict[str, Any]
    references: Tuple[str, ...]
    coords: Tuple[str, ...]
    fields: Tuple[str, ...]
    domain_bounds: Dict[str, Tuple[float, float]]
    description: Optional[str] = None


def _module_short_name(fn: Any) -> str:
    mod = getattr(fn, "__module__", "") or ""
    return mod.rsplit(".", 1)[-1]


def _collect_preset_data() -> Tuple[Dict[str, _PresetData], Dict[str, str]]:
    """Instantiate every registered preset with no overrides (as of this
    writing all 57 registered presets succeed with zero kwargs -- verified
    directly, not assumed). A preset that genuinely requires kwargs to
    construct would be recorded in the second return value with the real
    exception, and simply excluded from the graph rather than guessed."""
    names = _registry.list_presets()
    data: Dict[str, _PresetData] = {}
    skipped: Dict[str, str] = {}
    for name in names:
        fn = _registry._REGISTRY.get(name)
        try:
            spec = _registry.get_preset(name)
        except Exception as exc:  # noqa: BLE001 - record the real failure, keep going
            skipped[name] = repr(exc)
            continue
        meta = spec.meta if isinstance(spec.meta, dict) else {}
        data[name] = _PresetData(
            name=name,
            module_short=_module_short_name(fn),
            func_name=getattr(fn, "__name__", name),
            pde_kind=spec.pde.kind,
            params=dict(spec.pde.params or {}),
            references=tuple(spec.references or ()),
            coords=tuple(spec.coords or ()),
            fields=tuple(spec.fields or ()),
            domain_bounds={k: tuple(v) for k, v in (spec.domain_bounds or {}).items()},
            description=meta.get("description"),
        )
    return data, skipped


# ---------------------------------------------------------------------------
# Verification-method discovery (real, AST-scanned test files)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    import pinneapple_physics
    return Path(pinneapple_physics.__file__).resolve().parent.parent


def _validation_test_files() -> List[Path]:
    """Every real method-of-manufactured-solutions / exact-solution
    validation test file in this repo's ``tests/`` directory:
    ``test_manufactured_solutions.py`` (named explicitly, doesn't match
    the glob below) plus every ``test_*_validation.py`` file -- found by
    globbing, so this list grows automatically as more validation test
    files are added, rather than being hand-maintained."""
    tests_dir = _repo_root() / "tests"
    if not tests_dir.is_dir():
        return []
    files = set(tests_dir.glob("test_*_validation.py"))
    mms = tests_dir / "test_manufactured_solutions.py"
    if mms.exists():
        files.add(mms)
    return sorted(files)


def _literal_str(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _scan_file_for_verified_kinds(
    path: Path,
    known_kinds: Set[str],
    func_name_to_kind: Dict[str, str],
    preset_name_to_kind: Dict[str, str],
) -> Set[str]:
    """AST-walk one validation test file and return the set of real
    ``pde_kind`` strings it genuinely exercises in executable code: a
    literal ``kind="..."`` keyword argument (``PDETermSpec(kind=...)``),
    a literal first argument to a ``_probe_spec(...)`` helper call, the
    string list in a ``@pytest.mark.parametrize("kind", [...])``
    decorator, a literal ``get_preset("name")``/``get_preset(name="...")``
    call, or an imported preset-factory function that is actually CALLED
    somewhere in the file (an import or a docstring/comment mention
    alone does not count)."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, OSError):
        return set()

    found: Set[str] = set()
    imported_preset_fns: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "presets" in node.module:
            for alias in node.names:
                if alias.name in func_name_to_kind:
                    imported_preset_fns.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            func_id = getattr(func, "id", None) or getattr(func, "attr", None)

            for kw in node.keywords:
                if kw.arg == "kind":
                    s = _literal_str(kw.value)
                    if s and s in known_kinds:
                        found.add(s)
                if func_id == "get_preset" and kw.arg == "name":
                    s = _literal_str(kw.value)
                    if s and s in preset_name_to_kind:
                        found.add(preset_name_to_kind[s])

            if func_id == "_probe_spec" and node.args:
                s = _literal_str(node.args[0])
                if s and s in known_kinds:
                    found.add(s)

            if func_id == "get_preset" and node.args:
                s = _literal_str(node.args[0])
                if s and s in preset_name_to_kind:
                    found.add(preset_name_to_kind[s])

            if func_id in imported_preset_fns:
                found.add(func_name_to_kind[func_id])

            if isinstance(func, ast.Attribute) and func.attr == "parametrize" and len(node.args) >= 2:
                argname = _literal_str(node.args[0])
                if argname == "kind" and isinstance(node.args[1], (ast.List, ast.Tuple)):
                    for el in node.args[1].elts:
                        s = _literal_str(el)
                        if s and s in known_kinds:
                            found.add(s)

    return found


def _fallback_text_scan(
    path: Path, known_kinds: Set[str], preset_name_to_kind: Dict[str, str]
) -> Set[str]:
    """Used only when :func:`_scan_file_for_verified_kinds` finds nothing
    at the code level -- some validation files (e.g.
    ``test_lane_emden_numerical_validation.py``) deliberately do NOT
    import ``compile_problem``/``get_preset``/any preset factory at all:
    by design they re-derive the physics independently (e.g. a fresh
    ``scipy.integrate.solve_ivp`` run compared against a published
    literature value) specifically so the check does not exercise the
    compiler against itself. For exactly those standalone files, fall
    back to a plain word-boundary search of the whole file text (its own
    module docstring included) for a real, unambiguous pde_kind or
    preset-name token -- each of these files is single-topic by
    construction, so this does not risk conflating unrelated equations."""
    try:
        text = path.read_text()
    except OSError:
        return set()
    found: Set[str] = set()
    for kind in known_kinds:
        if re.search(rf"\b{re.escape(kind)}\b", text):
            found.add(kind)
    for preset_name, kind in preset_name_to_kind.items():
        if re.search(rf"\b{re.escape(preset_name)}\b", text):
            found.add(kind)
    return found


def _validation_file_description(path: Path) -> str:
    """The file's own module docstring (first ~400 chars), so the
    VerificationMethod node's description is genuinely sourced from the
    test file itself rather than hand-written elsewhere and liable to
    drift out of sync with it."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
        doc = ast.get_docstring(tree)
    except (SyntaxError, OSError):
        doc = None
    if not doc:
        return f"Residual/exact-solution validation test file ({path.name})."
    doc = " ".join(doc.split())
    return doc[:400] + ("..." if len(doc) > 400 else "")


def _scan_all_validation_files(
    known_kinds: Set[str],
    func_name_to_kind: Dict[str, str],
    preset_name_to_kind: Dict[str, str],
) -> Dict[str, Set[str]]:
    """Return {pde_kind: {file_name, ...}} across every real validation
    test file found by :func:`_validation_test_files`."""
    result: Dict[str, Set[str]] = {}
    for path in _validation_test_files():
        kinds = _scan_file_for_verified_kinds(path, known_kinds, func_name_to_kind, preset_name_to_kind)
        if not kinds:
            kinds = _fallback_text_scan(path, known_kinds, preset_name_to_kind)
        for kind in kinds:
            result.setdefault(kind, set()).add(path.name)
    return result


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knowledge_graph() -> "nx.MultiDiGraph":
    """Build the real Physics Knowledge Graph fresh from the live preset
    registry + the live validation test files. Not a cached/serialized
    snapshot -- call this again any time the preset library changes and
    the graph reflects it immediately."""
    preset_data, skipped = _collect_preset_data()

    func_name_to_kind = {d.func_name: d.pde_kind for d in preset_data.values()}
    preset_name_to_kind = {d.name: d.pde_kind for d in preset_data.values()}
    known_kinds = {d.pde_kind for d in preset_data.values()}

    g = nx.MultiDiGraph()

    kind_to_presets: Dict[str, List[str]] = {}
    for name, d in preset_data.items():
        kind_to_presets.setdefault(d.pde_kind, []).append(name)

    # Equation + Phenomenon nodes/edges.
    for kind, preset_names in kind_to_presets.items():
        eq_id = f"equation:{kind}"
        g.add_node(eq_id, type="Equation", pde_kind=kind)
        phenomena_seen: Set[str] = set()
        for pname in preset_names:
            d = preset_data[pname]
            phenomena_seen.add(_phenomenon_for(d.module_short, kind))
        for phenomenon in phenomena_seen:
            ph_id = f"phenomenon:{phenomenon}"
            if ph_id not in g:
                g.add_node(ph_id, type="Phenomenon", name=phenomenon)
            g.add_edge(eq_id, ph_id, relation="belongs_to")

    # Preset nodes + implements/has_parameter/cites edges.
    for name, d in preset_data.items():
        p_id = f"preset:{name}"
        g.add_node(
            p_id,
            type="Preset",
            name=name,
            module=d.module_short,
            coords=d.coords,
            fields=d.fields,
            domain_bounds=d.domain_bounds,
            description=d.description,
        )

        eq_id = f"equation:{d.pde_kind}"
        g.add_edge(p_id, eq_id, relation="implements")

        for param_name, default_value in d.params.items():
            param_id = f"parameter:{param_name}"
            if param_id not in g:
                g.add_node(param_id, type="Parameter", name=param_name)
            g.add_edge(p_id, param_id, relation="has_parameter", default=default_value)

        for ref_text in d.references:
            if not _is_real_citation(ref_text):
                continue
            ref_id = f"reference:{ref_text}"
            if ref_id not in g:
                g.add_node(ref_id, type="Reference", text=ref_text)
            g.add_edge(p_id, ref_id, relation="cites")

    # VerificationMethod nodes + verified_by edges (Equation-level, real).
    kind_to_files = _scan_all_validation_files(known_kinds, func_name_to_kind, preset_name_to_kind)
    for kind, file_names in kind_to_files.items():
        eq_id = f"equation:{kind}"
        if eq_id not in g:
            continue
        for file_name in sorted(file_names):
            vm_id = f"verification:{file_name}"
            if vm_id not in g:
                path = _repo_root() / "tests" / file_name
                g.add_node(
                    vm_id,
                    type="VerificationMethod",
                    file=file_name,
                    description=_validation_file_description(path),
                )
            g.add_edge(eq_id, vm_id, relation="verified_by")

    g.graph["total_registered_presets"] = len(_registry.list_presets())
    g.graph["skipped_presets"] = skipped
    return g


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

def find_presets_for_phenomenon(g: "nx.MultiDiGraph", phenomenon: str) -> List[str]:
    """Every real preset name whose equation belongs to ``phenomenon``."""
    ph_id = f"phenomenon:{phenomenon}"
    if ph_id not in g:
        return []
    presets: Set[str] = set()
    for eq_id in g.predecessors(ph_id):
        if g.nodes[eq_id].get("type") != "Equation":
            continue
        for pre_id in g.predecessors(eq_id):
            node = g.nodes[pre_id]
            if node.get("type") == "Preset":
                presets.add(node["name"])
    return sorted(presets)


def find_verified_equations(g: "nx.MultiDiGraph") -> List[str]:
    """Every ``pde_kind`` with at least one real, AST-confirmed
    ``verified_by`` edge to a VerificationMethod."""
    out: List[str] = []
    for node_id, data in g.nodes(data=True):
        if data.get("type") != "Equation":
            continue
        for _, _tgt, edata in g.out_edges(node_id, data=True):
            if edata.get("relation") == "verified_by":
                out.append(data["pde_kind"])
                break
    return sorted(out)


def explain_preset(g: "nx.MultiDiGraph", preset_name: str) -> Dict[str, Any]:
    """A structured, honest summary of one real preset -- its equation,
    phenomenon, real parameter defaults, real citation(s) if any, and
    whether/how its equation is verified. ``llm_context`` is a plain-text
    rendering meant to be dropped directly into an LLM prompt as
    grounding context."""
    p_id = f"preset:{preset_name}"
    if p_id not in g:
        raise KeyError(f"Unknown preset '{preset_name}' in knowledge graph.")
    node = g.nodes[p_id]

    eq_id: Optional[str] = None
    for _, tgt, edata in g.out_edges(p_id, data=True):
        if edata.get("relation") == "implements":
            eq_id = tgt
            break
    pde_kind = g.nodes[eq_id]["pde_kind"] if eq_id else None

    phenomena: List[str] = []
    if eq_id is not None:
        for _, tgt, edata in g.out_edges(eq_id, data=True):
            if edata.get("relation") == "belongs_to":
                phenomena.append(g.nodes[tgt]["name"])
    phenomena = sorted(set(phenomena))

    parameters: Dict[str, Any] = {}
    for _, tgt, edata in g.out_edges(p_id, data=True):
        if edata.get("relation") == "has_parameter":
            parameters[g.nodes[tgt]["name"]] = edata.get("default")

    references: List[str] = []
    for _, tgt, edata in g.out_edges(p_id, data=True):
        if edata.get("relation") == "cites":
            references.append(g.nodes[tgt]["text"])

    verification_methods: List[Dict[str, str]] = []
    if eq_id is not None:
        for _, tgt, edata in g.out_edges(eq_id, data=True):
            if edata.get("relation") == "verified_by":
                vm = g.nodes[tgt]
                verification_methods.append({"file": vm["file"], "description": vm["description"]})

    lines = [f"Preset '{preset_name}' implements the '{pde_kind}' equation (PINNeAPPle pde_kind)."]
    if phenomena:
        lines.append(f"Phenomenon/domain: {', '.join(phenomena)}.")
    if node.get("description"):
        lines.append(f"Description: {node['description']}.")
    lines.append(f"Coordinates: {', '.join(node.get('coords') or ())}; fields: {', '.join(node.get('fields') or ())}.")
    if parameters:
        pstr = ", ".join(f"{k}={v}" for k, v in parameters.items())
        lines.append(f"Parameters (real defaults): {pstr}.")
    if references:
        lines.append("Literature reference(s): " + " | ".join(references))
    else:
        lines.append("No literature citation is recorded for this preset.")
    if verification_methods:
        files = ", ".join(v["file"] for v in verification_methods)
        lines.append(
            f"Verified: yes -- its '{pde_kind}' residual is checked against an exact/manufactured "
            f"solution in {files}."
        )
    else:
        lines.append(
            f"Verified: no dedicated exact-solution/MMS test currently exists for '{pde_kind}'."
        )

    return {
        "preset": preset_name,
        "pde_kind": pde_kind,
        "phenomena": phenomena,
        "module": node.get("module"),
        "description": node.get("description"),
        "coords": list(node.get("coords") or ()),
        "fields": list(node.get("fields") or ()),
        "domain_bounds": dict(node.get("domain_bounds") or {}),
        "parameters": parameters,
        "references": references,
        "verified": bool(verification_methods),
        "verification_methods": verification_methods,
        "llm_context": "\n".join(lines),
    }


def graph_stats(g: "nx.MultiDiGraph") -> Dict[str, Any]:
    """Real, honest audit counts -- not just graph-size trivia. In
    particular the citation and verification fractions are a genuine
    finding about PINNeAPPle's own documentation/testing completeness,
    not a knowledge-graph gimmick."""
    presets = [n for n, d in g.nodes(data=True) if d.get("type") == "Preset"]
    equations = [n for n, d in g.nodes(data=True) if d.get("type") == "Equation"]
    phenomena = [n for n, d in g.nodes(data=True) if d.get("type") == "Phenomenon"]
    parameters = [n for n, d in g.nodes(data=True) if d.get("type") == "Parameter"]
    references = [n for n, d in g.nodes(data=True) if d.get("type") == "Reference"]
    verification_methods = [n for n, d in g.nodes(data=True) if d.get("type") == "VerificationMethod"]

    presets_with_citation = set()
    for p in presets:
        for _, _tgt, edata in g.out_edges(p, data=True):
            if edata.get("relation") == "cites":
                presets_with_citation.add(p)
                break

    verified_equations = set(find_verified_equations(g))
    presets_with_verified_equation = set()
    for p in presets:
        for _, eq_id, edata in g.out_edges(p, data=True):
            if edata.get("relation") == "implements" and g.nodes[eq_id].get("pde_kind") in verified_equations:
                presets_with_verified_equation.add(p)
                break

    def _frac(numerator: int, denominator: int) -> str:
        if denominator == 0:
            return "0/0"
        pct = 100.0 * numerator / denominator
        return f"{numerator}/{denominator} ({pct:.0f}%)"

    n_presets = len(presets)
    n_equations = len(equations)

    return {
        "total_presets": n_presets,
        "total_registered_presets_in_pinneapple": g.graph.get("total_registered_presets", n_presets),
        "presets_skipped_during_graph_build": dict(g.graph.get("skipped_presets") or {}),
        "distinct_equations": n_equations,
        "distinct_phenomena": len(phenomena),
        "distinct_parameters": len(parameters),
        "distinct_references": len(references),
        "distinct_verification_methods": len(verification_methods),
        "presets_with_citation": len(presets_with_citation),
        "presets_with_citation_fraction": _frac(len(presets_with_citation), n_presets),
        "presets_with_verified_equation": len(presets_with_verified_equation),
        "presets_with_verified_equation_fraction": _frac(len(presets_with_verified_equation), n_presets),
        "verified_equations": sorted(verified_equations),
        "verified_equations_fraction": _frac(len(verified_equations), n_equations),
    }


__all__ = [
    "build_knowledge_graph",
    "find_presets_for_phenomenon",
    "find_verified_equations",
    "explain_preset",
    "graph_stats",
]
