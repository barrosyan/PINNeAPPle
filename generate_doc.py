from __future__ import annotations

import os
import ast
import json
import shutil
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# CONFIG
# =========================
TOP_PACKAGES = [
    "pinneaple_data",
    "pinneaple_geom",
    "pinneaple_models",
    "pinneaple_pdb",
    "pinneaple_pinn",
    "pinneaple_problemdesign",
    "pinneaple_solvers",
    "pinneaple_timeseries",
    "pinneaple_train",
]

SITE_NAME = "PINNeAPPle Docs"

# Examples folder name candidates (repo root)
EXAMPLES_DIR_CANDIDATES = ["examples", "Examples"]

# File types to include in Examples docs
EXAMPLE_EXTS = {".py", ".md"}

# GitHub Pages settings (set via env vars or edit defaults)
# - Project Pages: https://<user>.github.io/<repo>/
SITE_URL = os.environ.get("DOCS_SITE_URL", "https://barrosyan.github.io/PINNeAPPle/")
REPO_URL = os.environ.get("DOCS_REPO_URL", "https://github.com/barrosyan/PINNeAPPle")

# Ignore examples folders starting with these prefixes
EXAMPLES_IGNORE_PREFIXES = ("_", ".")  # e.g. "_runs", ".cache"


# =========================
# PATHS
# =========================
REPO_ROOT = Path(os.getcwd()).resolve()
OUTPUT_JSON = REPO_ROOT / "package_map.json"

DOCS_DIR = REPO_ROOT / "docs"
REF_DIR = DOCS_DIR / "reference"
EX_DOCS_DIR = DOCS_DIR / "examples"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"


def find_examples_root() -> Optional[Path]:
    for name in EXAMPLES_DIR_CANDIDATES:
        p = REPO_ROOT / name
        if p.exists() and p.is_dir():
            return p
    return None


EXAMPLES_ROOT = find_examples_root()


# =========================
# SMALL HELPERS
# =========================
def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_clean_docs():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for d in [REF_DIR, EX_DOCS_DIR]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)


def write_pages_file(dir_path: Path, nav_items: List[Any]):
    """
    Writes .pages YAML for mkdocs-awesome-pages-plugin.
    nav_items can contain strings or dicts {Title: path}.
    """
    lines: List[str] = ["nav:"]
    for item in nav_items:
        if isinstance(item, str):
            lines.append(f"  - {item}")
        elif isinstance(item, dict):
            for k, v in item.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append(f"  - {str(item)}")
    write_text(dir_path / ".pages", "\n".join(lines) + "\n")


def natural_sort_key(name: str) -> Tuple[int, str]:
    """
    Sort like: 01_foo, 02_bar, 10_baz, then others alphabetically.
    """
    m = re.match(r"^(\d+)[-_ ].*", name)
    if m:
        return (int(m.group(1)), name.lower())
    return (10**9, name.lower())


def rel_link(from_doc: Path, to_doc: Path) -> str:
    """
    Build a relative link from one docs markdown file to another.
    Always uses forward slashes.
    """
    rel = os.path.relpath(to_doc, start=from_doc.parent)
    return rel.replace("\\", "/")


def make_quick_summary_text(
    title: str,
    package_doc: Optional[str],
    subpackages: List[str],
    modules: List[str],
    top_classes: List[str],
    top_functions: List[str],
) -> str:
    lines: List[str] = []
    lines.append(f"# `{title}`")
    lines.append("")
    lines.append("## Quick summary")
    lines.append("")
    if package_doc and package_doc.strip():
        lines.append(package_doc.strip())
        lines.append("")
    else:
        lines.append(
            "This package page is auto-generated. Add a package-level docstring in `__init__.py` to improve this summary."
        )
        lines.append("")

    if subpackages:
        lines.append("### Subpackages")
        lines.append("")
        for s in subpackages[:40]:
            lines.append(f"- `{s}`")
        lines.append("")

    if modules:
        lines.append("### Key modules")
        lines.append("")
        for m in modules[:40]:
            lines.append(f"- `{m}`")
        lines.append("")

    if top_classes:
        lines.append("### Key classes")
        lines.append("")
        for c in top_classes[:30]:
            lines.append(f"- `{c}`")
        lines.append("")

    if top_functions:
        lines.append("### Key functions")
        lines.append("")
        for f in top_functions[:30]:
            lines.append(f"- `{f}`")
        lines.append("")

    lines.append('!!! tip "Improve this page"')
    lines.append("    Add docstrings to modules/classes/functions. They will be shown in module pages automatically.")
    lines.append("")
    return "\n".join(lines)


# =========================
# AST PARSING (no imports)
# =========================
def ast_unparse(node: ast.AST) -> Optional[str]:
    try:
        return ast.unparse(node)
    except Exception:
        return None


def get_signature(fn: ast.AST) -> str:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "()"

    a = fn.args
    parts: List[str] = []

    def fmt_arg(arg: ast.arg) -> str:
        return arg.arg

    posonly = getattr(a, "posonlyargs", [])
    for arg in posonly:
        parts.append(fmt_arg(arg))
    if posonly:
        parts.append("/")

    for arg in a.args:
        parts.append(fmt_arg(arg))

    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")

    for arg in a.kwonlyargs:
        parts.append(fmt_arg(arg))

    if a.kwarg:
        parts.append("**" + a.kwarg.arg)

    return "(" + ", ".join(parts) + ")"


def parse_python_file(py_path: Path) -> Dict[str, Any]:
    text = safe_read_text(py_path)
    try:
        tree = ast.parse(text, filename=str(py_path))
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e}"}

    module_doc = ast.get_docstring(tree)

    imports: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"type": "import", "name": alias.name, "as": alias.asname})
            continue

        if isinstance(node, ast.ImportFrom):
            imports.append(
                {
                    "type": "from",
                    "module": node.module,
                    "level": node.level,
                    "names": [{"name": a.name, "as": a.asname} for a in node.names],
                }
            )
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                {
                    "name": node.name,
                    "signature": get_signature(node),
                    "doc": ast.get_docstring(node),
                    "decorators": [s for s in (ast_unparse(d) for d in node.decorator_list) if s],
                }
            )
            continue

        if isinstance(node, ast.ClassDef):
            bases = [s for s in (ast_unparse(b) for b in node.bases) if s]
            class_doc = ast.get_docstring(node)

            methods: List[Dict[str, Any]] = []
            class_vars: List[str] = []

            for cnode in node.body:
                if isinstance(cnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        {
                            "name": cnode.name,
                            "signature": get_signature(cnode),
                            "doc": ast.get_docstring(cnode),
                            "decorators": [s for s in (ast_unparse(d) for d in cnode.decorator_list) if s],
                        }
                    )
                elif isinstance(cnode, ast.Assign):
                    for t in cnode.targets:
                        if isinstance(t, ast.Name):
                            class_vars.append(t.id)
                elif isinstance(cnode, ast.AnnAssign):
                    if isinstance(cnode.target, ast.Name):
                        class_vars.append(cnode.target.id)

            classes.append(
                {
                    "name": node.name,
                    "bases": bases,
                    "doc": class_doc,
                    "class_vars": sorted(set(class_vars)),
                    "methods": methods,
                }
            )

    return {"doc": module_doc, "imports": imports, "functions": functions, "classes": classes}


# =========================
# MAP BUILDER
# =========================
def iter_py_files(package_dir: Path) -> List[Path]:
    out: List[Path] = []
    for root, _, files in os.walk(package_dir):
        root_path = Path(root)
        for f in files:
            if f.endswith(".py"):
                out.append(root_path / f)
    return out


def build_package_map() -> Dict[str, Any]:
    packages_info: Dict[str, Any] = {}
    files_info: Dict[str, Any] = {}

    for pkg in TOP_PACKAGES:
        pkg_dir = REPO_ROOT / pkg
        packages_info[pkg] = {"exists": pkg_dir.exists(), "package_dir": str(pkg_dir)}
        if not pkg_dir.exists():
            continue

        for py_path in iter_py_files(pkg_dir):
            rel = py_path.relative_to(REPO_ROOT).as_posix()
            files_info[rel] = {"package": pkg, "path": rel, "parsed": parse_python_file(py_path)}

    return {
        "repo_root": str(REPO_ROOT),
        "top_packages": TOP_PACKAGES,
        "packages": packages_info,
        "files": files_info,
        "examples_root": str(EXAMPLES_ROOT) if EXAMPLES_ROOT else None,
    }


# =========================
# RENDERING
# =========================
def module_name_from_relpath(relpath: str) -> str:
    p = relpath.replace("\\", "/")
    if p.endswith("/__init__.py"):
        p = p[: -len("/__init__.py")]
    elif p.endswith(".py"):
        p = p[: -len(".py")]
    return p.replace("/", ".")


def render_import(imp: Dict[str, Any]) -> str:
    if imp.get("type") == "import":
        return f"`import {imp['name']}" + (f" as {imp['as']}`" if imp.get("as") else "`")

    names = ", ".join(
        [f"{n['name']}" + (f" as {n['as']}" if n.get("as") else "") for n in imp.get("names", [])]
    )
    lvl = "." * int(imp.get("level", 0))
    mod = imp.get("module") or ""
    return f"`from {lvl}{mod} import {names}`"


def render_module_page(
    mod_name: str,
    parsed: Dict[str, Any],
    rel_file: str,
    example_links: List[str],
    current_doc_path: Path,
) -> str:
    """
    current_doc_path: absolute path to the markdown page we are generating
                      e.g. docs/reference/pinneaple_train/viz.md

    example_links: docs-relative example paths (no leading slash)
                   e.g. examples/pinneaple_train/01_audited_training.md
    """
    lines: List[str] = []
    lines.append(f"# `{mod_name}`")
    lines.append("")
    lines.append(f"**Source file:** `{rel_file}`")
    lines.append("")

    # Quick summary: docstring + key symbols
        # Quick summary (ONLY if useful content exists)
    if not parsed.get("error"):
        mod_doc = (parsed.get("doc") or "").strip()
        funcs = parsed.get("functions", []) or []
        clss = parsed.get("classes", []) or []

        key_classes = [c["name"] for c in clss[:10] if c.get("name")]
        key_functions = [f["name"] for f in funcs[:10] if f.get("name")]

        has_any_summary = bool(mod_doc or key_classes or key_functions)

        if has_any_summary:
            lines.append("## Quick summary")
            lines.append("")
            if mod_doc:
                lines.append(mod_doc)
                lines.append("")

            if key_classes:
                lines.append("**Key classes:** " + ", ".join([f"`{n}`" for n in key_classes]))
                lines.append("")

            if key_functions:
                lines.append("**Key functions:** " + ", ".join([f"`{n}`" for n in key_functions]))
                lines.append("")

            lines.append("")

    if example_links:
        lines.append("## Examples")
        lines.append("")
        lines.append("Related examples:")
        lines.append("")
        for link in example_links[:80]:
            title = Path(link).stem
            target = DOCS_DIR / link
            link_rel = rel_link(current_doc_path, target)
            lines.append(f"- [{title}]({link_rel})")
        lines.append("")

    if parsed.get("error"):
        lines.append('!!! danger "Parsing error"')
        lines.append("")
        lines.append(f"    `{parsed['error']}`")
        lines.append("")
        return "\n".join(lines)

    # Overview (full module docstring)
    doc = (parsed.get("doc") or "").strip()
    if doc:
        lines.append("## Overview")
        lines.append("")
        lines.append(doc)
        lines.append("")

    imports = parsed.get("imports", [])
    if imports:
        lines.append('??? note "Imports"')
        lines.append("")
        for imp in imports[:500]:
            lines.append(f"    - {render_import(imp)}")
        lines.append("")

    funcs = parsed.get("functions", [])
    if funcs:
        lines.append('??? info "Functions"')
        lines.append("")
        for fn in funcs:
            sig = fn.get("signature", "()")
            fn_doc = (fn.get("doc") or "").strip()
            if fn_doc:
                lines.append(f"    - `{fn['name']}{sig}` — {fn_doc.splitlines()[0]}")
            else:
                lines.append(f"    - `{fn['name']}{sig}`")
        lines.append("")

    classes = parsed.get("classes", [])
    if classes:
        lines.append("## Classes")
        lines.append("")
        for cls in classes:
            lines.append(f"### `{cls['name']}`")
            lines.append("")
            bases = cls.get("bases") or []
            lines.append("**Bases:** " + (", ".join([f"`{b}`" for b in bases]) if bases else "_(none)_"))
            lines.append("")
            cdoc = (cls.get("doc") or "").strip()
            lines.append(cdoc if cdoc else "_No class docstring._")
            lines.append("")
            methods = cls.get("methods", [])
            if methods:
                lines.append('??? tip "Methods"')
                lines.append("")
                for m in methods:
                    sig = m.get("signature", "()")
                    mdoc = (m.get("doc") or "").strip()
                    if mdoc:
                        lines.append(f"    - `{m['name']}{sig}` — {mdoc.splitlines()[0]}")
                    else:
                        lines.append(f"    - `{m['name']}{sig}`")
                lines.append("")

    return "\n".join(lines).strip() + "\n"


# =========================
# EXAMPLES -> DOCS
# =========================
def discover_examples() -> Dict[str, List[Path]]:
    """
    Mirror entire examples structure, ignoring folders starting with '_' or '.'.
    """
    out: Dict[str, List[Path]] = {}
    if not EXAMPLES_ROOT:
        return out

    for root, dirs, files in os.walk(EXAMPLES_ROOT):
        root_path = Path(root)

        # ignore private dirs
        dirs[:] = [d for d in dirs if not d.startswith(EXAMPLES_IGNORE_PREFIXES)]

        for f in files:
            p = root_path / f
            if p.suffix.lower() not in EXAMPLE_EXTS:
                continue

            rel = p.relative_to(EXAMPLES_ROOT).as_posix()
            folder = str(Path(rel).parent).replace("\\", "/")  # e.g. pinneaple_pinn/factory
            out.setdefault(folder, []).append(p)

    for k in out:
        out[k] = sorted(out[k], key=lambda x: natural_sort_key(x.name))

    return out


def extract_top_docstring_from_python_text(py_text: str) -> Optional[str]:
    try:
        t = ast.parse(py_text)
        return ast.get_docstring(t)
    except Exception:
        return None


def render_example_page(example_file: Path) -> str:
    ext = example_file.suffix.lower()
    rel = example_file.relative_to(REPO_ROOT).as_posix()

    if ext == ".md":
        content = safe_read_text(example_file).strip()
        if not content.lstrip().startswith("#"):
            content = f"# `{example_file.stem}`\n\n" + content
        return content + "\n"

    if ext == ".py":
        code = safe_read_text(example_file).rstrip()
        desc = (extract_top_docstring_from_python_text(code) or "").strip()

        lines = [
            f"# `{example_file.stem}`",
            "",
            f"**Source:** `{rel}`",
            "",
        ]

        if desc:
            lines += ["## What this example does", "", desc, ""]
        else:
            lines += [
                "## What this example does",
                "",
                "This example page is auto-generated. Add a top-level docstring to the example file to describe the goal and steps.",
                "",
            ]

        lines += ["## Code", "", "```python", code, "```", ""]
        return "\n".join(lines)

    if ext == ".ipynb":
        return "\n".join(
            [
                f"# `{example_file.stem}`",
                "",
                f"**Notebook:** `{rel}`",
                "",
                "> This notebook is not auto-converted. Open the `.ipynb` file in Jupyter/VSCode.",
                "",
            ]
        )

    # txt/other
    text = safe_read_text(example_file).rstrip()
    return "\n".join(
        [
            f"# `{example_file.stem}`",
            "",
            f"**File:** `{rel}`",
            "",
            "```",
            text,
            "```",
            "",
        ]
    )


def build_examples_docs() -> Dict[str, List[str]]:
    """
    Builds docs/examples mirroring examples/ structure.
    Returns: mapping folder_rel -> list of docs-relative links to generated example pages (NO leading slash).
    """
    examples = discover_examples()
    links_by_folder: Dict[str, List[str]] = {}

    if not examples:
        return links_by_folder

    dir_index: Dict[Path, Dict[str, Any]] = {}

    def touch_dir(d: Path):
        if d not in dir_index:
            dir_index[d] = {"subdirs": set(), "pages": []}

    # create pages
    for folder_rel, files in sorted(examples.items(), key=lambda x: x[0].lower()):
        parts = Path(folder_rel).parts
        if not parts:
            continue

        out_folder = EX_DOCS_DIR / Path(*parts)
        out_folder.mkdir(parents=True, exist_ok=True)

        # Folder landing
        if not (out_folder / "index.md").exists():
            # add small UX text for the folder
            folder_index = "\n".join(
                [
                    f"# `{folder_rel}`",
                    "",
                    "## Quick summary",
                    "",
                    "This folder contains runnable examples mirrored from the repository.",
                    "",
                    "Use **Next / Previous** at the bottom to follow numbered tutorials (e.g. `01_`, `02_`, ...).",
                    "",
                ]
            )
            write_text(out_folder / "index.md", folder_index)

        touch_dir(out_folder)

        # Example files
        for ef in files:
            out_name = ef.stem + ".md"
            out_path = out_folder / out_name
            write_text(out_path, render_example_page(ef))
            dir_index[out_folder]["pages"].append((ef.stem, out_name))

        # register in parent
        parent = out_folder.parent
        touch_dir(parent)
        dir_index[parent]["subdirs"].add(out_folder.name)

        # IMPORTANT: store docs-relative links WITHOUT leading slash
        links = [(out_folder / (ef.stem + ".md")).relative_to(DOCS_DIR).as_posix() for ef in files]
        links_by_folder[folder_rel] = links

    # write .pages ordering for every dir
    for d, info in dir_index.items():
        nav_items: List[Any] = []
        if (d / "index.md").exists():
            nav_items.append({"Overview": "index.md"})
        for sub in sorted(info["subdirs"], key=lambda s: natural_sort_key(s)):
            nav_items.append({sub: f"{sub}/"})
        for title, fname in sorted(info["pages"], key=lambda x: natural_sort_key(x[0])):
            nav_items.append({title: fname})
        write_pages_file(d, nav_items)

    # Examples landing
    top_groups = sorted({k.split("/")[0] for k in examples.keys() if k}, key=natural_sort_key)
    home = "\n".join(
        [
            "# Examples",
            "",
            "This section contains real-world, runnable examples mirrored from the repository `examples/` folder.",
            "",
            "!!! info \"How to use\"",
            "    - Browse by categories in the sidebar (e.g. `end_to_end`, `app`, `pinneaple_pinn`, ...).",
            "    - Use **Next / Previous** at the bottom to walk through numbered tutorials (e.g. `01_`, `02_`, ...).",
            "",
            "## Example groups",
            "",
        ]
        + [f"- `{g}`" for g in top_groups]
        + [
            "",
            "Tip: inside **Reference** pages, you will also see **Related examples** when available.",
            "",
        ]
    )
    write_text(EX_DOCS_DIR / "index.md", home)

    # root .pages for examples
    write_pages_file(
        EX_DOCS_DIR,
        [{"Overview": "index.md"}] + [{g: f"{g}/"} for g in top_groups if (EX_DOCS_DIR / g).exists()],
    )

    return links_by_folder


def find_examples_for_module(rel_file: str, links_by_folder: Dict[str, List[str]]) -> List[str]:
    """
    Heuristic:
      module: pinneaple_pinn/factory/pinn_factory.py
      look for examples folder: pinneaple_pinn/factory
      fallback: pinneaple_pinn
    """
    src = Path(rel_file)
    parts = src.parts
    if len(parts) < 2:
        return []
    pkg = parts[0]
    folder = "/".join(parts[:-1])
    if folder in links_by_folder:
        return links_by_folder[folder]
    if pkg in links_by_folder:
        return links_by_folder[pkg]
    return []


# =========================
# REFERENCE -> DOCS
# =========================
def build_reference_tree(data: Dict[str, Any], links_by_folder: Dict[str, List[str]]):
    files: Dict[str, Any] = data.get("files", {})

    # Collect package-level docstrings from __init__.py (if present)
    pkg_doc: Dict[str, str] = {}
    for rel_file, meta in files.items():
        if rel_file.endswith("/__init__.py") or rel_file.endswith("\\__init__.py"):
            pkg = meta.get("package")
            if pkg and Path(rel_file).parts[0] == pkg:
                doc = (meta.get("parsed", {}).get("doc") or "").strip()
                if doc:
                    pkg_doc[pkg] = doc

    # package root indexes with quick summary
    for pkg in TOP_PACKAGES:
        pkg_dir = REF_DIR / pkg
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # Build quick summary from discovered modules
        module_names = []
        top_classes = []
        top_functions = []
        subpackages = set()

        for rel_file, meta in files.items():
            if meta.get("package") != pkg:
                continue
            if not rel_file.endswith(".py"):
                continue
            # module name
            mod = module_name_from_relpath(rel_file)
            module_names.append(mod)

            # subpackage (first folder under pkg)
            parts = Path(rel_file).parts
            if len(parts) >= 2 and parts[0] == pkg:
                subpackages.add(parts[1])

            parsed = meta.get("parsed", {}) or {}
            for c in (parsed.get("classes") or [])[:2]:
                top_classes.append(f"{mod}.{c['name']}")
            for f in (parsed.get("functions") or [])[:2]:
                top_functions.append(f"{mod}.{f['name']}")

        module_names_sorted = sorted(set(module_names), key=lambda s: s.lower())
        subpackages_sorted = sorted({s for s in subpackages if s and s != "__pycache__"}, key=natural_sort_key)

        pkg_index = make_quick_summary_text(
            title=pkg,
            package_doc=pkg_doc.get(pkg),
            subpackages=subpackages_sorted,
            modules=[m for m in module_names_sorted if not m.endswith(".__init__")],
            top_classes=sorted(set(top_classes), key=lambda s: s.lower()),
            top_functions=sorted(set(top_functions), key=lambda s: s.lower()),
        )
        write_text(pkg_dir / "index.md", pkg_index)

    # directory index for .pages
    dir_index: Dict[Path, Dict[str, Any]] = {}

    def touch_dir(d: Path):
        if d not in dir_index:
            dir_index[d] = {"subdirs": set(), "modules": []}

    for rel_file, meta in sorted(files.items(), key=lambda x: x[0].lower()):
        pkg = meta.get("package")
        parsed = meta.get("parsed", {})
        if not pkg:
            continue

        src = Path(rel_file)

        try:
            relative_inside_pkg = src.relative_to(Path(pkg))
        except Exception:
            parts = src.parts
            relative_inside_pkg = Path(*parts[1:]) if parts and parts[0] == pkg else src

        out_pkg_root = REF_DIR / pkg

        # __init__.py => folder index (do not overwrite package root index)
        if relative_inside_pkg.name == "__init__.py":
            folder_dir = out_pkg_root.joinpath(*relative_inside_pkg.parent.parts)
            folder_dir.mkdir(parents=True, exist_ok=True)

            if relative_inside_pkg.parent.parts:
                # subpackage index (folder index) only
                if not (folder_dir / "index.md").exists():
                    title = f"{pkg}/" + "/".join(relative_inside_pkg.parent.parts)
                    subtitle = "Subpackage overview. Add docstrings to `__init__.py` to improve this page."
                    write_text(folder_dir / "index.md", f"# `{title}`\n\n## Quick summary\n\n{subtitle}\n")

            parent_dir = folder_dir.parent
            touch_dir(parent_dir)
            dir_index[parent_dir]["subdirs"].add(folder_dir.name)
            touch_dir(folder_dir)
            continue

        mod_name = module_name_from_relpath(rel_file)
        folder_dir = out_pkg_root.joinpath(*relative_inside_pkg.parent.parts)
        folder_dir.mkdir(parents=True, exist_ok=True)

        if not (folder_dir / "index.md").exists():
            title = (
                f"{pkg}/" + "/".join(relative_inside_pkg.parent.parts)
                if relative_inside_pkg.parent.parts
                else pkg
            )
            # folder summary (not the package root, which we already overwrote)
            if relative_inside_pkg.parent.parts:
                write_text(
                    folder_dir / "index.md",
                    f"# `{title}`\n\n## Quick summary\n\nFolder in `{pkg}`.\n\nUse the sidebar to navigate.\n",
                )

        md_name = relative_inside_pkg.stem + ".md"
        out_md = folder_dir / md_name

        example_links = find_examples_for_module(rel_file, links_by_folder)
        write_text(out_md, render_module_page(mod_name, parsed, rel_file, example_links, out_md))

        touch_dir(folder_dir)
        dir_index[folder_dir]["modules"].append((relative_inside_pkg.stem, md_name))

        parent_dir = folder_dir.parent
        touch_dir(parent_dir)
        dir_index[parent_dir]["subdirs"].add(folder_dir.name)

    # write .pages ordering
    for d, info in dir_index.items():
        nav_items: List[Any] = []
        if (d / "index.md").exists():
            nav_items.append({"Overview": "index.md"})
        for sub in sorted(info["subdirs"], key=natural_sort_key):
            nav_items.append({sub: f"{sub}/"})
        for title, mdfile in sorted(info["modules"], key=lambda x: natural_sort_key(x[0])):
            nav_items.append({title: mdfile})
        write_pages_file(d, nav_items)

    # reference landing
    if not (REF_DIR / "index.md").exists():
        write_text(
            REF_DIR / "index.md",
            "# `Reference`\n\n## Quick summary\n\nAPI reference organized by packages and folders.\n",
        )
    write_pages_file(
        REF_DIR,
        [{"Overview": "index.md"}] + [{pkg: f"{pkg}/"} for pkg in TOP_PACKAGES],
    )

        # reference landing (ALWAYS create)
    write_text(
        REF_DIR / "index.md",
        "# Reference\n\nAPI reference organized by packages and folders.\n",
    )
    write_pages_file(
        REF_DIR,
        [{"Overview": "index.md"}] + [{pkg: f"{pkg}/"} for pkg in TOP_PACKAGES],
    )


# =========================
# HOME + MKDOCS
# =========================
def build_home_pages():
    home = "\n".join(
        [
            "# PINNeAPPle",
            "",
            "Welcome to the **PINNeAPPle** documentation.",
            "",
            "This site is automatically generated from the repository source tree.",
            "",
            '!!! info "How to use this dashboard"',
            "    - **Reference**: API documentation organized by package and folder structure.",
            "    - **Examples**: real-world examples mirrored from the repository `examples/` folder.",
            "    - Use **Search** (top bar) to quickly find modules, classes, and functions.",
            "",
            "## Sections",
            "",
            "- 📚 **Reference**: browse the API by package",
            "- 🧪 **Examples**: browse runnable workflows and tutorials",
            "",
            "## Tips",
            "",
            '??? tip "Walk through tutorials"',
            "    In **Examples**, open `end_to_end/` and use **Next / Previous** at the bottom to follow numbered steps.",
            "",
            '??? tip "Implementation + example"',
            "    In **Reference**, module pages show **Related examples** when available.",
            "",
        ]
    )
    write_text(DOCS_DIR / "index.md", home)


def write_root_pages():
    nav = [{"Home": "index.md"}, {"Reference": "reference/"}]
    if EXAMPLES_ROOT:
        nav.append({"Examples": "examples/"})
    write_pages_file(DOCS_DIR, nav)


def write_mkdocs_yml():
    content = f"""site_name: {SITE_NAME}
site_url: {SITE_URL}
repo_url: {REPO_URL}
use_directory_urls: true

theme:
  name: material
  features:
    - navigation.expand
    - navigation.instant
    - navigation.tracking
    - navigation.path
    - navigation.sections
    - navigation.indexes
    - navigation.top
    - navigation.footer
    - content.code.copy
    - search.suggest
    - search.highlight


plugins:
  - search
  - awesome-pages
  - section-index

markdown_extensions:
  - admonition
  - toc:
      permalink: true
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
"""
    MKDOCS_YML.write_text(content, encoding="utf-8")


def ensure_pages_for_all_dirs(root: Path):
    for d in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: str(p).lower()):
        pages = d / ".pages"
        if pages.exists():
            continue
        items: List[Any] = []
        if (d / "index.md").exists():
            items.append({"Overview": "index.md"})
        # subfolders
        subs = [p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")]
        for sub in sorted(subs, key=lambda p: natural_sort_key(p.name)):
            items.append({sub.name: f"{sub.name}/"})
        # pages
        md_files = [p for p in d.iterdir() if p.is_file() and p.suffix == ".md" and p.name != "index.md"]
        for mf in sorted(md_files, key=lambda p: natural_sort_key(p.stem)):
            items.append({mf.stem: mf.name})
        write_pages_file(d, items)


# =========================
# MAIN
# =========================
def main():
    print("REPO_ROOT =", REPO_ROOT)

    data = build_package_map()
    OUTPUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("✅ package_map.json generated:", OUTPUT_JSON)

    ensure_clean_docs()
    build_home_pages()

    links_by_folder: Dict[str, List[str]] = {}
    if EXAMPLES_ROOT:
        print("✅ examples folder found:", EXAMPLES_ROOT)
        links_by_folder = build_examples_docs()
    else:
        print("ℹ️ No examples/ folder found. Skipping Examples section.")

    build_reference_tree(data, links_by_folder)
    write_root_pages()
    write_mkdocs_yml()

    ensure_pages_for_all_dirs(REF_DIR)
    ensure_pages_for_all_dirs(EX_DOCS_DIR)

    print("✅ Docs generated in:", DOCS_DIR)
    print("✅ mkdocs.yml generated in:", MKDOCS_YML)
    print("\nNext:\n  mkdocs serve\n")


if __name__ == "__main__":
    main()
