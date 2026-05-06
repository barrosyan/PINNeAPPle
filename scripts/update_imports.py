"""
Step 1: Update all Python import statements to use the new mega-module paths.
"""

import os
import re
import sys

REPO_ROOT = r"c:\Users\laisc\Documents\GitHub\PINNeAPPle"

# Old package dirs to SKIP entirely (they will be deleted later)
OLD_PKG_DIRS = {
    "pinneaple_environment",
    "pinneaple_pinn",
    "pinneaple_symbolic",
    "pinneaple_models",
    "pinneaple_train",
    "pinneaple_inference",
    "pinneaple_uq",
    "pinneaple_validate",
    "pinneaple_inverse",
    "pinneaple_transfer",
    "pinneaple_meta",
    "pinneaple_solvers",
    "pinneaple_dynamics",
    "pinneaple_integrations",
    "pinneaple_timeseries",
    "pinneaple_cosim",
    "pinneaple_digital_twin",
    "pinneaple_geom",
    "pinneaple_design_opt",
    "pinneaple_viz",
    "pinneaple_export",
    "pinneaple_researcher",
    "pinneaple_arena",
    "pinneaple_backend",
    "pinneaple_learning",
}

# Dirs to skip entirely
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "dist", "build", "artifacts"} | OLD_PKG_DIRS

# Rename mapping: old_package -> new_dotted_path
# Order matters: more specific (sub-module) names first, then bare names
RENAME_MAP = [
    ("pinneaple_environment",  "pinneaple_physics.pde_environment"),
    ("pinneaple_pinn",         "pinneaple_physics.pinn_solver"),
    ("pinneaple_symbolic",     "pinneaple_physics.symbolic_pde"),
    ("pinneaple_models",       "pinneaple_neural.architectures"),
    ("pinneaple_train",        "pinneaple_neural.trainer"),
    ("pinneaple_inference",    "pinneaple_neural.predictor"),
    ("pinneaple_uq",           "pinneaple_analysis.uncertainty"),
    ("pinneaple_validate",     "pinneaple_analysis.validation"),
    ("pinneaple_inverse",      "pinneaple_analysis.inverse_problems"),
    ("pinneaple_transfer",     "pinneaple_adaptation.transfer_learning"),
    ("pinneaple_meta",         "pinneaple_adaptation.meta_learning"),
    ("pinneaple_solvers",      "pinneaple_simulation.numerical_solvers"),
    ("pinneaple_dynamics",     "pinneaple_simulation.particle_dynamics"),
    ("pinneaple_integrations", "pinneaple_simulation.external_solvers"),
    ("pinneaple_timeseries",   "pinneaple_systems.time_series"),
    ("pinneaple_cosim",        "pinneaple_systems.cosimulation"),
    ("pinneaple_digital_twin", "pinneaple_systems.digital_twin"),
    ("pinneaple_geom",         "pinneaple_design.geometry"),
    ("pinneaple_design_opt",   "pinneaple_design.design_optimizer"),
    ("pinneaple_viz",          "pinneaple_tools.visualization"),
    ("pinneaple_export",       "pinneaple_tools.model_export"),
    ("pinneaple_researcher",   "pinneaple_tools.hpo_experiments"),
    ("pinneaple_arena",        "pinneaple_tools.benchmark_suite"),
    ("pinneaple_backend",      "pinneaple_tools.compute_backends"),
    ("pinneaple_learning",     "pinneaple_neural"),
]

def build_patterns(old, new):
    """Return list of (compiled_pattern, replacement_string) pairs for one mapping."""
    esc = re.escape(old)
    patterns = []

    # 1. from OLD.sub.sub import X  ->  from NEW.sub.sub import X
    patterns.append((
        re.compile(rf'\bfrom\s+{esc}(\.[^\s]+)\s+import\b'),
        rf'from {new}\1 import'
    ))

    # 2. from OLD import X  ->  from NEW import X
    patterns.append((
        re.compile(rf'\bfrom\s+{esc}\s+import\b'),
        rf'from {new} import'
    ))

    # 3. import OLD as alias  ->  import NEW as alias
    patterns.append((
        re.compile(rf'\bimport\s+{esc}\s+as\b'),
        rf'import {new} as'
    ))

    # 4. import OLD  (bare, not followed by . or word char)  ->  import NEW as OLD
    patterns.append((
        re.compile(rf'\bimport\s+{esc}(?![\.\w])'),
        rf'import {new} as {old}'
    ))

    return patterns

# Pre-compile all patterns
ALL_PATTERNS = []
for old, new in RENAME_MAP:
    ALL_PATTERNS.extend(build_patterns(old, new))

def process_file(filepath):
    """Apply all substitutions to a file. Returns (changed: bool, num_subs: int)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
    except Exception as e:
        print(f"  SKIP (read error): {filepath}: {e}")
        return False, 0

    content = original
    total_subs = 0
    for pattern, repl in ALL_PATTERNS:
        content, n = pattern.subn(repl, content)
        total_subs += n

    if content != original:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"  SKIP (write error): {filepath}: {e}")
            return False, 0
        return True, total_subs

    return False, 0

def should_skip_dir(dirname):
    return dirname in SKIP_DIRS

files_changed = 0
total_substitutions = 0
files_examined = 0

for root, dirs, files in os.walk(REPO_ROOT, topdown=True):
    # Prune skip dirs in-place
    dirs[:] = [d for d in dirs if not should_skip_dir(d)]

    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        files_examined += 1
        changed, n_subs = process_file(fpath)
        if changed:
            files_changed += 1
            total_substitutions += n_subs
            rel = os.path.relpath(fpath, REPO_ROOT)
            print(f"  UPDATED ({n_subs} subs): {rel}")

print()
print("=" * 60)
print(f"Files examined   : {files_examined}")
print(f"Files changed    : {files_changed}")
print(f"Total substitutions: {total_substitutions}")
print("=" * 60)
