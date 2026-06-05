"""
Step 1: Update all Python import statements to use the new mega-module paths.
"""

import os
import re
import sys

REPO_ROOT = r"c:\Users\laisc\Documents\GitHub\PINNeAPPle"

# Old package dirs to SKIP entirely (they will be deleted later)
OLD_PKG_DIRS = {
    "pinneapple_environment",
    "pinneapple_pinn",
    "pinneapple_symbolic",
    "pinneapple_models",
    "pinneapple_train",
    "pinneapple_inference",
    "pinneapple_uq",
    "pinneapple_validate",
    "pinneapple_inverse",
    "pinneapple_transfer",
    "pinneapple_meta",
    "pinneapple_solvers",
    "pinneapple_dynamics",
    "pinneapple_integrations",
    "pinneapple_timeseries",
    "pinneapple_cosim",
    "pinneapple_digital_twin",
    "pinneapple_design.geometry",
    "pinneapple_design_opt",
    "pinneapple_viz",
    "pinneapple_export",
    "pinneapple_researcher",
    "pinneapple_arena",
    "pinneapple_backend",
    "pinneapple_learning",
}

# Dirs to skip entirely
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "dist", "build", "artifacts"} | OLD_PKG_DIRS

# Rename mapping: old_package -> new_dotted_path
# Order matters: more specific (sub-module) names first, then bare names
RENAME_MAP = [
    ("pinneapple_environment",  "pinneapple_physics.pde_environment"),
    ("pinneapple_pinn",         "pinneapple_physics.pinn_solver"),
    ("pinneapple_symbolic",     "pinneapple_physics.symbolic_pde"),
    ("pinneapple_models",       "pinneapple_neural.architectures"),
    ("pinneapple_train",        "pinneapple_neural.trainer"),
    ("pinneapple_inference",    "pinneapple_neural.predictor"),
    ("pinneapple_uq",           "pinneapple_analysis.uncertainty"),
    ("pinneapple_validate",     "pinneapple_analysis.validation"),
    ("pinneapple_inverse",      "pinneapple_analysis.inverse_problems"),
    ("pinneapple_transfer",     "pinneapple_adaptation.transfer_learning"),
    ("pinneapple_meta",         "pinneapple_adaptation.meta_learning"),
    ("pinneapple_solvers",      "pinneapple_simulation.numerical_solvers"),
    ("pinneapple_dynamics",     "pinneapple_simulation.particle_dynamics"),
    ("pinneapple_integrations", "pinneapple_simulation.external_solvers"),
    ("pinneapple_timeseries",   "pinneapple_systems.time_series"),
    ("pinneapple_cosim",        "pinneapple_systems.cosimulation"),
    ("pinneapple_digital_twin", "pinneapple_systems.digital_twin"),
    ("pinneapple_design.geometry",         "pinneapple_design.geometry"),
    ("pinneapple_design_opt",   "pinneapple_design.design_optimizer"),
    ("pinneapple_viz",          "pinneapple_tools.visualization"),
    ("pinneapple_export",       "pinneapple_tools.model_export"),
    ("pinneapple_researcher",   "pinneapple_tools.hpo_experiments"),
    ("pinneapple_arena",        "pinneapple_tools.benchmark_suite"),
    ("pinneapple_backend",      "pinneapple_tools.compute_backends"),
    ("pinneapple_learning",     "pinneapple_neural"),
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
