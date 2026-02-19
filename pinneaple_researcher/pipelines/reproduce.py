from __future__ import annotations

import json
import os

from ..agents.reproducer import ReproducerAgent
from ..agents.verifier import VerifierAgent
from ..models import RankedItem
from ..providers.gemini_provider import GeminiProvider
from ..utils.verify_runtime import project_tree, py_compile_all, smoke_run


def _apply_patches(project_dir: str, patches: list[dict]) -> None:
    for p in patches:
        path = os.path.join(project_dir, p["path"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = p.get("mode", "overwrite")
        content = p.get("content", "")
        if mode == "append":
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)


def reproduce(
    *,
    item: RankedItem,
    kb_index_dir: str,
    out_dir: str | None = None,
    max_fix_iters: int = 2,
) -> str:
    provider = GeminiProvider()
    reproducer = ReproducerAgent(provider)
    verifier = VerifierAgent(provider)

    manifest_path = os.path.join(kb_index_dir, "manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            snippet = f.read()[:12000]
    except Exception:
        snippet = ""

    out_dir = out_dir or os.path.join(kb_index_dir, "reproductions")
    os.makedirs(out_dir, exist_ok=True)

    res = reproducer.run(item=item, kb_snippet=snippet, out_dir=out_dir)
    project_dir = res.project_dir

    logs_acc = []

    for it in range(max_fix_iters + 1):
        tree = project_tree(project_dir)

        c = py_compile_all(project_dir)
        logs_acc.append(f"\n\n=== ITER {it} | COMPILE ===\n{c.step}\nRC={c.returncode}\nSTDOUT:\n{c.stdout}\nSTDERR:\n{c.stderr}\n")
        if not c.ok:
            vr = verifier.run(project_tree=tree, logs="".join(logs_acc))
            _apply_patches(project_dir, vr.patches)
            logs_acc.append(f"\n\n=== VERIFIER DIAG ===\n{vr.diagnosis}\nNOTES:\n{vr.notes}\n")
            continue

        s = smoke_run(project_dir, timeout_s=180)
        logs_acc.append(f"\n\n=== ITER {it} | SMOKE ===\n{s.step}\nRC={s.returncode}\nSTDOUT:\n{s.stdout}\nSTDERR:\n{s.stderr}\n")
        if s.ok:
            break

        vr = verifier.run(project_tree=tree, logs="".join(logs_acc))
        _apply_patches(project_dir, vr.patches)
        logs_acc.append(f"\n\n=== VERIFIER DIAG ===\n{vr.diagnosis}\nNOTES:\n{vr.notes}\n")

    # persist logs
    with open(os.path.join(project_dir, "reproduce_verify.log"), "w", encoding="utf-8") as f:
        f.write("".join(logs_acc))

    return project_dir
