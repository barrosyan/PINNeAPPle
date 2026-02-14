from __future__ import annotations

import importlib
import sys


def _check(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def main() -> int:
    checks = {
        "torch": _check("torch"),
        "pandas": _check("pandas"),
        "pyarrow": _check("pyarrow"),
        "yaml": _check("yaml"),
        # optional
        "physicsnemo.sym": _check("physicsnemo.sym"),
    }

    print("=== Sanity check ===")
    for k, ok in checks.items():
        print(f"{k:15s} : {'OK' if ok else 'MISSING'}")

    print("\nNotes:")
    print("- physicsnemo.sym is optional (only needed for PhysicsNeMo backend).")
    print("- Omniverse modules are not checked here (they exist only inside Kit).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
