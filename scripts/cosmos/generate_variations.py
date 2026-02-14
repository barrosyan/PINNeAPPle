from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml


def read_text(p: str | Path) -> str:
    return Path(p).read_text(encoding="utf-8")


def write_json(p: str | Path, obj: Any) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Path to spec.yaml (generated or example)")
    ap.add_argument("--out", required=True, help="Output variations json path")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--strategy", default="grid", choices=["grid", "random"])
    ap.add_argument("--base-url", default=os.environ.get("COSMOS_BASE_URL", "").strip())
    args = ap.parse_args()

    spec_yaml = read_text(args.spec)
    yaml.safe_load(spec_yaml)

    if not args.base_url:
        # local deterministic variations (no placeholders): just sweep nu
        base = yaml.safe_load(spec_yaml)
        variations: List[Dict[str, Any]] = []
        nu0 = float(base["parameters"]["nu"])
        for i in range(int(args.n)):
            nu = max(1e-6, nu0 * (0.7 + 0.03 * i))
            variations.append(
                {
                    "id": f"var_{i:03d}",
                    "nu": float(nu),
                    "obstacle_circle": dict(base["geometry"]["obstacle_circle"]),
                }
            )
        write_json(args.out, {"variations": variations, "source": "local_sweep"})
        print("[WARN] COSMOS_BASE_URL not set. Generated local nu sweep variations.")
        print(f"[OK] {args.out}")
        return 0

    url = args.base_url.rstrip("/") + "/generate_variations"
    payload = {"spec_yaml": spec_yaml, "n": int(args.n), "strategy": str(args.strategy)}

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()

    vars_ = out.get("variations")
    if not isinstance(vars_, list) or not vars_:
        raise RuntimeError("Invalid response: missing variations list.")

    write_json(args.out, out)
    print(f"[OK] Wrote variations to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
