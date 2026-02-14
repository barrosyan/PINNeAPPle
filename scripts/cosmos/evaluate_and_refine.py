from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import yaml


def read_text(p: str | Path) -> str:
    return Path(p).read_text(encoding="utf-8")


def read_json(p: str | Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_text(p: str | Path, s: str) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def write_json(p: str | Path, obj: Any) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--metrics", required=True, help="Path to metrics.json from a run")
    ap.add_argument("--out", required=True, help="Path to write recommendations.json")
    ap.add_argument("--base-url", default=os.environ.get("COSMOS_BASE_URL", "").strip())
    args = ap.parse_args()

    spec_yaml = read_text(args.spec)
    yaml.safe_load(spec_yaml)
    metrics = read_json(args.metrics)

    if not args.base_url:
        # local rule-based recommendations (no placeholders)
        rec = {"recommendations": {}}
        pde = float(metrics.get("test_pde_rms", 0.0))
        div = float(metrics.get("test_div_rms", 0.0))
        bc = float(metrics.get("bc_mse", 0.0))

        # deterministic tuning suggestions
        if bc > 1e-3:
            rec["recommendations"]["increase_bc_weight"] = True
            rec["recommendations"]["bc_weight_factor"] = 2.0
        if div > 1e-2:
            rec["recommendations"]["increase_pde_weight"] = True
            rec["recommendations"]["pde_weight_factor"] = 1.5
        if pde > 1e-2 and div < 1e-2:
            rec["recommendations"]["try_more_steps_or_epochs"] = True
            rec["recommendations"]["steps_factor"] = 1.5

        write_json(args.out, rec)
        print("[WARN] COSMOS_BASE_URL not set. Wrote local rule-based recommendations.")
        print(f"[OK] {args.out}")
        return 0

    url = args.base_url.rstrip("/") + "/evaluate_and_refine"
    payload = {"spec_yaml": spec_yaml, "metrics": metrics, "history": []}

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()

    write_json(args.out, out)
    print(f"[OK] Wrote recommendations to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
