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


def write_text(p: str | Path, s: str) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Path to problemdesign prompt yaml (e.g. configs/problemdesign/flow_obstacle_2d_prompt.yaml)")
    ap.add_argument("--schema", default="configs/problemdesign/flow_obstacle_2d_spec_schema.yaml")
    ap.add_argument("--out", required=True, help="Output spec yaml path (e.g. data/cache/cosmos_outputs/spec.yaml)")
    ap.add_argument("--base-url", default=os.environ.get("COSMOS_BASE_URL", "").strip(), help="Cosmos endpoint base url (env COSMOS_BASE_URL)")
    args = ap.parse_args()

    prompt_yaml = read_text(args.scene)
    schema_yaml = read_text(args.schema)

    if not args.base_url:
        # fallback: emit local spec example
        example = read_text("configs/problemdesign/flow_obstacle_2d_spec_example.yaml")
        write_text(args.out, example)
        print("[WARN] COSMOS_BASE_URL not set. Wrote local example spec instead.")
        print(f"[OK] {args.out}")
        return 0

    url = args.base_url.rstrip("/") + "/scene_to_spec"
    payload = {"prompt_yaml": prompt_yaml, "schema_yaml": schema_yaml, "context": {}}

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()

    spec_yaml = out.get("spec_yaml")
    if not isinstance(spec_yaml, str) or not spec_yaml.strip():
        raise RuntimeError(f"Invalid response: missing spec_yaml. Got keys: {list(out.keys())}")

    # Validate it's parseable YAML
    yaml.safe_load(spec_yaml)

    write_text(args.out, spec_yaml)
    print(f"[OK] Wrote spec to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
