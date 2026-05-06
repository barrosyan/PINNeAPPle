"""07_validation_and_quality_gates.py

Showcase: lightweight validation / quality gates for PhysicalSample-like objects.

The validator in pinneaple_data.validators is intentionally small/fast, so you can
run it:
  - during data generation
  - before writing to disk
  - inside a DataLoader worker as a guardrail

It operates on a PhysicalSample-like interface with:
  - fields: Dict[str, Tensor]
  - coords: Dict[str, Tensor]
  - meta: Dict[str, Any]

Run:
  python examples/pinneaple_data/07_validation_and_quality_gates.py
"""

from __future__ import annotations

from pinneaple_data.synth.pde import PDESynthGenerator
from pinneaple_data.validators import assert_valid_physical_sample, validate_physical_sample


def main() -> None:
    gen = PDESynthGenerator(seed=3)

    out = gen.sample(kind="heat1d", T=32, X=64, alpha=0.05)
    s = out.sample  # SimplePhysicalSample with fields/coords/meta

    # Enrich meta with units and UPD version (recommended for traceability)
    s.meta.setdefault("upd", {})
    s.meta["upd"].setdefault("version", "0.1")
    s.meta.setdefault("units", {})
    s.meta["units"].update({"u": "arb", "t": "s", "x": "m"})

    issues = validate_physical_sample(
        s,
        units_policy="warn",
        required_units=["u"],
        ranges={"u": (-5.0, 5.0)},
        monotonic_dims=["t", "x"],
    )

    if issues:
        print("Issues found:")
        for it in issues:
            print(f"- {it.level.upper()}: {it.message}")
    else:
        print("No issues.")

    # Fail-fast gate (raises ValueError if any errors exist)
    assert_valid_physical_sample(
        s,
        units_policy="strict",
        required_units=["u"],
        ranges={"u": (-10.0, 10.0)},
        monotonic_dims=["t", "x"],
    )
    print("Strict validation passed.")

    # Example: show a failing case
    s_bad = out.sample
    s_bad.fields["u"] = s_bad.fields["u"] * 1_000  # blow up range
    bad_issues = validate_physical_sample(s_bad, ranges={"u": (-10.0, 10.0)})
    print("Failing example: num_issues=", len(bad_issues))


if __name__ == "__main__":
    main()
