"""Component registry + toolbox + physics-informed fit, end to end.

Builds a "pipe" component (VanillaPINN backbone, mass-conservation
physics), trains it on synthetic collocation data, evaluates it, and
shows the installed "Fluid Mechanics Toolbox" it belongs to.

Run: python examples/systems/01_component_library_toolbox.py
"""
from __future__ import annotations

import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from pinneapple_systems.component_library import ComponentRegistry, ToolboxRegistry


def _collate(batch):
    return {
        "x": torch.stack([b[0] for b in batch]),
        "y": torch.stack([b[1] for b in batch]),
    }


def main() -> None:
    print("Available component types:", ComponentRegistry.component_types())
    print("Available toolboxes:", ToolboxRegistry.list())

    toolbox = ToolboxRegistry.install("Fluid Mechanics Toolbox")
    print(f"\nInstalled '{toolbox.name} v{toolbox.version}' — components:",
          [c.name for c in toolbox.components()])

    pipe = ComponentRegistry.build("PipeVanillaPINN", in_dim=3, out_dim=3, hidden=(64, 64, 64))
    print(f"\nBuilt {pipe._component_name} — physics: {pipe.physics.name}")

    x = torch.rand(256, 3)
    y = torch.rand(256, 3)  # synthetic velocity targets, standing in for real CFD data
    loader = DataLoader(TensorDataset(x, y), batch_size=32, collate_fn=_collate)

    result = pipe.fit(loader, loader, epochs=5, lr=1e-3, run_name="pipe_demo", log_dir=tempfile.mkdtemp())
    print(f"\nTrained 5 epochs — best_val={result['best_val']:.4f}")

    metrics = pipe.evaluate(torch.rand(64, 3), torch.rand(64, 3))
    print("Held-out metrics:", metrics)


if __name__ == "__main__":
    main()
