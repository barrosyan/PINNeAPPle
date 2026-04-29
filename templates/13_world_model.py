"""13_world_model.py — World model integration (CosmosAdapter + SimToRealAdapter).

Demonstrates:
- CosmosAdapter with physics fallback (no API key required)
- PhysicsVideoDataset for generating physics-consistent frame sequences
- SimToRealAdapter for bridging simulation frames to a PINN model
- Visualizing generated frames

Note: By default this template uses the physics-based fallback generator.
To use the NVIDIA NIM API, set COSMOS_API_KEY in your environment and pass
use_api=True to WorldModelConfig.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_worldmodel.adapter import (
    CosmosAdapter,
    WorldModelConfig,
    PhysicsVideoDataset,
    SimToRealAdapter,
)


# ---------------------------------------------------------------------------
# Simple PINN for 2D heat diffusion (used as the "simulation" model)
# ---------------------------------------------------------------------------

class HeatDiffusionPINN(nn.Module):
    """Minimal PINN surrogate for 2D heat u(x, y, t)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)

    def field_at_t(self, t: float, H: int = 32, W: int = 32) -> torch.Tensor:
        """Return a (1, H, W) temperature field snapshot at time t."""
        x = torch.linspace(0, 1, W)
        y = torch.linspace(0, 1, H)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        t_col = torch.full_like(xx, t)
        xyt = torch.stack([xx.ravel(), yy.ravel(), t_col.ravel()], dim=1)
        with torch.no_grad():
            u = self.forward(xyt).reshape(H, W)
        # Normalise to [0, 1] for visualisation
        u = (u - u.min()) / (u.max() - u.min() + 1e-8)
        return u.unsqueeze(0)   # (1, H, W)


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")   # physics fallback runs on CPU
    print("World model integration demo (physics fallback mode)")

    # --- CosmosAdapter configuration (no API key → physics fallback) -----
    config = WorldModelConfig(
        model_name="cosmos-1.0-diffusion-7b",
        physics_prior_weight=0.1,
        n_frames=12,
        resolution=(64, 64),
        use_api=False,           # True requires COSMOS_API_KEY
        use_huggingface=False,   # True requires transformers + NVIDIA license
    )

    adapter = CosmosAdapter(config=config)
    print(f"CosmosAdapter backend: {adapter._backend}")

    # --- Generate a video sequence from an initial condition --------------
    pinn = HeatDiffusionPINN().to(device)

    # Initial state: sine-bump temperature field
    H, W = config.resolution
    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    init_field = (torch.sin(math.pi * xx) * torch.sin(math.pi * yy)).unsqueeze(0)  # (1, H, W)

    print(f"Generating {config.n_frames} frames from initial condition...")
    frames = adapter.generate(
        initial_state=init_field,
        condition_text="heat diffusion in 2D domain",
        n_frames=config.n_frames,
    )
    print(f"Generated frames shape: {frames.shape}")   # (T, 3, H, W)

    # --- PhysicsVideoDataset ----------------------------------------------
    # Dataset wraps a generator (or pre-generated frames) for training loops
    dataset = PhysicsVideoDataset(
        frames=frames,                  # (T, C, H, W)
        pinn_model=pinn,
        physics_weight=config.physics_prior_weight,
    )
    print(f"Dataset length (frame pairs): {len(dataset)}")

    # Example: get one sample
    sample = dataset[0]
    print(f"  sample keys: {list(sample.keys())}")

    # --- SimToRealAdapter -------------------------------------------------
    # Bridge: adapt a simulation frame to "real" style using a thin adapter
    sim_to_real = SimToRealAdapter(
        adapter_type="affine",     # lightweight affine transform
        n_channels=3,
    ).to(device)

    # Apply to first generated frame
    frame0 = frames[0:1]           # (1, 3, H, W)
    with torch.no_grad():
        adapted = sim_to_real(frame0)
    print(f"SimToReal output shape: {adapted.shape}")

    # --- Visualize frames -------------------------------------------------
    n_show = min(6, config.n_frames)
    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))

    for i in range(n_show):
        raw_frame = frames[i].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, i].imshow(raw_frame)
        axes[0, i].set_title(f"t={i}", fontsize=8)
        axes[0, i].axis("off")

        # PINN snapshot for comparison
        t_frac = i / max(config.n_frames - 1, 1)
        pinn_field = pinn.field_at_t(t=t_frac, H=H, W=W)  # (1, H, W)
        axes[1, i].imshow(pinn_field.squeeze().numpy(), cmap="hot")
        axes[1, i].set_title(f"PINN t={t_frac:.2f}", fontsize=8)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("World model", fontsize=9)
    axes[1, 0].set_ylabel("PINN reference", fontsize=9)

    plt.suptitle("World model frames vs PINN snapshots", fontsize=11)
    plt.tight_layout()
    plt.savefig("13_world_model_result.png", dpi=120)
    print("Saved 13_world_model_result.png")


if __name__ == "__main__":
    main()
