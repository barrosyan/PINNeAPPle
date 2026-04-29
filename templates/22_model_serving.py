"""22_model_serving.py — REST inference server with pinneaple_serve.

Demonstrates:
- ModelServer: FastAPI wrapper around a trained PINN
- InferenceEndpoint: /predict and /health routes
- ServeConfig: port, workers, timeout, batch size
- Client-side request using requests (or httpx)

Run this script to start the server, then query it via:
    curl -X POST http://localhost:8080/predict \
         -H "Content-Type: application/json" \
         -d '{"inputs": [[0.25, 0.75], [0.5, 0.5]]}'
"""

import math
import torch
import torch.nn as nn
import numpy as np

from pinneaple_serve.server import ModelServer
from pinneaple_serve.config import ServeConfig
from pinneaple_serve.endpoint import InferenceEndpoint


# ---------------------------------------------------------------------------
# Tiny trained model — Poisson PINN (pre-trained weights loaded inline)
# ---------------------------------------------------------------------------

class PoissonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        # phi(x,y) enforces Dirichlet u=0 on boundary
        self._phi = lambda xy: (xy[:, 0:1] * (1 - xy[:, 0:1]) *
                                 xy[:, 1:2] * (1 - xy[:, 1:2]))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self._phi(xy) * self.net(xy)


def train_poisson(model: nn.Module, device, n_epochs: int = 3000) -> None:
    """Quick training pass so the served model has meaningful weights."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        opt.zero_grad()
        xy = torch.rand(2048, 2, device=device, requires_grad=True)
        u  = model(xy)
        g  = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_xx = torch.autograd.grad(g[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(g[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        f   = -2.0 * math.pi**2 * torch.sin(math.pi * xy[:, 0:1]) * \
               torch.sin(math.pi * xy[:, 1:2])
        loss = (u_xx + u_yy - f).pow(2).mean()
        loss.backward()
        opt.step()
    print(f"  final PDE loss = {loss.item():.4e}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Train model ---------------------------------------------------------
    print("Training Poisson PINN ...")
    model = PoissonNet().to(device)
    train_poisson(model, device, n_epochs=3000)
    model.eval()

    # --- Save weights --------------------------------------------------------
    torch.save(model.state_dict(), "22_poisson_weights.pt")
    print("Weights saved to 22_poisson_weights.pt")

    # --- Build inference endpoint --------------------------------------------
    endpoint = InferenceEndpoint(
        model=model,
        input_shape=(-1, 2),          # expects (N, 2) float32 inputs
        output_names=["u"],
        device=str(device),
        preprocess_fn=None,           # raw tensor passthrough
        postprocess_fn=None,
    )

    # --- Serve config --------------------------------------------------------
    config = ServeConfig(
        host="0.0.0.0",
        port=8080,
        workers=1,
        timeout=30,
        max_batch_size=1024,
        enable_cors=True,
        log_level="info",
    )

    # --- Start server --------------------------------------------------------
    server = ModelServer(endpoint=endpoint, config=config)

    print(f"\nStarting inference server on {config.host}:{config.port}")
    print("Routes:")
    print("  GET  /health   — liveness check")
    print("  POST /predict  — JSON {inputs: [[x,y], ...]} → {u: [...]}")
    print("  GET  /metrics  — request counters and latency")
    print("\nExample curl:")
    print('  curl -X POST http://localhost:8080/predict \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"inputs": [[0.25, 0.75], [0.5, 0.5]]}\'')
    print("\nPress Ctrl+C to stop.\n")

    server.run()   # blocking; ctrl+c to stop


if __name__ == "__main__":
    main()
