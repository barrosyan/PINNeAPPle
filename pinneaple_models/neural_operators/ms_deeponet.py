from __future__ import annotations
"""Multi-scale DeepONet for multi-resolution operator learning."""
import torch
import torch.nn as nn

from .deeponet import DeepONet
from .base import OperatorOutput


class MultiScaleDeepONet(DeepONet):
    """
    Multi-scale DeepONet:
      - multiple trunk nets at different resolutions
      - summed contribution
    """
    def __init__(self, branch_dim, trunk_dim, out_dim, scales=(32, 64, 128)):
        super().__init__(branch_dim, trunk_dim, out_dim, modes=scales[0])
        self.trunks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_dim, s),
                nn.GELU(),
                nn.Linear(s, s),
            )
            for s in scales
        ])
        self.out = nn.Linear(sum(scales), out_dim)

    def forward(self, u, coords, **kw):
        b = self.branch(u)
        feats = []
        for t in self.trunks:
            feats.append(torch.einsum("bm,nm->bnm", b, t(coords)))
        y = torch.cat(feats, dim=-1)
        y = self.out(y)
        return OperatorOutput(y=y, losses={"total": torch.tensor(0.0, device=y.device)}, extras={})
