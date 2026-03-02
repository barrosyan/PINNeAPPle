from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import torch


@dataclass
class GeneticSDF2D:
    """A simple 'genetic' implicit geometry as an SDF with parameters.

    SDF is union of a circle and an ellipse with smooth min (soft union).

    Params:
      - circle center (cx,cy), radius r
      - ellipse center (ex,ey), radii (a,b), rotation theta
      - smoothness k for softmin
    """
    cx: float = 0.5
    cy: float = 0.5
    r: float = 0.30

    ex: float = 0.55
    ey: float = 0.45
    a: float = 0.22
    b: float = 0.12
    theta: float = 0.35

    k: float = 40.0

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # circle
        dc = torch.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2) - self.r

        # ellipse (rotated)
        ct = torch.cos(torch.tensor(self.theta, device=x.device, dtype=x.dtype))
        st = torch.sin(torch.tensor(self.theta, device=x.device, dtype=x.dtype))
        xr = (x - self.ex) * ct + (y - self.ey) * st
        yr = -(x - self.ex) * st + (y - self.ey) * ct
        de = torch.sqrt((xr / self.a) ** 2 + (yr / self.b) ** 2) - 1.0
        # scale to approximate distance
        de = de * min(self.a, self.b)

        # soft union: smin(dc, de)
        k = torch.tensor(self.k, device=x.device, dtype=x.dtype)
        # avoid overflow with clamp
        h = torch.clamp(0.5 + 0.5 * (de - dc) * k, 0.0, 1.0)
        smin = torch.lerp(de, dc, h) - (1.0 / k) * h * (1.0 - h)
        return smin


def make_grid(H: int, W: int, device, dtype):
    ys = torch.linspace(0, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def marching_squares_zero_level(sdf: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
    """Extract approximate boundary points for sdf=0 via marching squares.

    Inputs:
      sdf: (H,W) scalar field
      xx,yy: (H,W) coordinates in [0,1]
    Returns:
      points: (M,2) tensor in (x,y)
    """
    H, W = sdf.shape
    pts: List[torch.Tensor] = []

    def interp(p1, p2, v1, v2):
        # linear interpolation to v=0
        t = v1 / (v1 - v2 + 1e-12)
        return p1 + t * (p2 - p1)

    for i in range(H - 1):
        for j in range(W - 1):
            v00 = sdf[i, j]
            v10 = sdf[i, j + 1]
            v01 = sdf[i + 1, j]
            v11 = sdf[i + 1, j + 1]

            s00 = v00 <= 0
            s10 = v10 <= 0
            s01 = v01 <= 0
            s11 = v11 <= 0

            code = (1 if s00 else 0) | (2 if s10 else 0) | (4 if s11 else 0) | (8 if s01 else 0)
            if code == 0 or code == 15:
                continue

            p00 = torch.stack([xx[i, j], yy[i, j]])
            p10 = torch.stack([xx[i, j + 1], yy[i, j + 1]])
            p01 = torch.stack([xx[i + 1, j], yy[i + 1, j]])
            p11 = torch.stack([xx[i + 1, j + 1], yy[i + 1, j + 1]])

            # edges: e0 p00-p10, e1 p10-p11, e2 p11-p01, e3 p01-p00
            inters = []
            if s00 != s10:
                inters.append(interp(p00, p10, v00, v10))
            if s10 != s11:
                inters.append(interp(p10, p11, v10, v11))
            if s11 != s01:
                inters.append(interp(p11, p01, v11, v01))
            if s01 != s00:
                inters.append(interp(p01, p00, v01, v00))

            if len(inters) >= 2:
                # store midpoints of segments (robust & simple)
                for k in range(0, len(inters) - 1, 2):
                    mid = 0.5 * (inters[k] + inters[k + 1])
                    pts.append(mid)

    if not pts:
        return torch.zeros((0, 2), device=sdf.device, dtype=sdf.dtype)
    return torch.stack(pts, dim=0)
