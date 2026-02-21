"""Finite Volume Method solver scaffold with flux/source callbacks."""
from __future__ import annotations
from typing import Dict, Optional, Callable, Any

import torch

from .base import SolverBase, SolverOutput


class FVMSolver(SolverBase):
    """
    FVM Solver (scaffold MVP).

    Design:
      - build discrete operator from mesh volumes/faces
      - compute fluxes across faces
      - time integration (explicit Euler MVP)

    User provides:
      - flux_fn(state_cell, state_neighbor, face_info, params) -> flux
      - source_fn(state_cell, cell_info, params) -> source
    """
    def __init__(
        self,
        *,
        flux_fn: Callable[[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]], torch.Tensor],
        source_fn: Optional[Callable[[torch.Tensor, Dict[str, Any], Dict[str, Any]], torch.Tensor]] = None,
    ):
        super().__init__()
        self.flux_fn = flux_fn
        self.source_fn = source_fn

    def forward(
        self,
        *,
        mesh: Any,
        u0: torch.Tensor,              # (Nc, C) cell states
        steps: int,
        dt: float,
        topology: Dict[str, Any],       # faces, neighbors, areas, volumes...
        params: Dict[str, Any],
    ) -> SolverOutput:
        """
        topology expected keys (typical):
          - "faces": (F,2) cell indices (left,right), right=-1 for boundary
          - "areas": (F,1)
          - "normals": (F,dim)
          - "volumes": (Nc,1)
          - "boundary_state" optional callback or tensor for ghost cells
        """
        u = u0.clone()
        faces = topology["faces"]
        areas = topology["areas"]
        vols = topology["volumes"]

        traj = [u]

        for _ in range(int(steps)):
            dudt = torch.zeros_like(u)

            for f in range(faces.shape[0]):
                i = int(faces[f, 0].item())
                j = int(faces[f, 1].item())

                ui = u[i]
                if j >= 0:
                    uj = u[j]
                else:
                    # boundary: simple "copy" (Neumann-ish) as MVP
                    uj = ui

                face_info = {k: topology[k][f] for k in topology.keys() if k not in {"faces", "volumes"}}
                flux = self.flux_fn(ui, uj, face_info, params)  # (C,)

                Ai = areas[f]
                dudt[i] = dudt[i] - (Ai * flux) / vols[i]
                if j >= 0:
                    dudt[j] = dudt[j] + (Ai * flux) / vols[j]

            if self.source_fn is not None:
                # user can vectorize; MVP loops
                for c in range(u.shape[0]):
                    cell_info = {"volume": vols[c]}
                    dudt[c] = dudt[c] + self.source_fn(u[c], cell_info, params)

            u = u + float(dt) * dudt
            traj.append(u)

        out = torch.stack(traj, dim=0)  # (steps+1, Nc, C)
        return SolverOutput(
            result=out,
            losses={"total": torch.tensor(0.0, device=u.device)},
            extras={"dt": dt, "steps": steps},
        )
