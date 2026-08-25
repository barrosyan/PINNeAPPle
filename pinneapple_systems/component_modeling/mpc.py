"""pinneapple_systems.component_modeling.mpc — Model Predictive Control
(receding horizon) around an arbitrary differentiable component.

Real MPC in the textbook sense: at each outer step it optimizes a whole
future control sequence against the plant model over a horizon, applies only
the first action, then re-optimizes at the next step (receding horizon). The
plant itself IS the prediction model — the control sequence is optimized by
gradient descent straight through it (no finite differences, no
linearization), so `plant` must be a differentiable ``nn.Module`` (any
architecture; this module has no notion of a specific component or
registry).

Contrast with ``control.PIDController``: PID is reactive (responds to
current error only); MPC here is predictive (plans ``horizon`` steps ahead,
trading off tracking error against control effort via ``action_weight``).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def run_mpc(
    plant: nn.Module,
    *,
    in_dim: int,
    setpoint: float,
    control_input_index: int = 0,
    observed_output_index: int = 0,
    horizon: int = 10,
    steps: int = 30,
    opt_iters: int = 30,
    lr: float = 0.1,
    action_weight: float = 0.01,
    action_bounds: Tuple[float, float] = (-2.0, 2.0),
    fixed_inputs: Optional[Dict[int, float]] = None,
    seed: int = 0,
) -> Dict[str, List[float]]:
    """Receding-horizon MPC: at each of ``steps`` outer steps, optimize a
    length-``horizon`` control sequence (Adam, ``opt_iters`` iterations,
    gradients flowing through ``plant`` itself) minimizing::

        sum_k (y_k - setpoint)^2 + action_weight * u_k^2

    then apply only the first action. Returns the closed-loop trajectory
    ``{"time", "setpoint", "action", "output", "planned_cost"}``.

    ``plant`` is used exactly as given — no pretraining, no assumptions
    about its weights. The caller is responsible for supplying a plant whose
    ``control_input_index -> observed_output_index`` map is meaningful
    (trained, hand-built, or otherwise).
    """
    torch.manual_seed(seed)
    plant.eval()
    fixed_inputs = fixed_inputs or {}
    if control_input_index >= in_dim:
        raise ValueError(f"control_input_index={control_input_index} out of range for in_dim={in_dim}.")

    # The plant is frozen during control optimization — only the action
    # sequence is a decision variable, never the model weights.
    for p in plant.parameters():
        p.requires_grad_(False)

    def _plant_output(action: torch.Tensor) -> torch.Tensor:
        """Differentiable single-point evaluation at one control action."""
        x = torch.zeros(1, in_dim, dtype=torch.float32)
        for idx, val in fixed_inputs.items():
            x[0, idx] = val
        mask = torch.zeros(1, in_dim)
        mask[0, control_input_index] = 1.0
        x_in = x + mask * action
        out = plant(x_in)
        if hasattr(out, "y"):
            out = out.y
        if out.shape[1] <= observed_output_index:
            raise ValueError(
                f"observed_output_index={observed_output_index} out of range for "
                f"plant output width {out.shape[1]}."
            )
        return out[0, observed_output_index]

    history: Dict[str, List[float]] = {
        "time": [], "setpoint": [], "action": [], "output": [], "planned_cost": [],
    }
    # Warm-started across outer steps (standard receding-horizon practice:
    # shift last plan by one, repeat the tail).
    plan = torch.zeros(horizon, requires_grad=True)

    lo, hi = action_bounds
    for step in range(steps):
        optimizer = torch.optim.Adam([plan], lr=lr)
        cost = torch.zeros(())
        for _ in range(opt_iters):
            optimizer.zero_grad(set_to_none=True)
            cost = torch.zeros(())
            for k in range(horizon):
                u_k = plan[k].clamp(lo, hi)
                y_k = _plant_output(u_k)
                cost = cost + (y_k - setpoint) ** 2 + action_weight * u_k ** 2
            cost.backward()
            optimizer.step()

        with torch.no_grad():
            action = float(plan[0].clamp(lo, hi).item())
            output = float(_plant_output(torch.tensor(action)).item())
            planned_cost = float(cost.item())
            # shift plan: drop applied action, repeat tail as warm start
            plan[:-1] = plan[1:].clone()

        history["time"].append(step)
        history["setpoint"].append(setpoint)
        history["action"].append(action)
        history["output"].append(output)
        history["planned_cost"].append(planned_cost)

    return history
