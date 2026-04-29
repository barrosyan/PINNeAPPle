"""31_problem_design_llm.py — LLM-assisted PDE problem design.

Demonstrates:
- ProblemDesigner: natural-language → PDE specification pipeline
- ProblemSpec: structured representation of a PDE problem
- LLMBackend: configure which LLM provider to use for elicitation
- AutoPINNBuilder: turn a ProblemSpec into a runnable PINN config
- Interactive (or scripted) problem elicitation loop
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_problemdesign.designer import ProblemDesigner, DesignerConfig
from pinneaple_problemdesign.spec import ProblemSpec, PDESpec, BoundaryConditionSpec
from pinneaple_problemdesign.builder import AutoPINNBuilder
from pinneaple_problemdesign.llm_backend import LLMBackend, LLMBackendConfig


# ---------------------------------------------------------------------------
# This template demonstrates two modes:
#
# 1. Scripted mode (default, no API key required):
#    A pre-defined natural-language description is parsed into a ProblemSpec
#    using the rule-based fallback parser, then handed to AutoPINNBuilder.
#
# 2. LLM mode (set USE_LLM=True + configure your API key in the env):
#    ProblemDesigner calls the configured LLM to elicit and validate the spec.
# ---------------------------------------------------------------------------

USE_LLM = False   # set True to use live LLM elicitation

PROBLEM_DESCRIPTION = """
Solve the 2D steady-state heat equation on a unit square [0,1]^2.
The thermal diffusivity is k = 0.1.
Boundary conditions: T = 100 on the bottom edge (y=0),
T = 0 on the top edge (y=1), and zero-flux (Neumann) on left and right walls.
The heat source is Q = 50 * sin(pi*x).
Use a SIREN network with 4 hidden layers of width 64.
Train for 5000 epochs with Adam at learning rate 1e-3.
"""


def run_scripted_design() -> ProblemSpec:
    """Parse the problem description using the rule-based fallback."""
    designer = ProblemDesigner(
        config=DesignerConfig(
            use_llm=False,
            validate_spec=True,
        )
    )
    spec = designer.parse_description(PROBLEM_DESCRIPTION)
    return spec


def run_llm_design() -> ProblemSpec:
    """Use a live LLM to elicit a ProblemSpec from the description."""
    llm_cfg = LLMBackendConfig(
        provider="openai",             # "openai" | "anthropic" | "local"
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=1024,
    )
    backend = LLMBackend(config=llm_cfg)
    designer = ProblemDesigner(
        config=DesignerConfig(use_llm=True, validate_spec=True),
        llm_backend=backend,
    )
    spec = designer.elicit(PROBLEM_DESCRIPTION)
    return spec


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Problem description:\n  {PROBLEM_DESCRIPTION.strip()}\n")

    # --- Step 1: Elicit ProblemSpec ----------------------------------------
    if USE_LLM:
        print("Eliciting ProblemSpec via LLM ...")
        spec = run_llm_design()
    else:
        print("Parsing ProblemSpec (rule-based fallback) ...")
        spec = run_scripted_design()

    print("\nProblemSpec:")
    print(f"  PDE:            {spec.pde.equation_str}")
    print(f"  Domain:         {spec.domain}")
    print(f"  BCs:            {[bc.type for bc in spec.boundary_conditions]}")
    print(f"  Network arch:   {spec.model_config.get('arch', 'unknown')}")
    print(f"  Hidden layers:  {spec.model_config.get('hidden_layers', '?')}")
    print(f"  Hidden width:   {spec.model_config.get('hidden_dim', '?')}")
    print(f"  n_epochs:       {spec.training_config.get('n_epochs', '?')}")

    # --- Step 2: AutoPINNBuilder -------------------------------------------
    print("\nBuilding PINN from spec ...")
    builder = AutoPINNBuilder(device=str(device))
    pinn_config = builder.build(spec)

    model     = pinn_config.model.to(device)
    loss_fn   = pinn_config.loss_fn
    optimizer = pinn_config.optimizer
    n_epochs  = pinn_config.n_epochs

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Epochs: {n_epochs}")

    # --- Step 3: Run training ----------------------------------------------
    print("\nTraining PINN ...")
    history = []
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model, epoch)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if epoch % 1000 == 0:
            print(f"  epoch {epoch:5d} | loss = {loss.item():.4e}")

    print("Training complete.")

    # --- Step 4: Visualise -------------------------------------------------
    n_vis = 60
    x_ = np.linspace(0, 1, n_vis, dtype=np.float32)
    xx, yy = np.meshgrid(x_, x_)
    xy_vis = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), device=device
    )
    with torch.no_grad():
        T_pred = model(xy_vis).cpu().numpy().reshape(n_vis, n_vis)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im = axes[0].contourf(xx, yy, T_pred, levels=30, cmap="hot")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("Predicted T(x,y) — AutoPINN from LLM spec")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].semilogy(history)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Total loss")
    axes[1].set_title("Training loss")
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("31_problem_design_llm_result.png", dpi=120)
    print("Saved 31_problem_design_llm_result.png")


if __name__ == "__main__":
    main()
