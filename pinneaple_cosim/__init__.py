"""PINNeAPPle native co-simulation engine.

A graph-based, differentiable co-simulation framework with first-class
support for PyTorch models and Physics-Informed Neural Networks (PINNs).

Architecture
------------
- **Nodes** — computational units with named input/output ports.
- **Connections** — directed edges between node ports.
- **CoSimGraph** — registry of nodes and connections with topology analysis
  (Tarjan SCC, condensation order for algebraic loops).
- **CoSimEngine** — time-stepped execution; handles feedback loops via
  Gauss-Seidel or Jacobi iteration.
- **TrajectoryRecorder** — logs port values over time for analysis/plotting.
- **CoSimLoss** — composite weighted loss (data + physics + coupling).

Quick start::

    from pinneaple_cosim import (
        CoSimGraph, CoSimEngine, TrajectoryRecorder,
        AnalyticalNode, PINNNode,
        CoSimLoss,
    )

    # 1. Build graph
    graph = CoSimGraph()
    graph.add_node(forcing_node).add_node(mass_node)
    graph.connect("forcing.F", "mass.F_ext")
    graph.connect("mass.x",    "mass.x_prev")   # feedback loop

    # 2. Simulate
    recorder = TrajectoryRecorder().watch("mass", "x").watch("mass", "v")
    engine = CoSimEngine(graph, recorder=recorder)
    engine.reset()
    engine.run(T=10.0, dt=0.01)

    # 3. Plot
    traj = recorder.get("mass", "x")
    import matplotlib.pyplot as plt
    plt.plot(traj.times, traj.values)
"""
from .node import CoSimNode, TorchNode, AnalyticalNode, PINNNode, BlackBoxNode
from .connection import Connection
from .graph import CoSimGraph
from .recorder import Trajectory, TrajectoryRecorder
from .engine import CoSimEngine
from .losses import DataLoss, PhysicsLoss, CouplingLoss, CoSimLoss

# PINNeAPPle ecosystem adapters (import individually to avoid hard deps at package level)
from .adapters import (
    PINNeProblemNode,
    PINNeAPPleModelNode,
    SymbolicPDENode,
    TimeSeriesCoSimNode,
)
from .trainer import CoSimTrainer

__all__ = [
    # --- Core nodes ---
    "CoSimNode",
    "TorchNode",
    "AnalyticalNode",
    "PINNNode",
    "BlackBoxNode",
    # --- PINNeAPPle adapter nodes ---
    "PINNeProblemNode",        # ProblemSpec + compiled physics + ModelRegistry
    "PINNeAPPleModelNode",     # any BaseModel (DeepONet, FNO, surrogate, …)
    "SymbolicPDENode",         # SymPy expression → autograd residual
    "TimeSeriesCoSimNode",     # TSModelBase (LSTM, FFT+LSTM, TFT, …)
    # --- Graph primitives ---
    "Connection",
    "CoSimGraph",
    # --- Recording ---
    "Trajectory",
    "TrajectoryRecorder",
    # --- Execution ---
    "CoSimEngine",
    # --- Training ---
    "CoSimTrainer",
    # --- Losses ---
    "DataLoss",
    "PhysicsLoss",
    "CouplingLoss",
    "CoSimLoss",
]
