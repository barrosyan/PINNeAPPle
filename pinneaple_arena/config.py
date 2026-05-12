"""ArenaConfig — structured configuration for PINNeAPPle Arena benchmarks.

Supports loading from YAML, JSON, or direct Python construction.

Example YAML
------------
problem:
  name: kovasznay_ns
  physics_preset: ns_incompressible_2d_default   # optional: use pinneaple_physics preset
  params:
    re: 40.0
  grid_n: 40
  n_col: 2000
  n_bc: 500

models:
  - name: VanillaPINN
    type: vanilla_pinn
    network:
      hidden: [128, 128, 128, 128]
      activation: tanh
    training:
      epochs: 2000
      lr: 5.0e-4
      grad_clip: 1.0

  - name: FNO-2D
    type: fno2d
    network:
      width: 32
      modes: 12
      layers: 4
    training:
      epochs: 600
      lr: 1.0e-3

  - name: MeshGraphNet
    type: meshgraphnet
    network:
      hidden_dim: 128
      n_message_passing: 6
    training:
      epochs: 600
      lr: 1.0e-3

# Optional: load a pinneaple_data dataset instead of a built-in problem
# dataset:
#   dataset_id: navier_stokes_2d
#   input_fields: [x, y]
#   output_fields: [u, v, p]
#   n_train: 1000

# Optional: inverse problem mode
# inverse:
#   enabled: true
#   params: [re]
#   n_obs: 100
#   noise_std: 0.01
#   method: adam
#   n_iters: 2000
#   lambda_reg: 1.0e-3

# Optional: uncertainty quantification
# uq:
#   enabled: true
#   method: mc_dropout     # mc_dropout | ensemble | aleatoric | decompose
#   n_samples: 50
#   dropout_rate: 0.1

output:
  dir: outputs/
  prefix: benchmark
  save_figures: true
  dpi: 150
  dark_theme: true
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    """Neural network architecture hyperparameters."""
    # Shared / PINN
    hidden: List[int]          = field(default_factory=lambda: [128, 128, 128, 128])
    activation: str            = "tanh"
    # FNO
    width: int                 = 32
    modes: int                 = 12
    modes2: int                = 0      # 0 = same as modes
    layers: int                = 4
    use_grid: bool             = True
    # MeshGraphNet / GNN
    hidden_dim: int            = 128
    n_layers: int              = 2
    n_message_passing: int     = 6
    edge_in_dim: int           = 0
    dropout: float             = 0.0
    # SIREN
    omega_0: float             = 30.0
    # Transformer
    d_model: int               = 128
    nhead: int                 = 4
    num_layers: int            = 4
    dim_feedforward: int       = 256
    # DeepONet / Neural Operators
    branch_dim: int            = 0     # 0 = same as in_dim
    trunk_dim: int             = 0     # 0 = same as in_dim
    # NeuralODE / continuous
    state_dim: int             = 0     # 0 = same as in_dim
    # Extra kwargs forwarded verbatim to the constructor
    extra: Dict[str, Any]      = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        known_kw = {k: v for k, v in d.items() if k in known}
        extra    = {k: v for k, v in d.items() if k not in known}
        obj = cls(**known_kw)
        obj.extra = extra
        return obj


@dataclass
class TrainingConfig:
    """Training hyperparameters for a single model run."""
    epochs: int                = 2000
    lr: float                  = 1e-3
    weight_decay: float        = 0.0
    grad_clip: float           = 1.0
    optimizer: str             = "adam"        # "adam" | "adamw" | "sgd"
    scheduler: str             = "cosine"      # "cosine" | "step" | "none"
    batch_size: int            = 16            # used only for data-driven models
    seed: Optional[int]        = None

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ModelConfig:
    """Full specification for one model in the benchmark."""
    name: str                          # display name, e.g. "VanillaPINN"
    type: str                          # model key from ModelRegistry, e.g. "vanilla_pinn"
    network: NetworkConfig             = field(default_factory=NetworkConfig)
    training: TrainingConfig           = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        name = d.get("name", d.get("type", "model"))
        mtype = d.get("type", "vanilla_pinn")
        net   = NetworkConfig.from_dict(d.get("network", {}))
        trn   = TrainingConfig.from_dict(d.get("training", {}))
        return cls(name=name, type=mtype, network=net, training=trn)


@dataclass
class ProblemConfig:
    """Physics problem selection and parameters."""
    name: str                         # e.g. kovasznay_ns | burgers_1d | poisson_2d
    params: Dict[str, Any]            = field(default_factory=dict)
    physics_preset: Optional[str]     = None    # override with pinneaple_physics preset name
    grid_n: int                       = 40       # evaluation grid resolution
    n_col: int                        = 2000     # PINN interior collocation
    n_bc: int                         = 500      # PINN boundary collocation
    n_train_supervised: int           = 200      # training samples for data-driven models
    n_mesh_nodes: int                 = 600      # nodes for MeshGraphNet

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        obj_kw = {k: v for k, v in d.items() if k in known and k != "params"}
        params  = d.get("params", {})
        return cls(**obj_kw, params=params)


@dataclass
class OutputConfig:
    """Output / visualisation settings."""
    dir: str                  = "outputs"
    prefix: str               = "arena"
    save_figures: bool        = True
    dpi: int                  = 150
    dark_theme: bool          = True
    show: bool                = False

    @classmethod
    def from_dict(cls, d: dict) -> "OutputConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class InverseConfig:
    """Inverse problem configuration (identifying unknown physics parameters)."""
    enabled: bool             = False
    params: List[str]         = field(default_factory=list)  # unknown param names
    n_obs: int                = 100           # number of synthetic observations
    noise_std: float          = 0.01          # observation noise
    method: str               = "adam"        # "adam" | "lbfgs" | "eki" | "teki"
    n_iters: int              = 2000
    lambda_reg: float         = 1e-3
    sensor_locations: Optional[str] = None    # path to sensor location file (npy/csv)

    @classmethod
    def from_dict(cls, d: dict) -> "InverseConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        obj_kw = {k: v for k, v in d.items() if k in known and k != "params"}
        params = d.get("params", [])
        return cls(**obj_kw, params=params)


@dataclass
class UQConfig:
    """Uncertainty quantification configuration."""
    enabled: bool             = False
    method: str               = "mc_dropout"  # "mc_dropout" | "ensemble" | "aleatoric" | "decompose"
    n_samples: int            = 50
    dropout_rate: float       = 0.1
    coverage: float           = 0.95          # target coverage for conformal prediction

    @classmethod
    def from_dict(cls, d: dict) -> "UQConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DatasetConfig:
    """Use a pinneaple_data dataset instead of a built-in Arena problem."""
    dataset_id: str           = ""
    input_fields: List[str]   = field(default_factory=list)
    output_fields: List[str]  = field(default_factory=list)
    n_train: int              = 1000
    n_val: int                = 200
    split_seed: int           = 42

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        obj_kw = {k: v for k, v in d.items()
                  if k in known and k not in ("input_fields", "output_fields")}
        return cls(
            **obj_kw,
            input_fields=d.get("input_fields", []),
            output_fields=d.get("output_fields", []),
        )


@dataclass
class ArenaConfig:
    """Top-level Arena configuration."""
    problem: ProblemConfig
    models: List[ModelConfig]
    output: OutputConfig              = field(default_factory=OutputConfig)
    inverse: InverseConfig            = field(default_factory=InverseConfig)
    uq: UQConfig                      = field(default_factory=UQConfig)
    dataset: Optional[DatasetConfig]  = None

    @classmethod
    def from_dict(cls, d: dict) -> "ArenaConfig":
        problem  = ProblemConfig.from_dict(d.get("problem", {}))
        models   = [ModelConfig.from_dict(m) for m in d.get("models", [])]
        output   = OutputConfig.from_dict(d.get("output", {}))
        inverse  = InverseConfig.from_dict(d.get("inverse", {}))
        uq       = UQConfig.from_dict(d.get("uq", {}))
        dataset  = DatasetConfig.from_dict(d["dataset"]) if "dataset" in d else None
        return cls(problem=problem, models=models, output=output,
                   inverse=inverse, uq=uq, dataset=dataset)

    @classmethod
    def from_yaml(cls, path: str) -> "ArenaConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "ArenaConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        import yaml
        import dataclasses
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, default_flow_style=False, sort_keys=False)
