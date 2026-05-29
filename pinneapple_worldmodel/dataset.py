"""World model dataset: (state_t, params) → state_{t+1} pairs.

:class:`WorldModelDataset` wraps a list of :class:`~.simulator.TrajectoryData`
objects and exposes them as individual one-step (or multi-step) prediction
samples ready for :class:`~pinneapple_train.Trainer`.

Each sample is a dict::

    {
        "state_t"   : Tensor (C, *grid)   — current field
        "state_tp1" : Tensor (C, *grid)   — target field h steps later
        "params"    : Tensor (P,)         — PDE parameters (normalised)
        "context"   : Tensor (P + meta,)  — params + one-hot PDE kind + IC/BC
        "scenario"  : str                 — scenario name
    }

:class:`DatasetBuilder` orchestrates the full simulator → dataset pipeline,
optionally using ``pinneapple_validate`` to filter out physically inconsistent
trajectories.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset

from .scenario import PhysicsScenario, BUILTIN_SCENARIOS
from .simulator import PhysicsSimulator, TrajectoryData


# ---------------------------------------------------------------------------
# WorldModelDataset
# ---------------------------------------------------------------------------

class WorldModelDataset(Dataset):
    """One-step or multi-step prediction dataset from a list of trajectories.

    Parameters
    ----------
    trajectories : list of TrajectoryData
    horizon : int
        Number of steps between input and target.  1 = next-step prediction,
        >1 = multi-step (useful for pretraining rollout stability).
    normalize : bool
        If True, each field channel is normalised to zero mean / unit std
        computed across the dataset on first access.
    transform : optional callable applied to each sample dict after assembly.
    """

    def __init__(
        self,
        trajectories: List[TrajectoryData],
        *,
        horizon: int = 1,
        normalize: bool = True,
        transform: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        self.trajectories = trajectories
        self.horizon = horizon
        self.normalize = normalize
        self.transform = transform

        # Build flat index: (traj_idx, t) for each valid pair
        self._index: List[Tuple[int, int]] = []
        for ti, traj in enumerate(trajectories):
            T = traj.states.shape[0]
            for t in range(T - horizon):
                self._index.append((ti, t))

        # Normalisation stats (lazy, computed on first call)
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

        # Parameter encoding
        self._param_keys = self._collect_param_keys()
        self._pde_kinds = self._collect_pde_kinds()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ti, t = self._index[idx]
        traj = self.trajectories[ti]

        state_t   = traj.states[t].clone()
        state_tp1 = traj.states[t + self.horizon].clone()

        if self.normalize:
            mean, std = self._get_norm_stats()
            state_t   = (state_t   - mean) / std
            state_tp1 = (state_tp1 - mean) / std

        params_vec = self._encode_params(traj.params)
        context    = self._encode_context(traj)

        sample = {
            "state_t":   state_t,
            "state_tp1": state_tp1,
            "params":    params_vec,
            "context":   context,
            "scenario":  traj.scenario_name,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _get_norm_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._mean is None:
            all_states = torch.cat([t.states for t in self.trajectories], dim=0)
            dims = [0] + list(range(2, all_states.ndim))  # all but channel
            self._mean = all_states.mean(dim=dims, keepdim=True)[0]
            self._std  = all_states.std(dim=dims, keepdim=True)[0].clamp(min=1e-6)
        return self._mean, self._std  # type: ignore[return-value]

    @property
    def norm_stats(self) -> Dict[str, torch.Tensor]:
        """Return ``{"mean": ..., "std": ...}`` for denormalising predictions."""
        mean, std = self._get_norm_stats()
        return {"mean": mean, "std": std}

    def denormalize(self, state: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation for visualisation or evaluation."""
        mean, std = self._get_norm_stats()
        return state * std + mean

    def _recompute_norm_stats(self) -> None:
        """Force recompute normalisation stats (call after modifying trajectories)."""
        self._mean = None
        self._std = None
        self._param_keys = self._collect_param_keys()
        self._pde_kinds = self._collect_pde_kinds()

    def _build_samples(self) -> None:
        """Rebuild the flat (traj_idx, t) sample index."""
        self._index = []
        for ti, traj in enumerate(self.trajectories):
            T = traj.states.shape[0]
            for t in range(T - self.horizon):
                self._index.append((ti, t))

    def _normalize(self) -> None:
        """Pre-normalise all trajectory states in-place (optional)."""
        mean, std = self._get_norm_stats()
        for traj in self.trajectories:
            traj.states = (traj.states - mean) / std
        # After in-place normalisation, raw stats become identity
        self._mean = torch.zeros_like(mean)
        self._std = torch.ones_like(std)

    # ------------------------------------------------------------------
    # Parameter / context encoding
    # ------------------------------------------------------------------

    def _collect_param_keys(self) -> List[str]:
        keys: set = set()
        for t in self.trajectories:
            keys.update(t.params.keys())
        return sorted(keys)

    def _collect_pde_kinds(self) -> List[str]:
        kinds: set = set()
        for t in self.trajectories:
            kinds.add(t.scenario_name.split("_")[0])
        return sorted(kinds)

    def _encode_params(self, params: Dict[str, float]) -> torch.Tensor:
        """Return a (P,) float32 tensor of parameter values (0 for missing)."""
        return torch.tensor(
            [params.get(k, 0.0) for k in self._param_keys],
            dtype=torch.float32,
        )

    def _encode_context(self, traj: TrajectoryData) -> torch.Tensor:
        """Return (P + K,) context: params + one-hot PDE kind."""
        params_vec = self._encode_params(traj.params)
        kind = traj.scenario_name.split("_")[0]
        kind_oh = torch.zeros(len(self._pde_kinds))
        if kind in self._pde_kinds:
            kind_oh[self._pde_kinds.index(kind)] = 1.0
        return torch.cat([params_vec, kind_oh])

    @property
    def context_dim(self) -> int:
        """Dimension of the context vector fed to the world model."""
        return len(self._param_keys) + len(self._pde_kinds)

    @property
    def n_fields(self) -> int:
        """Number of field channels."""
        return self.trajectories[0].states.shape[1]

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        """Spatial grid shape."""
        return tuple(self.trajectories[0].states.shape[2:])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save trajectories and metadata to *directory*."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for i, traj in enumerate(self.trajectories):
            torch.save(traj.states, path / f"traj_{i:05d}_states.pt")
            with open(path / f"traj_{i:05d}_meta.json", "w") as f:
                json.dump({
                    "params": traj.params,
                    "scenario_name": traj.scenario_name,
                    "metadata": {k: str(v) for k, v in traj.metadata.items()},
                }, f)
        # Save index
        with open(path / "dataset_info.json", "w") as f:
            json.dump({
                "n_trajectories": len(self.trajectories),
                "horizon": self.horizon,
                "param_keys": self._param_keys,
                "pde_kinds": self._pde_kinds,
                "n_samples": len(self),
            }, f)

    @classmethod
    def load(
        cls,
        directory: str,
        *,
        horizon: int = 1,
        normalize: bool = True,
    ) -> "WorldModelDataset":
        """Load a previously saved dataset from *directory*."""
        path = Path(directory)
        trajs = []
        for states_path in sorted(path.glob("traj_*_states.pt")):
            stem = states_path.stem.replace("_states", "")
            meta_path = path / f"{stem}_meta.json"
            states = torch.load(states_path, map_location="cpu")
            with open(meta_path) as f:
                meta = json.load(f)
            trajs.append(TrajectoryData(
                states=states,
                params=meta["params"],
                scenario_name=meta["scenario_name"],
                metadata=meta.get("metadata", {}),
            ))
        return cls(trajs, horizon=horizon, normalize=normalize)


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for :class:`DatasetBuilder`.

    Parameters
    ----------
    scenarios : list of scenario names or PhysicsScenario objects.
    n_samples_per_scenario : int — trajectories to generate per scenario.
    horizon : int — prediction horizon (steps between state_t and state_tp1).
    normalize : bool — normalise fields to zero-mean / unit-std.
    validate_physics : bool — discard physically inconsistent trajectories.
    save_dir : str or None — if set, persist trajectories after generation.
    device : str — compute device for simulation.
    verbose : bool
    """
    scenarios: List[Any] = field(default_factory=lambda: ["heat_2d", "burgers_1d"])
    n_samples_per_scenario: int = 500
    horizon: int = 1
    normalize: bool = True
    validate_physics: bool = False
    save_dir: Optional[str] = None
    device: str = "cpu"
    verbose: bool = True


class DatasetBuilder:
    """Orchestrates physics simulation → :class:`WorldModelDataset` assembly.

    Parameters
    ----------
    config : DatasetConfig

    Example
    -------
    >>> builder = DatasetBuilder(DatasetConfig(
    ...     scenarios=["heat_2d", "burgers_1d"],
    ...     n_samples_per_scenario=500,
    ... ))
    >>> dataset = builder.build()
    >>> print(f"Dataset: {len(dataset)} samples, context_dim={dataset.context_dim}")
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def build(self) -> WorldModelDataset:
        """Run simulators and assemble the dataset."""
        cfg = self.config
        all_trajs: List[TrajectoryData] = []

        scenarios = self._resolve_scenarios()
        for sc in scenarios:
            if cfg.verbose:
                print(f"[DatasetBuilder] Simulating '{sc.name}' "
                      f"({cfg.n_samples_per_scenario} samples) …")

            sim = PhysicsSimulator(sc, device=cfg.device, verbose=cfg.verbose)
            trajs = sim.generate_batch(
                cfg.n_samples_per_scenario,
                base_seed=hash(sc.name) % (2 ** 31),
            )

            if cfg.validate_physics:
                trajs = self._filter_valid(trajs)

            all_trajs.extend(trajs)
            if cfg.verbose:
                print(f"  → {len(trajs)} valid trajectories "
                      f"({all_trajs[-1].states.shape})")

        dataset = WorldModelDataset(
            all_trajs,
            horizon=cfg.horizon,
            normalize=cfg.normalize,
        )

        if cfg.save_dir:
            if cfg.verbose:
                print(f"[DatasetBuilder] Saving to '{cfg.save_dir}' …")
            dataset.save(cfg.save_dir)

        return dataset

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_scenarios(self) -> List[PhysicsScenario]:
        out = []
        for s in self.config.scenarios:
            if isinstance(s, PhysicsScenario):
                out.append(s)
            elif isinstance(s, str):
                if s not in BUILTIN_SCENARIOS:
                    raise ValueError(
                        f"Unknown scenario '{s}'. Available: {sorted(BUILTIN_SCENARIOS)}"
                    )
                out.append(BUILTIN_SCENARIOS[s])
            else:
                raise TypeError(f"Expected str or PhysicsScenario, got {type(s)}")
        return out

    def _filter_valid(self, trajs: List[TrajectoryData]) -> List[TrajectoryData]:
        """Remove trajectories with NaN/Inf or unbounded growth."""
        valid = []
        for traj in trajs:
            s = traj.states
            if torch.isnan(s).any() or torch.isinf(s).any():
                continue
            ratio = s[-1].abs().max() / (s[0].abs().max() + 1e-8)
            if ratio > 1e4:
                continue
            valid.append(traj)
        return valid
