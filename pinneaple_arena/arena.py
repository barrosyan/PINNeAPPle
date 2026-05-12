"""Arena — main orchestrator for multi-model physics benchmarks.

Usage
-----
    arena = Arena.from_yaml("benchmark.yaml")
    arena.run()

    # or
    from pinneaple_arena import ArenaConfig, Arena
    cfg = ArenaConfig.from_dict({...})
    Arena(cfg).run()

Feature summary
---------------
  • ALL pinneaple_neural models via ModelRegistry.build()
  • pinneaple_physics compiled losses (when physics_preset is set)
  • pinneaple_data datasets (via DatasetConfig)
  • pinneaple_analysis inverse problems (via InverseConfig.enabled=True)
  • pinneaple_analysis UQ (via UQConfig.enabled=True)
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ArenaConfig, ModelConfig, InverseConfig, UQConfig, DatasetConfig
from .model_factory import build_model, is_graph_model, is_pinn_model
from .problems import get_problem, ArenaProblem
from .trainer import (
    TrainResult,
    train_pinn,
    train_supervised,
    train_graph,
    evaluate_model,
    run_uq,
    run_inverse,
    load_pinneaple_dataset,
)


# ── mesh builder for MeshGraphNet ─────────────────────────────────────────────

def _build_delaunay_graph(xy: np.ndarray):
    from scipy.spatial import Delaunay
    tri = Delaunay(xy)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            edges.add((min(a, b), max(a, b)))
    edges = list(edges)
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    edge_index = np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ], axis=0)
    edge_attr = np.concatenate([xy[dst] - xy[src], xy[src] - xy[dst]], axis=0)
    return edge_index, edge_attr


# ── Arena ─────────────────────────────────────────────────────────────────────

class Arena:
    """Multi-model physics benchmark runner.

    Parameters
    ----------
    config : ArenaConfig
    device : str  — torch device string (auto-detected if None)
    """

    def __init__(self, config: ArenaConfig, device: Optional[str] = None):
        self.cfg = config
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._problem: Optional[ArenaProblem] = None
        self._train_results: List[TrainResult] = []
        self._eval_results: List[Dict[str, Any]] = []
        self._data: Optional[Dict] = None

    # ── construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str, device: Optional[str] = None) -> "Arena":
        return cls(ArenaConfig.from_yaml(path), device=device)

    @classmethod
    def from_json(cls, path: str, device: Optional[str] = None) -> "Arena":
        return cls(ArenaConfig.from_json(path), device=device)

    @classmethod
    def from_config(cls, config: ArenaConfig, device: Optional[str] = None) -> "Arena":
        return cls(config, device=device)

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self) -> "Arena":
        """Train all models, evaluate, optionally run UQ/inverse, produce figures."""
        self._prepare_data()
        self._train_all()
        self._evaluate_all()
        if self.cfg.uq.enabled:
            self._run_uq_all()
        if self.cfg.inverse.enabled:
            self._run_inverse_all()
        if self.cfg.output.save_figures:
            self._visualize()
        self._print_summary()
        return self

    # ── data preparation ──────────────────────────────────────────────────────

    def _prepare_data(self):
        # ── Option A: external pinneaple_data dataset ─────────────────────
        if self.cfg.dataset is not None and self.cfg.dataset.dataset_id:
            self._prepare_from_dataset(self.cfg.dataset)
            return

        # ── Option B: built-in Arena problem ─────────────────────────────
        pc = self.cfg.problem
        self._problem = get_problem(pc.name)
        p = self._problem

        print(f"\n[Arena] Problem: {pc.name}  ({p.description})")
        print(f"        grid_n={pc.grid_n}  n_col={pc.n_col}  n_bc={pc.n_bc}")
        if p.physics_preset or pc.physics_preset:
            ps = pc.physics_preset or p.physics_preset
            print(f"        physics_preset={ps}")

        xy_int, Y_int, xy_bc, Y_bc, xy_eval, Y_eval, field_names = p.supervised_data(
            n_train=pc.n_train_supervised, n_bc=pc.n_bc, grid_n=pc.grid_n, **pc.params)

        # denser PINN collocation
        rng = np.random.default_rng(42)
        in_dim = p.input_dim
        lo = xy_int.min(axis=0); hi = xy_int.max(axis=0)
        xy_col = rng.uniform(lo, hi, (pc.n_col, in_dim))

        # mesh for GNN
        n_nodes = pc.n_mesh_nodes
        node_xy = rng.uniform(lo, hi, (n_nodes, in_dim))
        node_xy += rng.normal(0, 0.002, node_xy.shape)
        node_args = [node_xy[:, d] for d in range(in_dim)]
        node_fields = p.analytical(*node_args, **pc.params)
        if node_fields is not None:
            node_targets = np.stack([node_fields[f] for f in field_names], axis=1)
        else:
            from scipy.interpolate import NearestNDInterpolator
            node_targets = np.zeros((n_nodes, len(field_names)))
            for i, f in enumerate(field_names):
                ref_i = Y_eval[:, i] if Y_eval.ndim > 1 else Y_eval.ravel()
                node_targets[:, i] = NearestNDInterpolator(xy_eval, ref_i)(*node_args)

        node_feats = np.concatenate([node_xy,
            node_targets + rng.normal(0, 0.01, node_targets.shape)], axis=1)

        try:
            edge_index, edge_attr = _build_delaunay_graph(node_xy[:, :2])
        except Exception:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr  = np.zeros((0, 2),  dtype=np.float32)

        # try to get compiled physics losses from pinneaple_physics
        physics_preset = pc.physics_preset or p.physics_preset
        compiled = p.compiled_losses(physics_preset_override=pc.physics_preset, **pc.params)
        if compiled:
            print(f"        [OK] pinneaple_physics compiled losses loaded "
                  f"({len(compiled)} terms)")
        else:
            print(f"        [--] Using built-in autograd PINN residuals")

        self._data = {
            "xy_int": xy_int,   "Y_int": Y_int,
            "xy_bc":  xy_bc,    "Y_bc":  Y_bc,
            "xy_col": xy_col,
            "xy_eval": xy_eval, "Y_eval": Y_eval,
            "field_names": field_names,
            "node_xy": node_xy, "node_feats": node_feats,
            "node_targets": node_targets,
            "edge_index": edge_index, "edge_attr": edge_attr,
            "in_dim": in_dim,
            "out_dim": len(field_names),
            "compiled_losses": compiled,
        }

    def _prepare_from_dataset(self, ds_cfg: DatasetConfig):
        """Load data from pinneaple_data.datasets."""
        print(f"\n[Arena] Dataset: {ds_cfg.dataset_id}")
        X_train, Y_train, X_val, Y_val, field_names = load_pinneaple_dataset(
            ds_cfg.dataset_id,
            ds_cfg.input_fields, ds_cfg.output_fields,
            n_train=ds_cfg.n_train, n_val=ds_cfg.n_val,
            split_seed=ds_cfg.split_seed,
        )
        in_dim  = X_train.shape[1]
        out_dim = Y_train.shape[1] if Y_train.ndim > 1 else 1
        print(f"        X_train: {X_train.shape}  Y_train: {Y_train.shape}  "
              f"X_val: {X_val.shape}")

        # build minimal mesh for GNN
        rng = np.random.default_rng(42)
        n_nodes = min(500, len(X_train))
        idx = rng.choice(len(X_train), n_nodes, replace=False)
        node_xy = X_train[idx, :2] if in_dim >= 2 else np.stack(
            [X_train[idx, 0], np.zeros(n_nodes)], axis=1)
        node_targets = Y_train[idx]
        node_feats = np.concatenate([X_train[idx],
            node_targets + rng.normal(0, 0.01, node_targets.shape)], axis=1)
        try:
            edge_index, edge_attr = _build_delaunay_graph(node_xy)
        except Exception:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr  = np.zeros((0, 2),  dtype=np.float32)

        self._data = {
            "xy_int": X_train, "Y_int": Y_train,
            "xy_bc":  X_val,   "Y_bc":  Y_val,
            "xy_col": X_train,
            "xy_eval": X_val,  "Y_eval": Y_val,
            "field_names": field_names,
            "node_xy": node_xy, "node_feats": node_feats,
            "node_targets": node_targets,
            "edge_index": edge_index, "edge_attr": edge_attr,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "compiled_losses": None,   # no physics for raw datasets
        }

    # ── training ──────────────────────────────────────────────────────────────

    def _train_all(self):
        self._train_results = []
        for mcfg in self.cfg.models:
            print(f"\n[Arena] Training  {mcfg.name}  (type={mcfg.type})")
            result = self._train_one(mcfg)
            self._train_results.append(result)
            print(f"        done in {result.train_time:.1f}s")

    def _train_one(self, mcfg: ModelConfig) -> TrainResult:
        d = self._data
        in_dim  = d["in_dim"]
        out_dim = d["out_dim"]
        node_in = d["node_feats"].shape[1] if is_graph_model(mcfg) else in_dim
        edge_in = d["edge_attr"].shape[1] if is_graph_model(mcfg) and len(d["edge_attr"]) > 0 else in_dim
        model = build_model(mcfg, in_dim=node_in, out_dim=out_dim, edge_in_dim=edge_in)

        if is_pinn_model(mcfg) and self._problem is not None:
            return train_pinn(
                model, mcfg,
                pinn_residuals_fn=self._problem.pinn_residuals,
                xy_int=d["xy_col"],
                xy_bc=d["xy_bc"],
                uv_bc=d["Y_bc"],
                problem_params=self.cfg.problem.params,
                device=self.device,
                compiled_losses=d.get("compiled_losses"),
            )
        elif is_graph_model(mcfg):
            return train_graph(
                model, mcfg,
                node_feats=d["node_feats"],
                edge_index=d["edge_index"],
                edge_attr=d["edge_attr"],
                node_targets=d["node_targets"],
                device=self.device,
            )
        else:
            return train_supervised(
                model, mcfg,
                X_train=d["xy_int"],
                Y_train=d["Y_int"],
                device=self.device,
            )

    # ── evaluation ────────────────────────────────────────────────────────────

    def _evaluate_all(self):
        self._eval_results = []
        d = self._data
        for res in self._train_results:
            mcfg = self._mcfg_by_name(res.name)
            eval_out = evaluate_model(
                res, mcfg,
                xy_eval=d["xy_eval"],
                Y_ref=d["Y_eval"],
                field_names=d["field_names"],
                device=self.device,
                node_positions=d["node_feats"] if is_graph_model(mcfg) else None,
                edge_index=d["edge_index"] if is_graph_model(mcfg) else None,
                edge_attr=d["edge_attr"] if is_graph_model(mcfg) else None,
            )
            eval_out["name"] = res.name
            self._eval_results.append(eval_out)

    # ── UQ ────────────────────────────────────────────────────────────────────

    def _run_uq_all(self):
        print("\n[Arena] Running UQ analysis...")
        for tres in self._train_results:
            run_uq(tres, self._data["xy_eval"], self.cfg.uq, device=self.device)

    # ── inverse ───────────────────────────────────────────────────────────────

    def _run_inverse_all(self):
        print("\n[Arena] Running inverse problems...")
        for tres in self._train_results:
            run_inverse(tres, self._data["xy_eval"], self._data["Y_eval"],
                        self.cfg.inverse, device=self.device)

    # ── visualisation ─────────────────────────────────────────────────────────

    def _visualize(self):
        from .visualizer import (
            plot_field_comparison, plot_loss_curves,
            plot_metrics_table, plot_streamlines,
        )
        out = self.cfg.output
        os.makedirs(out.dir, exist_ok=True)
        prefix = os.path.join(out.dir, out.prefix)
        d = self._data
        kwargs = dict(dark_theme=out.dark_theme, dpi=out.dpi, show=out.show)

        print("\n[Arena] Saving figures...")
        plot_field_comparison(
            self._eval_results, d["field_names"],
            d["xy_eval"], self.cfg.problem.grid_n,
            problem_name=self.cfg.problem.name,
            save_path=f"{prefix}_fields.png", **kwargs)
        plot_loss_curves(
            self._train_results,
            save_path=f"{prefix}_losses.png", **kwargs)
        plot_metrics_table(
            self._eval_results, d["field_names"],
            self._train_results,
            save_path=f"{prefix}_metrics.png", **kwargs)
        if "u" in d["field_names"] and "v" in d["field_names"]:
            plot_streamlines(
                self._eval_results, d["xy_eval"],
                self.cfg.problem.grid_n,
                field_names=d["field_names"],
                problem_name=self.cfg.problem.name,
                save_path=f"{prefix}_streams.png", **kwargs)

        if self.cfg.uq.enabled:
            self._visualize_uq(prefix, kwargs)

    def _visualize_uq(self, prefix: str, kwargs: dict):
        try:
            from .visualizer import plot_uq
            for tres in self._train_results:
                if tres.uq_result is not None:
                    plot_uq(tres.uq_result, self._data["xy_eval"],
                            self._data["field_names"],
                            title=f"UQ — {tres.name}",
                            save_path=f"{prefix}_uq_{tres.name}.png", **kwargs)
        except Exception as e:
            warnings.warn(f"[Arena] UQ visualization failed: {e}")

    # ── summary ───────────────────────────────────────────────────────────────

    def _print_summary(self):
        d = self._data
        field_names = d["field_names"]
        print("\n" + "=" * 70)
        print(f"  ARENA RESULTS  >>  {self.cfg.problem.name}")
        print("=" * 70)
        header = f"  {'Model':<22}"
        for f in field_names:
            header += f"  L2-{f:<8}  rel-{f:<6}"
        header += "  Time(s)"
        print(header)
        print("-" * 70)
        for tres, eres in zip(self._train_results, self._eval_results):
            m = eres["metrics"]
            row = f"  {tres.name:<22}"
            for f in field_names:
                row += (f"  {m.get(f'L2_{f}', float('nan')):.3e}    "
                        f"{m.get(f'rel_{f}', float('nan')):.3e}  ")
            row += f"  {tres.train_time:6.1f}"
            print(row)
            if tres.uq_result is not None:
                try:
                    std = tres.uq_result.std
                    print(f"    {'UQ std':<20}  mean={float(std.mean()):.3e}")
                except Exception:
                    pass
            if tres.inverse_result is not None:
                try:
                    mf = tres.inverse_result.final_misfit
                    print(f"    {'Inverse misfit':<20}  {mf:.3e}")
                except Exception:
                    pass
        print("=" * 70)

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def results(self) -> List[Dict[str, Any]]:
        return self._eval_results

    @property
    def train_results(self) -> List[TrainResult]:
        return self._train_results

    def compare(self) -> None:
        self._print_summary()

    def _mcfg_by_name(self, name: str) -> ModelConfig:
        for m in self.cfg.models:
            if m.name == name:
                return m
        raise KeyError(name)
