"""Arena — main orchestrator for multi-model physics benchmarks.

Usage
-----
    arena = Arena.from_yaml("benchmark.yaml")
    arena.run()

    # or
    from pinneapple_arena import ArenaConfig, Arena
    cfg = ArenaConfig.from_dict({...})
    Arena(cfg).run()

Feature summary
---------------
  • ALL pinneapple_neural models via ModelRegistry.build()
  • pinneapple_physics compiled losses (when physics_preset is set)
  • pinneapple_data datasets (via DatasetConfig)
  • pinneapple_analysis inverse problems (via InverseConfig.enabled=True)
  • pinneapple_analysis UQ (via UQConfig.enabled=True)
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

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
    load_pinneapple_dataset,
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


# ── accuracy / physics-aware ranking ──────────────────────────────────────────

def _accuracy_score(eres: Dict[str, Any], field_names: List[str]) -> float:
    """Mean relative-L2 error across fields — lower is better. This is the
    same accuracy notion already shown in `_print_summary`'s ``rel-*``
    columns, just reduced to one scalar for ranking purposes."""
    m = eres.get("metrics", {})
    rels = [m[f"rel_{f}"] for f in field_names if f"rel_{f}" in m]
    if not rels:
        return float("nan")
    return float(np.mean(rels))


def rank_by_accuracy(train_results: List[TrainResult],
                      eval_results: List[Dict[str, Any]],
                      field_names: List[str]) -> List[Tuple[str, float]]:
    """Pure-accuracy ranking: (name, mean_rel_error) sorted ascending
    (best first). This is Arena's default, unchanged ranking behaviour."""
    by_name = {e["name"]: e for e in eval_results}
    scored = [(tres.name, _accuracy_score(by_name[tres.name], field_names))
              for tres in train_results if tres.name in by_name]
    scored.sort(key=lambda t: (t[1] != t[1], t[1]))  # NaN sorts last
    return scored


def physics_aware_rank(
    train_results: List[TrainResult],
    eval_results: List[Dict[str, Any]],
    field_names: List[str],
    n_std: float = 2.0,
    residual_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Additive, opt-in comparison view layered on top of the default
    pure-accuracy ranking.

    Keeps ``ranking`` identical to :func:`rank_by_accuracy` (accuracy is
    still what orders the models) but also flags any PINN-family model
    whose ``physics_residual`` is an outlier relative to the group —
    i.e. a model that *looks* accurate at the evaluation points but is
    violating the PDE it was trained against. Flags are additive
    annotations, never a silent re-ordering: a model that ranks #1 on
    accuracy and gets flagged still ranks #1.

    Parameters
    ----------
    n_std:
        A model is flagged if its ``physics_residual`` exceeds
        ``median + n_std * std`` of the group of models that report one.
        Ignored (outlier test skipped) if fewer than 2 models report a
        residual.
    residual_threshold:
        Optional absolute cutoff. If set, any model whose residual exceeds
        this value is flagged in addition to (not instead of) the
        relative-outlier test above.

    Returns
    -------
    dict with keys:
        ``ranking``               — same as :func:`rank_by_accuracy`
        ``physics_residuals``     — {name: residual} for models that have one
        ``flagged_high_residual`` — names flagged as outliers, in ranking order
        ``group_median``, ``group_std`` — stats used for the outlier test
                                           (``None`` if too few data points)
    """
    ranking = rank_by_accuracy(train_results, eval_results, field_names)
    residuals = {tres.name: tres.physics_residual
                 for tres in train_results if tres.physics_residual is not None}

    group_median: Optional[float] = None
    group_std: Optional[float] = None
    flagged: List[str] = []

    if residuals:
        vals = np.array(list(residuals.values()), dtype=float)
        if len(vals) >= 2:
            group_median = float(np.median(vals))
            group_std = float(np.std(vals))
        for name, r in residuals.items():
            is_relative_outlier = (
                group_median is not None and group_std is not None and group_std > 0
                and r > group_median + n_std * group_std
            )
            is_absolute_outlier = (
                residual_threshold is not None and r > residual_threshold
            )
            if is_relative_outlier or is_absolute_outlier:
                flagged.append(name)

    # preserve ranking order in the flagged list
    order = {name: i for i, (name, _) in enumerate(ranking)}
    flagged.sort(key=lambda n: order.get(n, len(order)))

    return {
        "ranking": ranking,
        "physics_residuals": residuals,
        "flagged_high_residual": flagged,
        "group_median": group_median,
        "group_std": group_std,
    }


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

    def run(self, physics_aware: bool = False) -> "Arena":
        """Train all models, evaluate, optionally run UQ/inverse, produce figures.

        Parameters
        ----------
        physics_aware:
            Opt-in. When True, additionally prints the physics-aware
            comparison view (see :func:`physics_aware_rank`) after the
            standard summary. Does not change the standard summary or the
            default pure-accuracy behaviour for callers who leave this off.
        """
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
        if physics_aware:
            self._print_physics_aware_summary()
        return self

    # ── data preparation ──────────────────────────────────────────────────────

    def _prepare_data(self):
        # ── Option A: external pinneapple_data dataset ─────────────────────
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

        # try to get compiled physics losses from pinneapple_physics
        physics_preset = pc.physics_preset or p.physics_preset
        compiled = p.compiled_losses(physics_preset_override=pc.physics_preset, **pc.params)
        if compiled:
            print(f"        [OK] pinneapple_physics compiled losses loaded "
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
        """Load data from pinneapple_data.datasets."""
        print(f"\n[Arena] Dataset: {ds_cfg.dataset_id}")
        X_train, Y_train, X_val, Y_val, field_names = load_pinneapple_dataset(
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
        show_residual = any(t.physics_residual is not None for t in self._train_results)
        print("\n" + "=" * 70)
        print(f"  ARENA RESULTS  >>  {self.cfg.problem.name}")
        print("=" * 70)
        header = f"  {'Model':<22}"
        for f in field_names:
            header += f"  L2-{f:<8}  rel-{f:<6}"
        header += "  Time(s)"
        if show_residual:
            header += "  Physics-Res"
        print(header)
        print("-" * 70)
        for tres, eres in zip(self._train_results, self._eval_results):
            m = eres["metrics"]
            row = f"  {tres.name:<22}"
            for f in field_names:
                row += (f"  {m.get(f'L2_{f}', float('nan')):.3e}    "
                        f"{m.get(f'rel_{f}', float('nan')):.3e}  ")
            row += f"  {tres.train_time:6.1f}"
            if show_residual:
                row += ("  " + (f"{tres.physics_residual:.3e}"
                                 if tres.physics_residual is not None else "n/a"))
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

    # ── physics-aware comparison (opt-in, additive) ──────────────────────────

    def physics_aware_summary(self, n_std: float = 2.0,
                              residual_threshold: Optional[float] = None
                              ) -> Dict[str, Any]:
        """Compute (without printing) the physics-aware comparison view.
        See :func:`physics_aware_rank` for the return shape."""
        return physics_aware_rank(
            self._train_results, self._eval_results, self._data["field_names"],
            n_std=n_std, residual_threshold=residual_threshold,
        )

    def _print_physics_aware_summary(self, n_std: float = 2.0,
                                     residual_threshold: Optional[float] = None):
        result = self.physics_aware_summary(n_std=n_std, residual_threshold=residual_threshold)
        print("\n" + "-" * 70)
        print("  PHYSICS-AWARE VIEW  (accuracy ranking unchanged; outliers flagged)")
        print("-" * 70)
        if not result["physics_residuals"]:
            print("  (no model in this run reports a physics_residual — nothing to flag)")
            print("-" * 70)
            return
        for rank, (name, score) in enumerate(result["ranking"], start=1):
            r = result["physics_residuals"].get(name)
            r_str = f"{r:.3e}" if r is not None else "n/a"
            flag = "  <-- FLAGGED: high physics residual" if name in result["flagged_high_residual"] else ""
            print(f"  #{rank}  {name:<22}  rel-err={score:.3e}  physics-res={r_str}{flag}")
        if result["group_median"] is not None:
            print(f"  (group median={result['group_median']:.3e}, "
                  f"std={result['group_std']:.3e}, threshold=median+{n_std}*std)")
        print("-" * 70)

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
