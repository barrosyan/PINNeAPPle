"""Benchmark report generator — produces plots and tables from ExperimentResult."""
from __future__ import annotations
import base64
import io
from typing import Any, Dict, List, Optional


def build_benchmark_payload(result) -> Dict[str, Any]:
    """Convert an ExperimentResult into a JSON-serialisable benchmark payload.

    Returns
    -------
    dict with keys:
        leaderboard  : list of dicts (sorted by best metric)
        charts       : dict of base64-encoded PNG images
        summary      : plain-text summary
    """
    leaderboard = result.leaderboard()
    charts = _build_charts(result)
    summary = _build_summary(result, leaderboard)
    errors = {
        name: r.error
        for name, r in result.model_results.items()
        if r.error
    }

    return {
        "experiment_id": result.experiment_id,
        "problem_name":  result.config.problem_name,
        "completed_at":  result.completed_at,
        "leaderboard":   leaderboard,
        "charts":        charts,
        "summary":       summary,
        "errors":        errors,
    }


def _build_charts(result) -> Dict[str, str]:
    """Produce all charts as base64-encoded PNG strings."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    charts = {}

    # 1. Loss curves for each model
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        labels = ("Total", "Physics", "BC")
        attr_keys = ("loss_history", "physics_loss_history", "bc_loss_history")
        for ax, label, key in zip(axes, labels, attr_keys):
            for name, r in result.model_results.items():
                if r.error:
                    continue
                hist = getattr(r, key, [])
                if hist:
                    ax.semilogy(hist, label=name, linewidth=1.5)
            ax.set_title(f"{label} Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Training Curves — {result.config.problem_name}", fontsize=11)
        plt.tight_layout()
        charts["loss_curves"] = _fig_to_b64(fig)
        plt.close(fig)
    except Exception:
        pass

    # 2. Metric bar chart (l2_relative)
    try:
        metric_key = "l2_relative"
        valid = {
            name: r.metrics.get(metric_key, float("nan"))
            for name, r in result.model_results.items()
            if not r.error and metric_key in r.metrics
        }
        if valid:
            import numpy as np
            names = list(valid.keys())
            vals  = [valid[n] for n in names]
            valid_vals = [v for v in vals if not (isinstance(v, float) and v != v)]
            if valid_vals:
                fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
                colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(names)))
                bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.5)
                ax.set_ylabel("Relative L2 Error")
                ax.set_title(f"Model Comparison — {metric_key}")
                ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
                ax.grid(True, axis="y", alpha=0.3)
                for bar, val in zip(bars, vals):
                    if val == val:  # not nan
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() * 1.02,
                                f"{val:.3e}", ha="center", va="bottom", fontsize=7)
                plt.tight_layout()
                charts["metric_comparison"] = _fig_to_b64(fig)
                plt.close(fig)
    except Exception:
        pass

    # 3. Training time vs accuracy scatter
    try:
        times = {}
        errors = {}
        for name, r in result.model_results.items():
            if r.error:
                continue
            t = r.train_time_s
            e = r.metrics.get("l2_relative", float("nan"))
            if t and e == e:
                times[name] = t
                errors[name] = e
        if len(times) >= 2:
            fig, ax = plt.subplots(figsize=(6, 5))
            for name in times:
                ax.scatter(times[name], errors[name], s=80, zorder=3)
                ax.annotate(name, (times[name], errors[name]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)
            ax.set_xlabel("Training Time (s)")
            ax.set_ylabel("Relative L2 Error")
            ax.set_title("Accuracy vs. Training Cost")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts["time_vs_accuracy"] = _fig_to_b64(fig)
            plt.close(fig)
    except Exception:
        pass

    # 4. Parameter count bar
    try:
        params = {name: r.n_params for name, r in result.model_results.items()
                  if not r.error and r.n_params > 0}
        if params:
            fig, ax = plt.subplots(figsize=(max(5, len(params) * 0.8), 4))
            ax.bar(params.keys(), params.values(), color="steelblue", edgecolor="white")
            ax.set_ylabel("# Parameters")
            ax.set_title("Model Size")
            ax.set_xticklabels(params.keys(), rotation=30, ha="right", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            charts["parameter_count"] = _fig_to_b64(fig)
            plt.close(fig)
    except Exception:
        pass

    return charts


def _build_summary(result, leaderboard: list) -> str:
    lines = [
        f"Experiment: {result.experiment_id}",
        f"Problem: {result.config.problem_name}",
        f"Models tested: {len(result.model_results)}",
        f"Epochs: {result.config.epochs}  |  LR: {result.config.lr}",
        "",
        "=== Leaderboard ===",
    ]
    for i, row in enumerate(leaderboard[:5], 1):
        l2 = row.get("l2_relative", "N/A")
        l2_str = f"{l2:.4e}" if isinstance(l2, float) and l2 == l2 else str(l2)
        lines.append(
            f"  {i}. {row['model']:<25s}  L2={l2_str}  "
            f"t={row.get('train_time_s', '?')}s  "
            f"params={row.get('n_params', '?')}"
        )

    failed = [n for n, r in result.model_results.items() if r.error]
    if failed:
        lines.append(f"\nFailed models: {', '.join(failed)}")

    return "\n".join(lines)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
