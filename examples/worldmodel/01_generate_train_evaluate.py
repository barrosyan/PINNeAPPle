"""Physics World Model — end-to-end example.

Generates a small dataset, trains a specialist FNO model, and evaluates it
with rollout predictions, error plots, and benchmark metrics.

Designed to run in ~2-5 minutes on CPU so you can quickly verify the idea.

Run
---
    cd <repo-root>
    python examples/worldmodel/01_generate_train_evaluate.py

    # Even faster (tiny sizes):
    python examples/worldmodel/01_generate_train_evaluate.py --fast

    # GPU:
    python examples/worldmodel/01_generate_train_evaluate.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── make sure repo root is on sys.path when running as a script ───────────────
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── config ────────────────────────────────────────────────────────────────────

def get_cfg(fast: bool = False, device: str = "cpu") -> dict:
    """Return experiment configuration."""
    if fast:
        return dict(
            scenarios   = ["burgers_1d"],
            sources     = ["solver"],
            n_train     = 20,
            n_test      = 5,
            epochs      = 5,
            n_modes     = 4,
            width       = 16,
            depth       = 2,
            batch_size  = 8,
            lr          = 1e-3,
            rollout_eval= 4,
            device      = device,
            output_dir  = "./outputs/worldmodel_fast",
        )
    return dict(
        scenarios   = ["burgers_1d", "heat_2d"],
        sources     = ["solver"],
        n_train     = 80,
        n_test      = 20,
        epochs      = 30,
        n_modes     = 8,
        width       = 32,
        depth       = 3,
        batch_size  = 16,
        lr          = 1e-3,
        rollout_eval= 8,
        device      = device,
        output_dir  = "./outputs/worldmodel_eval",
    )


# ── step 1: dataset generation ────────────────────────────────────────────────

def step_generate(cfg: dict):
    """Generate train + test trajectory datasets."""
    print("\n" + "=" * 60)
    print("  STEP 1 — Dataset generation")
    print("=" * 60)

    from pinneaple_worldmodel import generate_datasets
    from pinneaple_worldmodel.scenario import BUILTIN_SCENARIOS
    from pinneaple_worldmodel.simulator import PhysicsSimulator
    from pinneaple_worldmodel.dataset import WorldModelDataset

    t0 = time.perf_counter()

    # Training catalog (multi-scenario)
    catalog = generate_datasets(
        scenarios  = cfg["scenarios"],
        sources    = cfg["sources"],
        n_samples  = cfg["n_train"],
        output_dir = cfg["output_dir"] + "/data",
        device     = cfg["device"],
        validate   = False,
        verbose    = True,
    )

    # Held-out test trajectories (generated fresh, same distribution)
    test_sets: dict[str, WorldModelDataset] = {}
    for scenario_name in cfg["scenarios"]:
        scenario = BUILTIN_SCENARIOS[scenario_name]
        sim = PhysicsSimulator(scenario, device=cfg["device"])
        trajs = sim.generate_batch(cfg["n_test"], base_seed=9999)
        test_sets[scenario_name] = WorldModelDataset(trajs)
        print(f"  Test set '{scenario_name}': {len(test_sets[scenario_name])} samples")

    elapsed = time.perf_counter() - t0
    print(f"\n  Generation complete in {elapsed:.1f} s")
    return catalog, test_sets


# ── step 2: training ──────────────────────────────────────────────────────────

def step_train(catalog, cfg: dict):
    """Train specialist models (one per scenario)."""
    print("\n" + "=" * 60)
    print("  STEP 2 — Specialist training")
    print("=" * 60)

    from pinneaple_worldmodel import train_specialist

    t0 = time.perf_counter()
    zoo = train_specialist(
        catalog      = catalog,
        output_dir   = Path(cfg["output_dir"]) / "checkpoints",
        epochs       = cfg["epochs"],
        batch_size   = cfg["batch_size"],
        lr           = cfg["lr"],
        device       = cfg["device"],
        patience     = max(cfg["epochs"] // 3, 5),
        n_modes      = cfg["n_modes"],
        width        = cfg["width"],
        depth        = cfg["depth"],
        rollout_steps= 1,
        run_benchmark= False,   # we run our own detailed evaluation below
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Training complete in {elapsed:.1f} s")
    print(f"  Zoo: {zoo.list_names()}")
    return zoo


# ── step 3: evaluation ────────────────────────────────────────────────────────

def step_evaluate(zoo, catalog, test_sets: dict, cfg: dict) -> dict:
    """Evaluate every specialist with 1-step and rollout MSE."""
    print("\n" + "=" * 60)
    print("  STEP 3 — Evaluation")
    print("=" * 60)

    from pinneaple_worldmodel.model_zoo import ModelZoo

    results: dict[str, dict] = {}

    for scenario_name in cfg["scenarios"]:
        try:
            zoo_entry = zoo.get_entry(scenario_name)
        except KeyError:
            print(f"  [{scenario_name}] not found in zoo — skipping")
            continue

        model   = zoo_entry.model.to(cfg["device"])
        test_ds = test_sets.get(scenario_name)
        if test_ds is None or len(test_ds) == 0:
            print(f"  [{scenario_name}] no test data — skipping")
            continue

        model.eval()

        # ── 1-step MSE ──
        one_step_mse = _eval_one_step(model, test_ds, cfg["device"])

        # ── rollout MSE (multi-step) ──
        rollout_mse  = _eval_rollout(
            model, test_ds, cfg["device"], cfg["rollout_eval"]
        )

        # ── relative L2 ──
        rel_l2 = _eval_relative_l2(model, test_ds, cfg["device"])

        n_params = zoo_entry.n_params
        results[scenario_name] = {
            "one_step_mse"  : one_step_mse,
            "rollout_mse"   : rollout_mse,
            "rel_l2"        : rel_l2,
            "n_params"      : n_params,
        }

        print(f"\n  [{scenario_name}]")
        print(f"    Model params      : {n_params:,}")
        print(f"    1-step MSE        : {one_step_mse:.6f}")
        print(f"    Rollout MSE ({cfg['rollout_eval']} steps): {rollout_mse:.6f}")
        print(f"    Relative L2       : {rel_l2:.4f}  ({rel_l2*100:.2f} %)")
        print(f"    Quality           : {_quality_label(rel_l2)}")

    return results


def _context(n: int, context_dim: int, device: str) -> torch.Tensor:
    return torch.zeros(n, context_dim, device=device)


def _eval_one_step(model, dataset, device: str) -> float:
    mse_total, count = 0.0, 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    ctx_dim = model.config.context_dim if hasattr(model, "config") else 0
    with torch.no_grad():
        for batch in loader:
            st   = batch["state_t"].to(device)
            stp1 = batch["state_tp1"].to(device)
            ctx  = _context(len(st), ctx_dim, device) if ctx_dim > 0 else None
            pred = model(st, ctx)
            mse_total += torch.mean((pred - stp1) ** 2).item() * len(st)
            count += len(st)
    return mse_total / max(count, 1)


def _eval_rollout(model, dataset, device: str, n_steps: int) -> float:
    """Multi-step rollout: feed predictions back as inputs."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    ctx_dim = model.config.context_dim if hasattr(model, "config") else 0
    batch = next(iter(loader))
    state_0    = batch["state_t"].to(device)
    state_true = batch["state_tp1"].to(device)

    with torch.no_grad():
        ctx      = _context(len(state_0), ctx_dim, device) if ctx_dim > 0 else None
        rollout  = model.rollout(state_0, ctx, n_steps=n_steps)
        pred_last = rollout[:, -1]
        mse = torch.mean((pred_last - state_true) ** 2).item()
    return mse


def _eval_relative_l2(model, dataset, device: str) -> float:
    num, den = 0.0, 0.0
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    ctx_dim = model.config.context_dim if hasattr(model, "config") else 0
    with torch.no_grad():
        for batch in loader:
            st   = batch["state_t"].to(device)
            stp1 = batch["state_tp1"].to(device)
            ctx  = _context(len(st), ctx_dim, device) if ctx_dim > 0 else None
            pred = model(st, ctx)
            num += torch.sum((pred - stp1) ** 2).item()
            den += torch.sum(stp1 ** 2).item()
    return (num / max(den, 1e-12)) ** 0.5


def _quality_label(rel_l2: float) -> str:
    if rel_l2 < 0.01:  return "EXCELLENT (< 1 %)"
    if rel_l2 < 0.05:  return "GOOD      (< 5 %)"
    if rel_l2 < 0.15:  return "FAIR      (< 15 %)"
    if rel_l2 < 0.30:  return "POOR      (< 30 %)"
    return              "FAILING   (> 30 %)"


# ── step 4: visualisation ─────────────────────────────────────────────────────

def step_visualise(zoo, test_sets: dict, cfg: dict) -> None:
    """Plot predicted vs true fields and rollout error curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [viz] matplotlib not available — skipping plots")
        return

    out = Path(cfg["output_dir"]) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    ctx_dim = 64  # default

    for scenario_name in cfg["scenarios"]:
        try:
            zoo_entry = zoo.get_entry(scenario_name)
        except KeyError:
            continue

        model   = zoo_entry.model.to(cfg["device"])
        test_ds = test_sets.get(scenario_name)
        if test_ds is None or len(test_ds) == 0:
            continue

        model.eval()
        ctx_dim = model.config.context_dim if hasattr(model, "config") else 64

        # Pull one sample
        sample  = test_ds[0]
        state_0 = sample["state_t"].unsqueeze(0).to(cfg["device"])   # (1, C, *grid)
        state_1 = sample["state_tp1"].unsqueeze(0).to(cfg["device"])

        with torch.no_grad():
            ctx  = _context(1, ctx_dim, cfg["device"])
            pred = model(state_0, ctx)
            # rollout
            roll = model.rollout(state_0, ctx, n_steps=cfg["rollout_eval"])

        state_0_np = state_0[0, 0].cpu().numpy()   # first field
        state_1_np = state_1[0, 0].cpu().numpy()
        pred_np    = pred[0, 0].cpu().numpy()
        roll_np    = roll[0, :, 0].cpu().numpy()    # (n_steps, *grid)

        spatial_dim = state_0_np.ndim

        fig, axes = plt.subplots(
            2, 3, figsize=(14, 8),
            gridspec_kw={"hspace": 0.4, "wspace": 0.35}
        )
        fig.suptitle(
            f"World Model — {scenario_name}\n"
            f"n_modes={cfg['n_modes']}  width={cfg['width']}  "
            f"depth={cfg['depth']}  epochs={cfg['epochs']}",
            fontsize=12,
        )

        if spatial_dim == 1:
            x = np.linspace(0, 1, len(state_0_np))

            axes[0, 0].plot(x, state_0_np, label="t", color="steelblue")
            axes[0, 0].set_title("Input state  (t)"); axes[0, 0].set_xlabel("x")

            axes[0, 1].plot(x, state_1_np, label="true", color="green")
            axes[0, 1].plot(x, pred_np,   "--", label="pred", color="orange")
            axes[0, 1].legend(); axes[0, 1].set_title("True vs Predicted  (t+1)")

            err = np.abs(state_1_np - pred_np)
            axes[0, 2].plot(x, err, color="red")
            axes[0, 2].set_title(f"Pointwise error  (t+1)\nmax={err.max():.4f}")

            # Rollout: heatmap (time × space)
            im = axes[1, 0].imshow(
                roll_np, aspect="auto",
                extent=[0, 1, cfg["rollout_eval"], 0],
                cmap="RdBu_r",
            )
            plt.colorbar(im, ax=axes[1, 0])
            axes[1, 0].set_title("Rollout prediction  (time × x)")
            axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("step")

            # Rollout error per step (vs constant reference = state_1)
            step_mse = [
                float(np.mean((roll_np[i] - state_1_np) ** 2))
                for i in range(len(roll_np))
            ]
            axes[1, 1].semilogy(range(1, len(step_mse) + 1), step_mse, "o-", color="purple")
            axes[1, 1].set_title("Rollout MSE per step")
            axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("MSE (log)")

            # Final rollout step vs input
            axes[1, 2].plot(x, state_0_np,       label="t",      color="steelblue")
            axes[1, 2].plot(x, roll_np[-1], "--", label=f"t+{len(roll_np)}", color="darkorange")
            axes[1, 2].legend(); axes[1, 2].set_title(f"Input vs Final rollout step")

        else:  # 2D
            def _ishow(ax, data, title, cmap="RdBu_r"):
                im = ax.imshow(data, cmap=cmap, origin="lower")
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(title); ax.axis("off")

            _ishow(axes[0, 0], state_0_np, "Input  (t)")
            _ishow(axes[0, 1], state_1_np, "True   (t+1)", cmap="viridis")
            _ishow(axes[0, 2], pred_np,    "Predicted (t+1)", cmap="viridis")
            _ishow(axes[1, 0], np.abs(state_1_np - pred_np), "Error  (t+1)", cmap="hot")

            step_mse = [
                float(np.mean((roll_np[i] - state_1_np) ** 2))
                for i in range(len(roll_np))
            ]
            axes[1, 1].semilogy(range(1, len(step_mse) + 1), step_mse, "o-", color="purple")
            axes[1, 1].set_title("Rollout MSE per step")
            axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("MSE (log)")

            _ishow(axes[1, 2], roll_np[-1], f"Rollout step {len(roll_np)}", cmap="viridis")

        plot_path = out / f"{scenario_name}_eval.png"
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  [viz] saved -> {plot_path}")

    print(f"\n  Plots written to {out}/")


# ── step 5: benchmark ─────────────────────────────────────────────────────────

def step_benchmark(zoo, catalog, cfg: dict) -> None:
    """Run the built-in PhysicsBenchmark on available tasks."""
    print("\n" + "=" * 60)
    print("  STEP 5 — Standard Benchmark")
    print("=" * 60)

    try:
        from pinneaple_worldmodel.benchmark import PhysicsBenchmark, BUILTIN_TASKS

        # Keep only tasks whose scenario we trained on
        tasks = {
            k: v for k, v in BUILTIN_TASKS.items()
            if v.scenario_name in cfg["scenarios"]
        }
        if not tasks:
            print("  No matching benchmark tasks — skipping")
            return

        bench = PhysicsBenchmark(device=cfg["device"])

        def _ctx_fn(scenario_name: str, n: int) -> torch.Tensor:
            return torch.zeros(n, 64, device=cfg["device"])

        # Use the first specialist as the model to evaluate
        first_name = cfg["scenarios"][0]
        try:
            first_model = zoo.get(first_name)
        except KeyError:
            print("  No model found for first scenario — skipping")
            return

        results = bench.run(first_model, tasks, context_fn=_ctx_fn)
        bench.print_report(results)

        print("\n  Overall scores:")
        for name, res in sorted(results.items(), key=lambda x: -x[1].overall_score()):
            score = res.overall_score()
            bar   = "#" * int(score * 20)
            print(f"    {name:<30} {bar:<20} {score:.3f}")

    except Exception as exc:
        print(f"  Benchmark error (non-fatal): {exc}")


# ── final report ──────────────────────────────────────────────────────────────

def print_summary(results: dict, cfg: dict, total_time: float) -> None:
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Scenarios   : {cfg['scenarios']}")
    print(f"  Train samples: {cfg['n_train']} × {len(cfg['scenarios'])}")
    print(f"  Epochs       : {cfg['epochs']}")
    print(f"  Architecture : FNO  modes={cfg['n_modes']}  w={cfg['width']}  d={cfg['depth']}")
    print(f"  Device       : {cfg['device']}")
    print(f"  Total time   : {total_time:.1f} s")
    print()

    all_good = True
    for scenario, m in results.items():
        rl2 = m["rel_l2"]
        label = _quality_label(rl2)
        print(f"  {scenario:<25}  rel-L2={rl2:.4f}  {label}")
        if rl2 > 0.30:
            all_good = False

    print()
    if all_good:
        print("  VERDICT: Model is learning physics structure successfully.")
        print("           Try increasing n_samples + epochs to improve further.")
    else:
        print("  VERDICT: Model needs more training. Suggestions:")
        print("    - Increase --n-samples (try 200+) and --epochs (try 100+)")
        print("    - Increase --width (64) and --n-modes (16)")
        print("    - Use --device cuda for faster iteration")
    print("=" * 60 + "\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="World Model quick evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fast",       action="store_true", help="Tiny sizes, ~30 s")
    p.add_argument("--device",     default="cpu",       help="cpu / cuda")
    p.add_argument("--scenarios",  nargs="+",           default=None)
    p.add_argument("--n-samples",  type=int,            default=None)
    p.add_argument("--epochs",     type=int,            default=None)
    p.add_argument("--n-modes",    type=int,            default=None)
    p.add_argument("--width",      type=int,            default=None)
    p.add_argument("--depth",      type=int,            default=None)
    p.add_argument("--output",     default=None)
    p.add_argument("--no-plots",   action="store_true")
    p.add_argument("--no-bench",   action="store_true")
    args = p.parse_args()

    cfg = get_cfg(fast=args.fast, device=args.device)

    # Override with any explicit CLI args
    if args.scenarios:  cfg["scenarios"]  = args.scenarios
    if args.n_samples:  cfg["n_train"]    = args.n_samples
    if args.epochs:     cfg["epochs"]     = args.epochs
    if args.n_modes:    cfg["n_modes"]    = args.n_modes
    if args.width:      cfg["width"]      = args.width
    if args.depth:      cfg["depth"]      = args.depth
    if args.output:     cfg["output_dir"] = args.output

    print("\n" + "=" * 60)
    print("  Physics World Model — Quick Evaluation")
    print("=" * 60)
    print(f"  Mode       : {'FAST' if args.fast else 'NORMAL'}")
    print(f"  Scenarios  : {cfg['scenarios']}")
    print(f"  Train/test : {cfg['n_train']} / {cfg['n_test']}")
    print(f"  Epochs     : {cfg['epochs']}")
    print(f"  FNO arch   : modes={cfg['n_modes']}  w={cfg['width']}  d={cfg['depth']}")
    print(f"  Device     : {cfg['device']}")
    print(f"  Output     : {cfg['output_dir']}")

    t_start = time.perf_counter()

    # ── pipeline ──
    catalog,  test_sets = step_generate(cfg)
    zoo                 = step_train(catalog, cfg)
    results             = step_evaluate(zoo, catalog, test_sets, cfg)

    if not args.no_plots:
        step_visualise(zoo, test_sets, cfg)

    if not args.no_bench:
        step_benchmark(zoo, catalog, cfg)

    total = time.perf_counter() - t_start
    print_summary(results, cfg, total)


if __name__ == "__main__":
    main()
