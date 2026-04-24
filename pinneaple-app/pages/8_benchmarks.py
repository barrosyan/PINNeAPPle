"""Benchmarks — pre-defined problems, multi-model comparison."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from core.session import init as init_session

st.set_page_config(page_title="Benchmarks · PINNeAPPle", page_icon="🏆", layout="wide")
init_session()

st.title("🏆 Benchmarks")
st.markdown(
    "Run standardised benchmark problems and compare solvers. "
    "Results are computed fresh — no active project needed."
)

BENCHMARK_DEFS = {
    "Poisson 2D (FDM vs analytical)": {
        "key":  "poisson_fdm",
        "desc": "Unit-square Poisson equation. FDM 5-point stencil vs analytical solution "
                "u=sin(πx)sin(πy). Reports L2 and L∞ errors.",
        "category": "Classical PDE",
    },
    "Burgers 1D (PINN)": {
        "key":  "burgers_pinn",
        "desc": "Viscous Burgers equation, ν=0.01, PINN trained for 500 epochs. "
                "Reports final PDE residual loss.",
        "category": "PINN",
    },
    "Cylinder Flow LBM (Re=200)": {
        "key":  "cylinder_lbm",
        "desc": "Flow past a circular cylinder at Re=200. "
                "Reports max velocity, density range, and vortex cell count.",
        "category": "CFD",
    },
    "Lid-Driven Cavity LBM (Re=100)": {
        "key":  "cavity_lbm",
        "desc": "Classic lid-driven cavity at Re=100 on a 64×64 grid.",
        "category": "CFD",
    },
}

# ── Benchmark selection ───────────────────────────────────────────────────────
cat_filter = st.radio(
    "Filter",
    ["All", "Classical PDE", "PINN", "CFD"],
    horizontal=True,
)

filtered = {k: v for k, v in BENCHMARK_DEFS.items()
            if cat_filter == "All" or v["category"] == cat_filter}

selected_bench = st.selectbox("Select benchmark", list(filtered.keys()))
bench = filtered[selected_bench]

with st.expander("Description"):
    st.markdown(bench["desc"])

# ── Hyperparameter overrides ──────────────────────────────────────────────────
with st.expander("Hyperparameters"):
    if bench["key"] == "poisson_fdm":
        nx_b = st.slider("Grid size", 16, 256, 64)
        ny_b = nx_b
        iters_b = st.slider("Gauss-Seidel iterations", 100, 10000, 5000, step=100)
    elif bench["key"] == "burgers_pinn":
        epochs_b  = st.slider("Epochs",      50, 2000, 500)
        hidden_b  = st.slider("Hidden size", 16, 256, 64)
        n_layers_b = st.slider("Layers",      2, 10, 4)
        n_col_b   = st.slider("Collocation",  200, 5000, 2000)
        lr_b      = st.number_input("Learning rate", value=1e-3, format="%.5f")
    elif bench["key"] in ("cylinder_lbm", "cavity_lbm"):
        steps_b     = st.slider("Steps",     500, 10000, 4000, step=500)
        save_every_b = st.slider("Save every", 50,  2000, 1000, step=50)
        if bench["key"] == "cylinder_lbm":
            Re_b  = st.slider("Re", 50, 500, 200)
        else:
            Re_b  = st.slider("Re", 50, 1000, 100)

# ── Run benchmark ─────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    run_clicked = st.button("▶ Run Benchmark", type="primary", use_container_width=True)

if run_clicked:
    bench_key = bench["key"]

    # ── Poisson FDM ───────────────────────────────────────────────────────────
    if bench_key == "poisson_fdm":
        with st.spinner(f"Running Poisson FDM ({nx_b}×{ny_b}, {iters_b} iters)…"):
            import math
            t0 = time.time()
            x = np.linspace(0, 1, nx_b); y = np.linspace(0, 1, ny_b)
            X, Y = np.meshgrid(x, y, indexing="ij")
            dx = 1.0 / (nx_b - 1); dy = 1.0 / (ny_b - 1)
            f = 2 * math.pi**2 * np.sin(math.pi * X) * np.sin(math.pi * Y)
            u_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
            u = np.zeros_like(f)
            for _ in range(iters_b):
                u[1:-1, 1:-1] = 0.25 * (
                    u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                    + dx**2 * f[1:-1, 1:-1]
                )
            elapsed = time.time() - t0

        err = u - u_exact
        l2  = float(np.sqrt(np.mean(err**2)))
        linf = float(np.max(np.abs(err)))
        st.success(f"Done in {elapsed:.2f}s")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("L2 error",  f"{l2:.4e}")
        with col2: st.metric("L∞ error",  f"{linf:.4e}")
        with col3: st.metric("Grid",       f"{nx_b}×{ny_b}")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fig = px.imshow(u.T, origin="lower", color_continuous_scale="RdBu_r",
                            title="FDM solution", x=x, y=y)
            fig.update_layout(template="plotly_dark", height=280,
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.imshow(u_exact.T, origin="lower", color_continuous_scale="RdBu_r",
                            title="Analytical solution", x=x, y=y)
            fig.update_layout(template="plotly_dark", height=280,
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col_c:
            vmax_err = max(abs(err.min()), abs(err.max()))
            fig = px.imshow(err.T, origin="lower", color_continuous_scale="RdBu_r",
                            zmin=-vmax_err, zmax=vmax_err,
                            title="Error (FDM − exact)", x=x, y=y)
            fig.update_layout(template="plotly_dark", height=280,
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        df_res = pd.DataFrame([{
            "benchmark": selected_bench,
            "nx": nx_b, "ny": ny_b, "iters": iters_b,
            "L2_error": l2, "Linf_error": linf,
            "time_s": elapsed,
        }])
        st.session_state.setdefault("bench_results", [])
        st.session_state["bench_results"].append(df_res)

    # ── Burgers PINN ──────────────────────────────────────────────────────────
    elif bench_key == "burgers_pinn":
        from core.problem_library import get_problem
        from core.training_bridge import train_mlp

        prob = {**get_problem("burgers_1d"), "_preset_key": "burgers_1d"}

        progress_bar = st.progress(0, text="Initialising PINN…")
        history_live = []

        def cb(epoch, loss):
            progress_bar.progress(epoch / epochs_b,
                                  text=f"Epoch {epoch}/{epochs_b} — loss={loss:.4e}")
            history_live.append({"epoch": epoch, "loss": loss})

        t0 = time.time()
        with st.spinner("Training PINN on Burgers…"):
            res = train_mlp(
                problem=prob, n_epochs=epochs_b, lr=float(lr_b),
                hidden=hidden_b, n_layers=n_layers_b, n_interior=n_col_b,
                callback=cb,
            )
        elapsed = time.time() - t0
        progress_bar.progress(1.0, text=f"Done in {elapsed:.1f}s")
        st.success(f"PINN training complete in {elapsed:.1f}s")

        col1, col2 = st.columns(2)
        with col1: st.metric("Final loss", f"{res['final_loss']:.4e}")
        with col2: st.metric("Epochs", epochs_b)

        if history_live:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h["epoch"] for h in history_live],
                y=[h["loss"] for h in history_live],
                mode="lines", line=dict(color="#FF6B35", width=2),
            ))
            fig.update_layout(template="plotly_dark", height=300,
                              xaxis_title="Epoch", yaxis_title="Loss", yaxis_type="log",
                              title="Burgers PINN training loss",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        df_res = pd.DataFrame([{
            "benchmark": selected_bench,
            "epochs": epochs_b, "hidden": hidden_b, "n_layers": n_layers_b,
            "final_loss": res["final_loss"], "time_s": elapsed,
        }])
        st.session_state.setdefault("bench_results", [])
        st.session_state["bench_results"].append(df_res)

    # ── LBM benchmarks ────────────────────────────────────────────────────────
    elif bench_key in ("cylinder_lbm", "cavity_lbm"):
        from core.problem_library import get_problem
        from core.solver_bridge import run_lbm

        if bench_key == "cylinder_lbm":
            base_prob = get_problem("cylinder_flow")
            prob = {**base_prob}
            prob["params"] = {**base_prob.get("params", {}),
                              "Re": Re_b, "nx": 160, "ny": 64,
                              "u_in": 0.05,
                              "obstacle": {"type": "cylinder",
                                           "cx": 40, "cy": 32, "r": 8}}
        else:
            base_prob = get_problem("lid_driven_cavity")
            prob = {**base_prob}
            prob["params"] = {**base_prob.get("params", {}),
                              "Re": Re_b, "nx": 64, "ny": 64, "u_in": 0.05}

        with st.spinner(f"Running LBM ({steps_b} steps)…"):
            t0  = time.time()
            sim = run_lbm(prob, steps=steps_b, save_every=save_every_b)
            elapsed = time.time() - t0

        vm   = sim["vel_mag"]
        rho  = sim["rho"]
        obs  = sim.get("obstacle")

        if obs is not None:
            obs_np = np.array(obs, dtype=bool)
            vm_m   = np.where(obs_np, np.nan, vm)
            rho_m  = np.where(obs_np, np.nan, rho)
        else:
            vm_m = vm; rho_m = rho

        st.success(f"Done in {elapsed:.1f}s")

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("max |v|",  f"{np.nanmax(vm_m):.4f}")
        with col2: st.metric("min ρ",    f"{np.nanmin(rho_m):.4f}")
        with col3: st.metric("max ρ",    f"{np.nanmax(rho_m):.4f}")
        with col4: st.metric("Steps",    steps_b)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.imshow(vm_m.T, origin="lower", color_continuous_scale="viridis",
                            title="|velocity| — final step")
            fig.update_layout(template="plotly_dark", height=300,
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            ux = np.array(sim["ux"]); uy = np.array(sim["uy"])
            try:
                from pinneaple_viz.vortex import compute_q_criterion_2d
                Q = compute_q_criterion_2d(ux, uy, 1.0/sim["nx"], 1.0/sim["ny"])
                if obs is not None:
                    Q = np.where(obs_np, np.nan, Q)
                fig2 = px.imshow(Q.T, origin="lower", color_continuous_scale="hot",
                                 zmin=0, zmax=float(np.nanpercentile(Q, 99)),
                                 title="Q-criterion")
                fig2.update_layout(template="plotly_dark", height=300,
                                   margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig2, use_container_width=True)
                n_vortex = int(np.sum(Q > 0))
                st.caption(f"Q>0 cells (vortex cores): {n_vortex}")
            except Exception as e:
                fig2 = px.imshow(rho_m.T, origin="lower", color_continuous_scale="RdBu_r",
                                 title="Density ρ")
                fig2.update_layout(template="plotly_dark", height=300,
                                   margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig2, use_container_width=True)

        df_res = pd.DataFrame([{
            "benchmark": selected_bench,
            "Re": Re_b, "steps": steps_b,
            "max_vel": float(np.nanmax(vm_m)),
            "rho_min": float(np.nanmin(rho_m)),
            "rho_max": float(np.nanmax(rho_m)),
            "time_s": elapsed,
        }])
        st.session_state.setdefault("bench_results", [])
        st.session_state["bench_results"].append(df_res)

# ── Historical results ─────────────────────────────────────────────────────────
if st.session_state.get("bench_results"):
    st.divider()
    st.subheader("Run history")
    all_results = pd.concat(st.session_state["bench_results"], ignore_index=True)
    st.dataframe(all_results, use_container_width=True)
    st.download_button(
        "Download results CSV",
        all_results.to_csv(index=False).encode(),
        "benchmark_results.csv", "text/csv"
    )
    if st.button("Clear history"):
        st.session_state["bench_results"] = []
        st.rerun()
