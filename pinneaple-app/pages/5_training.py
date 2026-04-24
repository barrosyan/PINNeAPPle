"""Training — run models, live loss curves, progress."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from core.session import init as init_session, active_project, update_project, notify

st.set_page_config(page_title="Training · PINNeAPPle", page_icon="⚡", layout="wide")
init_session()

st.title("⚡ Training")

proj = active_project()
if proj is None:
    st.warning("No active project.")
    st.page_link("pages/2_problem_setup.py", label="Go to Problem Setup →")
    st.stop()

prob       = proj.get("problem", {})
model_cfg  = proj.get("model_config", {})
ts_data    = proj.get("timeseries_data")

if not model_cfg:
    st.warning("No model configured. Go to **Models** to select and configure one.")
    st.page_link("pages/4_models.py", label="Go to Models →")
    st.stop()

model_type = model_cfg.get("type", "")
model_name = model_cfg.get("name", model_type)

st.subheader(f"Model: {model_name}")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"Problem: **{prob.get('name','—')}**")
    with st.expander("Model config"):
        for k, v in model_cfg.items():
            if k not in ("type", "name"):
                st.text(f"{k}: {v}")

with col2:
    existing = proj.get("trained_model") or proj.get("simulation")
    if existing:
        st.info(f"Previous run available (type={existing.get('type','?')})")

# ─────────────────────────────────────────────────────────────────────────────
# PINN training
# ─────────────────────────────────────────────────────────────────────────────
if model_type == "pinn_mlp":
    if not prob:
        st.error("No problem defined.")
        st.stop()

    st.divider()
    if st.button("🚀 Start PINN training", type="primary"):
        from core.training_bridge import train_mlp

        progress_bar  = st.progress(0, text="Initialising…")
        loss_chart_ph = st.empty()
        status_ph     = st.empty()

        epochs = int(model_cfg.get("n_epochs", 500))
        history_live = []
        fig_live = go.Figure()
        fig_live.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="Epoch", yaxis_title="Loss", yaxis_type="log",
            margin=dict(l=0, r=0, t=20, b=0),
        )

        def cb(epoch, loss):
            pct = epoch / epochs
            progress_bar.progress(pct, text=f"Epoch {epoch}/{epochs} — loss={loss:.4e}")
            history_live.append({"epoch": epoch, "loss": loss})
            if len(history_live) % 5 == 0:
                eps = [h["epoch"] for h in history_live]
                lss = [h["loss"]  for h in history_live]
                fig_live.data = []
                fig_live.add_trace(go.Scatter(x=eps, y=lss, mode="lines",
                                              line=dict(color="#FF6B35", width=2)))
                loss_chart_ph.plotly_chart(fig_live, use_container_width=True)

        t0 = time.time()
        with st.spinner("Training PINN…"):
            # Merge problem params with model config obstacle
            merged_prob = {**prob, "params": {**prob.get("params", {}), **{
                k: v for k, v in model_cfg.items()
                if k not in ("type", "name", "n_epochs", "lr", "hidden", "n_layers", "n_interior")
            }}}
            result = train_mlp(
                problem    = merged_prob,
                n_epochs   = epochs,
                lr         = float(model_cfg.get("lr", 1e-3)),
                hidden     = int(model_cfg.get("hidden", 64)),
                n_layers   = int(model_cfg.get("n_layers", 4)),
                n_interior = int(model_cfg.get("n_interior", 2000)),
                callback   = cb,
            )

        elapsed = time.time() - t0
        progress_bar.progress(1.0, text=f"Done in {elapsed:.1f}s")

        # Store model (without the torch object — keep history/config)
        result_store = {k: v for k, v in result.items() if k != "model"}
        result_store["_torch_model"] = result["model"]   # keep ref in session
        update_project({"trained_model": result_store})
        notify(f"PINN training complete — final loss={result['final_loss']:.4e}", "success")
        status_ph.success(f"Training complete in {elapsed:.1f}s — "
                          f"final loss = {result['final_loss']:.4e}")

# ─────────────────────────────────────────────────────────────────────────────
# LBM simulation
# ─────────────────────────────────────────────────────────────────────────────
elif model_type == "lbm":
    if not prob:
        st.error("No problem defined.")
        st.stop()

    st.divider()
    if st.button("🚀 Run LBM simulation", type="primary"):
        from core.solver_bridge import run_lbm

        steps      = int(model_cfg.get("steps", 3000))
        save_every = int(model_cfg.get("save_every", 500))

        # Merge grid/obstacle params into problem params
        merged_prob = {**prob}
        merged_prob["params"] = {
            **prob.get("params", {}),
            "nx":     int(model_cfg.get("nx", 160)),
            "ny":     int(model_cfg.get("ny", 64)),
            "Re":     float(model_cfg.get("Re", 200.0)),
            "u_in":   float(model_cfg.get("u_in", 0.05)),
        }
        obs = model_cfg.get("obstacle")
        if obs:
            merged_prob["params"]["obstacle"] = obs

        with st.spinner(f"Running LBM ({steps} steps)…"):
            t0  = time.time()
            sim = run_lbm(merged_prob, steps=steps, save_every=save_every)
            elapsed = time.time() - t0

        update_project({"simulation": sim, "trained_model": None})
        notify(f"LBM simulation complete in {elapsed:.1f}s", "success")

        col_a, col_b = st.columns(2)
        import plotly.express as px
        with col_a:
            vm = sim["vel_mag"]
            fig = px.imshow(vm.T, origin="lower", color_continuous_scale="viridis",
                            title="|velocity| — final step")
            fig.update_layout(template="plotly_dark", height=280,
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            rho = sim["rho"]
            fig2 = px.imshow(rho.T, origin="lower", color_continuous_scale="RdBu_r",
                             title="Density ρ — final step")
            fig2.update_layout(template="plotly_dark", height=280,
                               margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.success(f"LBM done in {elapsed:.1f}s  |  "
                   f"Re={sim['Re']}  nx={sim['nx']}  ny={sim['ny']}")

# ─────────────────────────────────────────────────────────────────────────────
# FDM / FEM
# ─────────────────────────────────────────────────────────────────────────────
elif model_type in ("fdm", "fem"):
    st.divider()
    if st.button(f"🚀 Run {model_type.upper()} solver", type="primary"):
        from core.solver_bridge import run_fdm, run_fem
        import plotly.express as px

        nx = int(model_cfg.get("nx", 64))
        ny = int(model_cfg.get("ny", 64))

        with st.spinner("Solving…"):
            t0 = time.time()
            if model_type == "fdm":
                sim = run_fdm(prob, nx=nx, ny=ny)
            else:
                sim = run_fem(prob, nx=nx, ny=ny)
            elapsed = time.time() - t0

        update_project({"simulation": sim, "trained_model": None})
        notify(f"{model_type.upper()} solve complete in {elapsed:.1f}s", "success")

        field = np.array(sim.get("field", []))
        if field.ndim == 2:
            fig = px.imshow(field.T, origin="lower", color_continuous_scale="RdBu_r",
                            title=sim.get("label", "solution"))
            fig.update_layout(template="plotly_dark", height=350,
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.success(f"Done in {elapsed:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Timeseries models
# ─────────────────────────────────────────────────────────────────────────────
elif model_type in ("tcn", "lstm", "tft"):
    st.divider()
    if ts_data is None:
        st.warning("No timeseries data loaded. Go to **Data & Geometry** to upload CSV.")
        st.page_link("pages/3_data_geometry.py", label="Load data →")
        st.stop()

    if st.button(f"🚀 Train {model_type.upper()}", type="primary"):
        from core.training_bridge import train_timeseries

        y = np.array(ts_data, dtype=np.float32)
        input_len = int(model_cfg.get("input_len", 32))
        horizon   = int(model_cfg.get("horizon",   16))
        epochs    = int(model_cfg.get("epochs",    50))
        lr        = float(model_cfg.get("lr",      1e-3))

        with st.spinner(f"Training {model_type.upper()}…"):
            t0 = time.time()
            result = train_timeseries(y, model_type=model_type,
                                     input_len=input_len, horizon=horizon,
                                     epochs=epochs, lr=lr)
            elapsed = time.time() - t0

        result_store = {k: v for k, v in result.items() if k != "model"}
        result_store["_torch_model"] = result["model"]
        update_project({"trained_model": result_store})
        notify(f"{model_type.upper()} training complete in {elapsed:.1f}s", "success")

        hist = result.get("history", [])
        if hist:
            import plotly.express as px
            fig = px.line(x=[h["epoch"] for h in hist],
                          y=[h["loss"] for h in hist],
                          labels={"x": "Epoch", "y": "MSE"},
                          title="Training loss",
                          template="plotly_dark")
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.success(f"Done in {elapsed:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# FFT
# ─────────────────────────────────────────────────────────────────────────────
elif model_type == "fft":
    st.divider()
    if ts_data is None:
        st.warning("No timeseries data. Go to **Data & Geometry** to upload CSV.")
        st.stop()

    if st.button("🚀 Fit FFT Forecaster", type="primary"):
        try:
            from pinneaple_timeseries.decomposition.fft_forecaster import FFTForecaster
            y = np.array(ts_data, dtype=np.float64)
            n_harmonics = int(model_cfg.get("n_harmonics", 5))
            detrend     = bool(model_cfg.get("detrend", True))
            fft_model   = FFTForecaster(n_harmonics=n_harmonics, detrend=detrend)
            fft_model.fit(y)
            update_project({"trained_model": {
                "type": "fft",
                "n_harmonics": n_harmonics,
                "detrend": detrend,
                "_fft_model": fft_model,
            }})
            notify("FFT Forecaster fitted.", "success")
            st.success("FFT model fitted — go to **Inference** to forecast.")
        except ImportError as e:
            st.error(f"pinneaple_timeseries not available: {e}")

else:
    st.info(f"Model type `{model_type}` — no training UI implemented for this type yet.")

# ── Show existing results ─────────────────────────────────────────────────────
st.divider()
existing_model = proj.get("trained_model")
existing_sim   = proj.get("simulation")
if existing_model or existing_sim:
    st.subheader("Current results")
    result_obj = existing_model or existing_sim
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Type", result_obj.get("type", "—"))
    with col2:
        if "final_loss" in result_obj:
            st.metric("Final loss", f"{result_obj['final_loss']:.4e}")
    st.page_link("pages/7_visualization.py", label="Visualize results →")
