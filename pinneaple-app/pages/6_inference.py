"""Inference — run trained model on new inputs, generate predictions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from core.session import init as init_session, active_project, update_project, notify

st.set_page_config(page_title="Inference · PINNeAPPle", page_icon="🎯", layout="wide")
init_session()

st.title("🎯 Inference")

proj = active_project()
if proj is None:
    st.warning("No active project.")
    st.page_link("pages/2_problem_setup.py", label="Go to Problem Setup →")
    st.stop()

model_info = proj.get("trained_model")
sim_info   = proj.get("simulation")
prob       = proj.get("problem", {})

if model_info is None and sim_info is None:
    st.warning("No trained model or simulation available. Run training first.")
    st.page_link("pages/5_training.py", label="Go to Training →")
    st.stop()

result = model_info or sim_info
model_type = result.get("type", "")

st.subheader(f"Model: {model_type}")

# ─────────────────────────────────────────────────────────────────────────────
# PINN inference
# ─────────────────────────────────────────────────────────────────────────────
if model_type == "pinn_mlp":
    torch_model = result.get("_torch_model")
    if torch_model is None:
        st.error("PyTorch model not in session. Retrain the model.")
        st.stop()

    st.markdown("Evaluate the PINN on a grid over the problem domain.")

    domain = prob.get("domain", {"x": (0, 1), "y": (0, 1)})
    keys   = list(domain.keys())

    col1, col2 = st.columns(2)
    with col1:
        nx_inf = st.slider("Grid resolution (x)", 20, 200, 64)
    with col2:
        ny_inf = st.slider("Grid resolution (y/t)", 20, 200, 64)

    if st.button("Run inference", type="primary"):
        import torch

        x0, x1 = domain[keys[0]]
        y0, y1 = domain[keys[1]]
        x = np.linspace(x0, x1, nx_inf)
        y = np.linspace(y0, y1, ny_inf)
        X, Y = np.meshgrid(x, y, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
        pts_t = torch.tensor(pts)

        torch_model.eval()
        with torch.no_grad():
            u = torch_model(pts_t).numpy().reshape(nx_inf, ny_inf)

        fig = px.imshow(u.T, origin="lower",
                        x=np.linspace(x0, x1, nx_inf),
                        y=np.linspace(y0, y1, ny_inf),
                        color_continuous_scale="RdBu_r",
                        labels={"x": keys[0], "y": keys[1], "color": "u"},
                        title="PINN prediction — u(x,y)")
        fig.update_layout(template="plotly_dark", height=400,
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Store for export
        update_project({"inference_result": {
            "u": u.tolist(), "x": x.tolist(), "y": y.tolist(),
            "keys": keys,
        }})
        st.success("Inference complete.")

        # Download
        df_out = pd.DataFrame(
            np.column_stack([pts, u.ravel()]),
            columns=[keys[0], keys[1], "u"]
        )
        st.download_button("Download predictions CSV",
                           df_out.to_csv(index=False).encode(),
                           "pinn_predictions.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# LBM — display trajectory
# ─────────────────────────────────────────────────────────────────────────────
elif model_type == "lbm":
    st.markdown("Browse saved LBM trajectory snapshots.")

    traj_ux = sim_info.get("trajectory_ux", [])
    traj_uy = sim_info.get("trajectory_uy", [])
    vel_mag  = sim_info.get("vel_mag")
    obstacle = sim_info.get("obstacle")

    if traj_ux:
        n_snaps = len(traj_ux)
        snap_idx = st.slider("Snapshot", 0, n_snaps - 1, n_snaps - 1)
        ux_snap = np.array(traj_ux[snap_idx])
        uy_snap = np.array(traj_uy[snap_idx]) if traj_uy else np.zeros_like(ux_snap)
        vm_snap = np.sqrt(ux_snap**2 + uy_snap**2)
        if obstacle is not None:
            vm_snap = np.where(obstacle, np.nan, vm_snap)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(vm_snap.T, origin="lower", color_continuous_scale="viridis",
                            title=f"|v| — snapshot {snap_idx}")
            fig.update_layout(template="plotly_dark", height=300,
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.imshow(ux_snap.T, origin="lower", color_continuous_scale="RdBu_r",
                             title=f"ux — snapshot {snap_idx}")
            fig2.update_layout(template="plotly_dark", height=300,
                               margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
    elif vel_mag is not None:
        fig = px.imshow(vel_mag.T, origin="lower", color_continuous_scale="viridis",
                        title="|velocity| — final step")
        fig.update_layout(template="plotly_dark", height=350,
                          margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Timeseries inference
# ─────────────────────────────────────────────────────────────────────────────
elif model_type in ("tcn", "lstm", "tft"):
    import torch

    torch_model = result.get("_torch_model")
    ts_data     = proj.get("timeseries_data")

    if torch_model is None:
        st.error("PyTorch model not in session. Retrain the model.")
        st.stop()
    if ts_data is None:
        st.warning("No timeseries data to run inference on.")
        st.stop()

    y = np.array(ts_data, dtype=np.float32)
    input_len = result.get("input_len", 32)
    horizon   = result.get("horizon",   16)

    st.markdown(f"Using last **{input_len}** time steps as context to forecast **{horizon}** ahead.")

    if len(y) < input_len:
        st.error(f"Timeseries too short ({len(y)}) for input_len={input_len}.")
        st.stop()

    context = y[-input_len:]
    x_t = torch.tensor(context[None, :, None], dtype=torch.float32)  # (1, L, 1)

    torch_model.eval()
    with torch.no_grad():
        out = torch_model(x_t)
        forecast = out.y_hat.squeeze().numpy()  # (H,)

    t_ctx  = np.arange(len(y) - input_len, len(y))
    t_fore = np.arange(len(y), len(y) + horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y,
                             mode="lines", line=dict(color="#4ECDC4", width=1.5),
                             name="History"))
    fig.add_trace(go.Scatter(x=t_fore, y=forecast,
                             mode="lines+markers",
                             line=dict(color="#FF6B35", width=2, dash="dash"),
                             marker=dict(size=5),
                             name="Forecast"))
    fig.update_layout(template="plotly_dark", height=350,
                      xaxis_title="t", yaxis_title="value",
                      title=f"{model_type.upper()} forecast (horizon={horizon})",
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    df_out = pd.DataFrame({"t": t_fore, "forecast": forecast})
    st.download_button("Download forecast CSV",
                       df_out.to_csv(index=False).encode(),
                       "forecast.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# FFT inference
# ─────────────────────────────────────────────────────────────────────────────
elif model_type == "fft":
    fft_model = result.get("_fft_model")
    ts_data   = proj.get("timeseries_data")

    if fft_model is None:
        st.error("FFT model not in session. Refit the model.")
        st.stop()

    horizon = st.slider("Forecast horizon", 1, 500, 50)

    if st.button("Forecast", type="primary"):
        forecast = fft_model.predict(horizon)
        y_hist = np.array(ts_data) if ts_data else np.array([])
        t_hist = np.arange(len(y_hist))
        t_fore = np.arange(len(y_hist), len(y_hist) + horizon)

        fig = go.Figure()
        if len(y_hist) > 0:
            fig.add_trace(go.Scatter(x=t_hist, y=y_hist,
                                     mode="lines", line=dict(color="#4ECDC4"),
                                     name="History"))
        fig.add_trace(go.Scatter(x=t_fore, y=forecast,
                                 mode="lines", line=dict(color="#FF6B35", dash="dash"),
                                 name="FFT forecast"))
        fig.update_layout(template="plotly_dark", height=350,
                          xaxis_title="t", yaxis_title="value",
                          title="FFT harmonic forecast",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info(f"Inference UI for model type `{model_type}` not implemented yet.")
