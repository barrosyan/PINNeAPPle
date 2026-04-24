"""Visualization — Q-criterion, vorticity, LBM flow dashboards."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.session import init as init_session, active_project

st.set_page_config(page_title="Visualization · PINNeAPPle", page_icon="🌊", layout="wide")
init_session()

st.title("🌊 Visualization")

proj = active_project()
if proj is None:
    st.warning("No active project.")
    st.page_link("pages/2_problem_setup.py", label="Go to Problem Setup →")
    st.stop()

sim        = proj.get("simulation")
model_info = proj.get("trained_model")
ts_data    = proj.get("timeseries_data")

result = sim or model_info
if result is None:
    st.warning("No simulation or trained model to visualize.")
    st.page_link("pages/5_training.py", label="Go to Training →")
    st.stop()

model_type = result.get("type", "")

# ─────────────────────────────────────────────────────────────────────────────
# LBM — full CFD visualization suite
# ─────────────────────────────────────────────────────────────────────────────
if model_type == "lbm":
    tab_flow, tab_vortex, tab_qcrit, tab_anim = st.tabs(
        ["Flow Fields", "Vorticity", "Q-criterion", "Animation"]
    )

    ux       = np.array(result.get("ux", []))
    uy       = np.array(result.get("uy", []))
    rho      = np.array(result.get("rho", []))
    vel_mag  = np.array(result.get("vel_mag", []))
    obstacle = result.get("obstacle")
    nx       = result.get("nx", ux.shape[0] if ux.ndim >= 1 else 64)
    ny       = result.get("ny", ux.shape[1] if ux.ndim >= 2 else 64)

    if obstacle is not None:
        obs = np.array(obstacle, dtype=bool)
        ux_masked  = np.where(obs, np.nan, ux)
        uy_masked  = np.where(obs, np.nan, uy)
        vm_masked  = np.where(obs, np.nan, vel_mag)
        rho_masked = np.where(obs, np.nan, rho)
    else:
        ux_masked = ux; uy_masked = uy
        vm_masked = vel_mag; rho_masked = rho

    # ── Flow Fields tab ───────────────────────────────────────────────────────
    with tab_flow:
        st.subheader("Flow field overview")
        field_choice = st.radio("Field", ["|velocity|", "ux", "uy", "density ρ"],
                                horizontal=True)
        cmap_choice  = st.selectbox("Colormap", ["viridis", "RdBu_r", "plasma",
                                                  "inferno", "coolwarm"], index=0)

        if field_choice == "|velocity|":
            data = vm_masked
        elif field_choice == "ux":
            data = ux_masked
        elif field_choice == "uy":
            data = uy_masked
        else:
            data = rho_masked

        fig = px.imshow(data.T, origin="lower", color_continuous_scale=cmap_choice,
                        title=field_choice, labels={"color": field_choice})
        fig.update_layout(template="plotly_dark", height=400,
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("max |v|", f"{np.nanmax(vm_masked):.4f}")
        with col2:
            st.metric("Re", result.get("Re", "?"))
        with col3:
            st.metric("Grid", f"{nx}×{ny}")

        # Download
        df_out = {"ux": ux.ravel(), "uy": uy.ravel(), "vel_mag": vel_mag.ravel(),
                  "rho": rho.ravel()}
        import pandas as pd
        csv = pd.DataFrame(df_out).to_csv(index=False).encode()
        st.download_button("Download flow field CSV", csv, "flow_field.csv", "text/csv")

    # ── Vorticity tab ─────────────────────────────────────────────────────────
    with tab_vortex:
        st.subheader("Vorticity ωz = ∂v/∂x − ∂u/∂y")
        try:
            from pinneaple_viz.vortex import compute_vorticity_2d, plot_vorticity
            dx = 1.0 / nx; dy = 1.0 / ny
            omega = compute_vorticity_2d(ux, uy, dx, dy)
            if obstacle is not None:
                omega = np.where(obs, np.nan, omega)

            vmax = float(np.nanpercentile(np.abs(omega), 97))
            fig = px.imshow(omega.T, origin="lower", color_continuous_scale="RdBu_r",
                            zmin=-vmax, zmax=vmax,
                            title="Vorticity ωz", labels={"color": "ωz"})
            fig.update_layout(template="plotly_dark", height=400,
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Range: [{np.nanmin(omega):.3f}, {np.nanmax(omega):.3f}]  |  "
                       f"97th-pct |ω| = {vmax:.3f}")
        except Exception as e:
            st.error(f"pinneaple_viz not available: {e}")

    # ── Q-criterion tab ───────────────────────────────────────────────────────
    with tab_qcrit:
        st.subheader("Q-criterion — vortex identification")
        st.markdown(
            r"$Q = -\frac{\partial u}{\partial x}\frac{\partial v}{\partial y}"
            r"- \frac{\partial v}{\partial x}\frac{\partial u}{\partial y}$"
            r" &nbsp;— vortex cores where $Q > 0$"
        )

        try:
            from pinneaple_viz.vortex import compute_q_criterion_2d
            dx = 1.0 / nx; dy = 1.0 / ny
            Q = compute_q_criterion_2d(ux, uy, dx, dy)
            if obstacle is not None:
                Q = np.where(obs, np.nan, Q)

            q_thresh = st.slider("Q threshold (display)", 0.0,
                                 float(np.nanpercentile(Q, 99.5)), 0.0,
                                 step=float(np.nanpercentile(Q, 99.5)) / 100)

            Q_display = np.where(Q >= q_thresh, Q, np.nan)
            vmax_q    = float(np.nanpercentile(Q, 99))

            col_q1, col_q2 = st.columns(2)
            with col_q1:
                fig_q = px.imshow(Q.T, origin="lower", color_continuous_scale="hot",
                                  zmin=0, zmax=vmax_q,
                                  title="Q-criterion (full)", labels={"color": "Q"})
                fig_q.update_layout(template="plotly_dark", height=320,
                                    margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_q, use_container_width=True)
            with col_q2:
                fig_q2 = px.imshow(Q_display.T, origin="lower", color_continuous_scale="hot",
                                   title=f"Q > {q_thresh:.3f} (vortex cores)",
                                   labels={"color": "Q"})
                fig_q2.update_layout(template="plotly_dark", height=320,
                                     margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_q2, use_container_width=True)

            n_vortex = int(np.sum(Q > q_thresh)) if q_thresh > 0 else int(np.sum(Q > 0))
            pct_pos  = 100 * np.sum(Q > 0) / Q.size
            st.caption(f"Vortex core cells (Q>thresh): {n_vortex}  |  "
                       f"Q>0: {pct_pos:.1f}%  |  "
                       f"max Q: {np.nanmax(Q):.4f}")

        except Exception as e:
            st.error(f"Could not compute Q-criterion: {e}")

    # ── Animation tab ─────────────────────────────────────────────────────────
    with tab_anim:
        st.subheader("Trajectory animation")
        traj_ux = result.get("trajectory_ux", [])
        traj_uy = result.get("trajectory_uy", [])

        if not traj_ux:
            st.info("No trajectory snapshots saved. Re-run LBM with save_every > 0.")
        else:
            n_snaps = len(traj_ux)
            field_anim = st.radio("Field", ["|velocity|", "ux"], horizontal=True,
                                  key="anim_field")

            frames = []
            for i, (ux_s, uy_s) in enumerate(zip(traj_ux, traj_uy)):
                ux_s = np.array(ux_s); uy_s = np.array(uy_s)
                vm_s = np.sqrt(ux_s**2 + uy_s**2)
                if obstacle is not None:
                    vm_s = np.where(obs, np.nan, vm_s)
                    ux_s = np.where(obs, np.nan, ux_s)
                data_s = vm_s if field_anim == "|velocity|" else ux_s
                frames.append(go.Frame(
                    data=[go.Heatmap(z=data_s.T, colorscale="viridis",
                                     showscale=False)],
                    name=str(i),
                ))

            ux0 = np.array(traj_ux[0]); uy0 = np.array(traj_uy[0])
            vm0 = np.sqrt(ux0**2 + uy0**2)
            if obstacle is not None:
                vm0 = np.where(obs, np.nan, vm0)

            fig_anim = go.Figure(
                data=[go.Heatmap(z=vm0.T, colorscale="viridis")],
                frames=frames,
                layout=go.Layout(
                    template="plotly_dark",
                    height=350,
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=f"{field_anim} trajectory ({n_snaps} frames)",
                    updatemenus=[dict(
                        type="buttons", showactive=False, y=1.05,
                        buttons=[
                            dict(label="Play", method="animate",
                                 args=[None, dict(frame=dict(duration=150, redraw=True),
                                                  fromcurrent=True)]),
                            dict(label="Pause", method="animate",
                                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                    mode="immediate")]),
                        ],
                    )],
                    sliders=[dict(
                        steps=[dict(args=[[f.name], dict(mode="immediate",
                                                          frame=dict(duration=0, redraw=True))],
                                    method="animate", label=str(i))
                               for i, f in enumerate(frames)],
                        x=0, y=0, len=1.0, currentvalue=dict(prefix="Frame: "),
                    )],
                ),
            )
            st.plotly_chart(fig_anim, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PINN inference result
# ─────────────────────────────────────────────────────────────────────────────
elif model_type == "pinn_mlp":
    st.subheader("PINN solution")
    inf_result = proj.get("inference_result")

    if inf_result is None:
        st.info("Run inference first to generate predictions.")
        st.page_link("pages/6_inference.py", label="Go to Inference →")
    else:
        u = np.array(inf_result["u"])
        x = np.array(inf_result["x"])
        y = np.array(inf_result["y"])
        keys = inf_result.get("keys", ["x", "y"])

        fig = px.imshow(u.T, origin="lower",
                        x=x, y=y,
                        color_continuous_scale="RdBu_r",
                        labels={"x": keys[0], "y": keys[1], "color": "u"},
                        title="PINN prediction u(x,y)")
        fig.update_layout(template="plotly_dark", height=450,
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("min u", f"{np.min(u):.4f}")
        with col2: st.metric("max u", f"{np.max(u):.4f}")
        with col3: st.metric("Grid", f"{len(x)}×{len(y)}")

# ─────────────────────────────────────────────────────────────────────────────
# Timeseries visualization
# ─────────────────────────────────────────────────────────────────────────────
elif model_type in ("tcn", "lstm", "tft", "fft"):
    st.subheader("Timeseries visualization")

    hist = model_info.get("history", []) if model_info else []
    if hist:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[h["epoch"] for h in hist],
            y=[h["loss"] for h in hist],
            mode="lines", line=dict(color="#FF6B35", width=2),
            name="Training loss"
        ))
        fig.update_layout(template="plotly_dark", height=300,
                          xaxis_title="Epoch", yaxis_title="Loss", yaxis_type="log",
                          title="Training history",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    if ts_data:
        y = np.array(ts_data)
        fig2 = px.line(y=y, labels={"index": "t", "y": "value"},
                       title="Full timeseries", template="plotly_dark")
        fig2.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FDM / FEM
# ─────────────────────────────────────────────────────────────────────────────
elif model_type in ("fdm", "fem"):
    st.subheader(f"{model_type.upper()} solution")
    field = np.array(result.get("field", []))
    label = result.get("label", "u")

    if field.ndim == 2:
        fig = px.imshow(field.T, origin="lower",
                        color_continuous_scale="RdBu_r",
                        labels={"color": label},
                        title=label)
        fig.update_layout(template="plotly_dark", height=450,
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1: st.metric("min", f"{np.nanmin(field):.4f}")
        with col2: st.metric("max", f"{np.nanmax(field):.4f}")
    else:
        st.info("FEM result on unstructured mesh — scatter plot")
        nodes = np.array(result.get("nodes", []))
        if nodes.shape[1] >= 2 and len(field) == len(nodes):
            fig = go.Figure(go.Scatter(
                x=nodes[:, 0], y=nodes[:, 1],
                mode="markers",
                marker=dict(color=field, colorscale="RdBu_r",
                            size=5, showscale=True,
                            colorbar=dict(title=label)),
            ))
            fig.update_layout(template="plotly_dark", height=400,
                              xaxis_scaleanchor="y",
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"Visualization for model type `{model_type}` not implemented.")
