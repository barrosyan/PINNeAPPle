"""Dashboard — overview of active project, recent runs, quick stats."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.session import init as init_session, active_project, get_project

st.set_page_config(page_title="Dashboard · PINNeAPPle", page_icon="📊", layout="wide")
init_session()

st.title("📊 Dashboard")

proj = active_project()

# ── No project yet ────────────────────────────────────────────────────────────
if proj is None:
    st.info("No active project. Head to **Problem Setup** to create one.")
    st.page_link("pages/2_problem_setup.py", label="Go to Problem Setup →")
    st.stop()

# ── Project header ────────────────────────────────────────────────────────────
st.subheader(f"Project: {proj.get('name','Untitled')}")
prob  = proj.get("problem", {})
model = proj.get("trained_model")
sim   = proj.get("simulation")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Problem", prob.get("name", "—"))
with col2:
    st.metric("Category", prob.get("category", "—"))
with col3:
    if model:
        st.metric("Final Loss", f"{model.get('final_loss', float('nan')):.4e}")
    else:
        st.metric("Final Loss", "—")
with col4:
    status = "Trained" if model else ("Simulated" if sim else "Not trained")
    st.metric("Status", status)

st.divider()

# ── Loss curve ────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([2, 1])

with col_l:
    st.subheader("Training history")
    if model and model.get("history"):
        hist = model["history"]
        epochs = [h["epoch"] for h in hist]
        losses = [h["loss"]  for h in hist]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=losses, mode="lines",
                                 line=dict(color="#FF6B35", width=2), name="loss"))
        fig.update_layout(
            template="plotly_dark",
            height=300,
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis_type="log",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training history yet.")

with col_r:
    st.subheader("Problem spec")
    if prob:
        st.markdown(f"**Equations**")
        for eq in prob.get("equations", []):
            st.latex(eq.replace("∂", r"\partial").replace("∇", r"\nabla"))
        with st.expander("Domain"):
            for k, v in prob.get("domain", {}).items():
                st.text(f"{k}: [{v[0]}, {v[1]}]")
        with st.expander("Boundary conditions"):
            for bc in prob.get("bcs", []):
                st.markdown(f"- {bc}")
        with st.expander("Parameters"):
            params = prob.get("params", {})
            if params:
                for k, v in params.items():
                    st.text(f"{k} = {v}")
            else:
                st.caption("None defined")
    else:
        st.info("No problem defined yet.")

st.divider()

# ── Simulation preview ────────────────────────────────────────────────────────
if sim:
    st.subheader("Simulation preview")
    kind = sim.get("type", "")

    if kind == "lbm":
        vel  = sim.get("vel_mag")
        traj = sim.get("trajectory_ux", [])
        if vel is not None:
            vel_np = vel[-1] if hasattr(vel, "__len__") and vel.ndim == 3 else vel
            fig = px.imshow(vel_np.T, origin="lower", color_continuous_scale="viridis",
                            title="|velocity| — final step",
                            labels={"color": "|v|"})
            fig.update_layout(template="plotly_dark", height=300,
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Re={sim.get('Re','?')}  nx={sim.get('nx','?')}  ny={sim.get('ny','?')}")

    elif kind in ("fdm", "fem"):
        field = sim.get("field")
        if field is not None:
            fig = px.imshow(np.array(field).T if np.array(field).ndim == 2 else np.array(field).reshape(10,-1),
                            origin="lower", color_continuous_scale="RdBu_r",
                            title=sim.get("label", "field"))
            fig.update_layout(template="plotly_dark", height=300,
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ── Config snapshot ───────────────────────────────────────────────────────────
if model:
    with st.expander("Model config"):
        cfg = model.get("config", {})
        for k, v in cfg.items():
            st.text(f"{k}: {v}")
        st.text(f"type: {model.get('type','—')}")
