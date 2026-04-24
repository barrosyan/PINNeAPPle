"""
PINNeAPPle — Physics AI Platform
Main entry point for the Streamlit multi-page app.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from core.session import init as init_session, active_project, pop_notifications

st.set_page_config(
    page_title="PINNeAPPle",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* sidebar nav tweaks */
[data-testid="stSidebarNav"] { padding-top: 0.5rem; }
[data-testid="stSidebarNav"] li { font-size: 0.9rem; }

/* metric cards */
div[data-testid="metric-container"] {
    background: #16213E;
    border: 1px solid #FF6B3522;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}

/* accent buttons */
div.stButton > button {
    background: linear-gradient(135deg, #FF6B35, #e8521a);
    color: white;
    border: none;
    border-radius: 6px;
}
div.stButton > button:hover { filter: brightness(1.1); }

/* expander header */
details > summary { font-weight: 600; }

/* monospace code blocks */
pre { font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session bootstrap ───────────────────────────────────────────────────────
init_session()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/PINNeAPPle/assets/main/logo.png",
             use_column_width=True, caption="") if False else None  # logo placeholder
    st.markdown("## 🍍 PINNeAPPle")
    st.caption("Physics-Informed Neural Network Platform")
    st.divider()

    proj = active_project()
    if proj:
        st.markdown(f"**Active project:** `{proj.get('name','—')}`")
        prob = proj.get("problem", {})
        if prob.get("name"):
            st.caption(f"Problem: {prob['name']}")
        model_info = proj.get("trained_model")
        if model_info:
            st.caption(f"Model: {model_info.get('type','—')}  "
                       f"loss={model_info.get('final_loss', 0):.4f}")
    else:
        st.info("No active project. Go to **Problem Setup** to begin.")

    st.divider()
    st.markdown("**Quick links**")
    st.page_link("pages/1_dashboard.py",    label="📊 Dashboard")
    st.page_link("pages/2_problem_setup.py", label="🔬 Problem Setup")
    st.page_link("pages/3_data_geometry.py", label="📐 Data & Geometry")
    st.page_link("pages/4_models.py",        label="🧠 Models")
    st.page_link("pages/5_training.py",      label="⚡ Training")
    st.page_link("pages/6_inference.py",     label="🎯 Inference")
    st.page_link("pages/7_visualization.py", label="🌊 Visualization")
    st.page_link("pages/8_benchmarks.py",    label="🏆 Benchmarks")

# ── Notifications ────────────────────────────────────────────────────────────
for note in pop_notifications():
    level = note.get("level", "info")
    msg   = note.get("msg", "")
    if level == "success": st.success(msg)
    elif level == "error": st.error(msg)
    elif level == "warning": st.warning(msg)
    else: st.info(msg)

# ── Home content ─────────────────────────────────────────────────────────────
st.title("🍍 PINNeAPPle Physics AI Platform")
st.markdown(
    "Seamlessly integrate **Physics-Informed Neural Networks**, "
    "classical solvers, and ML forecasting in one environment."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Built-in Problems", "9")
with col2:
    from core.problem_library import PROBLEMS
    st.metric("Problem Library", f"{len(PROBLEMS)} PDEs")
with col3:
    st.metric("Solvers", "LBM · FDM · FEM · PINN")
with col4:
    st.metric("TS Models", "TCN · LSTM · TFT · FNO")

st.divider()

st.subheader("Getting started")
steps = [
    ("🔬 Problem Setup",    "pages/2_problem_setup.py",
     "Describe your problem in plain language or choose from the library."),
    ("📐 Data & Geometry",  "pages/3_data_geometry.py",
     "Upload STL geometry, CSV data, or generate collocation points."),
    ("🧠 Select a Model",   "pages/4_models.py",
     "Browse PINN, LBM, FDM, FEM, TCN, LSTM, TFT and configure hyperparameters."),
    ("⚡ Train",            "pages/5_training.py",
     "Run training with live loss curves and early stopping."),
    ("🌊 Visualize",        "pages/7_visualization.py",
     "Q-criterion, vorticity, flow dashboards, and forecast plots."),
]
cols = st.columns(len(steps))
for col, (title, page, desc) in zip(cols, steps):
    with col:
        st.markdown(f"**{title}**")
        st.caption(desc)
        st.page_link(page, label="Open →")

st.divider()
st.caption("PINNeAPPle · BIATech · ybarros@biatech.com")
