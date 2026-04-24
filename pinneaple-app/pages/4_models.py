"""Models — browse catalogue, configure hyperparameters."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
from core.session import init as init_session, active_project, update_project, notify
from core.ai_formulator import suggest_models

st.set_page_config(page_title="Models · PINNeAPPle", page_icon="🧠", layout="wide")
init_session()

st.title("🧠 Models")

proj = active_project()
if proj is None:
    st.warning("No active project.")
    st.page_link("pages/2_problem_setup.py", label="Go to Problem Setup →")
    st.stop()

prob = proj.get("problem", {})

# ── Suggested models ──────────────────────────────────────────────────────────
st.subheader("Recommended for this problem")
suggestions = suggest_models(prob) if prob else []
if suggestions:
    scols = st.columns(len(suggestions))
    for col, s in zip(scols, suggestions):
        with col:
            with st.container(border=True):
                st.markdown(f"**{s['model']}**")
                st.caption(s["reason"])
                score_color = "#4ECDC4" if s["score"] >= 85 else "#FFE66D"
                st.markdown(f"<span style='color:{score_color}'>Score: {s['score']}</span>",
                            unsafe_allow_html=True)
else:
    st.info("No suggestions — define a problem first.")

st.divider()
st.subheader("Model catalogue")

# ── Model catalogue ───────────────────────────────────────────────────────────
MODEL_CATALOGUE = {
    "PINN (MLP)": {
        "type": "pinn_mlp",
        "category": "PINN",
        "description": "Fully-connected Physics-Informed Neural Network. "
                       "Minimises PDE residual + boundary condition loss.",
        "params": {
            "n_epochs":   {"label": "Epochs",          "type": "int",   "default": 500,   "min": 50,   "max": 5000},
            "lr":         {"label": "Learning rate",   "type": "float", "default": 1e-3,  "min": 1e-5, "max": 0.1},
            "hidden":     {"label": "Hidden size",     "type": "int",   "default": 64,    "min": 16,   "max": 512},
            "n_layers":   {"label": "Layers",          "type": "int",   "default": 4,     "min": 2,    "max": 12},
            "n_interior": {"label": "Colloc. points",  "type": "int",   "default": 2000,  "min": 200,  "max": 20000},
        },
    },
    "LBM Solver": {
        "type": "lbm",
        "category": "Classical CFD",
        "description": "Lattice Boltzmann Method — D2Q9 BGK with Zou-He BCs, "
                       "bounce-back obstacles, optional Smagorinsky LES.",
        "params": {
            "steps":       {"label": "Steps",          "type": "int",   "default": 3000,  "min": 100,  "max": 50000},
            "save_every":  {"label": "Save every",     "type": "int",   "default": 500,   "min": 10,   "max": 5000},
            "nx":          {"label": "Grid nx",        "type": "int",   "default": 160,   "min": 32,   "max": 1024},
            "ny":          {"label": "Grid ny",        "type": "int",   "default": 64,    "min": 16,   "max": 512},
            "Re":          {"label": "Reynolds number","type": "float", "default": 200.0, "min": 10.0, "max": 2000.0},
            "u_in":        {"label": "Inlet velocity", "type": "float", "default": 0.05,  "min": 0.01, "max": 0.2},
        },
    },
    "FDM Solver": {
        "type": "fdm",
        "category": "Classical PDE",
        "description": "Finite Difference Method — 5-point Poisson/Laplace solver.",
        "params": {
            "nx": {"label": "Grid nx", "type": "int", "default": 64, "min": 16, "max": 512},
            "ny": {"label": "Grid ny", "type": "int", "default": 64, "min": 16, "max": 512},
        },
    },
    "FEM Solver": {
        "type": "fem",
        "category": "Classical PDE",
        "description": "Finite Element Method — 2D linear triangular elements.",
        "params": {
            "nx": {"label": "Elements x", "type": "int", "default": 20, "min": 5, "max": 200},
            "ny": {"label": "Elements y", "type": "int", "default": 20, "min": 5, "max": 200},
        },
    },
    "TCN Forecaster": {
        "type": "tcn",
        "category": "Timeseries",
        "description": "Temporal Convolutional Network for sequence forecasting.",
        "params": {
            "input_len": {"label": "Input length", "type": "int",   "default": 32,   "min": 8,  "max": 256},
            "horizon":   {"label": "Horizon",       "type": "int",   "default": 16,   "min": 1,  "max": 128},
            "epochs":    {"label": "Epochs",        "type": "int",   "default": 50,   "min": 5,  "max": 500},
            "lr":        {"label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-5, "max": 0.1},
        },
    },
    "LSTM Forecaster": {
        "type": "lstm",
        "category": "Timeseries",
        "description": "Long Short-Term Memory recurrent network.",
        "params": {
            "input_len": {"label": "Input length", "type": "int",   "default": 32,  "min": 8,  "max": 256},
            "horizon":   {"label": "Horizon",       "type": "int",   "default": 16,  "min": 1,  "max": 128},
            "epochs":    {"label": "Epochs",        "type": "int",   "default": 50,  "min": 5,  "max": 500},
            "lr":        {"label": "Learning rate", "type": "float", "default": 1e-3,"min": 1e-5,"max": 0.1},
        },
    },
    "TFT Forecaster": {
        "type": "tft",
        "category": "Timeseries",
        "description": "Temporal Fusion Transformer — interpretable attention-based model.",
        "params": {
            "input_len": {"label": "Input length", "type": "int",   "default": 32,  "min": 8,  "max": 512},
            "horizon":   {"label": "Horizon",       "type": "int",   "default": 16,  "min": 1,  "max": 256},
            "epochs":    {"label": "Epochs",        "type": "int",   "default": 30,  "min": 5,  "max": 200},
            "lr":        {"label": "Learning rate", "type": "float", "default": 5e-4,"min": 1e-5,"max": 0.05},
        },
    },
    "FFT Forecaster": {
        "type": "fft",
        "category": "Decomposition",
        "description": "Harmonic decomposition via FFT — fast, interpretable, no training needed.",
        "params": {
            "n_harmonics": {"label": "Harmonics",     "type": "int",  "default": 5, "min": 1, "max": 50},
            "detrend":     {"label": "Linear detrend","type": "bool", "default": True},
        },
    },
}

# Render catalogue with config panel
selected = st.session_state.get("selected_model_type", None)
model_names = list(MODEL_CATALOGUE.keys())

category_filter = st.selectbox(
    "Category",
    ["All", "PINN", "Classical CFD", "Classical PDE", "Timeseries", "Decomposition"],
)

filtered = {k: v for k, v in MODEL_CATALOGUE.items()
            if category_filter == "All" or v["category"] == category_filter}

# Grid display
cols_per_row = 4
items = list(filtered.items())
for row_start in range(0, len(items), cols_per_row):
    row = items[row_start:row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, (name, meta) in zip(cols, row):
        with col:
            is_sel = st.session_state.get("selected_model_type") == meta["type"]
            border_color = "#FF6B35" if is_sel else "#1e3a5f"
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(meta["category"])
                st.markdown(meta["description"][:100] + "…" if len(meta["description"]) > 100
                            else meta["description"])
                if st.button("Configure" if not is_sel else "✓ Selected",
                             key=f"cfg_{meta['type']}"):
                    st.session_state["selected_model_type"] = meta["type"]
                    st.session_state["selected_model_name"] = name
                    st.rerun()

# ── Configuration panel ───────────────────────────────────────────────────────
sel_type = st.session_state.get("selected_model_type")
if sel_type:
    sel_name = st.session_state.get("selected_model_name", sel_type)
    meta = next((v for v in MODEL_CATALOGUE.values() if v["type"] == sel_type), None)
    if meta:
        st.divider()
        st.subheader(f"Configure: {sel_name}")
        cfg_values = {}
        param_cols = st.columns(min(len(meta["params"]), 4))
        for i, (pname, pmeta) in enumerate(meta["params"].items()):
            with param_cols[i % len(param_cols)]:
                if pmeta["type"] == "int":
                    cfg_values[pname] = st.number_input(
                        pmeta["label"], min_value=pmeta["min"], max_value=pmeta["max"],
                        value=pmeta["default"], step=1, key=f"p_{sel_type}_{pname}"
                    )
                elif pmeta["type"] == "float":
                    cfg_values[pname] = st.number_input(
                        pmeta["label"], min_value=float(pmeta["min"]),
                        max_value=float(pmeta["max"]),
                        value=float(pmeta["default"]), format="%.5f",
                        key=f"p_{sel_type}_{pname}"
                    )
                elif pmeta["type"] == "bool":
                    cfg_values[pname] = st.checkbox(
                        pmeta["label"], value=pmeta["default"],
                        key=f"p_{sel_type}_{pname}"
                    )

        if sel_type == "lbm":
            st.markdown("**Obstacle**")
            obs_type = st.selectbox("Type", ["none", "cylinder", "rectangle"],
                                    key="obs_type")
            if obs_type == "cylinder":
                c1, c2, c3 = st.columns(3)
                nx_ = cfg_values.get("nx", 160)
                ny_ = cfg_values.get("ny", 64)
                with c1: cx = st.number_input("cx", value=nx_//4, key="obs_cx")
                with c2: cy = st.number_input("cy", value=ny_//2, key="obs_cy")
                with c3: r  = st.number_input("r",  value=ny_//8, key="obs_r")
                cfg_values["obstacle"] = {"type": "cylinder", "cx": cx, "cy": cy, "r": r}
            elif obs_type == "rectangle":
                c1, c2, c3, c4 = st.columns(4)
                with c1: x0 = st.number_input("x0", value=20, key="obs_x0")
                with c2: x1 = st.number_input("x1", value=30, key="obs_x1")
                with c3: y0 = st.number_input("y0", value=20, key="obs_y0")
                with c4: y1 = st.number_input("y1", value=44, key="obs_y1")
                cfg_values["obstacle"] = {"type": "rectangle",
                                          "x0": x0, "x1": x1, "y0": y0, "y1": y1}

        if st.button("Save model configuration", type="primary"):
            update_project({
                "model_config": {"type": sel_type, "name": sel_name, **cfg_values}
            })
            notify(f"Model '{sel_name}' configured.", "success")
            st.rerun()

        saved = proj.get("model_config", {})
        if saved.get("type") == sel_type:
            st.success("Configuration saved for this project.")
