"""Problem Setup — AI-assisted + expert mode."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import json
from core.session import init as init_session, new_project, update_project, notify
from core.problem_library import PROBLEMS, CATEGORIES, get_problem
from core.ai_formulator import formulate_with_ai, suggest_models

st.set_page_config(page_title="Problem Setup · PINNeAPPle", page_icon="🔬", layout="wide")
init_session()

st.title("🔬 Problem Setup")

tab_ai, tab_library, tab_expert = st.tabs(["AI Formulator", "Problem Library", "Expert Mode"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — AI Formulator
# ─────────────────────────────────────────────────────────────────────────────
with tab_ai:
    st.markdown(
        "Describe your physical problem in plain language. "
        "Claude will extract the governing equations, domain, boundary conditions, "
        "and suggest appropriate solvers."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "Set the `ANTHROPIC_API_KEY` environment variable to enable the Claude backend. "
            "Keyword matching and templates will be used as fallback."
        )

    description = st.text_area(
        "Problem description",
        placeholder="e.g. Simulate the flow of air past a circular cylinder at Re=200 "
                    "using a 160×64 grid. I want to observe the von Kármán vortex street.",
        height=120,
    )
    proj_name = st.text_input("Project name", value="My Physics Project")

    if st.button("Formulate problem", type="primary"):
        if not description.strip():
            st.error("Please enter a description.")
        else:
            with st.spinner("Analysing problem…"):
                spec = formulate_with_ai(description)

            source = spec.pop("_source", "unknown")
            preset_key = spec.get("_preset_key", "")

            st.success(f"Formulated via **{source}**")
            st.session_state["_pending_spec"] = spec

    if "_pending_spec" in st.session_state:
        spec = st.session_state["_pending_spec"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{spec.get('name','')}**  \n_{spec.get('category','')}_")
            st.markdown(spec.get("description", ""))
            st.markdown("**Equations**")
            for eq in spec.get("equations", []):
                st.latex(eq.replace("∂", r"\partial").replace("∇", r"\nabla")
                           .replace("²", "^2").replace("·", r"\cdot"))
        with col2:
            st.markdown("**Domain**")
            st.json(spec.get("domain", {}))
            st.markdown("**Parameters**")
            st.json(spec.get("params", {}))
            st.markdown("**Suggested solvers**")
            suggestions = suggest_models(spec)
            for s in suggestions:
                st.markdown(f"- **{s['model']}** — {s['reason']}  `score={s['score']}`")

        if st.button("Create project with this problem"):
            new_project(proj_name or spec.get("name", "Project"), spec)
            notify("Project created.", "success")
            del st.session_state["_pending_spec"]
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Problem Library
# ─────────────────────────────────────────────────────────────────────────────
with tab_library:
    st.markdown("Choose a built-in benchmark problem.")

    cat_filter = st.selectbox("Filter by category", ["All"] + CATEGORIES)
    cols_per_row = 3
    items = list(PROBLEMS.items())
    if cat_filter != "All":
        items = [(k, v) for k, v in items if v["category"] == cat_filter]

    proj_name_lib = st.text_input("Project name", value="Benchmark Run", key="lib_proj_name")

    for row_start in range(0, len(items), cols_per_row):
        row_items = items[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (key, prob) in zip(cols, row_items):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{prob['name']}**")
                    st.caption(prob["category"])
                    st.markdown(prob["description"][:120] + "…" if len(prob["description"]) > 120
                                else prob["description"])
                    st.caption(f"Solvers: {', '.join(prob.get('solvers', []))}")
                    if st.button("Select", key=f"sel_{key}"):
                        spec = {**prob, "_preset_key": key, "_source": "library"}
                        new_project(proj_name_lib or prob["name"], spec)
                        notify(f"Project '{prob['name']}' created.", "success")
                        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Expert Mode (manual JSON)
# ─────────────────────────────────────────────────────────────────────────────
with tab_expert:
    st.markdown(
        "Define the problem spec manually as JSON. "
        "Useful for custom PDEs not covered by the library."
    )

    default_spec = {
        "name": "Custom PDE",
        "category": "Other",
        "description": "My custom PDE",
        "equations": ["∂u/∂t = D ∇²u"],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
        "params": {"D": 0.01},
        "bcs": ["u=0 on ∂Ω"],
        "ics": ["u(x,y,0) = sin(πx)sin(πy)"],
        "dim": 3,
        "tags": ["parabolic", "diffusion"],
        "solvers": ["fdm", "pinn"],
        "ref": "",
    }

    json_text = st.text_area(
        "Problem spec (JSON)",
        value=json.dumps(default_spec, indent=2),
        height=400,
    )
    proj_name_exp = st.text_input("Project name", value="Custom Project", key="exp_proj_name")

    if st.button("Create project from JSON"):
        try:
            spec = json.loads(json_text)
            spec["_source"] = "expert"
            new_project(proj_name_exp or spec.get("name", "Custom"), spec)
            notify("Expert project created.", "success")
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
