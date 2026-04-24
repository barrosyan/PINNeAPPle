"""Data & Geometry — upload STL/CSV, generate collocation points."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
from core.session import init as init_session, active_project, update_project, notify
from core.problem_library import generate_collocation_points

st.set_page_config(page_title="Data & Geometry · PINNeAPPle", page_icon="📐", layout="wide")
init_session()

st.title("📐 Data & Geometry")

proj = active_project()
if proj is None:
    st.warning("No active project. Please set up a problem first.")
    st.page_link("pages/2_problem_setup.py", label="Go to Problem Setup →")
    st.stop()

prob = proj.get("problem", {})

tab_col, tab_geo, tab_data = st.tabs(["Collocation Points", "Geometry (STL)", "Measurement Data"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Collocation Points
# ─────────────────────────────────────────────────────────────────────────────
with tab_col:
    st.markdown("Generate random interior and boundary collocation points for PINN training.")

    col1, col2 = st.columns(2)
    with col1:
        n_interior = st.slider("Interior points", 100, 10000, 2000, step=100)
    with col2:
        n_boundary = st.slider("Boundary points", 50, 2000, 400, step=50)

    if st.button("Generate points", type="primary"):
        with st.spinner("Sampling…"):
            col_data = generate_collocation_points(prob, n_interior=n_interior,
                                                   n_boundary=n_boundary)
        update_project({"collocation": col_data})
        notify("Collocation points generated.", "success")
        st.rerun()

    col_data = proj.get("collocation")
    if col_data is not None:
        interior = col_data["interior"]
        boundary = col_data["boundary"]
        coord_names = col_data.get("coord_names", ["x", "y"])
        n_dim = interior.shape[1]

        st.success(f"Interior: {len(interior)} pts  |  Boundary: {len(boundary)} pts  |  dim={n_dim}")

        # 2D scatter preview
        if n_dim >= 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=interior[:, 0], y=interior[:, 1],
                mode="markers", marker=dict(size=2, color="#4ECDC4", opacity=0.5),
                name="Interior"
            ))
            fig.add_trace(go.Scatter(
                x=boundary[:, 0], y=boundary[:, 1],
                mode="markers", marker=dict(size=4, color="#FF6B35"),
                name="Boundary"
            ))
            fig.update_layout(
                template="plotly_dark", height=350,
                xaxis_title=coord_names[0] if len(coord_names) > 0 else "x",
                yaxis_title=coord_names[1] if len(coord_names) > 1 else "y",
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download
        df_int = pd.DataFrame(interior, columns=[f"{c}_int" for c in coord_names])
        df_bnd = pd.DataFrame(boundary, columns=[f"{c}_bnd" for c in coord_names])
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("Download interior CSV",
                               df_int.to_csv(index=False).encode(),
                               "interior_points.csv", "text/csv")
        with col_dl2:
            st.download_button("Download boundary CSV",
                               df_bnd.to_csv(index=False).encode(),
                               "boundary_points.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Geometry (STL)
# ─────────────────────────────────────────────────────────────────────────────
with tab_geo:
    st.markdown("Upload an STL file to define obstacle geometry for LBM / PINN.")
    uploaded_stl = st.file_uploader("STL file", type=["stl"])

    if uploaded_stl:
        try:
            import struct
            raw = uploaded_stl.read()
            # Detect ASCII vs binary STL
            try:
                text = raw.decode("ascii")
                is_ascii = text.strip().startswith("solid")
            except Exception:
                is_ascii = False

            if is_ascii:
                # Parse ASCII STL normals/vertices for preview
                lines = text.splitlines()
                verts = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("vertex"):
                        coords = list(map(float, line.split()[1:4]))
                        verts.append(coords)
                verts = np.array(verts)
            else:
                # Binary STL: 80-byte header, 4-byte triangle count, then 50*n bytes
                n_tri = struct.unpack_from("<I", raw, 80)[0]
                verts = []
                for i in range(n_tri):
                    offset = 84 + i * 50
                    for j in range(3):
                        v = struct.unpack_from("<fff", raw, offset + 12 + j * 12)
                        verts.append(v)
                verts = np.array(verts)

            st.success(f"Loaded STL: {len(verts)//3} triangles, "
                       f"{len(verts)} vertices")

            if len(verts) > 0:
                fig = go.Figure(data=[go.Scatter3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    mode="markers",
                    marker=dict(size=1, color=verts[:, 2],
                                colorscale="Viridis", opacity=0.6),
                )])
                fig.update_layout(
                    template="plotly_dark", height=400,
                    scene=dict(aspectmode="data"),
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

            update_project({"geometry": {"stl_vertices": verts.tolist(),
                                          "filename": uploaded_stl.name}})
            notify("STL geometry loaded.", "success")

        except Exception as e:
            st.error(f"Failed to parse STL: {e}")
    else:
        geo = proj.get("geometry")
        if geo:
            st.info(f"Geometry loaded: `{geo.get('filename','?')}` "
                    f"({len(geo.get('stl_vertices',[]))//3} triangles)")
        else:
            st.info("No geometry loaded. STL upload is optional — "
                    "analytical obstacles (cylinder, rectangle) are configured in the solver.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Measurement Data
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown(
        "Upload experimental or simulation data for physics-informed training "
        "or timeseries forecasting."
    )

    uploaded_csv = st.file_uploader("CSV file", type=["csv"])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(20), use_container_width=True)

            # Basic stats
            with st.expander("Statistics"):
                st.dataframe(df.describe(), use_container_width=True)

            # Column selection for timeseries
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                ts_col = st.selectbox("Select timeseries column", numeric_cols)
                if ts_col:
                    fig = px.line(df, y=ts_col, title=f"Column: {ts_col}",
                                  template="plotly_dark")
                    fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                    if st.button("Use this column for timeseries modelling"):
                        update_project({"timeseries_data": df[ts_col].values.tolist(),
                                        "timeseries_col": ts_col,
                                        "dataframe": df.to_dict("list")})
                        notify(f"Timeseries data set: {ts_col}", "success")
                        st.rerun()

            update_project({"upload_df": df.to_dict("list")})
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    else:
        ts_data = proj.get("timeseries_data")
        if ts_data:
            st.info(f"Timeseries loaded: {len(ts_data)} samples  "
                    f"(col=`{proj.get('timeseries_col','?')}`)")
            fig = px.line(y=ts_data, template="plotly_dark",
                          labels={"index": "t", "y": "value"})
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No measurement data loaded.")
