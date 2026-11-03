"""Optional localhost Streamlit interface, launched by ``holod3-smooth ui``."""

import json

import pandas as pd
import streamlit as st

from holod3_smoother import SmootherConfig, smooth_linker_output

st.set_page_config(page_title="HoloD3 Smoother", layout="wide")
st.caption("HOLOD3 / TRAJECTORY ANALYSIS")
st.title("Robust trajectory smoothing")
st.write("Load a HoloD3-Linker result, adjust the model, then inspect the fit and parameter search.")
with st.form("fit"):
    source = st.text_input("Linker output directory or track CSV")
    output = st.text_input("New output directory", "runs/ui-fit")
    left, middle, right = st.columns(3)
    with left:
        selection = st.selectbox("Parameter selection", ["auto", "fixed"])
        degree = st.number_input("Fixed polynomial degree", 0, 6, 2)
        penalty = st.number_input("Fixed lambda_other", min_value=0.0, value=0.0001, format="%.8f")
        alpha = st.number_input("Fixed alpha", min_value=0.0, value=0.0, format="%.6f")
    with middle:
        degree_grid = st.text_input("Degree grid", "1,2")
        penalty_grid = st.text_input("Lambda grid", "1e-8,1e-6,1e-4,1e-2,1e-1,1")
        alpha_grid = st.text_input("Alpha grid (degree ≥ 2)", "0,1e-4,1e-2")
        iterations = st.number_input("EM iteration limit", 1, 10000, 100)
    with right:
        window = st.number_input("XY smoothing window (points)", 1, 101, 7)
        xy_mode = st.selectbox("XY time handling", ["auto", "local", "legacy"])
        max_gap = st.number_input("Split above frame gap (0 = disabled)", 0, 100000, 0)
        min_points = st.number_input("Minimum tuning track points", 2, 100000, 8)
    overwrite = st.checkbox("Replace a previous HoloD3-Smoother output at this destination")
    submitted = st.form_submit_button("Fit and compare", type="primary")
if submitted:
    st.session_state.pop("result", None)
    try:
        if not source.strip():
            raise ValueError("Enter a linker output directory or track CSV")
        config = SmootherConfig(
            selection=selection,
            degree=degree,
            lambda_other=penalty,
            alpha=alpha,
            degree_grid=tuple(int(s) for s in degree_grid.split(",")),
            lambda_other_grid=tuple(float(s) for s in penalty_grid.split(",")),
            alpha_grid=tuple(float(s) for s in alpha_grid.split(",")),
            em_max_iter=iterations,
            tune_em_max_iter=iterations,
            sg_window=window,
            xy_mode=xy_mode,
            max_gap=max_gap or None,
            tune_min_points=min_points,
        )
        with st.spinner("Fitting trajectories…"):
            st.session_state["result"] = smooth_linker_output(
                source, output_dir=output, config=config, overwrite=overwrite
            )
    except (ValueError, OSError, FloatingPointError) as error:
        st.error(str(error))
result = st.session_state.get("result")
if result is not None:
    summary = result.summary()
    st.success(f"Saved {summary['counts']['tracks']} tracks / {summary['counts']['points']} points")
    if not summary["fit"]["em_converged"]:
        st.warning("Final EM fit did not converge. Inspect the diagnostics or increase the iteration limit.")
    tracks = result.tracks()
    track_id = st.selectbox("Track", tracks.track_id.unique().tolist())
    axis = st.selectbox("Axis", ["z", "x", "y"])
    selected = tracks.loc[tracks.track_id == track_id].set_index("frame")
    st.line_chart(selected[[f"{axis}_raw", f"{axis}_smooth"]])
    st.line_chart(selected[["postOutProb"]], color=["#c65331"])
    tuning = json.loads(result.tuning_json.read_text(encoding="utf-8"))
    st.subheader("Parameter search")
    st.caption(
        "Approximate robust PRESS; only converged candidates can be selected. Full output retains every row."
    )
    st.dataframe(pd.DataFrame(tuning.get("candidates", [])))
    st.download_button(
        "Download selected configuration",
        result.config_json.read_text(encoding="utf-8"),
        "selected_config.json",
    )
    st.download_button(
        "Download offline HTML report",
        result.report_html.read_text(encoding="utf-8"),
        "report.html",
        "text/html",
    )
    st.code(str(result.output_dir))
