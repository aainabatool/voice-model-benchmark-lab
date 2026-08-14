"""Streamlit dashboard for the Voice Model Benchmark Lab.

Talks to the FastAPI backend over HTTP -- never touches the database
directly, matching the dashboard -> API -> orchestrator architecture.

Requires the API to be running first:
    uv run uvicorn apps.api.main:app --reload

Then:
    uv run streamlit run apps/dashboard/app.py
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Voice Model Benchmark Lab", layout="wide")

API_BASE = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")


def api_get(path: str) -> dict | list:
    resp = requests.get(f"{API_BASE}{path}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json_body: dict) -> dict:
    resp = requests.post(f"{API_BASE}{path}", json=json_body, timeout=10)
    resp.raise_for_status()
    return resp.json()


st.title("Voice Model Benchmark Lab")

# --------------------------------------------------------------------------
# Sidebar: check API connectivity, trigger a new run
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Run a benchmark")

    try:
        models: dict[str, str] = api_get("/models")  # type: ignore[assignment]
        api_ok = True
    except requests.RequestException as exc:
        st.error(f"Can't reach API at {API_BASE}\n\n{exc}")
        models = {}
        api_ok = False

    if api_ok:
        model_names = list(models.keys())
        selected_models = st.multiselect(
            "Models",
            options=model_names,
            default=model_names[:2],
            format_func=lambda name: f"{name} - {models[name]}",
        )
        dataset_path = st.text_input(
            "Dataset manifest path", value="tests/fixtures/tiny_dataset.json"
        )

        if st.button("Run benchmark", disabled=not selected_models, type="primary"):
            try:
                result = api_post(
                    "/experiments", {"dataset_path": dataset_path, "models": selected_models}
                )
                st.session_state["last_experiment_id"] = result["experiment_id"]
                st.success(f"Started: {result['experiment_id']}")
            except requests.RequestException as exc:
                st.error(f"Failed to start run: {exc}")

# --------------------------------------------------------------------------
# Main: experiment list + detail view
# --------------------------------------------------------------------------
st.subheader("Experiments")

try:
    experiments = api_get("/experiments")
except requests.RequestException as exc:
    st.error(f"Can't reach API at {API_BASE}\n\n{exc}")
    st.stop()

if not experiments:
    st.info("No experiments yet. Trigger one from the sidebar.")
    st.stop()

exp_df = pd.DataFrame(experiments)[
    ["experiment_id", "benchmark_name", "status", "start_time", "end_time"]
].sort_values("start_time", ascending=False)

st.dataframe(exp_df, use_container_width=True, hide_index=True)

all_ids = exp_df["experiment_id"].tolist()
default_id = st.session_state.get("last_experiment_id", all_ids[0])
default_index = all_ids.index(default_id) if default_id in all_ids else 0

selected_id = st.selectbox("Inspect an experiment", options=all_ids, index=default_index)

detail = api_get(f"/experiments/{selected_id}")
experiment = detail["experiment"]
results = detail["results"]

st.subheader(f"Experiment: {selected_id}")
col1, col2, col3 = st.columns(3)
col1.metric("Status", experiment["status"])
col2.metric("Dataset version", experiment["dataset_version"])
col3.metric("Results recorded", len(results))

if experiment["status"] == "running":
    st.info("Still running - click refresh to check progress.")
    if st.button("Refresh"):
        st.rerun()

hw = experiment.get("hardware")
if hw:
    gpu_desc = f"{hw['gpu']} ({hw['vram_gb']}GB VRAM)" if hw.get("gpu") else "CPU only"
    st.caption(
        f"Ran on: {hw['os']} - Python {hw['python_version']} - {hw['cpu']} - "
        f"{hw['ram_gb']}GB RAM - GPU: {gpu_desc}"
    )

if results:
    res_df = pd.DataFrame(results)

    st.markdown("#### Per-model comparison")
    summary = (
        res_df.groupby("model")
        .agg(
            avg_wer=("wer", "mean"),
            avg_cer=("cer", "mean"),
            avg_rtf=("rtf", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            failures=("failed", "sum"),
        )
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Average WER by model**")
        st.bar_chart(summary.set_index("model")["avg_wer"])
    with chart_col2:
        st.markdown("**Average RTF by model** (lower = faster than real-time)")
        st.bar_chart(summary.set_index("model")["avg_rtf"])

    st.markdown("#### All results")
    st.dataframe(
        res_df[
            [
                "model",
                "test_case_id",
                "reference",
                "prediction",
                "wer",
                "cer",
                "rtf",
                "latency_ms",
                "failed",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No results recorded yet for this experiment.")
