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

from voice_benchmark.analytics.pareto import ParetoPoint, compute_pareto_frontier

st.set_page_config(page_title="Voice Model Benchmark Lab", layout="wide")

API_BASE = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")


def api_get(path: str) -> dict | list:
    resp = requests.get(f"{API_BASE}{path}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def api_get_text(path: str) -> str:
    resp = requests.get(f"{API_BASE}{path}", timeout=10)
    resp.raise_for_status()
    return resp.text


def api_post(path: str, json_body: dict) -> dict:
    resp = requests.post(f"{API_BASE}{path}", json=json_body, timeout=10)
    resp.raise_for_status()
    return resp.json()


st.title("Voice Model Benchmark Lab")

tab_stt, tab_tts = st.tabs(["STT Benchmarks", "TTS Benchmarks"])

# ============================================================================
# STT tab
# ============================================================================
with tab_stt:
    st.subheader("Run a new STT benchmark")

    try:
        stt_models: dict[str, str] = api_get("/models")  # type: ignore[assignment]
        stt_api_ok = True
    except requests.RequestException as exc:
        st.error(f"Can't reach API at {API_BASE}\n\n{exc}")
        stt_models = {}
        stt_api_ok = False

    if stt_api_ok:
        stt_model_names = list(stt_models.keys())
        stt_selected_models = st.multiselect(
            "Models",
            options=stt_model_names,
            default=stt_model_names[:2],
            format_func=lambda name: f"{name} - {stt_models[name]}",
            key="stt_models_select",
        )
        stt_dataset_path = st.text_input(
            "Dataset manifest path",
            value="tests/fixtures/tiny_dataset.json",
            key="stt_dataset_path",
        )

        if st.button(
            "Run STT benchmark",
            disabled=not stt_selected_models,
            type="primary",
            key="stt_run_button",
        ):
            try:
                result = api_post(
                    "/experiments",
                    {"dataset_path": stt_dataset_path, "models": stt_selected_models},
                )
                st.session_state["last_stt_experiment_id"] = result["experiment_id"]
                st.success(f"Started: {result['experiment_id']}")
            except requests.RequestException as exc:
                st.error(f"Failed to start run: {exc}")

    st.divider()
    st.subheader("STT Experiments")

    try:
        experiments = api_get("/experiments")
    except requests.RequestException as exc:
        st.error(f"Can't reach API at {API_BASE}\n\n{exc}")
        experiments = []

    if not experiments:
        st.info("No STT experiments yet. Trigger one above.")
    else:
        exp_df = pd.DataFrame(experiments)[
            ["experiment_id", "benchmark_name", "status", "start_time", "end_time"]
        ].sort_values("start_time", ascending=False)

        st.dataframe(exp_df, use_container_width=True, hide_index=True)

        all_ids = exp_df["experiment_id"].tolist()
        default_id = st.session_state.get("last_stt_experiment_id", all_ids[0])
        default_index = all_ids.index(default_id) if default_id in all_ids else 0

        selected_id = st.selectbox(
            "Inspect an experiment", options=all_ids, index=default_index, key="stt_select_exp"
        )

        detail = api_get(f"/experiments/{selected_id}")
        experiment = detail["experiment"]
        results = detail["results"]

        st.markdown(f"#### Experiment: {selected_id}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", experiment["status"])
        col2.metric("Dataset version", experiment["dataset_version"])
        col3.metric("Results recorded", len(results))
        with col4:
            st.write("")
            try:
                report_text = api_get_text(f"/experiments/{selected_id}/report")
                st.download_button(
                    "Download report (.md)",
                    data=report_text,
                    file_name=f"{selected_id}.md",
                    mime="text/markdown",
                    key="stt_download",
                )
            except requests.RequestException:
                pass

        if experiment["status"] == "running":
            st.info("Still running - click refresh to check progress.")
            if st.button("Refresh", key="stt_refresh"):
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

            if summary["model"].nunique() > 1:
                st.markdown("#### Pareto analysis: accuracy vs. speed tradeoff")
                pareto_points = [
                    ParetoPoint(row["model"], x=row["avg_wer"], y=row["avg_rtf"])
                    for _, row in summary.iterrows()
                ]
                frontier = compute_pareto_frontier(pareto_points)
                summary["pareto"] = summary["model"].apply(
                    lambda m: "Pareto-optimal" if m in frontier else "Dominated"
                )
                st.scatter_chart(summary, x="avg_wer", y="avg_rtf", color="pareto", size=120)
                st.caption(
                    "Bottom-left is best (lower WER, lower RTF). A 'Dominated' model is beaten "
                    "on both axes by some other model here -- there's no accuracy/speed reason "
                    "to pick it over a Pareto-optimal one."
                )
                pareto_names = summary.loc[summary["pareto"] == "Pareto-optimal", "model"].tolist()
                st.write("**Pareto-optimal models:** " + ", ".join(pareto_names))

            if "condition" in res_df.columns and res_df["condition"].nunique() > 1:
                st.markdown("#### Robustness curve (WER vs. noise condition)")
                condition_order = [
                    "noise_neg5db",
                    "noise_0db",
                    "noise_5db",
                    "noise_10db",
                    "noise_20db",
                    "clean",
                ]
                present = [c for c in condition_order if c in res_df["condition"].unique()]
                robustness = (
                    res_df.groupby(["condition", "model"])["wer"]
                    .mean()
                    .reset_index()
                    .pivot(index="condition", columns="model", values="wer")
                    .reindex(present)
                )
                st.line_chart(robustness)
                st.caption(
                    "Left = noisiest (-5dB SNR) -> right = clean. A model that stays flatter "
                    "across this curve is more robust to noise, even if a less robust model "
                    "wins on clean audio alone."
                )

            st.markdown("#### All results")
            display_cols = [
                "model",
                "test_case_id",
                "condition",
                "reference",
                "prediction",
                "wer",
                "cer",
                "rtf",
                "latency_ms",
                "failed",
            ]
            display_cols = [c for c in display_cols if c in res_df.columns]
            st.dataframe(res_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No results recorded yet for this experiment.")

# ============================================================================
# TTS tab
# ============================================================================
with tab_tts:
    st.subheader("Run a new TTS benchmark")

    try:
        tts_models: dict[str, str] = api_get("/models/tts")  # type: ignore[assignment]
        tts_api_ok = True
    except requests.RequestException as exc:
        st.error(f"Can't reach API at {API_BASE}\n\n{exc}")
        tts_models = {}
        tts_api_ok = False

    if tts_api_ok:
        tts_model_names = list(tts_models.keys())
        tts_selected_models = st.multiselect(
            "Models",
            options=tts_model_names,
            default=tts_model_names,
            format_func=lambda name: f"{name} - {tts_models[name]}",
            key="tts_models_select",
        )
        tts_dataset_path = st.text_input(
            "Dataset manifest path",
            value="tests/fixtures/tiny_tts_dataset.json",
            key="tts_dataset_path",
        )
        tts_output_dir = st.text_input(
            "Output audio directory", value="artifacts/audio", key="tts_output_dir"
        )

        if st.button(
            "Run TTS benchmark",
            disabled=not tts_selected_models,
            type="primary",
            key="tts_run_button",
        ):
            try:
                result = api_post(
                    "/tts-experiments",
                    {
                        "dataset_path": tts_dataset_path,
                        "models": tts_selected_models,
                        "output_dir": tts_output_dir,
                    },
                )
                st.session_state["last_tts_experiment_id"] = result["experiment_id"]
                st.success(f"Started: {result['experiment_id']}")
            except requests.RequestException as exc:
                st.error(f"Failed to start run: {exc}")

    st.divider()
    st.subheader("TTS Experiments")

    try:
        tts_experiments_list = api_get("/tts-experiments")
    except requests.RequestException as exc:
        st.error(f"Can't reach API at {API_BASE}\n\n{exc}")
        tts_experiments_list = []

    if not tts_experiments_list:
        st.info(
            "No TTS experiments yet. Trigger one above -- note a TTS run only "
            "appears here once its first result is written, unlike the STT list."
        )
    else:
        tts_exp_df = pd.DataFrame(tts_experiments_list)[
            ["experiment_id", "benchmark_name", "status", "start_time", "end_time"]
        ].sort_values("start_time", ascending=False)

        st.dataframe(tts_exp_df, use_container_width=True, hide_index=True)

        tts_all_ids = tts_exp_df["experiment_id"].tolist()
        tts_default_id = st.session_state.get("last_tts_experiment_id", tts_all_ids[0])
        tts_default_index = (
            tts_all_ids.index(tts_default_id) if tts_default_id in tts_all_ids else 0
        )

        tts_selected_id = st.selectbox(
            "Inspect a TTS experiment",
            options=tts_all_ids,
            index=tts_default_index,
            key="tts_select_exp",
        )

        tts_detail = api_get(f"/tts-experiments/{tts_selected_id}")
        tts_experiment = tts_detail["experiment"]
        tts_results = tts_detail["results"]

        st.markdown(f"#### Experiment: {tts_selected_id}")
        tcol1, tcol2, tcol3 = st.columns(3)
        tcol1.metric("Status", tts_experiment["status"])
        tcol2.metric("Dataset version", tts_experiment["dataset_version"])
        tcol3.metric("Results recorded", len(tts_results))

        if tts_experiment["status"] == "running":
            st.info("Still running - click refresh to check progress.")
            if st.button("Refresh", key="tts_refresh"):
                st.rerun()

        tts_hw = tts_experiment.get("hardware")
        if tts_hw:
            gpu_desc = (
                f"{tts_hw['gpu']} ({tts_hw['vram_gb']}GB VRAM)" if tts_hw.get("gpu") else "CPU only"
            )
            st.caption(
                f"Ran on: {tts_hw['os']} - Python {tts_hw['python_version']} - {tts_hw['cpu']} - "
                f"{tts_hw['ram_gb']}GB RAM - GPU: {gpu_desc}"
            )

        if tts_results:
            tts_res_df = pd.DataFrame(tts_results)

            st.markdown("#### Per-model comparison")
            tts_summary = (
                tts_res_df.groupby("model")
                .agg(
                    avg_rtf=("rtf", "mean"),
                    avg_duration_sec=("output_duration_sec", "mean"),
                    avg_speech_rate_wpm=("speech_rate_wpm", "mean"),
                    avg_silence_ratio=("silence_ratio", "mean"),
                    failures=("failed", "sum"),
                )
                .reset_index()
            )
            st.dataframe(tts_summary, use_container_width=True, hide_index=True)

            st.markdown("**Average RTF by model** (lower = faster than real-time)")
            st.bar_chart(tts_summary.set_index("model")["avg_rtf"])

            st.markdown("#### All results")
            tts_display_cols = [
                "model",
                "test_case_id",
                "output_path",
                "output_duration_sec",
                "rtf",
                "speech_rate_wpm",
                "sample_rate",
                "silence_ratio",
                "failed",
            ]
            tts_display_cols = [c for c in tts_display_cols if c in tts_res_df.columns]
            st.dataframe(tts_res_df[tts_display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No results recorded yet for this experiment.")
