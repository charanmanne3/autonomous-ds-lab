"""Modern Streamlit dashboard for Autonomous Data Science Lab."""

from pathlib import Path
import threading
import time
from typing import Any, Dict

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/run-pipeline"
MLFLOW_UI_URL = "http://127.0.0.1:5000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_PIPELINE_TIMEOUT_SECONDS = 90


def _inject_styles() -> None:
    """Apply custom CSS to give the app a platform-like feel."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
        .section-title { margin-top: 0.6rem; margin-bottom: 0.35rem; }
        .card {
            background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            color: #f9fafb;
        }
        .card-title { color: #9ca3af; font-size: 0.85rem; margin-bottom: 0.25rem; }
        .card-value { font-size: 1.15rem; font-weight: 650; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _call_pipeline(task: str, fast_demo_mode: bool) -> Dict[str, Any]:
    """Send pipeline request to FastAPI backend and return JSON payload."""
    response = requests.post(
        API_URL,
        json={"task": task, "fast_demo_mode": fast_demo_mode},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _check_backend_health() -> str:
    """Return backend health state for lightweight status polling."""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if response.ok:
            return "Online"
    except requests.RequestException:
        pass
    return "Offline"


def _run_pipeline_background(task: str, fast_demo_mode: bool, output: Dict[str, Any]) -> None:
    """Execute pipeline in worker thread and write to a plain dict."""
    try:
        output["payload"] = _call_pipeline(task, fast_demo_mode)
        output["fast_demo_mode"] = fast_demo_mode
    except requests.RequestException as exc:
        output["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        output["error"] = str(exc)


def _finalize_worker_if_done() -> None:
    """Sync worker result back to session state from main thread only."""
    worker = st.session_state.get("pipeline_worker")
    output = st.session_state.get("pipeline_output", {})
    if not worker or worker.is_alive():
        return

    payload = output.get("payload")
    if payload:
        st.session_state["pipeline_payload"] = payload
        st.session_state["pipeline_result"] = payload.get("result", {})
        st.session_state["last_fast_demo_mode"] = output.get("fast_demo_mode", True)
        status = payload.get("status")
        if status in {"failed", "already_running"}:
            st.session_state["pipeline_error"] = payload.get(
                "error",
                f"Pipeline request returned status '{status}'.",
            )
        else:
            st.session_state["pipeline_error"] = None
    elif output.get("error"):
        st.session_state["pipeline_error"] = output["error"]

    st.session_state["pipeline_running"] = False
    st.session_state["pipeline_worker"] = None
    st.session_state["pipeline_output"] = {}


def _normalize_model_name(model_name: str) -> str:
    """Convert API model names to the naming convention used in leaderboard and file names."""
    mapping = {
        "LinearRegression": "Linear Regression",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
    }
    return mapping.get(model_name, model_name)


def _model_file_for_name(model_name: str) -> Path:
    """Return the saved model file path in storage/models for a model name."""
    filename_map = {
        "LinearRegression": "linearregression.joblib",
        "Linear Regression": "linearregression.joblib",
        "RandomForest": "randomforest.joblib",
        "Random Forest": "randomforest.joblib",
        "XGBoost": "xgboost.joblib",
    }
    filename = filename_map.get(model_name, f"{model_name.lower().replace(' ', '')}.joblib")
    return PROJECT_ROOT / "storage" / "models" / filename


def _parse_leaderboard(report_path: str, best_model_name: str) -> pd.DataFrame:
    """Read model comparison table from report markdown and build leaderboard dataframe."""
    normalized_best = _normalize_model_name(best_model_name)
    path = Path(report_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / report_path

    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            if "Model" in line or "---" in line:
                continue
            cells = [part.strip() for part in line.strip("|").split("|")]
            if len(cells) != 4:
                continue
            model = cells[0]
            try:
                rows.append(
                    {
                        "Model": model,
                        "RMSE": float(cells[1]),
                        "MAE": float(cells[2]),
                        "R2": float(cells[3]),
                    }
                )
            except ValueError:
                continue

    # Fallback rows guarantee a usable leaderboard even before report generation.
    if not rows:
        rows = [
            {"Model": "Linear Regression", "RMSE": None, "MAE": None, "R2": None},
            {"Model": "Random Forest", "RMSE": None, "MAE": None, "R2": None},
            {"Model": "XGBoost", "RMSE": None, "MAE": None, "R2": None},
        ]

    leaderboard = pd.DataFrame(rows)
    leaderboard["Best"] = leaderboard["Model"].apply(
        lambda name: "🏆 Best" if name == normalized_best else ""
    )
    return leaderboard


def _render_feature_importance(best_model_name: str) -> None:
    """Render feature importance chart for tree-based models."""
    normalized_best = _normalize_model_name(best_model_name)
    if normalized_best not in {"Random Forest", "XGBoost"}:
        st.info("Feature importance is available for Random Forest and XGBoost models.")
        return

    model_path = _model_file_for_name(normalized_best)
    if not model_path.exists():
        st.info(f"Model artifact not found at `{model_path}`.")
        return

    try:
        pipeline = joblib.load(model_path)
        preprocessor = pipeline.named_steps.get("preprocessor")
        model = pipeline.named_steps.get("model")

        if not hasattr(model, "feature_importances_"):
            st.info("The selected model does not expose feature importances.")
            return

        importances = model.feature_importances_
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = list(preprocessor.get_feature_names_out())
        else:
            feature_names = [f"feature_{idx}" for idx in range(len(importances))]

        feature_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(15)
            .sort_values("Importance", ascending=True)
        )

        fig = px.bar(
            feature_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top Feature Importances - {normalized_best}",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=500, margin=dict(l=8, r=8, t=48, b=8))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not render feature importance chart: {exc}")


def _render_report_section(report_path: str) -> None:
    """Display report path, open button, and markdown preview."""
    st.markdown("### 📄 Generated Report", help="Pipeline-generated markdown report.")
    path = Path(report_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / report_path

    st.code(str(path), language="text")
    if path.exists():
        st.link_button("Open Model Report", f"file://{path.resolve()}")
        with st.expander("Preview Report Content"):
            st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.info("Report file not found yet. Run the pipeline to generate it.")


def _format_currency_metric(value: Any) -> str:
    """Format RMSE/MAE with currency style for dashboard readability."""
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_r2_metric(value: Any) -> str:
    """Format R2 with three decimal places."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def main() -> None:
    """Render the dashboard and handle all user interactions."""
    st.set_page_config(page_title="Autonomous Data Science Lab", layout="wide")
    _inject_styles()

    st.title("Autonomous Data Science Lab")
    st.caption(
        "An AI analytics platform that orchestrates dataset ingestion, cleaning, feature engineering, "
        "model training, evaluation, and reporting."
    )

    # Initialize runtime state for asynchronous pipeline execution.
    st.session_state.setdefault("pipeline_running", False)
    st.session_state.setdefault("pipeline_started_at", None)
    st.session_state.setdefault("pipeline_error", None)
    st.session_state.setdefault("last_fast_demo_mode", True)
    st.session_state.setdefault("pipeline_worker", None)
    st.session_state.setdefault("pipeline_output", {})

    # Resolve completed worker runs first so UI cannot remain stuck in running.
    _finalize_worker_if_done()

    # Top layout: controls on left, live status cards on right.
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### 🎯 Pipeline Task")
        task = st.text_input("Enter ML task", value="Predict house prices")
        fast_demo_mode = st.toggle(
            "Fast Demo Mode",
            value=True,
            help="ON: speed-optimized training profile. OFF: fuller training profile.",
        )
        run_clicked = st.button("Run AI Pipeline 🚀", type="primary", use_container_width=True)
        auto_refresh = st.toggle(
            "Auto-refresh while pipeline is running",
            value=True,
            help="Refreshes the dashboard every 2 seconds during execution.",
        )
    with right:
        st.markdown("### ⚙️ Pipeline Snapshot")
        previous_result = st.session_state.get("pipeline_result", {})
        if st.session_state.get("pipeline_running"):
            previous_status = "running"
        else:
            previous_status = previous_result.get("pipeline_status", "Not started")
        previous_model = previous_result.get("best_model_name", "N/A")
        backend_state = _check_backend_health()
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Pipeline Status</div>
                <div class="card-value">{previous_status}</div>
            </div>
            <div class="card">
                <div class="card-title">Best Model</div>
                <div class="card-value">🏆 {_normalize_model_name(previous_model)}</div>
            </div>
            <div class="card">
                <div class="card-title">Backend Health</div>
                <div class="card-value">🩺 {backend_state}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Trigger asynchronous pipeline execution.
    if run_clicked:
        if not task.strip():
            st.warning("Please enter a valid task before running the pipeline.")
        elif st.session_state.get("pipeline_running"):
            st.info("Pipeline is already running. Please wait for completion.")
        else:
            st.session_state["pipeline_running"] = True
            st.session_state["pipeline_started_at"] = time.time()
            st.session_state["pipeline_error"] = None
            st.session_state["pipeline_output"] = {}
            worker = threading.Thread(
                target=_run_pipeline_background,
                args=(task.strip(), fast_demo_mode, st.session_state["pipeline_output"]),
                daemon=True,
            )
            st.session_state["pipeline_worker"] = worker
            worker.start()

    if st.session_state.get("pipeline_running"):
        started_at = st.session_state.get("pipeline_started_at") or time.time()
        elapsed = int(time.time() - started_at)
        worker = st.session_state.get("pipeline_worker")

        # Client-side timeout safeguard to prevent permanent running UI state.
        if elapsed > UI_PIPELINE_TIMEOUT_SECONDS:
            st.session_state["pipeline_running"] = False
            st.session_state["pipeline_worker"] = None
            st.session_state["pipeline_output"] = {}
            st.session_state["pipeline_error"] = (
                f"UI timeout after {UI_PIPELINE_TIMEOUT_SECONDS}s. "
                "Please retry the pipeline run."
            )
            st.rerun()

        # If thread finished between reruns, finalize immediately.
        if worker is not None and not worker.is_alive():
            _finalize_worker_if_done()
            st.rerun()

        with st.spinner("Running AI pipeline..."):
            st.info(f"Pipeline is running in background. Elapsed time: {elapsed}s")

        # Auto-refresh gives a near-real-time status experience during execution.
        if auto_refresh:
            time.sleep(2)
            st.rerun()

    payload = st.session_state.get("pipeline_payload")
    pipeline_error = st.session_state.get("pipeline_error")
    if pipeline_error and not st.session_state.get("pipeline_running"):
        st.error(f"Pipeline request failed: {pipeline_error}")
        st.info("Start backend with: `uvicorn api.main:app --reload`")

    if not payload:
        st.info("Run the pipeline to view analytics, leaderboard, charts, and report output.")
        return

    result = payload.get("result", {})
    best_model_name = result.get("best_model_name", "N/A")
    metrics = result.get("best_model_metrics", {})
    report_path = result.get("report_path", "reports/model_report.md")
    mode_label = (
        "Demo Fast"
        if st.session_state.get("last_fast_demo_mode", True)
        else "Standard"
    )

    st.markdown("### 🧭 Run Profile")
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Latest Pipeline Mode</div>
            <div class="card-value">⚡ {mode_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key metric cards.
    st.markdown("### 📊 Model Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", _format_currency_metric(metrics.get("rmse")))
    c2.metric("MAE", _format_currency_metric(metrics.get("mae")))
    c3.metric("R2", _format_r2_metric(metrics.get("r2")))

    # Simple metrics visualization for the selected best model.
    st.markdown("### 📈 Metrics Visualization")
    metric_df = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "R2"],
            "Value": [
                metrics.get("rmse"),
                metrics.get("mae"),
                metrics.get("r2"),
            ],
        }
    ).set_index("Metric")
    st.bar_chart(metric_df)

    # Leaderboard section with best-model highlighting.
    st.markdown("### 🏆 Model Leaderboard")
    leaderboard_df = _parse_leaderboard(report_path, best_model_name)
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    # Feature importance for tree-based best models.
    st.markdown("### 🔍 Feature Importance")
    _render_feature_importance(best_model_name)

    # Experiment history quick access.
    st.markdown("### 🧪 Experiment History (MLflow)")
    st.link_button("View full experiment history", MLFLOW_UI_URL)

    # Report section with open button and markdown preview.
    _render_report_section(report_path)


if __name__ == "__main__":
    main()
