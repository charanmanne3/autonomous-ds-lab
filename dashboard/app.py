"""Modern Streamlit dashboard for Autonomous Data Science Lab."""

import json
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
CHAT_URL = "http://127.0.0.1:8000/chat"
CHAT_STREAM_URL = "http://127.0.0.1:8000/chat/stream"
STATUS_URL = "http://127.0.0.1:8000/pipeline-status"
PLAN_URL = "http://127.0.0.1:8000/plan-task"
MEMORY_SIMILAR_URL = "http://127.0.0.1:8000/memory/similar"
MEMORY_HISTORY_URL = "http://127.0.0.1:8000/memory/history"
MLFLOW_UI_URL = "http://127.0.0.1:5000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_PIPELINE_TIMEOUT_SECONDS = 90
STEP_LABELS = {
    "dataset_ingestion": "Dataset Ingestion",
    "data_cleaning": "Data Cleaning",
    "feature_engineering": "Feature Engineering",
    "model_training": "Model Training",
    "model_evaluation": "Model Evaluation",
    "report_generation": "Report Generation",
}


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


def _call_pipeline(
    task: str,
    fast_demo_mode: bool,
    dataset_csv: str | None,
    target_column: str | None,
    task_type: str,
) -> Dict[str, Any]:
    """Send pipeline request to FastAPI backend and return JSON payload."""
    response = requests.post(
        API_URL,
        json={
            "task": task,
            "fast_demo_mode": fast_demo_mode,
            "dataset_csv": dataset_csv,
            "target_column": target_column,
            "task_type": task_type,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _get_pipeline_status() -> Dict[str, Any]:
    """Fetch real-time pipeline step status from backend."""
    try:
        response = requests.get(STATUS_URL, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {
            "running": False,
            "current_step": 0,
            "total_steps": 6,
            "message": "Status unavailable",
            "status": "unknown",
            "pipeline_plan": [],
        }


def _chat_with_backend(message: str) -> Dict[str, Any]:
    """Send chat message to backend LangGraph workflow."""
    response = requests.post(CHAT_URL, json={"message": message}, timeout=180)
    response.raise_for_status()
    return response.json()


def _stream_chat_with_backend(message: str):
    """Stream NDJSON events from backend chat endpoint."""
    with requests.post(
        CHAT_STREAM_URL,
        json={"message": message},
        timeout=300,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _get_pipeline_plan(task: str, task_type: str | None, target: str | None) -> Dict[str, Any]:
    """Fetch planner-generated pipeline plan from backend."""
    try:
        response = requests.get(
            PLAN_URL,
            params={"task": task, "task_type": task_type, "target": target},
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {
            "task": task,
            "task_type": (task_type or "regression").lower(),
            "target": target or "target",
            "pipeline_steps": [
                "dataset_ingestion",
                "data_cleaning",
                "feature_engineering",
                "model_training",
                "model_evaluation",
                "report_generation",
            ],
        }


def _get_memory_preview(task: str) -> Dict[str, Any]:
    """Fetch similar experiments and recommendations for a given task."""
    try:
        response = requests.get(MEMORY_SIMILAR_URL, params={"task": task, "limit": 3}, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {
            "similar_experiments": [],
            "recommended_models": [],
            "ai_insight": "Memory service unavailable.",
        }


def _get_experiment_history(limit: int = 20) -> list[dict]:
    """Fetch experiment history table from backend memory."""
    try:
        response = requests.get(MEMORY_HISTORY_URL, params={"limit": limit}, timeout=5)
        response.raise_for_status()
        return response.json().get("experiment_history", [])
    except requests.RequestException:
        return []


def _check_backend_health() -> str:
    """Return backend health state for lightweight status polling."""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if response.ok:
            return "Online"
    except requests.RequestException:
        pass
    return "Offline"


def _run_pipeline_background(
    task: str,
    fast_demo_mode: bool,
    dataset_csv: str | None,
    target_column: str | None,
    task_type: str,
    output: Dict[str, Any],
) -> None:
    """Execute pipeline in worker thread and write to a plain dict."""
    try:
        output["payload"] = _call_pipeline(
            task=task,
            fast_demo_mode=fast_demo_mode,
            dataset_csv=dataset_csv,
            target_column=target_column,
            task_type=task_type,
        )
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
    }
    return mapping.get(model_name, model_name)


def _model_file_for_name(model_name: str) -> Path:
    """Return the saved model file path in storage/models for a model name."""
    filename_map = {
        "LinearRegression": "linearregression.joblib",
        "Linear Regression": "linearregression.joblib",
        "RandomForest": "randomforest.joblib",
        "Random Forest": "randomforest.joblib",
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
        ]

    leaderboard = pd.DataFrame(rows)
    leaderboard["Best"] = leaderboard["Model"].apply(
        lambda name: "🏆 Best" if name == normalized_best else ""
    )
    return leaderboard


def _build_comparison_df(model_comparison: list[dict], best_model_name: str) -> pd.DataFrame:
    """Create normalized model comparison dataframe from API payload."""
    if not model_comparison:
        return _parse_leaderboard("reports/model_report.md", best_model_name)

    rows = []
    for row in model_comparison:
        rows.append(
            {
                "Model": _normalize_model_name(row.get("model", "")),
                "RMSE": row.get("rmse"),
                "MAE": row.get("mae"),
                "R2": row.get("r2"),
            }
        )
    df = pd.DataFrame(rows)
    df["Best"] = df["Model"].apply(
        lambda name: "🏆 Best" if name == _normalize_model_name(best_model_name) else ""
    )
    return df


def _render_pipeline_plan(plan_steps: list[str]) -> None:
    """Render numbered pipeline plan for transparency."""
    st.markdown("### 🧠 Pipeline Plan")
    for idx, step in enumerate(plan_steps, start=1):
        st.markdown(f"{idx}. {STEP_LABELS.get(step, step.replace('_', ' ').title())}")


def _render_memory_sections(
    recommended_models: list[str],
    ai_insight: str,
    experiment_history: list[dict],
) -> None:
    """Render recommendations, AI insight, and experiment history UI."""
    st.markdown("### 🤖 Model Recommendation System")
    if recommended_models:
        st.markdown(
            "\n".join([f"- `{_normalize_model_name(model)}`" for model in recommended_models])
        )
    else:
        st.caption("No recommendations yet. Run more experiments to build memory.")

    st.markdown("### 💡 AI Insights")
    st.info(ai_insight)

    st.markdown("### 🗂️ Experiment History")
    if experiment_history:
        history_df = pd.DataFrame(experiment_history)
        keep_cols = ["task", "best_model", "rmse", "r2", "timestamp"]
        history_df = history_df[[col for col in keep_cols if col in history_df.columns]].copy()
        history_df = history_df.rename(
            columns={
                "task": "Task",
                "best_model": "Best Model",
                "rmse": "RMSE",
                "r2": "R2",
                "timestamp": "Date",
            }
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No experiment history yet.")


def _render_chatbot_interface() -> None:
    """Render ChatGPT-style interface backed by /chat endpoint."""
    st.markdown("### 🤖 LLM Multi-Agent Chatbot")
    st.caption(
        "Powered by LangChain + LangGraph agents: Research -> Analysis -> Code -> Visualization -> Report."
    )

    st.session_state.setdefault("messages", [])
    if not st.session_state["messages"]:
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "I am your Autonomous DS Lab assistant. Ask me to analyze datasets, suggest models, "
                    "or generate ML implementation plans."
                ),
            }
        )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask a data science or ML engineering question...")
    if not user_prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        progress_placeholder = st.empty()
        response_placeholder = st.empty()
        assistant_text = ""
        similar_history = []

        try:
            for event in _stream_chat_with_backend(user_prompt):
                event_type = event.get("type")
                if event_type == "status":
                    progress_placeholder.info(event.get("content", "Running agents..."))
                elif event_type == "token":
                    assistant_text += event.get("content", "")
                    response_placeholder.markdown(assistant_text)
                elif event_type == "done":
                    similar_history = event.get("similar_history", [])
                    break
                elif event_type == "error":
                    assistant_text = event.get("content", "Unknown error")
                    response_placeholder.error(assistant_text)
                    break
        except requests.RequestException as exc:
            assistant_text = f"Chat backend error: {exc}"
            response_placeholder.error(assistant_text)

        if not assistant_text:
            assistant_text = "No response generated."
            response_placeholder.markdown(assistant_text)

        if similar_history:
            with st.expander("Retrieved similar memory"):
                for item in similar_history:
                    st.markdown(f"- **Q:** {item.get('question', '')}")
                    st.markdown(f"  **A:** {item.get('response', '')[:250]}...")
    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})


def _render_workflow_diagram(plan_steps: list[str], current_step: int) -> None:
    """Render lightweight workflow diagram with active step highlighting."""
    st.markdown("### 🔄 Pipeline Workflow")
    chips = []
    for idx, step in enumerate(plan_steps, start=1):
        label = STEP_LABELS.get(step, step.replace("_", " ").title())
        if idx < current_step:
            color = "#16a34a"
            icon = "✓"
        elif idx == current_step:
            color = "#ea580c"
            icon = "▶"
        else:
            color = "#334155"
            icon = "•"
        chips.append(
            f"<span style='background:{color};color:white;padding:6px 10px;border-radius:8px;'>"
            f"{icon} {label}</span>"
        )
    st.markdown(" ➜ ".join(chips), unsafe_allow_html=True)


def _render_feature_importance(best_model_name: str) -> None:
    """Render feature importance chart for tree-based models."""
    normalized_best = _normalize_model_name(best_model_name)
    if normalized_best not in {"Random Forest"}:
        st.info("Feature importance is available for Random Forest model.")
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
    app_mode = st.radio(
        "Mode",
        options=["LLM Chatbot", "Pipeline Lab"],
        horizontal=True,
        index=0,
    )
    if app_mode == "LLM Chatbot":
        _render_chatbot_interface()
        return

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
        task_type_override = st.selectbox(
            "Task Type Override",
            options=["Auto", "Regression", "Classification"],
            index=0,
            help="Auto uses the AI Task Planner keyword inference.",
        )
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        uploaded_target_col = None
        uploaded_dataset_csv = None
        if uploaded_file is not None:
            uploaded_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            uploaded_dataset_csv = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            uploaded_target_col = st.selectbox(
                "Select Target Column",
                options=list(uploaded_preview.columns),
            )
            st.caption(f"Uploaded rows: {len(uploaded_preview)} | columns: {len(uploaded_preview.columns)}")

        planner_task_type = None if task_type_override == "Auto" else task_type_override
        plan_preview = _get_pipeline_plan(task, planner_task_type, uploaded_target_col)
        memory_preview = _get_memory_preview(task)
        history_preview = _get_experiment_history(limit=10)
        _render_pipeline_plan(plan_preview.get("pipeline_steps", []))
        st.caption(
            f"Planner inferred task type: **{plan_preview.get('task_type', 'regression')}** | "
            f"target hint: **{plan_preview.get('target', 'target')}**"
        )
        _render_memory_sections(
            recommended_models=memory_preview.get("recommended_models", []),
            ai_insight=memory_preview.get("ai_insight", "No insights available yet."),
            experiment_history=history_preview,
        )

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
        elif uploaded_file is not None and not uploaded_target_col:
            st.warning("Please select a target column for the uploaded CSV.")
        elif st.session_state.get("pipeline_running"):
            st.info("Pipeline is already running. Please wait for completion.")
        else:
            st.session_state["pipeline_running"] = True
            st.session_state["pipeline_started_at"] = time.time()
            st.session_state["pipeline_error"] = None
            st.session_state["pipeline_output"] = {}
            worker = threading.Thread(
                target=_run_pipeline_background,
                args=(
                    task.strip(),
                    fast_demo_mode,
                    uploaded_dataset_csv,
                    uploaded_target_col,
                    planner_task_type or plan_preview.get("task_type", "regression"),
                    st.session_state["pipeline_output"],
                ),
                daemon=True,
            )
            st.session_state["pipeline_worker"] = worker
            worker.start()

    if st.session_state.get("pipeline_running"):
        started_at = st.session_state.get("pipeline_started_at") or time.time()
        elapsed = int(time.time() - started_at)
        worker = st.session_state.get("pipeline_worker")
        backend_progress = _get_pipeline_status()
        plan_steps = backend_progress.get("pipeline_plan", plan_preview.get("pipeline_steps", []))

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

        st.markdown("### 🚦 Pipeline Status")
        progress_total = max(int(backend_progress.get("total_steps", 6)), 1)
        progress_step = int(backend_progress.get("current_step", 0))
        st.progress(min(progress_step / progress_total, 1.0))
        st.caption(backend_progress.get("message", "Running pipeline..."))
        _render_workflow_diagram(plan_steps, max(progress_step, 1))
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
    best_model_name = payload.get("best_model") or result.get("best_model_name", "N/A")
    metrics = payload.get("metrics") or result.get("best_model_metrics", {})
    report_path = payload.get("report_path") or result.get("report_path", "reports/model_report.md")
    model_comparison = payload.get("model_comparison") or result.get("model_comparison", [])
    recommended_models = payload.get("recommended_models", [])
    ai_insight = payload.get("ai_insight", "No insights available yet.")
    experiment_history = payload.get("experiment_history", [])
    task_type_used = payload.get("task_type") or result.get("task_type", "regression")
    plan_used = payload.get("pipeline_plan") or result.get("pipeline_plan", plan_preview.get("pipeline_steps", []))
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
    st.caption(f"Task type used: **{task_type_used}**")
    _render_workflow_diagram(plan_used, len(plan_used))

    st.markdown("### 🏁 Best Model")
    st.success(f"Best model selected: {_normalize_model_name(best_model_name)}")

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
    st.markdown("### 🏆 Model Comparison")
    leaderboard_df = _build_comparison_df(model_comparison, best_model_name)
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    # Model-wise RMSE comparison chart for quick performance scan.
    rmse_chart_df = leaderboard_df[["Model", "RMSE"]].dropna()
    if not rmse_chart_df.empty:
        rmse_chart_df = rmse_chart_df.set_index("Model")
        st.bar_chart(rmse_chart_df)

    # Feature importance for tree-based best models.
    st.markdown("### 🔍 Feature Importance")
    _render_feature_importance(best_model_name)

    # Experiment history quick access.
    st.markdown("### 🧪 Experiment History (MLflow)")
    st.link_button("View full experiment history", MLFLOW_UI_URL)

    _render_memory_sections(
        recommended_models=recommended_models,
        ai_insight=ai_insight,
        experiment_history=experiment_history,
    )

    # Report section with open button and markdown preview.
    _render_report_section(report_path)


if __name__ == "__main__":
    main()
