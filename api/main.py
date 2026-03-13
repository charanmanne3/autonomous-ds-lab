import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from memory import ExperimentMemory
from orchestrator.task_planner import TaskPlanner

app = FastAPI(title="Autonomous Data Science Lab API", version="1.0.0")
planner = TaskPlanner()
experiment_memory = ExperimentMemory()
logger = logging.getLogger("api.main")

PIPELINE_TIMEOUT_SECONDS = 60
pipeline_lock = threading.Lock()
pipeline_running = False
pipeline_started_at: float | None = None
pipeline_progress: Dict[str, Any] = {
    "running": False,
    "current_step": 0,
    "total_steps": 6,
    "message": "Idle",
    "status": "idle",
}


class PipelineRequest(BaseModel):
    task: str = Field(..., description="Task description. Example: Predict house prices")
    fast_demo_mode: bool = Field(
        True,
        description="If true, runs a speed-optimized demo training profile.",
    )
    dataset_csv: str | None = Field(
        default=None,
        description="Optional CSV content from dashboard upload.",
    )
    target_column: str | None = Field(
        default=None,
        description="Target column name for uploaded datasets.",
    )
    task_type: str | None = Field(
        default=None,
        description="Optional override: Regression or Classification. If empty, planner infers.",
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "pipeline_running": pipeline_running,
        "pipeline_progress": pipeline_progress,
    }


@app.get("/pipeline-status")
def pipeline_status():
    return pipeline_progress


@app.get("/plan-task")
def plan_task(task: str, task_type: str | None = None, target: str | None = None):
    """Preview planner output before running the pipeline."""
    return planner.create_plan(task_description=task, task_type=task_type, target_hint=target)


@app.get("/memory/similar")
def memory_similar(task: str, limit: int = 3):
    similar = experiment_memory.query_similar(task=task, limit=limit)
    return {
        "similar_experiments": similar,
        "recommended_models": experiment_memory.recommend_models(task=task, limit=limit),
        "ai_insight": experiment_memory.generate_insight(task=task),
    }


@app.get("/memory/history")
def memory_history(limit: int = 20):
    return {"experiment_history": experiment_memory.get_history(limit=limit)}


def _execute_pipeline(
    task_description: str,
    fast_demo_mode: bool,
    dataset_csv: str | None,
    target_column: str | None,
    task_type: str,
    output: Dict[str, Any],
) -> None:
    """Run the planner and pass result/errors via a shared dict."""
    def _progress_cb(step_idx: int, total_steps: int, message: str) -> None:
        pipeline_progress["running"] = True
        pipeline_progress["current_step"] = step_idx
        pipeline_progress["total_steps"] = total_steps
        pipeline_progress["message"] = message
        pipeline_progress["status"] = "running"

    try:
        output["result"] = planner.run_pipeline(
            task_description=task_description,
            fast_demo_mode=fast_demo_mode,
            dataset_csv=dataset_csv,
            target_column=target_column,
            task_type=task_type,
            progress_callback=_progress_cb,
        )
    except Exception as exc:  # noqa: BLE001
        output["error"] = str(exc)


@app.post("/run-pipeline")
def run_pipeline(payload: PipelineRequest):
    global pipeline_running
    global pipeline_started_at

    now = time.time()
    # Timeout safeguard for stale state from previous runs.
    if pipeline_running and pipeline_started_at and now - pipeline_started_at > PIPELINE_TIMEOUT_SECONDS:
        logger.warning("Resetting stale pipeline state after timeout safeguard.")
        pipeline_running = False
        pipeline_started_at = None
        pipeline_progress.update(
            {
                "running": False,
                "status": "failed",
                "message": f"Timed out after {PIPELINE_TIMEOUT_SECONDS}s",
                "current_step": 0,
                "total_steps": 6,
            }
        )
        if pipeline_lock.locked():
            try:
                pipeline_lock.release()
            except RuntimeError:
                # Lock may already be unlocked by another request path.
                pass

    if not pipeline_lock.acquire(blocking=False):
        pipeline_progress.update(
            {
                "running": True,
                "status": "already_running",
                "message": "A pipeline run is already in progress.",
            }
        )
        return {
            "status": "already_running",
            "error": "A pipeline run is already in progress. Please wait and retry.",
        }

    pipeline_running = True
    pipeline_started_at = time.time()
    initial_plan = planner.create_plan(
        task_description=payload.task,
        task_type=payload.task_type,
        target_hint=payload.target_column,
    )
    pipeline_progress.update(
        {
            "running": True,
            "current_step": 0,
            "total_steps": 6,
            "message": "Pipeline request accepted",
            "status": "running",
            "task_type": initial_plan.get("task_type"),
            "pipeline_plan": initial_plan.get("pipeline_steps"),
        }
    )
    try:
        output: Dict[str, Any] = {}
        worker = threading.Thread(
            target=_execute_pipeline,
            args=(
                payload.task,
                payload.fast_demo_mode,
                payload.dataset_csv,
                payload.target_column,
                payload.task_type,
                output,
            ),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=PIPELINE_TIMEOUT_SECONDS)

        if worker.is_alive():
            logger.error("Pipeline execution timed out after %s seconds.", PIPELINE_TIMEOUT_SECONDS)
            pipeline_progress.update(
                {
                    "running": False,
                    "status": "failed",
                    "message": f"Timed out after {PIPELINE_TIMEOUT_SECONDS}s",
                }
            )
            return {
                "status": "failed",
                "error": f"Pipeline execution timed out after {PIPELINE_TIMEOUT_SECONDS} seconds.",
            }

        if "error" in output:
            logger.error("Pipeline execution failed: %s", output["error"])
            pipeline_progress.update(
                {
                    "running": False,
                    "status": "failed",
                    "message": output["error"],
                }
            )
            return {"status": "failed", "error": output["error"]}

        result = output.get("result", {})
        metrics = result.get("best_model_metrics", {})
        experiment_memory.save_experiment(
            {
                "task": result.get("task", payload.task),
                "dataset_name": result.get("dataset_name", "unknown"),
                "best_model": result.get("best_model_name"),
                "rmse": metrics.get("rmse", 0.0),
                "mae": metrics.get("mae", 0.0),
                "r2": metrics.get("r2", 0.0),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        similar_experiments = experiment_memory.query_similar(task=result.get("task", payload.task), limit=3)
        recommended_models = experiment_memory.recommend_models(task=result.get("task", payload.task), limit=3)
        ai_insight = experiment_memory.generate_insight(task=result.get("task", payload.task))
        experiment_history = experiment_memory.get_history(limit=20)
        pipeline_progress.update(
            {
                "running": False,
                "status": "completed",
                "current_step": 6,
                "total_steps": 6,
                "message": "Pipeline completed successfully",
            }
        )
        return {
            "status": "completed",
            "task": result.get("task"),
            "task_type": result.get("task_type"),
            "pipeline_plan": result.get("pipeline_plan", []),
            "best_model": result.get("best_model_name"),
            "metrics": result.get("best_model_metrics", {}),
            "model_comparison": result.get("model_comparison", []),
            "report_path": result.get("report_path"),
            "similar_experiments": similar_experiments,
            "recommended_models": recommended_models,
            "ai_insight": ai_insight,
            "experiment_history": experiment_history,
            "result": result,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pipeline endpoint failure: %s", exc)
        pipeline_progress.update(
            {
                "running": False,
                "status": "failed",
                "message": str(exc),
            }
        )
        return {"status": "failed", "error": str(exc)}
    finally:
        # Always reset in finally so dashboard cannot remain in a permanent running state.
        pipeline_running = False
        pipeline_started_at = None
        if pipeline_lock.locked():
            pipeline_lock.release()

