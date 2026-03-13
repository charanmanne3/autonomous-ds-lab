import logging
import threading
import time
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from orchestrator.task_planner import TaskPlanner

app = FastAPI(title="Autonomous Data Science Lab API", version="1.0.0")
planner = TaskPlanner()
logger = logging.getLogger("api.main")

PIPELINE_TIMEOUT_SECONDS = 60
pipeline_lock = threading.Lock()
pipeline_running = False
pipeline_started_at: float | None = None


class PipelineRequest(BaseModel):
    task: str = Field(..., description="Task description. Example: Predict house prices")
    fast_demo_mode: bool = Field(
        True,
        description="If true, runs a speed-optimized demo training profile.",
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_running": pipeline_running}


def _execute_pipeline(task_description: str, fast_demo_mode: bool, output: Dict[str, Any]) -> None:
    """Run the planner and pass result/errors via a shared dict."""
    try:
        output["result"] = planner.run_pipeline(
            task_description=task_description,
            fast_demo_mode=fast_demo_mode,
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
        if pipeline_lock.locked():
            try:
                pipeline_lock.release()
            except RuntimeError:
                # Lock may already be unlocked by another request path.
                pass

    if not pipeline_lock.acquire(blocking=False):
        return {
            "status": "already_running",
            "error": "A pipeline run is already in progress. Please wait and retry.",
        }

    pipeline_running = True
    pipeline_started_at = time.time()
    try:
        output: Dict[str, Any] = {}
        worker = threading.Thread(
            target=_execute_pipeline,
            args=(payload.task, payload.fast_demo_mode, output),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=PIPELINE_TIMEOUT_SECONDS)

        if worker.is_alive():
            logger.error("Pipeline execution timed out after %s seconds.", PIPELINE_TIMEOUT_SECONDS)
            return {
                "status": "failed",
                "error": f"Pipeline execution timed out after {PIPELINE_TIMEOUT_SECONDS} seconds.",
            }

        if "error" in output:
            logger.error("Pipeline execution failed: %s", output["error"])
            return {"status": "failed", "error": output["error"]}

        return {"status": "completed", "result": output.get("result", {})}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pipeline endpoint failure: %s", exc)
        return {"status": "failed", "error": str(exc)}
    finally:
        # Always reset in finally so dashboard cannot remain in a permanent running state.
        pipeline_running = False
        pipeline_started_at = None
        if pipeline_lock.locked():
            pipeline_lock.release()

