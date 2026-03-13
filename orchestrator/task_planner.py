import logging
from typing import Any, Callable, Dict, Optional

from registry.agent_registry import get_agent_registry


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


class TaskPlanner:
    """Task planner + orchestrator for autonomous DS workflow."""

    DEFAULT_PIPELINE_STEPS = [
        "dataset_ingestion",
        "data_cleaning",
        "feature_engineering",
        "model_training",
        "model_evaluation",
        "report_generation",
    ]

    def __init__(self) -> None:
        configure_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = get_agent_registry()

    def create_plan(
        self,
        task_description: str,
        task_type: Optional[str] = None,
        target_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an AI task plan from natural language user intent."""
        task_text = task_description.strip().lower()
        regression_keywords = {"predict", "forecast"}
        classification_keywords = {"classify", "detect"}

        if task_type and task_type.strip().lower() in {"regression", "classification"}:
            inferred_task_type = task_type.strip().lower()
        elif any(keyword in task_text for keyword in classification_keywords):
            inferred_task_type = "classification"
        elif any(keyword in task_text for keyword in regression_keywords):
            inferred_task_type = "regression"
        else:
            inferred_task_type = "regression"

        inferred_target = target_hint
        if not inferred_target:
            if "price" in task_text or "prices" in task_text:
                inferred_target = "price"
            elif "class" in task_text or "label" in task_text:
                inferred_target = "label"
            elif task_text.startswith("predict "):
                inferred_target = task_text.replace("predict ", "", 1).strip().split()[-1]
            elif task_text.startswith("forecast "):
                inferred_target = task_text.replace("forecast ", "", 1).strip().split()[-1]
            else:
                inferred_target = "target"

        plan = {
            "task": task_description,
            "task_type": inferred_task_type,
            "target": inferred_target,
            "pipeline_steps": self.DEFAULT_PIPELINE_STEPS.copy(),
        }
        self.logger.info("Generated plan: %s", plan)
        return plan

    def run_pipeline(
        self,
        task_description: str,
        fast_demo_mode: bool = True,
        dataset_csv: Optional[str] = None,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        plan = self.create_plan(
            task_description=task_description,
            task_type=task_type,
            target_hint=target_column,
        )
        self.logger.info(
            "Pipeline started for task: %s | fast_demo_mode=%s | task_type=%s",
            task_description,
            fast_demo_mode,
            plan["task_type"],
        )
        total_steps = 6

        def _emit(step_idx: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(step_idx, total_steps, message)

        try:
            _emit(1, "[1/6] Loading dataset")
            self.logger.info("Step 1/6: dataset_agent")
            dataset_output = self.registry["dataset"].run(
                {
                    "task": task_description,
                    "dataset_csv": dataset_csv,
                    "target_column": target_column,
                    "task_type": plan["task_type"],
                }
            )
            dataset_output["runtime_config"] = {
                "fast_demo_mode": fast_demo_mode,
                "task_plan": plan,
            }
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at dataset_agent: %s", exc)
            raise

        try:
            _emit(2, "[2/6] Cleaning data")
            self.logger.info("Step 2/6: cleaning_agent")
            cleaning_output = self.registry["cleaning"].run(dataset_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at cleaning_agent: %s", exc)
            raise

        try:
            _emit(3, "[3/6] Creating features")
            self.logger.info("Step 3/6: feature_agent")
            feature_output = self.registry["feature"].run(cleaning_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at feature_agent: %s", exc)
            raise

        try:
            _emit(4, "[4/6] Training models")
            self.logger.info("Step 4/6: training_agent")
            training_output = self.registry["training"].run(feature_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at training_agent: %s", exc)
            raise

        try:
            _emit(5, "[5/6] Evaluating models")
            self.logger.info("Step 5/6: evaluation_agent")
            evaluation_output = self.registry["evaluation"].run(training_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at evaluation_agent: %s", exc)
            raise

        try:
            _emit(6, "[6/6] Generating report")
            self.logger.info("Step 6/6: report_agent")
            report_output = self.registry["report"].run(evaluation_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at report_agent: %s", exc)
            raise

        self.logger.info("Pipeline completed successfully")
        return {
            "task": task_description,
            "task_type": plan["task_type"],
            "pipeline_plan": plan["pipeline_steps"],
            "plan_target": plan["target"],
            "dataset_name": dataset_output.get("metadata", {}).get("name", "unknown"),
            "pipeline_status": report_output["status"],
            "best_model_name": report_output["best_model_name"],
            "best_score": report_output["best_score"],
            "best_model_metrics": report_output["best_model_metrics"],
            "model_comparison": report_output.get("model_comparison", []),
            "report_path": report_output["report_path"],
        }

