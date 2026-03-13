import logging
from typing import Any, Dict

from registry.agent_registry import get_agent_registry


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


class TaskPlanner:
    """Orchestrates the full autonomous data science workflow."""

    def __init__(self) -> None:
        configure_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = get_agent_registry()

    def run_pipeline(self, task_description: str, fast_demo_mode: bool = True) -> Dict[str, Any]:
        self.logger.info(
            "Pipeline started for task: %s | fast_demo_mode=%s",
            task_description,
            fast_demo_mode,
        )

        try:
            self.logger.info("Step 1/6: dataset_agent")
            dataset_output = self.registry["dataset"].run({"task": task_description})
            dataset_output["runtime_config"] = {"fast_demo_mode": fast_demo_mode}
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at dataset_agent: %s", exc)
            raise

        try:
            self.logger.info("Step 2/6: cleaning_agent")
            cleaning_output = self.registry["cleaning"].run(dataset_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at cleaning_agent: %s", exc)
            raise

        try:
            self.logger.info("Step 3/6: feature_agent")
            feature_output = self.registry["feature"].run(cleaning_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at feature_agent: %s", exc)
            raise

        try:
            self.logger.info("Step 4/6: training_agent")
            training_output = self.registry["training"].run(feature_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at training_agent: %s", exc)
            raise

        try:
            self.logger.info("Step 5/6: evaluation_agent")
            evaluation_output = self.registry["evaluation"].run(training_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at evaluation_agent: %s", exc)
            raise

        try:
            self.logger.info("Step 6/6: report_agent")
            report_output = self.registry["report"].run(evaluation_output)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed at report_agent: %s", exc)
            raise

        self.logger.info("Pipeline completed successfully")
        return {
            "task": task_description,
            "pipeline_status": report_output["status"],
            "best_model_name": report_output["best_model_name"],
            "best_score": report_output["best_score"],
            "best_model_metrics": report_output["best_model_metrics"],
            "report_path": report_output["report_path"],
        }

