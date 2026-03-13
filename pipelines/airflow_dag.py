from datetime import datetime
import pickle
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator

from registry.agent_registry import get_agent_registry


def _registry() -> Dict[str, Any]:
    return get_agent_registry()


def _artifact_dir() -> Path:
    path = Path("storage/datasets/airflow_artifacts")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_artifact(name: str, data: Dict[str, Any]) -> str:
    path = _artifact_dir() / f"{name}.pkl"
    with path.open("wb") as file:
        pickle.dump(data, file)
    return str(path)


def _load_artifact(path: str) -> Dict[str, Any]:
    with open(path, "rb") as file:
        return pickle.load(file)


def run_dataset_task(**context):
    output = _registry()["dataset"].run({"task": "Predict house prices"})
    artifact_path = _save_artifact("dataset_output", output)
    context["ti"].xcom_push(key="dataset_output_path", value=artifact_path)


def run_cleaning_task(**context):
    dataset_output_path = context["ti"].xcom_pull(
        key="dataset_output_path", task_ids="dataset_task"
    )
    dataset_output = _load_artifact(dataset_output_path)
    output = _registry()["cleaning"].run(dataset_output)
    artifact_path = _save_artifact("cleaning_output", output)
    context["ti"].xcom_push(key="cleaning_output_path", value=artifact_path)


def run_feature_task(**context):
    cleaning_output_path = context["ti"].xcom_pull(
        key="cleaning_output_path", task_ids="cleaning_task"
    )
    cleaning_output = _load_artifact(cleaning_output_path)
    output = _registry()["feature"].run(cleaning_output)
    artifact_path = _save_artifact("feature_output", output)
    context["ti"].xcom_push(key="feature_output_path", value=artifact_path)


def run_training_task(**context):
    feature_output_path = context["ti"].xcom_pull(
        key="feature_output_path", task_ids="feature_task"
    )
    feature_output = _load_artifact(feature_output_path)
    output = _registry()["training"].run(feature_output)
    artifact_path = _save_artifact("training_output", output)
    context["ti"].xcom_push(key="training_output_path", value=artifact_path)


def run_evaluation_task(**context):
    training_output_path = context["ti"].xcom_pull(
        key="training_output_path", task_ids="training_task"
    )
    training_output = _load_artifact(training_output_path)
    output = _registry()["evaluation"].run(training_output)
    artifact_path = _save_artifact("evaluation_output", output)
    context["ti"].xcom_push(key="evaluation_output_path", value=artifact_path)


def run_report_task(**context):
    evaluation_output_path = context["ti"].xcom_pull(
        key="evaluation_output_path", task_ids="evaluation_task"
    )
    evaluation_output = _load_artifact(evaluation_output_path)
    output = _registry()["report"].run(evaluation_output)
    context["ti"].xcom_push(key="report_output", value=output)


default_args = {
    "owner": "autonomous-ds-lab",
    "depends_on_past": False,
}

with DAG(
    dag_id="autonomous_ds_lab_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["data-science", "multi-agent"],
) as dag:
    dataset_task = PythonOperator(
        task_id="dataset_task",
        python_callable=run_dataset_task,
    )

    cleaning_task = PythonOperator(
        task_id="cleaning_task",
        python_callable=run_cleaning_task,
    )

    feature_task = PythonOperator(
        task_id="feature_task",
        python_callable=run_feature_task,
    )

    training_task = PythonOperator(
        task_id="training_task",
        python_callable=run_training_task,
    )

    evaluation_task = PythonOperator(
        task_id="evaluation_task",
        python_callable=run_evaluation_task,
    )

    report_task = PythonOperator(
        task_id="report_task",
        python_callable=run_report_task,
    )

    dataset_task >> cleaning_task >> feature_task >> training_task >> evaluation_task >> report_task

