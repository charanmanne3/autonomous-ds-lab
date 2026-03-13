import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.datasets import fetch_california_housing


class DatasetAgent:
    """Discovers and loads datasets for downstream agents."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        task = input_data.get("task", "unspecified task")
        task_type = input_data.get("task_type", "Regression")
        self.logger.info("Starting dataset discovery for task: %s", task)

        project_root = Path(__file__).resolve().parents[2]
        dataset_dir = project_root / "storage" / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv = input_data.get("dataset_csv")
        target_column = input_data.get("target_column")

        if dataset_csv:
            self.logger.info("Using user-uploaded dataset from dashboard input")
            dataset = pd.read_csv(StringIO(dataset_csv))
            if not target_column:
                raise ValueError("Target column is required when using an uploaded dataset.")
            if target_column not in dataset.columns:
                raise ValueError(f"Target column '{target_column}' not found in uploaded dataset.")
            dataset_path = dataset_dir / "uploaded_dataset.csv"
            source = "dashboard_upload_csv"
            dataset_name = "Uploaded Dataset"
            feature_names = [col for col in dataset.columns if col != target_column]
            target_multiplier = 1.0
            target_unit = "native"
        else:
            housing = fetch_california_housing(as_frame=True)
            dataset = housing.frame.copy()
            target_column = housing.target.name
            dataset_path = dataset_dir / "california_housing.csv"
            source = "sklearn.datasets.fetch_california_housing"
            dataset_name = "California Housing"
            feature_names = list(housing.feature_names)
            # California housing target is provided in 100k USD units.
            target_multiplier = 100000
            target_unit = "USD"

        dataset.to_csv(dataset_path, index=False)
        self.logger.info("Dataset saved to %s", dataset_path)

        metadata = {
            "name": dataset_name,
            "source": source,
            "target_column": target_column,
            "target_unit": target_unit,
            "target_value_multiplier": target_multiplier,
            "num_rows": int(dataset.shape[0]),
            "num_columns": int(dataset.shape[1]),
            "feature_names": feature_names,
            "task": task,
            "task_type": task_type,
            "dataset_path": str(dataset_path),
        }

        self.logger.info(
            "Dataset loaded successfully: %s rows, %s columns",
            metadata["num_rows"],
            metadata["num_columns"],
        )
        return {"dataset": dataset, "metadata": metadata}

