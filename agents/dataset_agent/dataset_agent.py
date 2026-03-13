import logging
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
        self.logger.info("Starting dataset discovery for task: %s", task)

        housing = fetch_california_housing(as_frame=True)
        dataset: pd.DataFrame = housing.frame.copy()

        project_root = Path(__file__).resolve().parents[2]
        dataset_dir = project_root / "storage" / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / "california_housing.csv"
        dataset.to_csv(dataset_path, index=False)
        self.logger.info("Dataset saved to %s", dataset_path)

        metadata = {
            "name": "California Housing",
            "source": "sklearn.datasets.fetch_california_housing",
            "target_column": housing.target.name,
            "target_unit": "USD",
            # California housing target is provided in 100k USD units.
            "target_value_multiplier": 100000,
            "num_rows": int(dataset.shape[0]),
            "num_columns": int(dataset.shape[1]),
            "feature_names": list(housing.feature_names),
            "task": task,
            "dataset_path": str(dataset_path),
        }

        self.logger.info(
            "Dataset loaded successfully: %s rows, %s columns",
            metadata["num_rows"],
            metadata["num_columns"],
        )
        return {"dataset": dataset, "metadata": metadata}

