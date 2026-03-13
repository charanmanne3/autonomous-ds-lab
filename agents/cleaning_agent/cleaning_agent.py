import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class CleaningAgent:
    """Handles missing values, encoding, and feature/target split."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting data cleaning workflow")
        df: pd.DataFrame = input_data["dataset"].copy()
        metadata: Dict[str, Any] = input_data["metadata"]
        target_col = metadata["target_column"]

        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()

        numeric_columns: List[str] = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns: List[str] = X.select_dtypes(exclude=["number"]).columns.tolist()

        if numeric_columns:
            numeric_imputer = SimpleImputer(strategy="median")
            X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])

        if categorical_columns:
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
            # Encode categorical values for model readiness when present.
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            X[categorical_columns] = encoder.fit_transform(X[categorical_columns])

        cleaning_summary = {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values_after_cleaning": int(X.isna().sum().sum()),
            "feature_shape": tuple(X.shape),
            "target_shape": tuple(y.shape),
        }

        self.logger.info(
            "Cleaning complete. Missing values remaining: %s",
            cleaning_summary["missing_values_after_cleaning"],
        )
        return {
            "X": X,
            "y": y,
            "metadata": metadata,
            "runtime_config": input_data.get("runtime_config", {}),
            "cleaning_summary": cleaning_summary,
        }

