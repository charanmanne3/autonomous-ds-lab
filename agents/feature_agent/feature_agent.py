import logging
from typing import Any, Dict

import pandas as pd


class FeatureAgent:
    """Creates derived features for improved model learning."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting feature engineering")
        X: pd.DataFrame = input_data["X"].copy()
        y: pd.Series = input_data["y"].copy()
        metadata: Dict[str, Any] = input_data["metadata"]

        for required_col in [
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
        ]:
            if required_col not in X.columns:
                raise ValueError(f"Required column '{required_col}' not found for feature engineering.")

        households = X["Population"] / X["AveOccup"].replace(0, pd.NA)
        households = households.fillna(households.median())

        X["rooms_per_household"] = X["AveRooms"] / households.replace(0, pd.NA)
        X["bedrooms_per_room"] = X["AveBedrms"] / X["AveRooms"].replace(0, pd.NA)
        X["population_per_household"] = X["Population"] / households.replace(0, pd.NA)

        # Replace residual NaNs from safe divisions.
        X = X.fillna(X.median(numeric_only=True))

        feature_summary = {
            "created_features": [
                "rooms_per_household",
                "bedrooms_per_room",
                "population_per_household",
            ],
            "output_columns": list(X.columns),
            "output_shape": tuple(X.shape),
        }

        self.logger.info(
            "Feature engineering complete. Total columns: %s",
            len(feature_summary["output_columns"]),
        )
        return {
            "X": X,
            "y": y,
            "metadata": metadata,
            "runtime_config": input_data.get("runtime_config", {}),
            "cleaning_summary": input_data.get("cleaning_summary", {}),
            "feature_summary": feature_summary,
        }

