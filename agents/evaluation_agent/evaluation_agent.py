import logging
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EvaluationAgent:
    """Evaluates candidate models and selects the best one."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting model evaluation")
        model_results: List[Dict[str, Any]] = input_data["model_results"]
        X_test: pd.DataFrame = input_data["X_test"]
        y_test = np.asarray(input_data["y_test"])
        target_scaler = input_data.get("target_scaler")
        metadata: Dict[str, Any] = input_data["metadata"]
        target_multiplier = float(metadata.get("target_value_multiplier", 1.0))

        if not model_results:
            raise ValueError("No model results provided for evaluation.")

        evaluated_results: List[Dict[str, Any]] = []
        for result in model_results:
            model = result.get("model")
            if model is None:
                model = joblib.load(result["model_path"])

            y_pred = model.predict(X_test)

            # Metrics must be reported in original business units.
            # If target scaling was used during training, reverse it for both
            # predictions and labels before computing RMSE/MAE/R2.
            if target_scaler is not None:
                y_pred_original = target_scaler.inverse_transform(
                    np.asarray(y_pred).reshape(-1, 1)
                ).ravel()
                y_test_original = target_scaler.inverse_transform(
                    np.asarray(y_test).reshape(-1, 1)
                ).ravel()
            else:
                y_pred_original = np.asarray(y_pred).ravel()
                y_test_original = np.asarray(y_test).ravel()

            y_pred_for_metrics = y_pred_original * target_multiplier
            y_test_for_metrics = y_test_original * target_multiplier
            rmse = float(mean_squared_error(y_test_for_metrics, y_pred_for_metrics, squared=False))
            mae = float(mean_absolute_error(y_test_for_metrics, y_pred_for_metrics))
            r2 = float(r2_score(y_test_for_metrics, y_pred_for_metrics))

            evaluated_results.append(
                {
                    "model_name": result["model_name"],
                    "model_path": result["model_path"],
                    "metrics": {
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                    },
                }
            )

        sorted_results = sorted(evaluated_results, key=lambda item: item["metrics"]["rmse"])
        best_result = sorted_results[0]

        comparison_table = [
            {
                "model": result["model_name"],
                "rmse": result["metrics"]["rmse"],
                "mae": result["metrics"]["mae"],
                "r2": result["metrics"]["r2"],
            }
            for result in sorted_results
        ]

        best_model_info = {
            "model_name": best_result["model_name"],
            "metrics": best_result["metrics"],
            "model_path": best_result["model_path"],
        }

        self.logger.info(
            "Best model selected: %s (RMSE=%.4f)",
            best_model_info["model_name"],
            best_model_info["metrics"]["rmse"],
        )
        return {
            "best_model_info": best_model_info,
            "best_model_name": best_model_info["model_name"],
            "best_score": best_model_info["metrics"]["rmse"],
            "model_comparison": comparison_table,
            "metadata": input_data["metadata"],
            "cleaning_summary": input_data.get("cleaning_summary", {}),
            "feature_summary": input_data.get("feature_summary", {}),
        }

