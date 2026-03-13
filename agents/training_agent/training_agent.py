import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


class TrainingAgent:
    """Trains multiple models and tracks experiments in MLflow."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _is_fast_demo_mode_from_env() -> bool:
        """Read FAST_DEMO_MODE env flag (default: enabled)."""
        return os.getenv("FAST_DEMO_MODE", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def _resolve_fast_demo_mode(self, input_data: Dict[str, Any]) -> bool:
        """Resolve mode from runtime payload first, then env default."""
        runtime_config = input_data.get("runtime_config", {})
        if isinstance(runtime_config, dict) and "fast_demo_mode" in runtime_config:
            return bool(runtime_config["fast_demo_mode"])
        return self._is_fast_demo_mode_from_env()

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting model training")
        X: pd.DataFrame = input_data["X"].copy()
        y: pd.Series = input_data["y"].copy()
        metadata: Dict[str, Any] = input_data["metadata"]
        target_multiplier = float(metadata.get("target_value_multiplier", 1.0))
        fast_demo_mode = self._resolve_fast_demo_mode(input_data)
        self.logger.info("FAST_DEMO_MODE=%s", fast_demo_mode)

        # Demo-speed optimization: cap training data size so end-to-end runs
        # complete quickly during live demos.
        if fast_demo_mode and len(X) > 5000:
            sampled_idx = X.sample(n=5000, random_state=42).index
            X = X.loc[sampled_idx].reset_index(drop=True)
            y = y.loc[sampled_idx].reset_index(drop=True)
            self.logger.info("Applied demo sampling: using 5000 rows for training")
        else:
            self.logger.info("Using full dataset for training")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Store and use a dedicated target scaler so downstream evaluation can
        # always recover original target units before metric calculation.
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.to_numpy().reshape(-1, 1)).ravel()

        numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_columns,
                ),
            ]
        )

        # FAST_DEMO_MODE uses smaller models for speed; standard mode keeps
        # stronger defaults for better quality.
        if fast_demo_mode:
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(
                    n_estimators=20,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                ),
                "XGBoost": XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=50,
                    learning_rate=0.1,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            }
        else:
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    n_jobs=-1,
                ),
                "XGBoost": XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            }

        project_root = Path(__file__).resolve().parents[2]
        model_dir = project_root / "storage" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        mlflow_tracking_dir = project_root / "mlflow"
        mlflow_tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{mlflow_tracking_dir.resolve()}")
        mlflow.set_experiment("autonomous-ds-lab")

        model_results: List[Dict[str, Any]] = []
        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )
            self.logger.info("Training model: %s", model_name)

            with mlflow.start_run(run_name=model_name):
                pipeline.fit(X_train, y_train_scaled)
                y_pred_scaled = pipeline.predict(X_test)

                # Convert both prediction and ground truth back to original
                # target units before logging metrics.
                y_pred_original = target_scaler.inverse_transform(
                    np.asarray(y_pred_scaled).reshape(-1, 1)
                ).ravel()
                y_test_original = target_scaler.inverse_transform(
                    np.asarray(y_test_scaled).reshape(-1, 1)
                ).ravel()
                # Convert to business-facing units (USD for this dataset).
                y_pred_for_metrics = y_pred_original * target_multiplier
                y_test_for_metrics = y_test_original * target_multiplier
                rmse = float(
                    mean_squared_error(y_test_for_metrics, y_pred_for_metrics, squared=False)
                )
                mae = float(mean_absolute_error(y_test_for_metrics, y_pred_for_metrics))
                r2 = float(r2_score(y_test_for_metrics, y_pred_for_metrics))

                mlflow.log_param("model_name", model_name)
                mlflow.log_param("train_rows", int(X_train.shape[0]))
                mlflow.log_param("feature_count", int(X_train.shape[1]))
                mlflow.log_param("profile", "demo_fast" if fast_demo_mode else "standard")
                mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
                model_path = model_dir / f"{model_name.lower()}.joblib"
                joblib.dump(pipeline, model_path)
                mlflow.log_param("model_path", str(model_path))
                # Keep MLflow lightweight for demo runtime: log params/metrics only.

                model_results.append(
                    {
                        "model_name": model_name,
                        "model": pipeline,
                        "model_path": str(model_path),
                    }
                )
                self.logger.info(
                    "Completed %s with RMSE=%.4f, MAE=%.4f, R2=%.4f | saved: %s",
                    model_name,
                    rmse,
                    mae,
                    r2,
                    model_path,
                )

        return {
            "model_results": model_results,
            "X_test": X_test,
            "y_test": y_test_scaled,
            "target_scaler": target_scaler,
            "metadata": metadata,
            "cleaning_summary": input_data.get("cleaning_summary", {}),
            "feature_summary": input_data.get("feature_summary", {}),
        }

