import logging
from pathlib import Path
from typing import Any, Dict, List


class ReportAgent:
    """Generates a markdown report summarizing the workflow."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Generating markdown report")

        metadata = input_data["metadata"]
        cleaning_summary = input_data.get("cleaning_summary", {})
        feature_summary = input_data.get("feature_summary", {})
        model_comparison: List[Dict[str, Any]] = input_data["model_comparison"]
        best_model_info = input_data["best_model_info"]

        report_lines = [
            "# Autonomous Data Science Lab - Model Report",
            "",
            "## Dataset Summary",
            f"- Dataset: {metadata.get('name')}",
            f"- Source: {metadata.get('source')}",
            f"- Rows: {metadata.get('num_rows')}",
            f"- Columns: {metadata.get('num_columns')}",
            f"- Target: {metadata.get('target_column')}",
            "",
            "## Data Cleaning Summary",
            f"- Numeric columns: {len(cleaning_summary.get('numeric_columns', []))}",
            f"- Categorical columns: {len(cleaning_summary.get('categorical_columns', []))}",
            f"- Missing values after cleaning: {cleaning_summary.get('missing_values_after_cleaning', 'n/a')}",
            "",
            "## Feature Engineering Summary",
            "- Created features:",
        ]

        for feature in feature_summary.get("created_features", []):
            report_lines.append(f"  - {feature}")

        report_lines.extend(
            [
                "",
                "## Model Comparison",
                "| Model | RMSE | MAE | R2 |",
                "|---|---:|---:|---:|",
            ]
        )

        for row in model_comparison:
            report_lines.append(
                f"| {row['model']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |"
            )

        report_lines.extend(
            [
                "",
                "## Best Model",
                f"- Model: {best_model_info['model_name']}",
                f"- RMSE: {best_model_info['metrics']['rmse']:.4f}",
                f"- MAE: {best_model_info['metrics']['mae']:.4f}",
                f"- R2: {best_model_info['metrics']['r2']:.4f}",
                f"- Model Path: {best_model_info['model_path']}",
                "",
                "## Recommendations",
                "- Tune hyperparameters of the best-performing model using cross-validation.",
                "- Add external features and run feature selection to reduce noise.",
                "- Schedule periodic retraining and drift monitoring in production.",
                "",
            ]
        )

        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "model_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        self.logger.info("Report generated at: %s", report_path)
        return {
            "status": "success",
            "report_path": str(report_path),
            "best_model_name": best_model_info["model_name"],
            "best_score": best_model_info["metrics"]["rmse"],
            "best_model_metrics": best_model_info["metrics"],
        }

