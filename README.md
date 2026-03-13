# Autonomous Data Science Lab

Autonomous Data Science Lab is a production-style multi-agent system that automates an end-to-end data science workflow.  
Given a task such as **"Predict house prices"**, the platform executes a sequential pipeline:

1. Discover dataset
2. Clean data
3. Engineer features
4. Train multiple models
5. Evaluate model performance
6. Generate a markdown report

---

## Architecture

The project follows a modular, clean architecture with clear responsibilities:

- `agents/`: domain agents for each stage of the workflow
- `registry/`: central registry for agent construction and lookup
- `orchestrator/`: task planner that controls execution flow
- `api/`: FastAPI service to trigger the pipeline
- `pipelines/`: Airflow DAG for scheduled/orchestrated runs
- `storage/`: local dataset and intermediate artifact storage
- `mlflow/`: local MLflow artifacts and tracking persistence
- `docker/`: container orchestration for API + MLflow + Postgres

---

## Agent Workflow

Each agent is implemented as a Python class with:

```python
def run(self, input_data):
    pass
```

### Implemented agents

- **DatasetAgent**: loads California Housing dataset from `sklearn`
- **CleaningAgent**: handles missing values, encoding, and scaling
- **FeatureAgent**: creates domain-inspired features automatically
- **TrainingAgent**: trains Linear Regression, Random Forest, and XGBoost with MLflow tracking
- **EvaluationAgent**: computes RMSE, MAE, and R2; selects best model
- **ReportAgent**: writes report to `reports/model_report.md`

---

## Run Locally

### 1) Setup environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3) Trigger pipeline

```bash
curl -X POST "http://localhost:8000/run-pipeline" \
  -H "Content-Type: application/json" \
  -d '{"task":"Predict house prices"}'
```

Expected outcome:

- Models trained and tracked in MLflow
- Best model selected
- Report generated at `reports/model_report.md`

---

## Airflow DAG

The DAG is located at `pipelines/airflow_dag.py` and includes:

- `dataset_task`
- `cleaning_task`
- `feature_task`
- `training_task`
- `evaluation_task`
- `report_task`

All tasks execute sequentially and pass outputs using artifact files + XCom paths.

---

## Docker

From project root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Services:

- API: `http://localhost:8000`
- MLflow: `http://localhost:5000`
- PostgreSQL backend for MLflow metadata

---

## API Contract

### `POST /run-pipeline`

Request:

```json
{
  "task": "Predict house prices"
}
```

Response (example):

```json
{
  "status": "completed",
  "result": {
    "task": "Predict house prices",
    "pipeline_status": "success",
    "best_model_name": "RandomForest",
    "best_model_metrics": {
      "rmse": 0.49,
      "mae": 0.32,
      "r2": 0.80
    },
    "report_path": "reports/model_report.md"
  }
}
```

---

## Notes

- Logging is enabled across agents and orchestrator.
- Code is modular and designed for extension (new datasets, agents, and models).
- Current dataset discovery implementation uses California Housing as a baseline source.

