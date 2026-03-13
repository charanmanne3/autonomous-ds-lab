# Autonomous Data Science Lab - Model Report

## Dataset Summary
- Dataset: California Housing
- Source: sklearn.datasets.fetch_california_housing
- Rows: 20640
- Columns: 9
- Target: MedHouseVal

## Data Cleaning Summary
- Numeric columns: 8
- Categorical columns: 0
- Missing values after cleaning: 0

## Feature Engineering Summary
- Created features:
  - rooms_per_household
  - bedrooms_per_room
  - population_per_household

## Model Comparison
| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| XGBoost | 55788.0789 | 38393.0971 | 0.7696 |
| RandomForest | 68894.1520 | 49354.7643 | 0.6486 |
| LinearRegression | 70803.1533 | 51741.6112 | 0.6288 |

## Best Model
- Model: XGBoost
- RMSE: 55788.0789
- MAE: 38393.0971
- R2: 0.7696
- Model Path: /Users/charanmanne/Documents/Programming/autonomous-ds-lab/storage/models/xgboost.joblib

## Recommendations
- Tune hyperparameters of the best-performing model using cross-validation.
- Add external features and run feature selection to reduce noise.
- Schedule periodic retraining and drift monitoring in production.
