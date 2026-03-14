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
| RandomForest | 68894.1520 | 49354.7643 | 0.6486 |
| LinearRegression | 70803.1533 | 51741.6112 | 0.6288 |

## Best Model
- Model: RandomForest
- RMSE: 68894.1520
- MAE: 49354.7643
- R2: 0.6486
- Model Path: /Users/charanmanne/Documents/Programming/autonomous-ds-lab/storage/models/randomforest.joblib

## Recommendations
- Tune hyperparameters of the best-performing model using cross-validation.
- Add external features and run feature selection to reduce noise.
- Schedule periodic retraining and drift monitoring in production.
