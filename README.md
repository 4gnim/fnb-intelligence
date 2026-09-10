# F&B Intelligence

End-to-End AI Engineering project for Food & Beverage business intelligence and decision support.

## Project Overview

F&B Intelligence is an AI-powered decision support system designed to address common F&B business problems:

- Sales demand forecasting
- Inventory optimization
- Customer segmentation
- Product recommendation
- AI model deployment through API
- Monitoring and MLOps

The project follows an end-to-end workflow:

Business Problem
→ Data Generation
→ Data Validation
→ EDA
→ Feature Engineering
→ ML Experimentation
→ Evaluation
→ Decision Support
→ API
→ Dashboard
→ MLOps

## Current Progress

### Completed

- Synthetic F&B dataset generation
- Data validation
- Data profiling
- Exploratory Data Analysis
- Forecasting feature engineering
- Forecasting baseline
- Random Forest experiment
- XGBoost experiment and tuning
- LSTM comparative experiment
- Inventory optimization
- Inventory policy simulation
- Inventory policy sensitivity analysis

### Current Best Forecasting Model

XGBoost:

- MAE: 13.52
- RMSE: 18.69
- MAPE: 21.22%
- R²: 0.760

Test period:

2025-10-01 → 2025-12-31

### Inventory Optimization

The current selected inventory policy is:

- Target service level: 85%
- Z-value: 1.036
- Review period: 1 day

The policy is selected based on the trade-off between service level and inventory investment.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Jupyter Notebook
- FastAPI
- PostgreSQL
- MLflow
- Docker

## Project Status

🚧 In Development

The next stages will include:

- Inventory recommendation layer
- FastAPI
- Dashboard
- MLOps
- Model monitoring
- Deployment

## Disclaimer

This project currently uses synthetic data for development and experimentation.

Model performance and business impact demonstrated in this repository should not be interpreted as real-world business performance.
