# F&B Intelligence

**AI-Powered Decision Support System for Food & Beverage Operations**

F&B Intelligence is an end-to-end AI engineering portfolio project
designed to support Food & Beverage operations through demand
forecasting and inventory decision support.

The system combines machine learning, business-oriented analytics, model
serving, model registry, containerization, and operational monitoring
into a production-oriented architecture.

> **Project status:** Production-oriented MVP\
> **Primary forecasting model:** XGBoost\
> **Model registry:** MLflow\
> **API:** FastAPI\
> **Dashboard:** Streamlit\
> **Deployment:** Docker Compose

---

## 1. Overview

Food & Beverage operations need to make decisions under demand
uncertainty. Forecasting demand too low can increase stockout risk,
while forecasting too high can lead to excess inventory.

This project addresses that problem by building a small end-to-end
decision support system with two implemented capabilities:

1.  **Demand Forecasting** --- predicts future daily product demand for
    a selected branch and product.
2.  **Inventory Optimization** --- converts product demand into
    ingredient-level demand and generates order/hold recommendations
    using a service-level-based inventory policy.

The project also demonstrates how an ML model can move beyond
experimentation into a deployable system:

```text
Data
  ↓
Feature Engineering
  ↓
Model Training & Evaluation
  ↓
MLflow Tracking
  ↓
MLflow Model Registry
  ↓
FastAPI
  ↓
Streamlit Dashboard
  ↓
Docker Compose
  ↓
Operational Monitoring
```

---

## 2. Business Problems

The system is designed around common F&B operational questions:

### Demand

- How much demand is expected for a product?
- Which dates are expected to have higher demand?
- How can operations prepare for demand fluctuations?

### Inventory

- Which ingredients should be reordered?
- How much should be ordered?
- How can inventory be balanced against service-level requirements?

### Management

The architecture can be extended to support customer segmentation,
product recommendation, branch/product performance analysis, promotion
analysis, and other F&B intelligence use cases.

These additional use cases are part of the broader system design, while
the current implemented MVP focuses on forecasting and inventory
optimization.

---

## 3. Key Capabilities

### 3.1 Demand Forecasting

The forecasting module uses historical daily demand with:

- calendar features,
- promotion discount,
- payday indicator,
- holiday indicator,
- lag features,
- rolling statistics.

The final production model is an XGBoost regressor.

### 3.2 Inventory Optimization

The inventory module:

1.  converts product demand into ingredient demand using product
    recipes,
2.  estimates ingredient consumption,
3.  applies lead-time and safety-stock logic,
4.  calculates target stock,
5.  generates `ORDER` or `HOLD` recommendations.

The selected policy is:

- **Policy:** Lean 85%
- **Target service level:** 85%
- **Z-value:** 1.036
- **Review period:** 1 day

### 3.3 Production API

FastAPI exposes forecasting, future recursive forecasting, inventory
recommendations, reference data, model information, health/readiness
checks, and operational metrics.

### 3.4 Monitoring

The API provides request count, successful requests, failed requests,
average latency, uptime, active MLflow model version, health status, and
readiness status.

---

## 4. System Architecture

```text
                         ┌──────────────────────┐
                         │   Synthetic Dataset  │
                         │ daily_demand, sales, │
                         │ recipes, inventory   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         │ Calendar + Lag +     │
                         │ Rolling Statistics   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Model Experiments    │
                         │ Baseline / RF /      │
                         │ XGBoost / LSTM        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     MLflow Tracking  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ MLflow Model Registry│
                         │ fb_demand_forecasting│
                         │      @ champion      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │             FastAPI              │
                  │ Forecast / Inventory / Monitoring│
                  └───────────────┬─────────────────┘
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
          ┌────────────────────┐     ┌────────────────────┐
          │ Streamlit Dashboard│     │ Operational Metrics│
          │ Forecast + Inventory│     │ Health / Readiness │
          └────────────────────┘     └────────────────────┘
```

---

## 5. Data

The project uses synthetic data generated specifically for development
and demonstration.

The synthetic dataset contains operational patterns such as
weekday/weekend behavior, payday effects, holiday effects, promotions,
weather-related effects, branch/product demand differences, and demand
noise.

The main forecasting table is `daily_demand`.

Other datasets support the wider business workflow:

- branches
- products
- ingredients
- product_ingredients
- customers
- sales
- promotions
- weather
- holidays
- inventory_transactions

### Important limitation

Because the dataset is synthetic, the reported model performance
represents performance on the simulated data-generating process. It
should **not** be interpreted as evidence of real-world performance on
an actual F&B company's data.

---

## 6. Forecasting Pipeline

### 6.1 Feature Engineering

The forecasting pipeline creates:

**Calendar features**

- `day_of_week`
- `day_of_month`
- `month`
- `week_of_year`
- `is_weekend`
- `is_payday`
- `is_holiday`

**Lag features**

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_21`
- `lag_28`

**Rolling statistics**

- `rolling_mean_7`
- `rolling_std_7`
- `rolling_mean_14`
- `rolling_std_14`
- `rolling_mean_28`
- `rolling_std_28`

The lag and rolling features are calculated using previous observations
to avoid same-day target leakage.

### 6.2 Temporal Evaluation

The forecasting experiment uses a chronological split rather than a
random train/test split.

The project evaluates training, validation, and test periods
chronologically, reflecting the real forecasting scenario where future
observations are unavailable during training.

### 6.3 Baselines and Model Experiments

The forecasting model was compared against simple baselines, including:

- Naive-1
- Seasonal Naive-7

Machine learning experiments included:

- Random Forest
- XGBoost
- LSTM

The final model was selected based on observed evaluation results rather
than model complexity alone.

---

## 7. Model Results

The final XGBoost forecasting model achieved the following test
performance in the project experiments:

Metric Test Result

---

MAE **13.52**
RMSE **18.69**
MAPE **21.22%**
R² **0.760**

Against the Seasonal Naive-7 baseline:

Metric Seasonal Naive-7 XGBoost Improvement

---

MAE 19.47 **13.52** **\~30.6%**
RMSE 26.85 **18.69** **\~30.4%**

The final XGBoost model was selected as the production forecasting
model.

---

## 8. Forecasting Model

The production model uses XGBoost with the following tuned
configuration:

```python
XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
```

Model artifacts include:

```text
models/forecasting/
├── metadata.json
├── preprocessor.joblib
└── xgb_forecasting.joblib
```

---

## 9. Inventory Optimization

Inventory demand is derived from product demand and recipes:

```text
Product Demand
      ↓
Product Recipe
      ↓
Ingredient Usage
      ↓
Ingredient Demand
      ↓
Inventory Policy
      ↓
ORDER / HOLD
```

The policy uses:

```text
Target Stock =
    Average Daily Usage × (Lead Time + Review Period)
    +
    z × Daily Usage Std × √(Lead Time + Review Period)
```

The inventory position is:

```text
Inventory Position =
    On Hand + On Order
```

Recommended order quantity is:

```text
Order Quantity =
    max(0, Target Stock - Inventory Position)
```

The selected policy is **Lean 85%**, chosen from the tested
service-level policies using the project's decision criterion of meeting
the service-level requirement while minimizing average inventory value.

---

## 10. MLOps

MLflow is used for experiment tracking and model registry.

The project tracks:

- model parameters,
- validation metrics,
- test metrics,
- model artifacts,
- experiment runs.

The production model is registered as:

```text
Model:
fb_demand_forecasting

Alias:
champion
```

The current production champion is **version 3**.

The FastAPI application retrieves the champion model through the MLflow
Model Registry rather than relying solely on a hard-coded local model
path.

---

## 11. Production API

FastAPI provides the serving layer.

### Reference Data

```text
GET /branches
GET /products
```

### Forecasting

```text
POST /forecast
POST /forecast/future
GET  /forecast/model
```

### Inventory

```text
GET /inventory/recommend
```

### Monitoring

```text
GET /health
GET /ready
GET /metrics
GET /model-info
```

### Interactive API Documentation

When the system is running:

```text
http://localhost:8000/docs
```

---

## 12. Example API Request

### Future Forecast

```json
{
  "branch_id": "B001",
  "product_id": "P024",
  "start_date": "2026-01-01",
  "horizon": 7,
  "discount_percentage": 0,
  "is_payday": 0,
  "is_holiday": 0
}
```

The API performs recursive forecasting, feeding previous predictions
into subsequent forecasting steps.

The request is validated with constraints including:

- horizon: 1--30 days,
- discount: 0--100%,
- payday: 0 or 1,
- holiday: 0 or 1.

---

## 13. Dashboard

The Streamlit dashboard provides a business-oriented interface for
interacting with the API.

Current dashboard capabilities include:

### Configuration

- Branch selector
- Product selector
- Forecast horizon
- Discount assumption
- Payday flag
- Holiday flag

### Forecast

- Total forecast
- Average demand/day
- Peak demand
- Peak date
- Forecast chart
- Forecast detail table
- Operational insight
- Model information

### Inventory

- Active inventory policy
- Number of order recommendations
- Number of hold recommendations
- Recommended quantity
- Estimated order value
- Ingredient-level recommendations

### Monitoring

- API status
- Readiness
- Model version
- Total requests
- Successful requests
- Failed requests
- Average latency
- Active model/alias

Dashboard:

```text
http://localhost:8501
```

---

## 14. Operational Monitoring

The API contains lightweight in-memory operational metrics.

The `/health` endpoint checks basic service availability.

The `/ready` endpoint verifies that the forecasting model can be loaded
and that the configured MLflow champion model exists.

The `/model-info` endpoint exposes the currently active registered model
version.

### Monitoring limitation

Current request metrics are stored in application memory. Therefore,
they reset when the API container restarts.

This is intentional for the MVP. A future production implementation
could replace this with persistent observability infrastructure such as
Prometheus/Grafana or a managed monitoring platform.

---

## 15. Docker Deployment

The system is containerized using Docker Compose.

Services:

```text
fnb-mlflow
fnb-api
fnb-dashboard
```

Ports:

```text
MLflow    → 5000
FastAPI   → 8000
Dashboard → 8501
```

The Docker setup separates the main runtime responsibilities:

```text
MLflow
  └── Experiment tracking + Model Registry

FastAPI
  └── Model serving + Business APIs

Streamlit
  └── User-facing dashboard
```

---

## 16. Running the Project

### Prerequisites

- Docker Desktop
- Docker Compose

### Start the system

From the project root:

```bash
docker compose up --build
```

### Check services

```bash
docker compose ps
```

Expected services:

```text
fnb-api
fnb-dashboard
fnb-mlflow
```

### Access the services

FastAPI:

```text
http://localhost:8000
```

Swagger/OpenAPI:

```text
http://localhost:8000/docs
```

Streamlit:

```text
http://localhost:8501
```

MLflow:

```text
http://localhost:5000
```

### Stop the system

```bash
docker compose down
```

---

## 17. Project Structure

The project is organized around separation of concerns:

```text
fb-intelligence/
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── raw/
│
├── docs/
│
├── mlruns/
│
├── models/
│   └── forecasting/
│
├── notebooks/
│
├── scripts/
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── features/
│   ├── inference/
│   ├── inventory/
│   ├── mlops/
│   └── config.py
│
├── tests/
│
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 18. Technology Stack

Area Technology

---

Language Python
Data Processing Pandas, NumPy
Machine Learning Scikit-learn, XGBoost
Deep Learning Experiment TensorFlow/Keras
API FastAPI, Uvicorn
Validation Pydantic
Dashboard Streamlit
Model Serialization Joblib
MLOps MLflow
Database/ORM Support SQLAlchemy
Containerization Docker, Docker Compose

---

## 19. Engineering Decisions

### Why temporal evaluation?

Random splitting can expose a forecasting model to information from
periods that occur after the observations used for prediction. A
chronological evaluation better represents the intended production
scenario.

### Why include baselines?

A machine learning model should demonstrate value against a simple
benchmark. Seasonal Naive-7 provides a meaningful baseline for data with
weekly seasonality.

### Why XGBoost?

XGBoost provided competitive performance without the additional
deployment complexity of the LSTM experiment. The final model was
selected based on observed evaluation results.

### Why MLflow?

The project needed more than a saved `.joblib` model. MLflow provides
experiment tracking, artifact management, model registration, and a
mechanism for identifying the production champion.

### Why FastAPI?

FastAPI provides a lightweight serving layer that allows the forecasting
model and business logic to be consumed independently from the
dashboard.

### Why Docker Compose?

The project contains multiple cooperating services. Docker Compose makes
the local production-like environment reproducible without introducing
unnecessary orchestration complexity.

---

## 20. Limitations

This project is a portfolio-oriented production MVP and has several
intentional limitations.

### Synthetic data

The dataset is simulated and does not represent actual F&B company data.

### Weather features

The forecasting model does not directly depend on same-day weather
observations because those values may not be available at prediction
time. A future implementation could integrate weather forecasts.

### Inventory scope

The current inventory recommendation endpoint produces ingredient-level
recommendations across the modeled inventory scope rather than accepting
a branch/product filter directly.

### Monitoring

Operational metrics are currently stored in memory.

### Deployment

The current deployment uses Docker Compose as a production-like local
environment. It is not yet a cloud deployment with managed
infrastructure, autoscaling, or Kubernetes.

### Data pipeline

The project currently focuses on a reproducible portfolio workflow
rather than a live ingestion pipeline from an operational POS/ERP
system.

---

## 21. Future Improvements

Potential next steps include:

- real POS/ERP data integration,
- automated data ingestion,
- weather forecast integration,
- branch-specific inventory recommendations,
- customer segmentation,
- product recommendation,
- promotion optimization,
- forecast confidence intervals,
- model drift detection,
- automated retraining,
- persistent monitoring,
- Prometheus/Grafana,
- CI/CD deployment,
- cloud deployment,
- authentication and authorization,
- automated model promotion based on evaluation criteria.

---

## 22. What This Project Demonstrates

This project is intentionally broader than a standalone machine learning
notebook.

It demonstrates an end-to-end workflow:

```text
Business Problem
      ↓
Synthetic Data Generation
      ↓
EDA
      ↓
Feature Engineering
      ↓
Baseline Modeling
      ↓
Model Experimentation
      ↓
Model Evaluation
      ↓
Business Optimization
      ↓
MLflow Tracking
      ↓
Model Registry
      ↓
FastAPI Model Serving
      ↓
Streamlit Dashboard
      ↓
Dockerization
      ↓
Production Hardening
      ↓
Monitoring
```

The main engineering principle is:

> **Baseline → Experiment → Evaluate → Explain → Deploy → Monitor**

---

## 23. Project Status

Module Status

---

Concept & Architecture ✅ Complete
Synthetic Data ✅ Complete
EDA & Feature Engineering ✅ Complete
Demand Forecasting ✅ Complete
Model Comparison ✅ Complete
Inventory Optimization ✅ Complete
FastAPI ✅ Complete
Streamlit Dashboard ✅ Complete
MLflow Tracking & Registry ✅ Complete
Docker Production Setup ✅ Complete
Production Hardening ✅ Complete
Monitoring ✅ Complete

---

## 24. Author

**Agni Musadad**

AI / Machine Learning Engineering Portfolio Project

Focus areas:

- Machine Learning
- AI Engineering
- MLOps
- Model Serving
- Data Analytics
- Decision Support Systems
