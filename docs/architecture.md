# System Architecture

## F&B Intelligence

F&B Intelligence is designed as an end-to-end AI engineering system for
F&B decision support.

The current implementation separates the system into four main concerns:

1.  **Data and feature layer**
2.  **Machine learning and model management**
3.  **API serving and business logic**
4.  **Dashboard and operational monitoring**

The system is containerized with Docker Compose.

---

## 1. High-Level Architecture

```text
┌──────────────────────────────┐
│        Synthetic Data        │
│                              │
│ sales / demand / inventory   │
│ products / ingredients       │
│ recipes / promotions / etc.  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Data & Feature Layer    │
│                              │
│ Pandas / NumPy               │
│ Feature Engineering          │
│ Lag & Rolling Features       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     ML Experiment Layer      │
│                              │
│ Baseline                     │
│ Random Forest                │
│ XGBoost                      │
│ LSTM                         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       MLflow Tracking        │
│       & Model Registry       │
│                              │
│ fb_demand_forecasting        │
│ alias: champion              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          FastAPI             │
│                              │
│ Forecasting                  │
│ Inventory Recommendation     │
│ Reference Data               │
│ Health / Readiness           │
│ Model Info / Metrics         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│        Streamlit             │
│        Dashboard             │
│                              │
│ Forecast Visualization       │
│ Inventory Recommendation     │
│ System Monitoring             │
└──────────────────────────────┘
```

---

## 2. Runtime Architecture

The current Docker Compose environment contains three services:

```text
                     ┌──────────────────┐
                     │     MLflow       │
                     │   Port 5000      │
                     │                  │
                     │ Tracking +       │
                     │ Model Registry   │
                     └────────┬─────────┘
                              │
                              │ model loading
                              ▼
┌──────────────────┐    ┌──────────────────┐
│    Dashboard     │───▶│       API        │
│    Streamlit     │    │     FastAPI      │
│    Port 8501     │    │    Port 8000     │
└──────────────────┘    └────────┬─────────┘
                                 │
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             Forecasting               Inventory
             Model                    Recommendation
```

The current `docker-compose.yml` defines `mlflow`, `api`, and
`dashboard` services. MLflow is exposed on port 5000, FastAPI on port
8000, and Streamlit on port 8501. Inside the Compose network, the API
connects to MLflow through `http://mlflow:5000`, while the dashboard
connects to the API through `http://api:8000`.
fileciteturn21file0L11-L79

---

## 3. Data Layer

The current project uses CSV files as the primary data source.

The API reads forecasting data from:

```text
data/raw/daily_demand.csv
```

Inventory processing additionally uses:

```text
data/raw/ingredients.csv
data/raw/inventory_transactions.csv
data/raw/product_ingredients.csv
```

The forecasting feature layer transforms daily demand into model-ready
features.

The API loads `daily_demand.csv` for forecasting and combines daily
demand, ingredient, inventory, and recipe data for inventory processing.
fileciteturn22file0L244-L291

### Why CSV?

CSV was selected for the current portfolio MVP because:

- the dataset is synthetic,
- the project focuses on demonstrating the AI engineering workflow,
- local development is simple,
- the data pipeline remains easy to inspect and reproduce.

The architecture can later replace the CSV layer with PostgreSQL or
another production data source without changing the model-serving
interface.

---

## 4. Feature Engineering Layer

The forecasting pipeline creates time-series features from historical
demand.

Core feature groups include:

### Calendar Features

- day of week,
- day of month,
- month,
- week of year,
- weekend indicator,
- payday indicator,
- holiday indicator.

### Historical Demand Features

- lag 1,
- lag 7,
- lag 14,
- lag 21,
- lag 28.

### Rolling Statistics

- 7-day rolling mean and standard deviation,
- 14-day rolling mean and standard deviation,
- 28-day rolling mean and standard deviation.

The important design principle is that historical demand features are
shifted so that information from the prediction target date does not
leak into the input.

---

## 5. Machine Learning Layer

The forecasting module follows:

```text
Baseline
   ↓
Experiment
   ↓
Evaluation
   ↓
Model Selection
```

Models explored during development included:

- Naive baseline,
- Seasonal Naive-7,
- Random Forest,
- XGBoost,
- LSTM.

The final serving model is XGBoost.

The deployed API loads the model from MLflow using the registered model
name and the `champion` alias. The preprocessor is loaded from a local
Joblib artifact. fileciteturn22file0L62-L94

---

## 6. Model Management with MLflow

MLflow provides two major responsibilities:

### Experiment Tracking

Model experiments record:

- model configuration,
- evaluation metrics,
- run information,
- model artifacts.

### Model Registry

The forecasting model is registered as:

```text
fb_demand_forecasting
```

The serving alias is:

```text
champion
```

The API therefore does not need to hard-code a specific model version.

Conceptually:

```text
fb_demand_forecasting
        │
        ├── Version 1
        ├── Version 2
        └── Version 3
                │
             champion
                │
                ▼
             FastAPI
```

The current application configuration defines the model name as
`fb_demand_forecasting` and the active alias as `champion`.
fileciteturn21file0L138-L149

This design allows the model registry to become the control point for
model promotion.

---

## 7. Forecasting Serving Flow

The online forecasting flow is:

```text
Client
  │
  │ POST /forecast/future
  ▼
FastAPI
  │
  ├── Validate request
  │
  ├── Load historical demand
  │
  ├── Build forecasting features
  │
  ├── Apply preprocessing
  │
  ├── Load champion model
  │
  └── Recursive prediction
  │
  ▼
Forecast Result
```

The future forecasting endpoint accepts:

- `branch_id`,
- `product_id`,
- `start_date`,
- forecast `horizon`,
- discount percentage,
- payday indicator,
- holiday indicator.

The horizon is constrained to a maximum of 30 days at the API schema
level. fileciteturn22file0L192-L241

The recursive inference layer generates future predictions using
previously generated predictions as part of subsequent forecasting
steps.

---

## 8. Inventory Architecture

Inventory processing follows a separate business-logic path:

```text
Daily Product Demand
        │
        ▼
Product-Ingredient Recipe
        │
        ▼
Ingredient Demand
        │
        ├── Current Stock
        ├── On Order
        ├── Lead Time
        └── Demand Variability
        │
        ▼
Inventory Policy
        │
        ▼
ORDER / HOLD
```

The inventory recommendation logic is not another ML model in the
current implementation.

Instead, it is a deterministic decision layer built on:

- expected usage,
- lead time,
- review period,
- demand variability,
- service-level assumption,
- current inventory position.

This separation is intentional: machine learning estimates demand, while
explicit business rules translate demand into an operational inventory
decision.

---

## 9. API Layer

FastAPI acts as the application and model-serving layer.

Current API responsibilities include:

### System

```text
GET /health
GET /ready
```

### Monitoring

```text
GET /metrics
GET /model-info
```

### Reference Data

```text
GET /branches
GET /products
```

### Forecasting

```text
POST /forecast
POST /forecast/future
GET /forecast/model
```

### Inventory

```text
POST /inventory/recommend
```

The API also contains request validation, structured request logging,
operational metrics, and exception handling.

The FastAPI application configures logging and tracks total requests,
successful requests, failed requests, and total latency in memory.
fileciteturn22file0L23-L38

---

## 10. Dashboard Layer

Streamlit provides the user-facing interface.

The dashboard communicates with FastAPI rather than loading the ML model
directly.

```text
User
 │
 ▼
Streamlit Dashboard
 │
 │ HTTP
 ▼
FastAPI
 │
 ▼
MLflow / Data / Business Logic
```

This separation is important because the dashboard is treated as a
client of the API.

It avoids coupling presentation logic directly to model implementation.

The Docker Compose configuration sets the dashboard API base URL to
`http://api:8000`. fileciteturn21file0L60-L79

---

## 11. Monitoring Architecture

The monitoring layer currently operates at the application level.

```text
FastAPI
   │
   ├── Request Logging
   │
   ├── Request Count
   │
   ├── Success / Failure Count
   │
   ├── Average Latency
   │
   ├── Uptime
   │
   ├── Model Version
   │
   └── Readiness
           │
           ▼
      Streamlit Monitoring
```

The `/metrics` endpoint exposes:

- total requests,
- successful requests,
- failed requests,
- average latency,
- uptime.

fileciteturn22file0L310-L336

The `/ready` endpoint verifies that the forecasting model can be loaded
and that the configured MLflow model alias resolves to a registered
model version.

---

## 12. Health vs Readiness

The system deliberately separates health and readiness.

### Health

```text
/health
```

Answers:

> Is the API process running?

### Readiness

```text
/ready
```

Answers:

> Is the API capable of serving predictions using the configured model?

This distinction becomes important when the application is later
deployed behind a container orchestrator or cloud platform.

---

## 13. Container Architecture

The Dockerfile uses:

```text
python:3.11-slim
```

and installs the project's Python dependencies before copying:

```text
src/
models/
data/
dashboard/
```

The API container exposes port 8000 and runs Uvicorn with:

```text
src.api.main:app
```

The same Dockerfile is also used as the build source for the dashboard
service. fileciteturn21file0L81-L101

---

## 14. Configuration

Runtime MLflow configuration is environment-based.

Default:

```text
MLFLOW_TRACKING_URI=http://localhost:5000
```

Inside Docker Compose, the API receives:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

This allows the same application code to run in both local and
containerized environments.

The project also keeps model paths, MLflow model name, model alias, and
inventory policy parameters in `src/config.py`.
fileciteturn21file0L117-L176

---

## 15. End-to-End Request Example

A future forecast request follows this path:

```text
User
 │
 │ branch = B001
 │ product = P024
 │ horizon = 7
 ▼
Streamlit
 │
 │ POST /forecast/future
 ▼
FastAPI
 │
 ├── Pydantic validation
 │
 ├── Load demand history
 │
 ├── Create features
 │
 ├── Preprocess features
 │
 ├── Resolve:
 │      fb_demand_forecasting@champion
 │
 ├── XGBoost prediction
 │
 └── Recursive forecasting
 │
 ▼
JSON response
 │
 ▼
Streamlit
 │
 ├── KPI cards
 ├── Forecast chart
 └── Forecast table
```

This creates a clear separation between:

```text
Presentation
     ↓
API
     ↓
Inference
     ↓
Model Registry
     ↓
Model
```

---

## 16. Engineering Principles

### Separation of Concerns

Data processing, feature engineering, inference, API serving, business
logic, and presentation are kept in separate layers.

### Reproducibility

Docker and pinned dependencies reduce environment differences.

### Model Registry-Based Serving

The API resolves the active model through an MLflow alias instead of
embedding a version directly in application code.

### Validation Before Inference

API requests are validated before they reach the forecasting logic.

### Observable Runtime

The API exposes operational metrics and logs request latency and status.

### Business Logic Outside the Model

Inventory decisions are implemented as explicit business logic rather
than hidden inside the forecasting model.

---

## 17. Current Architecture vs Future Architecture

### Current

```text
CSV
 ↓
Python Feature Engineering
 ↓
MLflow
 ↓
FastAPI
 ↓
Streamlit
```

### Future

```text
POS / ERP / Inventory / CRM
            ↓
     Data Ingestion Layer
            ↓
      PostgreSQL / Data Lake
            ↓
       Feature Pipeline
            ↓
     ML Training Pipeline
            ↓
      MLflow Tracking
            ↓
      Model Registry
            ↓
       FastAPI Serving
            ↓
      Business Services
            ↓
       Dashboard / Apps
            ↓
      Monitoring System
```

The current implementation intentionally stops before the infrastructure
complexity of a full cloud production platform.

---

## 18. Architecture Summary

F&B Intelligence is structured as an end-to-end system rather than a
standalone machine learning notebook.

The key flow is:

```text
Data
 ↓
Features
 ↓
Model
 ↓
MLflow Registry
 ↓
FastAPI
 ↓
Business Decision
 ↓
Dashboard
 ↓
Monitoring
```

The architecture demonstrates the complete path from machine learning
experimentation to model serving and operational consumption while
keeping the current implementation small enough to remain reproducible
and understandable.
