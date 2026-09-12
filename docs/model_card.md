# Model Card

## F&B Intelligence --- Demand Forecasting

## 1. Model Overview

**Model name:** `fb_demand_forecasting`

**Active registry alias:** `champion`

**Model type:** XGBoost Regressor

**Task:** Daily product demand forecasting

**Prediction unit:**

```text
branch × product × date
```

The model predicts expected product demand quantity for a selected
branch and product.

The active model is served through FastAPI and resolved from the MLflow
Model Registry using the `champion` alias.

---

## 2. Intended Use

The model is intended to support F&B operational decision-making.

Primary use cases:

- estimate future product demand,
- identify expected demand peaks,
- support preparation planning,
- provide an input for inventory planning,
- support operational scenario analysis.

The model is a **decision-support component**.

It should not be interpreted as an autonomous business decision-maker.

---

## 3. Users

Potential users include:

- Operations Manager,
- Management,
- Marketing / Business Team,
- AI Engineer / Data Scientist.

The current dashboard exposes forecasting results and operational
summaries through Streamlit.

---

## 4. Training Data

The model was trained using the project's synthetic F&B demand dataset.

The primary dataset is:

```text
data/raw/daily_demand.csv
```

The forecasting feature pipeline creates historical demand, calendar,
promotion, and rolling-statistical features.

The final model uses:

```text
branch_id
product_id
day_of_week
day_of_month
month
week_of_year
is_weekend
is_payday
is_holiday
discount_percentage
lag_1
lag_7
lag_14
lag_21
lag_28
rolling_mean_7
rolling_std_7
rolling_mean_14
rolling_std_14
rolling_mean_28
rolling_std_28
```

Target:

```text
quantity
```

---

## 5. Data Split

The model evaluation uses chronological splits.

### Training

```text
2024-01-29 → 2025-06-30
```

### Validation

```text
2025-07-01 → 2025-09-30
```

### Test

```text
2025-10-01 → 2025-12-31
```

Random splitting was avoided because the task is time-dependent.

---

## 6. Feature Engineering

Historical demand is represented through lag features:

```text
lag_1
lag_7
lag_14
lag_21
lag_28
```

Rolling statistics capture recent demand level and variability:

```text
rolling_mean_7
rolling_std_7
rolling_mean_14
rolling_std_14
rolling_mean_28
rolling_std_28
```

The rolling calculations are shifted so that the target date is not
included in its own historical feature values.

Calendar and contextual features include:

```text
day_of_week
day_of_month
month
week_of_year
is_weekend
is_payday
is_holiday
discount_percentage
```

---

## 7. Model Selection

The project followed:

```text
Baseline
   ↓
Experiment
   ↓
Evaluate
   ↓
Select
```

Approaches explored included:

- Naive forecasting,
- Seasonal Naive-7,
- Random Forest,
- XGBoost,
- LSTM.

The final production model is XGBoost.

The model was selected based on forecasting performance on chronological
validation/test data while also considering the practicality of serving
the model as part of the application.

---

## 8. Model Configuration

The selected XGBoost configuration used during the final modeling
workflow was:

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

The preprocessing pipeline is stored separately as:

```text
models/forecasting/preprocessor.joblib
```

The model artifact is also retained locally as:

```text
models/forecasting/xgb_forecasting.joblib
```

The deployed API, however, resolves the active forecasting model through
MLflow Model Registry.

---

## 9. Evaluation Metrics

The model is evaluated using:

### MAE

Mean Absolute Error.

Measures average absolute difference between actual and predicted
demand.

Lower is better.

### RMSE

Root Mean Squared Error.

Penalizes larger forecasting errors more strongly than MAE.

Lower is better.

### MAPE

Mean Absolute Percentage Error.

Expresses forecasting error in percentage terms.

Lower is better.

### R²

Coefficient of determination.

Measures how much variance in the target is explained by the model.

Higher is generally better.

---

## 10. Final Test Performance

The final XGBoost model achieved:

Metric Test Result

---

MAE 13.52
RMSE 18.69
MAPE 21.22%
R² 0.760

For comparison, the Seasonal Naive-7 baseline achieved:

Metric Test Result

---

MAE 19.47
RMSE 26.85

Approximate improvement of XGBoost over the Seasonal Naive-7 baseline:

Metric Improvement

---

MAE 30.6% lower
RMSE 30.4% lower

These improvements describe performance on the project's synthetic test
set.

---

## 11. Interpretation

The test results indicate that the XGBoost model captures meaningful
patterns in the simulated demand data beyond a simple weekly seasonal
baseline.

An R² of approximately 0.76 indicates that the model explains a
substantial portion of the variation in the synthetic test data.

However, an MAPE of approximately 21.22% also indicates that prediction
errors remain material.

The model should therefore be treated as an operational forecasting aid
rather than a source of exact demand values.

---

## 12. Recursive Forecasting

The API supports future forecasting through recursive prediction.

Conceptually:

```text
Historical Data
      ↓
Predict t+1
      ↓
Use prediction t+1
      ↓
Predict t+2
      ↓
Use prediction t+2
      ↓
...
```

This allows the API to produce multi-day forecasts without requiring
future observed target values.

The current API limits the requested forecast horizon to a maximum of 30
days.

---

## 13. Serving

The model is served through:

```text
FastAPI
```

The production-oriented serving flow is:

```text
Client
  ↓
FastAPI
  ↓
MLflow Model Registry
  ↓
fb_demand_forecasting@champion
  ↓
XGBoost
  ↓
Prediction
```

The active model is controlled by the MLflow alias:

```text
champion
```

This allows a new model version to be promoted without changing the
API's model-loading code.

---

## 14. Monitoring

The API currently provides operational monitoring endpoints:

```text
GET /health
GET /ready
GET /metrics
GET /model-info
```

### `/health`

Checks whether the API process is running.

### `/ready`

Checks whether the API can access the configured forecasting model
through MLflow.

### `/metrics`

Provides:

- total requests,
- successful requests,
- failed requests,
- average latency,
- uptime.

### `/model-info`

Provides active model metadata including:

- model name,
- alias,
- version,
- model type,
- run ID,
- registry status.

---

## 15. Model Version

The current registered production champion at the time of this model
card is:

```text
Model:
fb_demand_forecasting

Alias:
champion

Version:
3
```

The model registry is treated as the source of truth for the active
serving model.

---

## 16. Limitations

### Synthetic Data

The most important limitation is that the training and evaluation
dataset is synthetic.

The data contains intentionally designed patterns such as:

- weekly seasonality,
- payday effects,
- holiday effects,
- promotion effects,
- branch/product differences,
- external contextual effects.

Therefore, strong test performance does not prove that the model will
achieve the same performance on real F&B data.

---

### Distribution Shift

Real operations can experience events that are absent from the synthetic
dataset, including:

- sudden product popularity changes,
- menu changes,
- store closures,
- supply disruptions,
- unusual campaigns,
- competitor activity,
- extreme weather,
- operational constraints.

These can cause model performance to degrade.

---

### Long-Horizon Forecasting

Recursive forecasting can accumulate prediction errors as the horizon
increases because earlier predictions become inputs to later
predictions.

---

### External Variables

The current production feature set does not directly use same-day
observed weather as a forecasting input.

A future version could incorporate reliable weather forecasts or other
exogenous predictions.

---

### Monitoring Metrics

Current operational metrics are stored in application memory.

They reset when the API process restarts and are therefore not a
replacement for a persistent monitoring system such as
Prometheus/Grafana or a cloud observability platform.

---

## 17. Fairness and Safety Considerations

This is an operational demand forecasting model rather than a model used
for sensitive individual-level decisions.

The primary risks are operational:

- overestimating demand,
- underestimating demand,
- inappropriate inventory decisions,
- excessive confidence in model predictions.

Recommended operational practice:

> Use forecasts together with inventory state, operational constraints,
> and human judgment.

---

## 18. Business Impact Interpretation

The project does **not** claim:

- a percentage reduction in real stockouts,
- a percentage reduction in inventory costs,
- revenue uplift,
- waste reduction,
- increased customer retention.

Those metrics require real operational deployment and controlled
measurement.

The demonstrated results are model evaluation results on synthetic data.

---

## 19. Recommended Production Evaluation

Before real operational adoption, evaluate the model using:

### Offline Evaluation

- MAE,
- RMSE,
- MAPE,
- R²,
- performance by branch,
- performance by product,
- performance by forecast horizon.

### Operational Evaluation

- stockout rate,
- service level,
- inventory value,
- waste,
- forecast bias,
- order recommendation accuracy.

### Monitoring

Track:

```text
Data Drift
    ↓
Prediction Drift
    ↓
Forecast Error
    ↓
Business KPI
```

This provides a complete model lifecycle rather than monitoring only API
availability.

---

## 20. Future Improvements

Potential next iterations include:

1.  train on real POS data,
2.  add weather forecast features,
3.  add richer promotion features,
4.  evaluate branch/product-specific performance,
5.  introduce automated model retraining,
6.  persist monitoring metrics,
7.  add data drift detection,
8.  add prediction drift monitoring,
9.  implement automated model promotion,
10. evaluate inventory decisions against real procurement outcomes.

---

## 21. Model Lifecycle

The intended lifecycle is:

```text
Data
 ↓
Training
 ↓
Experiment Tracking
 ↓
Evaluation
 ↓
Model Registration
 ↓
Promotion
 ↓
Serving
 ↓
Monitoring
 ↓
Retraining
```

This lifecycle is the main engineering objective of the model component.

---

## 22. Summary

`fb_demand_forecasting` is an XGBoost-based daily demand forecasting
model designed as a component of the F&B Intelligence decision-support
system.

It achieved:

```text
MAE  = 13.52
RMSE = 18.69
MAPE = 21.22%
R²   = 0.760
```

on the synthetic chronological test set.

The model is registered in MLflow, served through FastAPI, consumed by
the Streamlit dashboard, and supported by operational health, readiness,
model-information, and request-metric endpoints.

Its primary limitation is the synthetic nature of the dataset. The model
should therefore be presented as a complete AI engineering prototype and
decision-support system, not as a validated production model for real
F&B operations.
