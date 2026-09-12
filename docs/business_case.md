# Business Case

## F&B Intelligence

### 1. Background

Food & Beverage operations must continuously balance customer demand
with product and ingredient availability.

Demand that is underestimated can increase the risk of stockouts and
operational disruption. Demand that is overestimated can contribute to
unnecessary inventory and holding costs.

F&B Intelligence was designed as a decision-support system that connects
demand forecasting with inventory optimization in an end-to-end AI
engineering workflow.

---

## 2. Business Problem

The project focuses on two connected operational problems.

### 2.1 Demand Uncertainty

F&B demand is not constant. It can vary according to:

- day of week,
- weekend behavior,
- payday,
- holidays,
- promotions,
- branch,
- product,
- historical demand patterns.

Operations therefore need a way to estimate future demand before making
preparation and inventory decisions.

### 2.2 Inventory Uncertainty

Product demand eventually translates into ingredient consumption.

If expected demand is not incorporated into inventory planning,
operations may face:

- insufficient ingredient stock,
- excessive inventory,
- inefficient ordering,
- higher holding costs,
- operational disruption.

The project therefore connects product-level demand forecasting with
ingredient-level inventory recommendations.

---

## 3. Proposed Solution

F&B Intelligence provides an end-to-end decision-support workflow:

```text
Historical Demand
       ↓
Feature Engineering
       ↓
Demand Forecast
       ↓
Product Demand
       ↓
Recipe Mapping
       ↓
Ingredient Demand
       ↓
Inventory Policy
       ↓
ORDER / HOLD Recommendation
```

The forecasting service predicts future demand for a selected branch and
product.

The inventory service uses product recipes to translate demand into
ingredient usage and applies an inventory policy based on lead time,
demand variability, safety stock, and service-level assumptions.

---

## 4. Target Users

### Operations Manager

Primary decisions:

- expected product demand,
- demand peaks,
- inventory preparation,
- ingredient reorder quantities.

The dashboard provides forecast summaries and inventory recommendations
intended to support these decisions.

### Management

Potential questions:

- Is demand increasing or decreasing?
- Which products or branches require attention?
- How much inventory is likely to be required?
- Is the forecasting system operating correctly?

### Marketing / Business Team

The forecasting API accepts promotion discount, payday, and holiday
assumptions. This allows the system architecture to be extended toward
promotion-aware planning.

### AI / Engineering Team

The system also provides technical interfaces for:

- model serving,
- model registry,
- model version tracking,
- operational monitoring,
- reproducible containerized execution.

---

## 5. Business Use Cases

### Use Case 1 --- Demand Forecasting

**Question:**

> How much demand is expected for a particular product at a particular
> branch?

**Input:**

- branch,
- product,
- forecast start date,
- forecast horizon,
- promotion assumption,
- payday assumption,
- holiday assumption.

**Output:**

- predicted demand for each future date,
- forecast summary,
- peak demand date,
- operational insight.

---

### Use Case 2 --- Inventory Recommendation

**Question:**

> Which ingredients should be ordered and how much should be ordered?

The system derives ingredient demand from product demand and recipe
relationships.

The inventory policy then calculates target stock and recommended order
quantity.

**Output:**

- ingredient,
- current stock,
- average daily usage,
- lead time,
- reorder point,
- target stock,
- recommended order quantity,
- estimated order value,
- action (`ORDER` / `HOLD`).

---

## 6. Decision Logic

The inventory policy uses a service-level-oriented approach.

The selected production policy is:

Parameter Value

---

Policy Lean 85%
Target service level 85%
Z-value 1.036
Review period 1 day

Target stock is calculated using:

```text
Target Stock =
    Average Daily Usage × (Lead Time + Review Period)
    +
    z × Daily Usage Std × √(Lead Time + Review Period)
```

Inventory position is:

```text
Inventory Position =
    On Hand + On Order
```

Recommended order quantity is:

```text
Order Quantity =
    max(0, Target Stock - Inventory Position)
```

The policy was selected from evaluated service-level alternatives using
the project's criterion of satisfying the service-level requirement
while minimizing average inventory value.

---

## 7. Business-Oriented Success Metrics

The project distinguishes model metrics from business metrics.

### Forecasting Metrics

- MAE
- RMSE
- MAPE
- R²

### Inventory Metrics

- stockout rate,
- service level,
- average inventory value,
- recommended order quantity,
- estimated order value.

### Potential Future Business Metrics

When real operational data becomes available, the system can
additionally evaluate:

- stockout reduction,
- inventory holding-cost reduction,
- forecast error by branch/product,
- order recommendation accuracy,
- waste reduction,
- revenue impact,
- customer retention,
- average order value.

These future metrics are not claimed as achieved business results
because the current project uses synthetic data.

---

## 8. Observed Model Result

The final XGBoost forecasting model achieved:

Metric Test Result

---

MAE 13.52
RMSE 18.69
MAPE 21.22%
R² 0.760

Compared with the Seasonal Naive-7 baseline:

Metric Seasonal Naive-7 XGBoost

---

MAE 19.47 13.52
RMSE 26.85 18.69

This corresponds to approximately:

- **30.6% lower MAE**
- **30.4% lower RMSE**

These are **model performance results on the project's synthetic test
data**, not measured business improvements in a real F&B operation.

---

## 9. Inventory Simulation Result

The inventory policy was evaluated through simulation using alternative
service-level settings.

The tested policies included service levels from 80% through 99%.

The selected Lean 85% policy achieved approximately:

- stockout rate: **1.958%**
- service level: **98.042%**
- average inventory value: **2.295 million**

The selection was based on the project's optimization criterion rather
than simply choosing the highest service level.

---

## 10. Why an End-to-End System?

A forecasting model alone does not automatically create an operational
decision.

The project therefore treats machine learning as one component of a
larger system:

```text
Model
  ↓
Prediction
  ↓
Business Logic
  ↓
Decision
  ↓
API
  ↓
User Interface
  ↓
Monitoring
```

This allows the forecasting output to be consumed by an application
instead of remaining inside a notebook.

---

## 11. Production-Oriented Workflow

The project follows:

> **Baseline → Experiment → Evaluate → Explain → Deploy → Monitor**

### Baseline

Simple forecasting methods establish a reference point.

### Experiment

Multiple approaches were evaluated, including Random Forest, XGBoost,
and LSTM.

### Evaluate

Models were evaluated using chronological forecasting splits and
standard regression metrics.

### Explain

The project documents model selection, assumptions, and limitations.

### Deploy

The selected model is registered with MLflow and served through FastAPI.

### Monitor

The API exposes health, readiness, model information, request counts,
failures, latency, and uptime.

---

## 12. Current Scope

The implemented MVP focuses on:

1.  demand forecasting,
2.  ingredient-level inventory optimization,
3.  API model serving,
4.  dashboard visualization,
5.  MLflow experiment tracking and model registry,
6.  Docker-based service orchestration,
7.  operational monitoring.

The broader architecture was designed with future F&B intelligence
capabilities in mind, including customer segmentation, product
recommendation, and deeper business analytics.

---

## 13. Limitations

### Synthetic Dataset

The current data is synthetic. Patterns were intentionally embedded to
create a realistic development scenario.

Therefore:

> Model performance demonstrates the behavior of the implemented system
> on simulated data, not its expected accuracy on a real company's
> operational data.

### Inventory Scope

The current inventory recommendation endpoint operates at the modeled
ingredient/branch scope and does not expose a direct branch/product
filter through its API request.

### Operational Monitoring

Current API metrics are stored in memory and reset when the API
restarts.

### Deployment

Docker Compose provides a reproducible production-like environment, but
the project is not yet a cloud production deployment with autoscaling or
Kubernetes.

---

## 14. Expected Value in a Real Deployment

With real POS, inventory, recipe, promotion, and operational data, the
same architecture could support:

```text
Real POS Data
      ↓
Demand Forecast
      ↓
Expected Product Sales
      ↓
Ingredient Requirement
      ↓
Inventory Decision
      ↓
Procurement Planning
```

Potential business value would then be evaluated empirically through
changes in:

- forecast accuracy,
- stockout frequency,
- inventory value,
- waste,
- service level,
- ordering efficiency.

The project deliberately does not claim these benefits without
real-world measurements.

---

## 15. Future Business Expansion

The architecture can be expanded into a broader F&B intelligence
platform:

### Customer Intelligence

Segment customers based on behavioral and transaction features.

### Product Recommendation

Recommend products using transaction history and product relationships.

### Promotion Intelligence

Estimate demand response under different promotion assumptions.

### Branch Intelligence

Compare demand, inventory, and operational performance between branches.

### Automated Decision Support

Eventually combine forecasts, inventory state, and business constraints
into automated planning recommendations.

---

## 16. Business Case Summary

The core business problem is:

> **How can an F&B operation use historical demand and operational data
> to anticipate demand and make better inventory decisions?**

The implemented solution is:

> **A production-oriented AI decision-support system that forecasts
> product demand and translates that forecast into ingredient-level
> inventory recommendations, exposed through APIs and a dashboard and
> supported by MLflow-based model management and operational
> monitoring.**
