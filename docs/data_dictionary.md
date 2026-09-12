# Data Dictionary

## F&B Intelligence

This document describes the datasets used by the F&B Intelligence
project and their roles in forecasting, inventory optimization, and
business analytics.

The current MVP uses CSV files under `data/raw/`.

---

## 1. Dataset Overview

---

Dataset Purpose Main Use

---

`branches.csv` Branch master data Branch reference

`products.csv` Product master data Product reference

`ingredients.csv` Ingredient master data Inventory reference

`product_ingredients.csv` Recipe / Convert product demand
product-ingredient to ingredient demand
mapping

`customers.csv` Customer master data Customer analytics

`sales.csv` Transaction-level sales Customer/product
analytics

`promotions.csv` Promotion information Demand feature /
scenario input

`weather.csv` Daily weather Exploratory/business
information analysis

`holidays.csv` Holiday calendar Forecasting feature

`inventory_transactions.csv` Ingredient inventory Inventory simulation
movement and recommendation

`daily_demand.csv` Daily demand by branch Primary forecasting
and product dataset

---

---

# 2. Forecasting Dataset

## `daily_demand.csv`

This is the primary dataset for the demand forecasting system.

The main grain is:

```text
one row = one date × one branch × one product
```

### Columns

---

Column Type Description Role

---

`date` date Demand date Time index

`branch_id` string Branch identifier Entity feature

`product_id` string Product Entity feature
identifier

`quantity` numeric Product demand Target
quantity

`discount_percentage` numeric Promotion Feature
discount  
 percentage

`is_payday` integer/binary Payday indicator Feature

`is_holiday` integer/binary Holiday indicator Feature

---

### Derived Forecasting Features

The raw demand data is transformed into additional features before model
training.

Feature Description

---

`day_of_week` Day of week extracted from date
`day_of_month` Day of month extracted from date
`month` Month extracted from date
`week_of_year` ISO week number
`is_weekend` Weekend indicator
`lag_1` Demand one day earlier
`lag_7` Demand seven days earlier
`lag_14` Demand fourteen days earlier
`lag_21` Demand twenty-one days earlier
`lag_28` Demand twenty-eight days earlier
`rolling_mean_7` Previous 7-day mean demand
`rolling_std_7` Previous 7-day demand standard deviation
`rolling_mean_14` Previous 14-day mean demand
`rolling_std_14` Previous 14-day demand standard deviation
`rolling_mean_28` Previous 28-day mean demand
`rolling_std_28` Previous 28-day demand standard deviation

The rolling statistics are shifted so that the target date is not
included in its own historical feature calculation.

---

# 3. Branch Master Data

## `branches.csv`

Contains the branch reference data used to identify operating locations.

Typical usage:

```text
branch_id → branch-level demand history
```

The forecasting model uses `branch_id` to distinguish demand behavior
between branches.

---

# 4. Product Master Data

## `products.csv`

Contains the product reference data.

The forecasting system uses `product_id` to distinguish demand behavior
between products.

Product information can also be joined with recipe relationships for
inventory calculations.

---

# 5. Ingredient Master Data

## `ingredients.csv`

Contains the ingredient reference data used by the inventory module.

The inventory system operates at the ingredient level because product
sales must ultimately be translated into ingredient consumption.

Important conceptual entities include:

```text
ingredient_id
ingredient_name
unit
cost
```

The exact columns should be treated according to the generated dataset
schema rather than assumed production ERP standards.

---

# 6. Recipe Mapping

## `product_ingredients.csv`

This table defines the relationship between products and ingredients.

Conceptually:

```text
Product
   │
   ├── Ingredient A × quantity
   ├── Ingredient B × quantity
   └── Ingredient C × quantity
```

This mapping is critical for converting product demand into ingredient
demand.

The API loads this dataset together with daily demand, ingredients, and
inventory transactions when constructing inventory recommendations.

---

# 7. Customer Master Data

## `customers.csv`

Contains customer-level reference information.

It exists primarily to support the broader F&B Intelligence scope,
particularly future customer segmentation and customer analytics.

The current production API focuses on forecasting and inventory rather
than exposing a customer segmentation endpoint.

---

# 8. Sales Transactions

## `sales.csv`

Contains transaction-level product sales.

The sales dataset is intended for:

- customer analytics,
- product analytics,
- transaction behavior analysis,
- recommendation experiments,
- future retention analysis.

### Important Data Grain

A `transaction_id` is not necessarily unique by itself because one
transaction can contain multiple products.

Therefore, transaction-level uniqueness should be evaluated using a
composite key such as:

```text
transaction_id + product_id
```

This is important when performing transaction aggregation or
customer/product analysis.

---

# 9. Promotions

## `promotions.csv`

Contains promotion information associated with demand.

Promotion-related information is used to represent the effect of
promotions in the synthetic demand-generation process and forecasting
scenarios.

The forecasting API accepts:

```text
discount_percentage
```

as an explicit future forecasting assumption.

---

# 10. Weather

## `weather.csv`

Contains daily weather-related variables used during exploratory
analysis and synthetic demand generation.

Weather variables can represent external demand drivers.

The current forecasting feature set does not directly depend on same-day
weather observations because actual future weather observations would
not necessarily be available at prediction time.

A future production implementation could use:

```text
Weather Forecast
      ↓
Forecasting Features
      ↓
Demand Prediction
```

instead of relying on observed same-day weather.

---

# 11. Holidays

## `holidays.csv`

Contains holiday calendar information.

Holiday information is used to create the `is_holiday` forecasting
feature.

This allows the forecasting model to distinguish ordinary dates from
holiday periods.

---

# 12. Inventory Transactions

## `inventory_transactions.csv`

Contains inventory movement information for ingredients.

The inventory module uses this data together with ingredient demand and
recipe mappings to simulate inventory behavior and calculate
recommendations.

Conceptual flow:

```text
Product Demand
      ↓
Recipe Mapping
      ↓
Ingredient Usage
      ↓
Inventory Transactions
      ↓
Inventory Position
      ↓
ORDER / HOLD
```

---

# 13. Relationships Between Datasets

The main entity relationships are:

```text
branches
   │
   └──────────────┐
                  │
products ─────────┤
   │              │
   │              ▼
   │        daily_demand
   │
   └── product_ingredients ── ingredients
                                  │
                                  ▼
                         inventory_transactions
```

Customer analytics follows a separate transaction path:

```text
customers
    │
    ▼
sales
    │
    ├── products
    └── branches
```

Business context:

```text
promotions ──┐
holidays ────┼──→ demand / forecasting
weather ─────┘
```

---

# 14. Forecasting Feature Contract

The final forecasting model uses the following feature groups:

### Categorical / Entity Features

```text
branch_id
product_id
```

### Calendar Features

```text
day_of_week
day_of_month
month
week_of_year
is_weekend
is_payday
is_holiday
```

### Promotion Feature

```text
discount_percentage
```

### Lag Features

```text
lag_1
lag_7
lag_14
lag_21
lag_28
```

### Rolling Features

```text
rolling_mean_7
rolling_std_7
rolling_mean_14
rolling_std_14
rolling_mean_28
rolling_std_28
```

---

# 15. Target Variable

The primary forecasting target is:

```text
quantity
```

It represents the daily demand quantity for a product at a branch.

The forecasting task is therefore:

```text
Historical demand + contextual features
                ↓
             XGBoost
                ↓
       Future quantity
```

---

# 16. Data Leakage Considerations

Time-series leakage is treated as a critical concern.

The forecasting feature pipeline uses historical lag and rolling
features rather than future target information.

For rolling statistics:

```text
rolling(t)
```

is calculated from observations before `t`.

For example:

```text
Target: 2025-01-10

7-day rolling mean:
2025-01-03 ... 2025-01-09
```

rather than including:

```text
2025-01-10
```

This prevents the target value from becoming part of its own input.

---

# 17. Data Split

The forecasting dataset uses chronological splitting rather than random
train/test splitting.

```text
Train
2024-01-29 → 2025-06-30

Validation
2025-07-01 → 2025-09-30

Test
2025-10-01 → 2025-12-31
```

This better represents the intended production scenario:

```text
Past → Train
Recent Past → Validation
Future Period → Test
```

---

# 18. Data Quality Expectations

The forecasting dataset is expected to satisfy:

- no negative demand,
- valid dates,
- valid branch identifiers,
- valid product identifiers,
- no impossible target values,
- chronological ordering within branch/product series,
- no target leakage through lag/rolling features.

Inventory data is expected to maintain consistent stock-flow
relationships.

---

# 19. Data Lineage

The forecasting lineage is:

```text
daily_demand.csv
       ↓
create_calendar_features()
       ↓
create_lag_features()
       ↓
create_forecasting_dataset()
       ↓
preprocessor
       ↓
XGBoost
       ↓
forecast
```

The inventory lineage is:

```text
daily_demand.csv
       +
product_ingredients.csv
       ↓
ingredient demand
       +
ingredients.csv
       +
inventory_transactions.csv
       ↓
inventory policy
       ↓
recommendation
```

---

# 20. Data Scope and Limitations

The current dataset is synthetic and was created specifically for
development and evaluation of the system.

Consequently:

- field meanings are designed for the project,
- demand patterns are simulated,
- relationships between variables are simulated,
- model performance should not be interpreted as real-world business
  performance.

For a production deployment, these datasets would be replaced or
supplemented by authoritative operational sources such as POS, ERP,
inventory, recipe/BOM, CRM, promotion, and external data systems.

---

# 21. Summary

The most important dataset for the current AI system is:

```text
daily_demand.csv
```

because it drives the demand forecasting model.

The most important inventory relationship is:

```text
daily_demand
      ↓
product_ingredients
      ↓
ingredient demand
      ↓
inventory_transactions
      ↓
inventory recommendation
```

The project intentionally separates **raw business data**, **derived
model features**, and **business decision logic** so that each layer can
be replaced or extended independently.
