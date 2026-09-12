from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.features.forecasting import (
    load_demand,
    create_forecasting_dataset,
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "F&B Intelligence - Demand Forecasting"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

TRAIN_END = pd.Timestamp("2025-06-30")
VAL_END = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2025-10-01")


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "branch_id",
    "product_id",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "is_payday",
    "is_holiday",
    "discount_percentage",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
]


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

EXPERIMENTS = {

    "XGBoost Baseline": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },

    "XGBoost Tuned": {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    "XGBoost Alternative": {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    non_zero_mask = y_true != 0

    if non_zero_mask.any():

        mape = (
            abs(
                (
                    y_true[non_zero_mask]
                    - y_pred[non_zero_mask]
                )
                / y_true[non_zero_mask]
            ).mean()
            * 100
        )

    else:

        mape = 0.0

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MLflow Forecasting Experiment Comparison")
    print("=" * 60)

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading forecasting dataset...")

    demand = load_demand()

    forecasting_df = create_forecasting_dataset(
        demand
    )

    print(
        f"Dataset shape: {forecasting_df.shape}"
    )

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train_df = forecasting_df[
        forecasting_df["date"] <= TRAIN_END
    ].copy()

    val_df = forecasting_df[
        (
            forecasting_df["date"] > TRAIN_END
        )
        & (
            forecasting_df["date"] <= VAL_END
        )
    ].copy()

    test_df = forecasting_df[
        forecasting_df["date"] >= TEST_START
    ].copy()

    print("\nSplit:")

    print(
        f"Train: {train_df['date'].min().date()} "
        f"→ {train_df['date'].max().date()}"
    )

    print(
        f"Val  : {val_df['date'].min().date()} "
        f"→ {val_df['date'].max().date()}"
    )

    print(
        f"Test : {test_df['date'].min().date()} "
        f"→ {test_df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train = train_df[FEATURES]
    y_train = train_df["quantity"]

    X_val = val_df[FEATURES]
    y_val = val_df["quantity"]

    X_test = test_df[FEATURES]
    y_test = test_df["quantity"]

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    categorical_features = [
        "branch_id",
        "product_id",
    ]

    numeric_features = [
        feature
        for feature in FEATURES
        if feature not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    print("\nFitting preprocessor...")

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_val_transformed = preprocessor.transform(
        X_val
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for run_name, params in EXPERIMENTS.items():

        print("\n" + "-" * 60)
        print(f"Running: {run_name}")
        print("-" * 60)

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        with mlflow.start_run(
            run_name=run_name
        ):

            # ------------------------------------------------
            # Tags
            # ------------------------------------------------

            mlflow.set_tags(
                {
                    "project": "F&B Intelligence",
                    "task": "demand_forecasting",
                    "model_family": "XGBoost",
                    "evaluation_strategy": "temporal_split",
                }
            )

            # ------------------------------------------------
            # Parameters
            # ------------------------------------------------

            mlflow.log_params(
                params
            )

            mlflow.log_param(
                "feature_count",
                len(FEATURES),
            )

            mlflow.log_param(
                "train_end",
                str(TRAIN_END.date()),
            )

            mlflow.log_param(
                "validation_end",
                str(VAL_END.date()),
            )

            mlflow.log_param(
                "test_start",
                str(TEST_START.date()),
            )

            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            print("Training...")

            model.fit(
                X_train_transformed,
                y_train,
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_predictions = model.predict(
                X_val_transformed
            )

            val_metrics = calculate_metrics(
                y_val.to_numpy(),
                val_predictions,
            )

            # ------------------------------------------------
            # Test
            # ------------------------------------------------

            test_predictions = model.predict(
                X_test_transformed
            )

            test_metrics = calculate_metrics(
                y_test.to_numpy(),
                test_predictions,
            )

            # ------------------------------------------------
            # Log metrics
            # ------------------------------------------------

            mlflow.log_metrics(
                {
                    f"val_{key}": value
                    for key, value
                    in val_metrics.items()
                }
            )

            mlflow.log_metrics(
                {
                    f"test_{key}": value
                    for key, value
                    in test_metrics.items()
                }
            )

            # ------------------------------------------------
            # Log model
            # ------------------------------------------------

            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
            )

            # ------------------------------------------------
            # Output
            # ------------------------------------------------

            print("\nValidation:")

            for metric, value in val_metrics.items():

                print(
                    f"{metric.upper():<6}: "
                    f"{value:.6f}"
                )

            print("\nTest:")

            for metric, value in test_metrics.items():

                print(
                    f"{metric.upper():<6}: "
                    f"{value:.6f}"
                )

            print(
                f"\nRun ID: "
                f"{mlflow.active_run().info.run_id}"
            )

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()