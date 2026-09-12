from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import (
    MODEL_DIR,
    PREPROCESSOR_PATH,
    MODEL_PATH,
    METADATA_PATH,
)

from src.features.forecasting import (
    load_demand,
    create_forecasting_dataset,
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "F&B Intelligence - Demand Forecasting"

TRAIN_END = pd.Timestamp("2025-06-30")
VAL_END = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2025-10-01")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


# ============================================================
# XGBOOST PARAMETERS
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
}


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
# HELPERS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate forecasting evaluation metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    # Avoid division by zero.
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


def save_feature_list(output_path: Path):
    """
    Save forecasting feature list as JSON.
    """

    output_path.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "feature_count": len(FEATURES),
            },
            indent=4,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MLflow Forecasting Experiment Tracking")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate artifacts
    # --------------------------------------------------------

    required_files = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        METADATA_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    print("\nArtifacts found:")
    print(f"Model        : {MODEL_PATH}")
    print(f"Preprocessor : {PREPROCESSOR_PATH}")
    print(f"Metadata     : {METADATA_PATH}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print(
        f"\nLoaded model: {type(model).__name__}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = json.loads(
        METADATA_PATH.read_text()
    )

    # --------------------------------------------------------
    # Load forecasting dataset
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
    # Validate feature availability
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in forecasting_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
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

    print("\nTemporal split:")
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

    X_val = val_df[FEATURES]
    y_val = val_df["quantity"]

    X_test = test_df[FEATURES]
    y_test = test_df["quantity"]

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    X_val_transformed = preprocessor.transform(
        X_val
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    val_predictions = model.predict(
        X_val_transformed
    )

    test_predictions = model.predict(
        X_test_transformed
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    val_metrics = calculate_metrics(
        y_val.to_numpy(),
        val_predictions,
    )

    test_metrics = calculate_metrics(
        y_test.to_numpy(),
        test_predictions,
    )

    print("\nValidation metrics:")

    for metric, value in val_metrics.items():

        print(
            f"{metric.upper():<6}: {value:.6f}"
        )

    print("\nTest metrics:")

    for metric, value in test_metrics.items():

        print(
            f"{metric.upper():<6}: {value:.6f}"
        )

    # --------------------------------------------------------
    # MLflow setup
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    experiment = mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # --------------------------------------------------------
    # Start run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="XGBoost Final Model"
    ) as run:

        print(
            f"\nMLflow Run ID: {run.info.run_id}"
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params(
            MODEL_PARAMS
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURES),
        )

        mlflow.log_param(
            "train_start",
            str(train_df["date"].min().date()),
        )

        mlflow.log_param(
            "train_end",
            str(TRAIN_END.date()),
        )

        mlflow.log_param(
            "validation_start",
            str(val_df["date"].min().date()),
        )

        mlflow.log_param(
            "validation_end",
            str(VAL_END.date()),
        )

        mlflow.log_param(
            "test_start",
            str(TEST_START.date()),
        )

        mlflow.log_param(
            "test_end",
            str(test_df["date"].max().date()),
        )

        mlflow.log_param(
            "model_artifact",
            MODEL_PATH.name,
        )

        # ----------------------------------------------------
        # Log validation metrics
        # ----------------------------------------------------

        mlflow.log_metrics(
            {
                f"val_{key}": value
                for key, value in val_metrics.items()
            }
        )

        # ----------------------------------------------------
        # Log test metrics
        # ----------------------------------------------------

        mlflow.log_metrics(
            {
                f"test_{key}": value
                for key, value in test_metrics.items()
            }
        )

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "project": "F&B Intelligence",
                "task": "demand_forecasting",
                "model_family": "XGBoost",
                "stage": "final_model",
                "evaluation_strategy": "temporal_split",
            }
        )

        # ----------------------------------------------------
        # Save and log feature list
        # ----------------------------------------------------

        feature_path = (
            MODEL_DIR / "mlflow_features.json"
        )

        save_feature_list(
            feature_path
        )

        mlflow.log_artifact(
            str(feature_path),
            artifact_path="configuration",
        )

        # ----------------------------------------------------
        # Log metadata
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(METADATA_PATH),
            artifact_path="configuration",
        )

        # ----------------------------------------------------
        # Log model artifacts
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(PREPROCESSOR_PATH),
            artifact_path="model",
        )

        print("\nMLflow tracking completed.")

        print(
            f"Experiment: {experiment.name}"
        )

        print(
            f"Run ID    : {run.info.run_id}"
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()