from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient


# ============================================================
# CONFIG
# ============================================================

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "fb_demand_forecasting"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "xgb_forecasting.joblib"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "preprocessor.joblib"
)


# ============================================================
# SETUP
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(
    tracking_uri=MLFLOW_TRACKING_URI
)


# ============================================================
# LOAD PRODUCTION ARTIFACT
# ============================================================

print("=" * 60)
print("MLflow Forecasting Model Registry")
print("=" * 60)

print("\nLoading production model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessor not found: {PREPROCESSOR_PATH}"
    )

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print(f"Model        : {MODEL_PATH}")
print(f"Preprocessor : {PREPROCESSOR_PATH}")
print(f"Model type   : {type(model).__name__}")


# ============================================================
# CREATE REGISTRY SOURCE RUN
# ============================================================

print("\n" + "-" * 60)
print("CREATING REGISTRY SOURCE RUN")
print("-" * 60)

with mlflow.start_run(
    run_name="XGBoost Production Champion"
) as run:

    run_id = run.info.run_id

    # --------------------------------------------------------
    # Model metadata
    # --------------------------------------------------------

    mlflow.set_tag(
        "model_type",
        "XGBoost"
    )

    mlflow.set_tag(
        "model_role",
        "production_champion"
    )

    mlflow.set_tag(
        "pipeline_stage",
        "production"
    )

    mlflow.set_tag(
        "source_artifact",
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # Known evaluation metrics
    # --------------------------------------------------------

    mlflow.log_metrics(
        {
            "test_mae": 13.521476,
            "test_rmse": 18.690569,
            "test_mape": 21.215963,
            "test_r2": 0.759691,
        }
    )

    # --------------------------------------------------------
    # Log model parameters
    # --------------------------------------------------------

    if hasattr(model, "get_params"):
        params = model.get_params()

        selected_params = {
            key: params[key]
            for key in [
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
            ]
            if key in params
        }

        mlflow.log_params(selected_params)

    # --------------------------------------------------------
    # Log model
    # --------------------------------------------------------

    mlflow.xgboost.log_model(
        model,
        name="model",
    )

    # --------------------------------------------------------
    # Log preprocessor
    # --------------------------------------------------------

    mlflow.log_artifact(
        str(PREPROCESSOR_PATH),
        artifact_path="preprocessor"
    )

    print(f"\nSource Run ID: {run_id}")


# ============================================================
# REGISTER MODEL
# ============================================================

print("\n" + "-" * 60)
print("REGISTERING MODEL")
print("-" * 60)

model_uri = f"runs:/{run_id}/model"

print(f"Model URI      : {model_uri}")
print(f"Registry name  : {MODEL_NAME}")


registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

version = registered_model.version

print("\nModel registered successfully.")
print(f"Model name     : {MODEL_NAME}")
print(f"Model version  : {version}")


# ============================================================
# MODEL VERSION DESCRIPTION
# ============================================================

description = (
    "Production champion XGBoost demand forecasting model. "
    "Selected based on the best test MAE among the evaluated "
    "forecasting experiments. "
    "Test MAE=13.521476, "
    "Test RMSE=18.690569, "
    "Test MAPE=21.215963, "
    "Test R2=0.759691."
)

client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description=description,
)


# ============================================================
# SET CHAMPION ALIAS
# ============================================================

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=version,
)

print("\nChampion alias assigned.")

print(f"Model   : {MODEL_NAME}")
print(f"Version : {version}")
print("Alias   : champion")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("MODEL REGISTRY COMPLETED")
print("=" * 60)

print(f"Registered model : {MODEL_NAME}")
print(f"Version          : {version}")
print("Alias            : champion")

print("\nEvaluation metrics:")
print("Test MAE         : 13.521476")
print("Test RMSE        : 18.690569")
print("Test MAPE        : 21.215963")
print("Test R2          : 0.759691")

print(f"\nSource Run ID    : {run_id}")

print("=" * 60)