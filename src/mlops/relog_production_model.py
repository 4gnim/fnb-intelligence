from pathlib import Path

import mlflow
import mlflow.xgboost
import joblib
import json


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "forecasting" / "xgb_forecasting.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "forecasting" / "preprocessor.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "forecasting" / "metadata.json"

MLFLOW_TRACKING_URI = "http://localhost:5000"

EXPERIMENT_NAME = "F&B Intelligence - Production Model"

REGISTERED_MODEL_NAME = "fb_demand_forecasting"


# ============================================================
# VALIDATION
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessor not found: {PREPROCESSOR_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("=" * 70)
print("RE-LOG PRODUCTION FORECASTING MODEL")
print("=" * 70)

print(f"Tracking URI : {MLFLOW_TRACKING_URI}")
print(f"Model        : {MODEL_PATH}")
print(f"Preprocessor : {PREPROCESSOR_PATH}")
print()


# ============================================================
# CREATE / SELECT EXPERIMENT
# ============================================================

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location="mlflow-artifacts:/",
    )

    print(f"Created experiment: {EXPERIMENT_NAME}")
    print(f"Experiment ID     : {experiment_id}")
else:
    experiment_id = experiment.experiment_id

    print(f"Using experiment  : {EXPERIMENT_NAME}")
    print(f"Experiment ID     : {experiment_id}")

print()


# ============================================================
# LOAD PRODUCTION ARTIFACTS
# ============================================================

print("Loading production artifacts...")

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print(f"Model type: {type(model).__name__}")
print()


# ============================================================
# CREATE RUN
# ============================================================

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(
    run_name="Production Champion - Artifact Migration"
) as run:

    run_id = run.info.run_id

    print(f"Run ID: {run_id}")
    print()

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_param("model_name", metadata.get("model_name", "XGBoost tuned"))
    mlflow.log_param("model_type", metadata.get("model_type", "XGBRegressor"))
    mlflow.log_param("artifact_source", "production_model")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mlflow.log_metric("test_mae", 13.521476)
    mlflow.log_metric("test_rmse", 18.690569)
    mlflow.log_metric("test_mape", 21.215963)
    mlflow.log_metric("test_r2", 0.759691)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    mlflow.log_dict(
        metadata,
        "configuration/metadata.json",
    )

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    mlflow.log_artifact(
        str(PREPROCESSOR_PATH),
        artifact_path="preprocessor",
    )

    # --------------------------------------------------------
    # XGBoost Model
    # --------------------------------------------------------

    print("Logging XGBoost model...")

    mlflow.xgboost.log_model(
        model,
        name="model",
    )

    print("Model logged successfully.")

    print()
    print("=" * 70)
    print("RUN CREATED")
    print("=" * 70)
    print(f"Run ID: {run_id}")


# ============================================================
# FIND NEW LOGGED MODEL
# ============================================================

client = mlflow.MlflowClient()

logged_models = client.search_logged_models(
    experiment_ids=[experiment_id],
    filter_string=f"source_run_id = '{run_id}'"
)

if not logged_models:
    raise RuntimeError(
        "No MLflow Logged Model was created."
    )

logged_model = logged_models[0]

print()
print("=" * 70)
print("LOGGED MODEL")
print("=" * 70)
print(f"Logged Model ID : {logged_model.model_id}")
print(f"Artifact Path   : {logged_model.artifact_location}")
print()


# ============================================================
# REGISTER MODEL
# ============================================================

print("=" * 70)
print("REGISTERING MODEL")
print("=" * 70)

model_version = client.create_model_version(
    name=REGISTERED_MODEL_NAME,
    source=f"models:/{logged_model.model_id}",
)

print(f"Registered Model : {REGISTERED_MODEL_NAME}")
print(f"Version          : {model_version.version}")
print(f"Source           : {model_version.source}")
print()


# ============================================================
# PROMOTE NEW VERSION
# ============================================================

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "champion",
    model_version.version,
)

print("=" * 70)
print("CHAMPION UPDATED")
print("=" * 70)

print(f"Model : {REGISTERED_MODEL_NAME}")
print(f"Version: {model_version.version}")
print("Alias : champion")
print()

print("Migration completed successfully.")