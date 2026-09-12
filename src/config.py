import os
from pathlib import Path



# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

MODEL_DIR = PROJECT_ROOT / "models" / "forecasting"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
MODEL_PATH = MODEL_DIR / "xgb_forecasting.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


# ============================================================
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_MODEL_NAME = "fb_demand_forecasting"

MLFLOW_MODEL_ALIAS = "champion"


# ============================================================
# INVENTORY POLICY
# ============================================================

FINAL_Z = 1.036

FINAL_REVIEW_PERIOD = 1

FINAL_POLICY_NAME = "Lean 85%"

FINAL_SERVICE_LEVEL = 0.85


# ============================================================
# API
# ============================================================

API_TITLE = "F&B Intelligence API"

API_DESCRIPTION = (
    "AI-powered decision support API for Food & Beverage operations. "
    "Provides demand forecasting and inventory recommendations."
)

API_VERSION = "0.1.0"