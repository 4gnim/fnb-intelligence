from datetime import date as Date
from functools import lru_cache
from pathlib import Path
from threading import Lock
import json
import mlflow
import joblib
import pandas as pd
import logging
import time

from fastapi import Request, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal

from src.inventory.recommendation import calculate_inventory_policy
from src.inference.forecasting import recursive_forecast


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("fnb-api")

metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency_seconds": 0.0,
}

metrics_lock = Lock()
application_start_time = time.time()

# ============================================================
# PROJECT PATHS
# ============================================================

from src.config import (
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    DATA_DIR,
    PREPROCESSOR_PATH,
    MODEL_PATH,
    METADATA_PATH,
    FINAL_Z,
    FINAL_REVIEW_PERIOD,
    FINAL_POLICY_NAME,
    FINAL_SERVICE_LEVEL,
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@lru_cache(maxsize=1)
def load_forecasting_preprocessor():
    """
    Load forecasting preprocessor from local artifact.
    """

    return joblib.load(PREPROCESSOR_PATH)


@lru_cache(maxsize=1)
def load_forecasting_model():
    """
    Load champion forecasting model from MLflow Model Registry.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = (
        f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    )

    return mlflow.xgboost.load_model(model_uri)

forecasting_model = load_forecasting_model()
preprocessor = load_forecasting_preprocessor()


with open(METADATA_PATH, "r", encoding="utf-8") as f:
    model_metadata = json.load(f)


# ============================================================
# INVENTORY POLICY
# ============================================================

FINAL_Z = 1.036
FINAL_REVIEW_PERIOD = 1
FINAL_POLICY_NAME = "Lean 85%"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.perf_counter()
    response = None

    try:
        response = await call_next(request)
        return response

    except Exception:
        duration = time.perf_counter() - start_time

        with metrics_lock:
            metrics["total_requests"] += 1
            metrics["failed_requests"] += 1
            metrics["total_latency_seconds"] += duration

        logger.exception(
            "%s %s | status=500 | latency=%.3fs",
            request.method,
            request.url.path,
            duration,
        )

        raise

    finally:
        if response is not None:
            duration = time.perf_counter() - start_time

            with metrics_lock:
                metrics["total_requests"] += 1

                if response.status_code < 400:
                    metrics["successful_requests"] += 1
                else:
                    metrics["failed_requests"] += 1

                metrics["total_latency_seconds"] += duration

            logger.info(
                "%s %s | status=%s | latency=%.3fs",
                request.method,
                request.url.path,
                response.status_code,
                duration,
            )

# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ForecastRequest(BaseModel):
    """
    Request schema for historical forecasting.
    """

    branch_id: str = Field(
        ...,
        description="Branch identifier.",
        examples=["B001"],
    )

    product_id: str = Field(
        ...,
        description="Product identifier.",
        examples=["P024"],
    )

    date: Date = Field(
        ...,
        description="Historical date available in the dataset.",
        examples=["2025-12-31"],
    )


class FutureForecastRequest(BaseModel):
    """
    Request schema for recursive future forecasting.
    """

    branch_id: str = Field(
        ...,
        description="Branch identifier.",
        examples=["B001"],
    )

    product_id: str = Field(
        ...,
        description="Product identifier.",
        examples=["P024"],
    )

    start_date: Date = Field(
        ...,
        description="First future date to forecast.",
        examples=["2026-01-01"],
    )

    horizon: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Number of future days to forecast. Maximum 30 days.",
        examples=[7],
    )

    discount_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Assumed promotion discount percentage.",
        examples=[0],
    )

    is_payday: Literal[0, 1] = Field(
        default=0,
        description="Whether the forecast date is treated as payday.",
        examples=[0],
    )

    is_holiday: Literal[0, 1] = Field(
        default=0,
        description="Whether the forecast date is treated as a holiday.",
        examples=[0],
    )


# ============================================================
# DATA LOADERS
# ============================================================

@lru_cache(maxsize=1)
def load_inventory_data():
    daily_demand = pd.read_csv(
        DATA_DIR / "daily_demand.csv",
        parse_dates=["date"],
    )

    ingredients = pd.read_csv(
        DATA_DIR / "ingredients.csv",
    )

    inventory = pd.read_csv(
        DATA_DIR / "inventory_transactions.csv",
        parse_dates=["date"],
    )

    recipes = pd.read_csv(
        DATA_DIR / "product_ingredients.csv",
    )

    ingredient_demand = daily_demand_to_ingredient_demand(
        daily_demand,
        recipes,
    )

    return ingredient_demand, ingredients, inventory


@lru_cache(maxsize=1)
def load_forecasting_data():
    return pd.read_csv(
        DATA_DIR / "daily_demand.csv",
        parse_dates=["date"],
    )

@lru_cache(maxsize=1)
def load_forecasting_features():
    from src.features.forecasting import create_forecasting_dataset

    daily_demand = load_forecasting_data()

    return create_forecasting_dataset(
        daily_demand
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
)
def health_check():
    return {
        "status": "healthy",
        "service": "fnb-intelligence-api",
        "version": "0.1.0",
    }

@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Get API operational metrics",
)
def get_metrics():
    with metrics_lock:
        total_requests = metrics["total_requests"]
        successful_requests = metrics["successful_requests"]
        failed_requests = metrics["failed_requests"]
        total_latency = metrics["total_latency_seconds"]

    average_latency = (
        total_latency / total_requests
        if total_requests > 0
        else 0.0
    )

    uptime = time.time() - application_start_time

    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "average_latency_seconds": round(average_latency, 4),
        "uptime_seconds": round(uptime, 2),
    }

# ============================================================
# FORECAST MODEL INFORMATION
# ============================================================

@app.get(
    "/forecast/model",
    tags=["Forecasting"],
    summary="Get forecasting model information",
)
def forecast_model_info():
    return {
        "model_name": model_metadata.get("model_name"),
        "model_type": model_metadata.get("model_type"),
        "target": model_metadata.get("target"),
        "features": model_metadata.get("features"),
        "test_metrics": model_metadata.get("test_metrics"),
    }

# ============================================================
# BRANCHES
# ============================================================

@app.get(
    "/branches",
    tags=["Reference Data"],
    summary="Get available branches",
)
def get_branches():
    try:
        daily_demand = load_forecasting_data()

        branches = (
            daily_demand["branch_id"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        return {
            "branches": branches,
            "count": len(branches),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# PRODUCTS
# ============================================================

@app.get(
    "/products",
    tags=["Reference Data"],
    summary="Get available products",
)
def get_products():
    try:
        daily_demand = load_forecasting_data()

        products = (
            daily_demand["product_id"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        return {
            "products": products,
            "count": len(products),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

# ============================================================
# HISTORICAL FORECAST
# ============================================================

@app.post(
    "/forecast",
    tags=["Forecasting"],
    summary="Forecast historical demand",
)
def forecast(request: ForecastRequest):

    try:
        forecast_date = pd.Timestamp(request.date)

        daily_demand = load_forecasting_data()

        # ----------------------------------------------------
        # Validate branch
        # ----------------------------------------------------

        if request.branch_id not in daily_demand["branch_id"].unique():
            raise HTTPException(
                status_code=404,
                detail=f"Branch not found: {request.branch_id}",
            )

        # ----------------------------------------------------
        # Validate product
        # ----------------------------------------------------

        if request.product_id not in daily_demand["product_id"].unique():
            raise HTTPException(
                status_code=404,
                detail=f"Product not found: {request.product_id}",
            )

        # ----------------------------------------------------
        # Validate date
        # ----------------------------------------------------

        available_dates = set(daily_demand["date"])

        if forecast_date not in available_dates:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Date {request.date} is not available in "
                    "historical dataset."
                ),
            )

        # ----------------------------------------------------
        # Create forecasting features
        # ----------------------------------------------------

        forecasting_df = load_forecasting_features()

        # ----------------------------------------------------
        # Select requested observation
        # ----------------------------------------------------

        row = forecasting_df[
            (forecasting_df["date"] == forecast_date)
            & (forecasting_df["branch_id"] == request.branch_id)
            & (forecasting_df["product_id"] == request.product_id)
        ].copy()

        if row.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Forecasting features are not available for "
                    f"{request.branch_id}, "
                    f"{request.product_id}, "
                    f"{request.date}. "
                    "The date may be too early to calculate "
                    "the required lag/rolling features."
                ),
            )

        # ----------------------------------------------------
        # Prepare features
        # ----------------------------------------------------

        features = model_metadata["features"]

        X = row[features].copy()

        X_encoded = preprocessor.transform(X)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = forecasting_model.predict(
            X_encoded
        )

        prediction = max(
            float(prediction[0]),
            0.0,
        )

        return {
            "branch_id": request.branch_id,
            "product_id": request.product_id,
            "date": request.date.isoformat(),
            "predicted_quantity": round(
                prediction,
                2,
            ),
            "model": {
                "name": model_metadata.get(
                    "model_name"
                ),
                "type": model_metadata.get(
                    "model_type"
                ),
            },
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# FUTURE FORECAST
# ============================================================

@app.post(
    "/forecast/future",
    tags=["Forecasting"],
    summary="Forecast future demand recursively",
)
def future_forecast(
    request: FutureForecastRequest,
):

    try:
        start_date = pd.Timestamp(
            request.start_date
        )

        daily_demand = load_forecasting_data()

        # ----------------------------------------------------
        # Validate branch
        # ----------------------------------------------------

        if request.branch_id not in daily_demand["branch_id"].unique():
            raise HTTPException(
                status_code=404,
                detail=f"Branch not found: {request.branch_id}",
            )

        # ----------------------------------------------------
        # Validate product
        # ----------------------------------------------------

        if request.product_id not in daily_demand["product_id"].unique():
            raise HTTPException(
                status_code=404,
                detail=f"Product not found: {request.product_id}",
            )

        # ----------------------------------------------------
        # Validate future date
        # ----------------------------------------------------

        historical_max_date = daily_demand["date"].max()

        if start_date <= historical_max_date:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"start_date must be after the last "
                    f"historical date "
                    f"({historical_max_date.date()})."
                ),
            )

        # ----------------------------------------------------
        # Prepare historical data
        # ----------------------------------------------------

        history = daily_demand[
            [
                "date",
                "branch_id",
                "product_id",
                "quantity",
                "discount_percentage",
                "is_payday",
                "is_holiday",
            ]
        ].copy()

        # ----------------------------------------------------
        # Recursive forecast
        # ----------------------------------------------------

        forecast_df = recursive_forecast(
            history=history,
            model=forecasting_model,
            preprocessor=preprocessor,
            features=model_metadata["features"],
            branch_id=request.branch_id,
            product_id=request.product_id,
            start_date=request.start_date,
            horizon=request.horizon,
            discount_percentage=request.discount_percentage,
            is_payday=request.is_payday,
            is_holiday=request.is_holiday,
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "branch_id": request.branch_id,
            "product_id": request.product_id,
            "start_date": request.start_date.isoformat(),
            "horizon": request.horizon,
            "assumptions": {
                "discount_percentage": request.discount_percentage,
                "is_payday": request.is_payday,
                "is_holiday": request.is_holiday,
            },
            "model": {
                "name": model_metadata.get(
                    "model_name"
                ),
                "type": model_metadata.get(
                    "model_type"
                ),
            },
            "forecast": forecast_df.to_dict(
                orient="records"
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# INVENTORY RECOMMENDATION
# ============================================================

@app.get(
    "/inventory/recommend",
    tags=["Inventory"],
    summary="Get inventory order recommendations",
)
def inventory_recommendation():

    try:
        ingredient_demand, ingredients, inventory = (
            load_inventory_data()
        )

        recommendations = calculate_inventory_policy(
            ingredient_demand=ingredient_demand,
            ingredients=ingredients,
            inventory=inventory,
            z_value=FINAL_Z,
            review_period=FINAL_REVIEW_PERIOD,
            cutoff_date="2025-06-30",
        )

        # ----------------------------------------------------
        # Select ORDER recommendations
        # ----------------------------------------------------

        orders = recommendations[
            recommendations["action"] == "ORDER"
        ].copy()

        orders = orders.sort_values(
            "estimated_order_value",
            ascending=False,
        )

        records = orders[
            [
                "branch_id",
                "ingredient_id",
                "current_stock",
                "avg_daily_usage",
                "lead_time_days",
                "reorder_point",
                "target_stock",
                "recommended_order_quantity",
                "estimated_order_value",
                "action",
            ]
        ].round(2).to_dict(
            orient="records"
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "policy": {
                "name": FINAL_POLICY_NAME,
                "target_service_level": FINAL_SERVICE_LEVEL,
                "z_value": FINAL_Z,
                "review_period_days": FINAL_REVIEW_PERIOD,
            },
            "summary": {
                "total_combinations": len(
                    recommendations
                ),
                "order_recommendations": len(
                    orders
                ),
                "hold_recommendations": int(
                    (
                        recommendations["action"]
                        == "HOLD"
                    ).sum()
                ),
                "total_recommended_quantity": round(
                    orders[
                        "recommended_order_quantity"
                    ].sum(),
                    2,
                ),
                "total_estimated_order_value": round(
                    orders[
                        "estimated_order_value"
                    ].sum(),
                    2,
                ),
            },
            "recommendations": records,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# HELPER
# ============================================================

def daily_demand_to_ingredient_demand(
    daily_demand,
    recipes,
):

    merged = daily_demand.merge(
        recipes,
        on="product_id",
        how="inner",
    )

    merged["ingredient_usage"] = (
        merged["quantity"]
        * merged["quantity_required"]
    )

    ingredient_demand = (
        merged.groupby(
            [
                "date",
                "branch_id",
                "ingredient_id",
            ],
            as_index=False,
        )["ingredient_usage"]
        .sum()
    )

    ingredient_demand["date"] = pd.to_datetime(
        ingredient_demand["date"]
    )

    return ingredient_demand

@app.get(
    "/model-info",
    tags=["Monitoring"],
    summary="Get active forecasting model information",
)
def get_model_info():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        client = mlflow.MlflowClient()

        model_version = client.get_model_version_by_alias(
            name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_MODEL_ALIAS,
        )

        return {
            "model_name": MLFLOW_MODEL_NAME,
            "alias": MLFLOW_MODEL_ALIAS,
            "version": model_version.version,
            "model_type": model_metadata.get(
                "model_type",
                "XGBRegressor",
            ),
            "run_id": model_version.run_id,
            "status": model_version.status,
        }

    except Exception as exc:
        logger.exception("Failed to retrieve model information")

        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve model information: {exc}",
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    logger.warning(
        "Validation error | %s %s | errors=%s",
        request.method,
        request.url.path,
        exc.errors(),
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed.",
            "details": exc.errors(),
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled exception | %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred.",
        },
    )

@app.get(
    "/ready",
    tags=["Monitoring"],
    summary="Check whether the API is ready to serve predictions",
)
def readiness_check():
    try:
        forecasting_model = load_forecasting_model()

        if forecasting_model is None:
            raise RuntimeError("Forecasting model is not available")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        client = mlflow.MlflowClient()

        model_version = client.get_model_version_by_alias(
            name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_MODEL_ALIAS,
        )

        return {
            "status": "ready",
            "service": "fnb-intelligence-api",
            "model": {
                "name": MLFLOW_MODEL_NAME,
                "alias": MLFLOW_MODEL_ALIAS,
                "version": model_version.version,
            },
        }

    except Exception as exc:
        logger.exception("Readiness check failed")

        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "service": "fnb-intelligence-api",
                "reason": str(exc),
            },
        )