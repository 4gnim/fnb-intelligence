from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# CONFIG
# ============================================================

TRACKING_URI = "sqlite:///mlflow.db"

MODEL_NAME = "fb_demand_forecasting"
CHAMPION_ALIAS = "champion"

# Candidate yang benar-benar memiliki MLflow logged model.
#
# M8.4 membuat source run ini dan melakukan:
#     mlflow.xgboost.log_model(model, name="model")
#
CANDIDATE_RUN_ID = "8dfdc45ab72e47308558bd4ce0a0a080"

# Jangan promote jika performanya sama atau lebih buruk.
MIN_IMPROVEMENT = 0.0


# ============================================================
# MLflow SETUP
# ============================================================

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(
    tracking_uri=TRACKING_URI
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("MLflow Forecasting Model Promotion")
print("=" * 60)


# ============================================================
# 1. LOAD CURRENT CHAMPION
# ============================================================

print("\nLoading current champion...")

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    CHAMPION_ALIAS,
)

champion_run_id = champion.run_id

champion_run = client.get_run(champion_run_id)

champion_mae = champion_run.data.metrics.get("test_mae")

if champion_mae is None:
    raise RuntimeError(
        f"Champion run {champion_run_id} "
        "does not contain test_mae."
    )


print(f"Model          : {MODEL_NAME}")
print(f"Champion       : Version {champion.version}")
print(f"Champion Run ID: {champion_run_id}")
print(f"Champion Test MAE: {champion_mae:.6f}")


# ============================================================
# 2. LOAD CANDIDATE
# ============================================================

print("\n" + "-" * 60)
print("LOADING CANDIDATE MODEL")
print("-" * 60)

candidate_run = client.get_run(CANDIDATE_RUN_ID)

candidate_mae = candidate_run.data.metrics.get("test_mae")

candidate_name = candidate_run.data.tags.get(
    "mlflow.runName",
    "Unnamed Model",
)

if candidate_mae is None:
    raise RuntimeError(
        f"Candidate run {CANDIDATE_RUN_ID} "
        "does not contain test_mae."
    )


print(f"Candidate Name    : {candidate_name}")
print(f"Candidate Run ID  : {CANDIDATE_RUN_ID}")
print(f"Candidate Test MAE: {candidate_mae:.6f}")


# ============================================================
# 3. CHECK MODEL ARTIFACT
# ============================================================

print("\nChecking candidate MLflow model artifact...")

artifacts = client.list_artifacts(
    CANDIDATE_RUN_ID,
    "model",
)

if not artifacts:
    raise RuntimeError(
        f"Candidate run {CANDIDATE_RUN_ID} "
        "does not contain a 'model' artifact.\n"
        "This run cannot be promoted."
    )

print("Model artifact: FOUND")


# ============================================================
# 4. MODEL COMPARISON
# ============================================================

print("\n" + "-" * 60)
print("MODEL COMPARISON")
print("-" * 60)

print("Champion:")
print(f"  Version : {champion.version}")
print(f"  MAE     : {champion_mae:.6f}")

print("\nCandidate:")
print(f"  Name    : {candidate_name}")
print(f"  Run ID  : {CANDIDATE_RUN_ID}")
print(f"  MAE     : {candidate_mae:.6f}")


# ============================================================
# 5. CALCULATE IMPROVEMENT
# ============================================================

improvement = (
    (champion_mae - candidate_mae)
    / champion_mae
) * 100


print(f"\nImprovement: {improvement:.4f}%")


# ============================================================
# 6. PROMOTION DECISION
# ============================================================

print("\n" + "-" * 60)
print("PROMOTION DECISION")
print("-" * 60)


# Lower MAE = better model.
#
# Equal MAE -> REJECT
# Higher MAE -> REJECT
# Lower MAE -> PROMOTE

if candidate_mae >= champion_mae:

    print("Decision: REJECT")

    if candidate_mae == champion_mae:
        print(
            "Reason: Candidate performance is identical "
            "to the current champion."
        )
    else:
        print(
            "Reason: Candidate has worse test MAE "
            "than the current champion."
        )

    print("\nChampion remains unchanged.")
    print("=" * 60)

    raise SystemExit(0)


# Minimum improvement check

if improvement <= MIN_IMPROVEMENT:

    print("Decision: REJECT")

    print(
        f"Reason: Improvement {improvement:.4f}% "
        f"does not exceed the required threshold."
    )

    print("\nChampion remains unchanged.")
    print("=" * 60)

    raise SystemExit(0)


# ============================================================
# 7. PROMOTE
# ============================================================

print("Decision: PROMOTE")

model_uri = (
    f"runs:/{CANDIDATE_RUN_ID}/model"
)

print("\nRegistering candidate model...")
print(f"Model URI: {model_uri}")


registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)


new_version = registered_model.version


print(
    f"Created model version: {new_version}"
)


# ============================================================
# 8. UPDATE CHAMPION ALIAS
# ============================================================

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias=CHAMPION_ALIAS,
    version=new_version,
)


print(
    f"Alias '{CHAMPION_ALIAS}' "
    f"updated to version {new_version}"
)


# ============================================================
# 9. UPDATE DESCRIPTION
# ============================================================

description = (
    f"Promoted candidate model '{candidate_name}'. "
    f"Source run: {CANDIDATE_RUN_ID}. "
    f"Test MAE: {candidate_mae:.6f}. "
    f"Improvement over previous champion: "
    f"{improvement:.4f}%."
)


client.update_model_version(
    name=MODEL_NAME,
    version=new_version,
    description=description,
)


# ============================================================
# 10. RESULT
# ============================================================

print("\n" + "=" * 60)
print("PROMOTION SUCCESSFUL")
print("=" * 60)

print(f"Model       : {MODEL_NAME}")
print(f"Version     : {new_version}")
print(f"Alias       : {CHAMPION_ALIAS}")
print(f"Test MAE    : {candidate_mae:.6f}")
print(f"Improvement : {improvement:.4f}%")

print("=" * 60)