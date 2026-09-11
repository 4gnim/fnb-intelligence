from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_inventory_policy(
    ingredient_demand: pd.DataFrame,
    ingredients: pd.DataFrame,
    inventory: pd.DataFrame,
    z_value: float = 1.036,
    review_period: int = 1,
    cutoff_date: str = "2025-06-30",
) -> pd.DataFrame:
    """
    Calculate inventory policy and order recommendations.

    Parameters
    ----------
    ingredient_demand:
        Historical ingredient usage data.

    ingredients:
        Ingredient master data containing:
        ingredient_id, lead_time_days, unit_cost.

    inventory:
        Inventory transaction history containing:
        date, branch_id, ingredient_id, closing_stock.

    z_value:
        Safety stock z-score.

    review_period:
        Inventory review period in days.

    cutoff_date:
        Date used as the end of the historical training period.

    Returns
    -------
    pd.DataFrame
        Inventory recommendation table.
    """

    cutoff_date = pd.Timestamp(cutoff_date)

    # ---------------------------------------------------------
    # 1. Historical demand
    # ---------------------------------------------------------

    train_demand = ingredient_demand[
        ingredient_demand["date"] <= cutoff_date
    ].copy()

    # ---------------------------------------------------------
    # 2. Demand statistics
    # ---------------------------------------------------------

    inventory_policy = (
        train_demand
        .groupby(["branch_id", "ingredient_id"])["ingredient_usage"]
        .agg(
            avg_daily_usage="mean",
            std_daily_usage="std",
        )
        .reset_index()
    )

    inventory_policy["std_daily_usage"] = (
        inventory_policy["std_daily_usage"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # 3. Ingredient information
    # ---------------------------------------------------------

    ingredient_info = ingredients[
        [
            "ingredient_id",
            "lead_time_days",
            "unit_cost",
        ]
    ].copy()

    inventory_policy = inventory_policy.merge(
        ingredient_info,
        on="ingredient_id",
        how="left",
    )

    # ---------------------------------------------------------
    # 4. Safety stock
    # ---------------------------------------------------------

    inventory_policy["safety_stock"] = (
        z_value
        * inventory_policy["std_daily_usage"]
        * np.sqrt(
            inventory_policy["lead_time_days"]
        )
    )

    # ---------------------------------------------------------
    # 5. Lead-time demand
    # ---------------------------------------------------------

    inventory_policy["lead_time_demand"] = (
        inventory_policy["avg_daily_usage"]
        * inventory_policy["lead_time_days"]
    )

    # ---------------------------------------------------------
    # 6. Reorder point
    # ---------------------------------------------------------

    inventory_policy["reorder_point"] = (
        inventory_policy["lead_time_demand"]
        + inventory_policy["safety_stock"]
    )

    # ---------------------------------------------------------
    # 7. Target stock
    # ---------------------------------------------------------

    inventory_policy["target_stock"] = (
        inventory_policy["avg_daily_usage"]
        * (
            inventory_policy["lead_time_days"]
            + review_period
        )
        +
        z_value
        * inventory_policy["std_daily_usage"]
        * np.sqrt(
            inventory_policy["lead_time_days"]
            + review_period
        )
    )

    # ---------------------------------------------------------
    # 8. Current inventory
    # ---------------------------------------------------------

    current_inventory = (
        inventory[
            inventory["date"] == cutoff_date
        ][
            [
                "branch_id",
                "ingredient_id",
                "closing_stock",
            ]
        ]
        .rename(
            columns={
                "closing_stock": "current_stock"
            }
        )
    )

    inventory_policy = inventory_policy.merge(
        current_inventory,
        on=[
            "branch_id",
            "ingredient_id",
        ],
        how="left",
    )

    inventory_policy["current_stock"] = (
        inventory_policy["current_stock"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # 9. Inventory position
    # ---------------------------------------------------------

    inventory_policy["on_order"] = 0.0

    inventory_policy["inventory_position"] = (
        inventory_policy["current_stock"]
        + inventory_policy["on_order"]
    )

    # ---------------------------------------------------------
    # 10. Order quantity
    # ---------------------------------------------------------

    inventory_policy["order_quantity"] = (
        inventory_policy["target_stock"]
        - inventory_policy["inventory_position"]
    ).clip(lower=0)

    # ---------------------------------------------------------
    # 11. Action
    # ---------------------------------------------------------

    inventory_policy["action"] = np.where(
        inventory_policy["inventory_position"]
        <= inventory_policy["reorder_point"],
        "ORDER",
        "HOLD",
    )

    # ---------------------------------------------------------
    # 12. Recommended quantity
    # ---------------------------------------------------------

    inventory_policy["recommended_order_quantity"] = np.where(
        inventory_policy["action"] == "ORDER",
        inventory_policy["order_quantity"],
        0,
    )

    # ---------------------------------------------------------
    # 13. Estimated order value
    # ---------------------------------------------------------

    inventory_policy["estimated_order_value"] = (
        inventory_policy["recommended_order_quantity"]
        * inventory_policy["unit_cost"]
    )

    return inventory_policy