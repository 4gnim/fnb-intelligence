from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


def normalize_dates(df, columns):
    """
    Convert date columns to pandas datetime.
    Invalid dates become NaT.
    """
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


def calculate_uplift(base, comparison):
    """
    Calculate percentage uplift.
    """
    if base == 0:
        return 0.0

    return (comparison / base - 1) * 100


def main():

    # =========================================================
    # LOAD DATA
    # =========================================================

    demand = pd.read_csv(
        DATA_DIR / "daily_demand.csv"
    )

    products = pd.read_csv(
        DATA_DIR / "products.csv"
    )

    weather = pd.read_csv(
        DATA_DIR / "weather.csv"
    )

    promotions = pd.read_csv(
        DATA_DIR / "promotions.csv"
    )

    holidays = pd.read_csv(
        DATA_DIR / "holidays.csv"
    )

    inventory = pd.read_csv(
        DATA_DIR / "inventory_transactions.csv"
    )

    sales = pd.read_csv(
        DATA_DIR / "sales.csv"
    )

    # =========================================================
    # NORMALIZE DATES
    # =========================================================

    demand = normalize_dates(
        demand,
        ["date"],
    )

    weather = normalize_dates(
        weather,
        ["date"],
    )

    promotions = normalize_dates(
        promotions,
        ["date"],
    )

    holidays = normalize_dates(
        holidays,
        ["date"],
    )

    inventory = normalize_dates(
        inventory,
        ["date"],
    )

    sales = normalize_dates(
        sales,
        ["timestamp"],
    )

    # =========================================================
    # HEADER
    # =========================================================

    print("=" * 60)
    print("BUSINESS SANITY CHECK")
    print("=" * 60)

    # =========================================================
    # 1. BASIC DEMAND CHECK
    # =========================================================

    print("\n" + "=" * 60)
    print("1. DEMAND")
    print("=" * 60)

    print("\nQuantity:")
    print(
        demand["quantity"].describe()
    )

    zero_pct = (
        demand["quantity"]
        .eq(0)
        .mean()
        * 100
    )

    negative_demand = (
        demand["quantity"]
        .lt(0)
        .sum()
    )

    print(
        f"\nZero-demand percentage : "
        f"{zero_pct:.2f}%"
    )

    print(
        f"Negative-demand rows   : "
        f"{negative_demand:,}"
    )

    # =========================================================
    # 2. DATE RANGE
    # =========================================================

    print("\n" + "=" * 60)
    print("2. DATE RANGE")
    print("=" * 60)

    print(
        f"Demand : "
        f"{demand['date'].min().date()} "
        f"→ "
        f"{demand['date'].max().date()}"
    )

    print(
        f"Weather: "
        f"{weather['date'].min().date()} "
        f"→ "
        f"{weather['date'].max().date()}"
    )

    # =========================================================
    # 3. WEEKEND EFFECT
    # =========================================================

    demand["is_weekend"] = (
        demand["date"]
        .dt.dayofweek
        >= 5
    )

    weekend_avg = demand.loc[
        demand["is_weekend"],
        "quantity",
    ].mean()

    weekday_avg = demand.loc[
        ~demand["is_weekend"],
        "quantity",
    ].mean()

    weekend_uplift = calculate_uplift(
        weekday_avg,
        weekend_avg,
    )

    print("\n" + "=" * 60)
    print("3. WEEKEND EFFECT")
    print("=" * 60)

    print(
        f"Weekday : {weekday_avg:.2f}"
    )

    print(
        f"Weekend : {weekend_avg:.2f}"
    )

    print(
        f"Uplift  : {weekend_uplift:.2f}%"
    )

    # =========================================================
    # 4. PAYDAY EFFECT
    # =========================================================

    demand["is_payday"] = (
        demand["date"]
        .dt.day
        >= 25
    )

    payday_avg = demand.loc[
        demand["is_payday"],
        "quantity",
    ].mean()

    normal_avg = demand.loc[
        ~demand["is_payday"],
        "quantity",
    ].mean()

    payday_uplift = calculate_uplift(
        normal_avg,
        payday_avg,
    )

    print("\n" + "=" * 60)
    print("4. PAYDAY EFFECT")
    print("=" * 60)

    print(
        f"Normal : {normal_avg:.2f}"
    )

    print(
        f"Payday : {payday_avg:.2f}"
    )

    print(
        f"Uplift : {payday_uplift:.2f}%"
    )

    # =========================================================
    # 5. HOLIDAY EFFECT
    # =========================================================

    demand["is_holiday"] = (
        demand["date"]
        .isin(
            holidays["date"]
        )
    )

    holiday_avg = demand.loc[
        demand["is_holiday"],
        "quantity",
    ].mean()

    normal_holiday_avg = demand.loc[
        ~demand["is_holiday"],
        "quantity",
    ].mean()

    holiday_uplift = calculate_uplift(
        normal_holiday_avg,
        holiday_avg,
    )

    print("\n" + "=" * 60)
    print("5. HOLIDAY EFFECT")
    print("=" * 60)

    print(
        f"Normal  : "
        f"{normal_holiday_avg:.2f}"
    )

    print(
        f"Holiday : "
        f"{holiday_avg:.2f}"
    )

    print(
        f"Uplift  : "
        f"{holiday_uplift:.2f}%"
    )

    # =========================================================
    # 6. PROMOTION EFFECT
    # =========================================================

    demand["has_promotion"] = (
        demand["discount_percentage"]
        > 0
    )

    promotion_avg = demand.loc[
        demand["has_promotion"],
        "quantity",
    ].mean()

    no_promotion_avg = demand.loc[
        ~demand["has_promotion"],
        "quantity",
    ].mean()

    promotion_uplift = calculate_uplift(
        no_promotion_avg,
        promotion_avg,
    )

    print("\n" + "=" * 60)
    print("6. PROMOTION EFFECT")
    print("=" * 60)

    print(
        f"No promotion : "
        f"{no_promotion_avg:.2f}"
    )

    print(
        f"Promotion    : "
        f"{promotion_avg:.2f}"
    )

    print(
        f"Uplift       : "
        f"{promotion_uplift:.2f}%"
    )

    # =========================================================
    # 7. WEATHER RELATIONSHIP
    # =========================================================

    weather_daily = (
        weather
        .groupby(
            [
                "date",
                "branch_id",
            ],
            as_index=False,
        )
        .agg(
            temperature=(
                "temperature",
                "mean",
            ),
            rainfall=(
                "rainfall",
                "mean",
            ),
        )
    )

    demand_daily = (
        demand
        .groupby(
            [
                "date",
                "branch_id",
            ],
            as_index=False,
        )
        .agg(
            demand=(
                "quantity",
                "sum",
            )
        )
    )

    merged = demand_daily.merge(
        weather_daily,
        on=[
            "date",
            "branch_id",
        ],
        how="left",
    )

    print("\n" + "=" * 60)
    print("7. WEATHER RELATIONSHIP")
    print("=" * 60)

    print(
        "\nDemand / weather correlations:"
    )

    print(
        merged[
            [
                "demand",
                "temperature",
                "rainfall",
            ]
        ].corr()
    )

    # =========================================================
    # 8. BEVERAGE / TEMPERATURE
    # =========================================================

    demand_product = demand.merge(
        products[
            [
                "product_id",
                "category",
            ]
        ],
        on="product_id",
        how="left",
    )

    beverage = demand_product[
        demand_product["category"]
        == "Beverage"
    ].copy()

    beverage_daily = (
        beverage
        .groupby(
            [
                "date",
                "branch_id",
            ],
            as_index=False,
        )
        .agg(
            demand=(
                "quantity",
                "sum",
            )
        )
    )

    beverage_daily = beverage_daily.merge(
        weather_daily,
        on=[
            "date",
            "branch_id",
        ],
        how="left",
    )

    print("\n" + "=" * 60)
    print("8. BEVERAGE / TEMPERATURE")
    print("=" * 60)

    print(
        beverage_daily[
            [
                "demand",
                "temperature",
                "rainfall",
            ]
        ].corr()
    )

    # =========================================================
    # 9. RAINFALL EFFECT
    # =========================================================

    merged["is_rainy"] = (
        merged["rainfall"] > 0
    )

    rainy_avg = merged.loc[
        merged["is_rainy"],
        "demand",
    ].mean()

    non_rainy_avg = merged.loc[
        ~merged["is_rainy"],
        "demand",
    ].mean()

    rainy_difference = calculate_uplift(
        non_rainy_avg,
        rainy_avg,
    )

    print("\n" + "=" * 60)
    print("9. RAINFALL EFFECT")
    print("=" * 60)

    print(
        f"Non-rainy : "
        f"{non_rainy_avg:.2f}"
    )

    print(
        f"Rainy     : "
        f"{rainy_avg:.2f}"
    )

    print(
        f"Difference: "
        f"{rainy_difference:.2f}%"
    )

    # =========================================================
    # 10. CATEGORY
    # =========================================================

    category_stats = (
        demand_product
        .groupby("category")[
            "quantity"
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
                "sum",
            ]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    print("\n" + "=" * 60)
    print("10. DEMAND BY CATEGORY")
    print("=" * 60)

    print(category_stats)

    # =========================================================
    # 11. BRANCH
    # =========================================================

    branch_stats = (
        demand
        .groupby("branch_id")[
            "quantity"
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
                "sum",
            ]
        )
        .sort_values(
            "sum",
            ascending=False,
        )
    )

    print("\n" + "=" * 60)
    print("11. DEMAND BY BRANCH")
    print("=" * 60)

    print(branch_stats)

    # =========================================================
    # 12. TOP PRODUCTS
    # =========================================================

    top_products = (
        demand_product
        .groupby(
            [
                "product_id",
                "category",
            ]
        )["quantity"]
        .sum()
        .sort_values(
            ascending=False,
        )
        .head(10)
    )

    print("\n" + "=" * 60)
    print("12. TOP PRODUCTS")
    print("=" * 60)

    print(top_products)

    # =========================================================
    # 13. INVENTORY CONSISTENCY
    # =========================================================

    inventory[
        "expected_closing"
    ] = (
        inventory["opening_stock"]
        + inventory["received_stock"]
        - inventory["used_stock"]
    )

    inventory[
        "inventory_error"
    ] = (
        inventory["closing_stock"]
        - inventory["expected_closing"]
    )

    max_inventory_error = (
        inventory["inventory_error"]
        .abs()
        .max()
    )

    negative_closing = (
        inventory["closing_stock"]
        < 0
    ).sum()

    print("\n" + "=" * 60)
    print("13. INVENTORY CONSISTENCY")
    print("=" * 60)

    print(
        f"Max absolute error : "
        f"{max_inventory_error:.6f}"
    )

    print(
        f"Negative closing stock : "
        f"{negative_closing:,}"
    )

    # =========================================================
    # 14. SALES
    # =========================================================

    unique_transactions = (
        sales["transaction_id"]
        .nunique()
    )

    unique_customers = (
        sales["customer_id"]
        .nunique()
    )

    unique_products = (
        sales["product_id"]
        .nunique()
    )

    invalid_quantity = (
        sales["quantity"]
        <= 0
    ).sum()

    negative_amount = (
        sales["total_amount"]
        < 0
    ).sum()

    basket_size = (
        sales
        .groupby("transaction_id")
        ["product_id"]
        .nunique()
    )

    multi_item_transactions = (
        basket_size > 1
    ).sum()

    multi_item_percentage = (
        multi_item_transactions
        / len(basket_size)
        * 100
        if len(basket_size) > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("14. SALES")
    print("=" * 60)

    print(
        f"Rows              : "
        f"{len(sales):,}"
    )

    print(
        f"Transactions      : "
        f"{unique_transactions:,}"
    )

    print(
        f"Customers         : "
        f"{unique_customers:,}"
    )

    print(
        f"Products          : "
        f"{unique_products:,}"
    )

    print(
        f"Invalid quantity  : "
        f"{invalid_quantity:,}"
    )

    print(
        f"Negative amount   : "
        f"{negative_amount:,}"
    )

    print(
        f"Multi-item tx     : "
        f"{multi_item_transactions:,}"
    )

    print(
        f"Multi-item ratio  : "
        f"{multi_item_percentage:.2f}%"
    )

    # =========================================================
    # FINAL
    # =========================================================

    print("\n" + "=" * 60)
    print("BUSINESS SANITY CHECK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()