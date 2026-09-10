from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


LAG_PERIODS = [
    1,
    7,
    14,
    21,
    28,
]

ROLLING_WINDOWS = [
    7,
    14,
    28,
]


def load_demand() -> pd.DataFrame:
    """
    Load daily demand dataset.
    """

    path = DATA_DIR / "daily_demand.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    demand = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    return demand


def create_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create calendar-based features.
    """

    df = df.copy()

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["date"].dt.day
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["week_of_year"] = (
        df["date"].dt.isocalendar().week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_payday"] = (
        df["is_payday"]
        .astype(int)
    )

    df["is_holiday"] = (
        df["is_holiday"]
        .astype(int)
    )

    return df


def create_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical demand features
    for each branch-product series.

    Shift is performed before rolling calculations
    to prevent target leakage.
    """

    df = df.copy()

    df = df.sort_values(
        [
            "branch_id",
            "product_id",
            "date",
        ]
    )

    group_columns = [
        "branch_id",
        "product_id",
    ]

    grouped = df.groupby(
        group_columns,
        sort=False,
    )["quantity"]

    # ---------------------------------------------------------
    # Lag features
    # ---------------------------------------------------------

    for lag in LAG_PERIODS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # ---------------------------------------------------------
    # Rolling features
    # ---------------------------------------------------------

    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            grouped.transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=window).mean()
            )
        )

        df[f"rolling_std_{window}"] = (
            grouped.transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=window).std()
            )
        )

    return df

def create_forecasting_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build final supervised learning dataset.
    """

    required_columns = [
        "date",
        "branch_id",
        "product_id",
        "quantity",
        "discount_percentage",
        "is_payday",
        "is_holiday",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df[
        required_columns
    ].copy()

    df = create_calendar_features(
        df
    )

    df = create_lag_features(
        df
    )

    # ---------------------------------------------------------
    # Remove rows without sufficient history
    # ---------------------------------------------------------

    feature_columns = [
        f"lag_{lag}"
        for lag in LAG_PERIODS
    ]

    feature_columns += [
        f"rolling_mean_{window}"
        for window in ROLLING_WINDOWS
    ]

    feature_columns += [
        f"rolling_std_{window}"
        for window in ROLLING_WINDOWS
    ]

    df = df.dropna(
        subset=feature_columns
    ).reset_index(
        drop=True
    )

    return df


def main():

    print("=" * 60)
    print("FORECASTING FEATURE ENGINEERING")
    print("=" * 60)

    demand = load_demand()

    print(
        f"Raw demand shape : "
        f"{demand.shape}"
    )

    forecasting_df = (
        create_forecasting_dataset(
            demand
        )
    )

    print(
        f"Final dataset    : "
        f"{forecasting_df.shape}"
    )

    print("\nColumns:")
    print(
        forecasting_df.columns.tolist()
    )

    print("\nMissing values:")
    print(
        forecasting_df.isna().sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nDate range:")
    print(
        forecasting_df["date"].min(),
        "→",
        forecasting_df["date"].max(),
    )

    print(
        "\nFeature engineering "
        "completed."
    )


if __name__ == "__main__":
    main()