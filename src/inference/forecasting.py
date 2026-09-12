from pathlib import Path

import pandas as pd


LAG_PERIODS = [1, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]


def create_future_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create calendar features for future dates.

    Future exogenous assumptions:
    - discount_percentage is supplied by caller
    - is_payday is supplied by caller
    - is_holiday is supplied by caller
    """

    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_payday"] = df["is_payday"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)

    return df


def create_recursive_features(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create lag and rolling features from a demand history.

    The implementation follows the same logic used during
    model training:
    - lag = shifted historical quantity
    - rolling = rolling statistics from shifted quantity
    """

    df = history.copy()

    df = df.sort_values(
        [
            "branch_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    group_columns = [
        "branch_id",
        "product_id",
    ]

    grouped = df.groupby(
        group_columns,
        sort=False,
    )["quantity"]

    for lag in LAG_PERIODS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            grouped.transform(
                lambda x: (
                    x.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
        )

        df[f"rolling_std_{window}"] = (
            grouped.transform(
                lambda x: (
                    x.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .std()
                )
            )
        )

    return df


def prepare_future_row(
    date: pd.Timestamp,
    branch_id: str,
    product_id: str,
    discount_percentage: float = 0.0,
    is_payday: int = 0,
    is_holiday: int = 0,
) -> pd.DataFrame:
    """
    Create one future row before lag/rolling features
    are calculated.
    """

    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(date),
                "branch_id": branch_id,
                "product_id": product_id,
                "quantity": float("nan"),
                "discount_percentage": float(
                    discount_percentage
                ),
                "is_payday": int(is_payday),
                "is_holiday": int(is_holiday),
            }
        ]
    )


def recursive_forecast(
    history: pd.DataFrame,
    model,
    preprocessor,
    features: list[str],
    branch_id: str,
    product_id: str,
    start_date: str,
    horizon: int,
    discount_percentage: float = 0.0,
    is_payday: int = 0,
    is_holiday: int = 0,
) -> pd.DataFrame:
    """
    Generate recursive multi-step forecasts.

    Each prediction is appended to the history and becomes
    available as historical information for the next prediction.
    """

    if horizon < 1:
        raise ValueError(
            "horizon must be at least 1"
        )

    if horizon > 30:
        raise ValueError(
            "horizon must not exceed 30 days"
        )

    start_date = pd.Timestamp(start_date)

    working_history = history.copy()

    working_history["date"] = pd.to_datetime(
        working_history["date"]
    )

    # ---------------------------------------------------------
    # Keep only requested branch-product series
    # ---------------------------------------------------------

    working_history = working_history[
        (working_history["branch_id"] == branch_id)
        & (
            working_history["product_id"]
            == product_id
        )
    ].copy()

    if working_history.empty:
        raise ValueError(
            "No historical demand found for "
            f"{branch_id} / {product_id}"
        )

    working_history = working_history.sort_values(
        "date"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Validate sufficient history
    #
    # The model requires up to lag 28 and rolling window 28.
    # Therefore at least 28 historical observations are needed.
    # ---------------------------------------------------------

    if len(working_history) < 28:
        raise ValueError(
            "At least 28 historical observations "
            "are required for forecasting."
        )

    # ---------------------------------------------------------
    # Recursive forecasting
    # ---------------------------------------------------------

    predictions = []

    for step in range(horizon):

        forecast_date = (
            start_date
            + pd.Timedelta(days=step)
        )

        future_row = prepare_future_row(
            date=forecast_date,
            branch_id=branch_id,
            product_id=product_id,
            discount_percentage=discount_percentage,
            is_payday=is_payday,
            is_holiday=is_holiday,
        )

        # Add future row to working history.
        combined = pd.concat(
            [
                working_history,
                future_row,
            ],
            ignore_index=True,
        )

        # Calendar features
        combined = create_future_calendar_features(
            combined
        )

        # Lag + rolling features
        combined = create_recursive_features(
            combined
        )

        # Get only the future row
        row = combined[
            combined["date"] == forecast_date
        ].copy()

        if row.empty:
            raise ValueError(
                f"Could not create features for "
                f"{forecast_date.date()}"
            )

        row = row.iloc[[0]]

        # Ensure every model feature exists
        missing_features = [
            feature
            for feature in features
            if feature not in row.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing forecasting features: "
                f"{missing_features}"
            )

        X = row[features].copy()

        # Check missing values
        if X.isna().any().any():
            missing = X.columns[
                X.isna().any()
            ].tolist()

            raise ValueError(
                "Insufficient historical data for "
                f"forecasting {forecast_date.date()}. "
                f"Missing features: {missing}"
            )

        # Preprocess
        X_encoded = preprocessor.transform(X)

        # Predict
        prediction = model.predict(
            X_encoded
        )

        prediction = max(
            float(prediction[0]),
            0.0,
        )

        prediction = round(
            prediction,
            2,
        )

        # Store result
        predictions.append(
            {
                "date": forecast_date.date().isoformat(),
                "predicted_quantity": prediction,
            }
        )

        # -----------------------------------------------------
        # IMPORTANT:
        # The prediction becomes historical quantity
        # for the next recursive step.
        # -----------------------------------------------------

        future_row["quantity"] = prediction

        working_history = pd.concat(
            [
                working_history,
                future_row,
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)