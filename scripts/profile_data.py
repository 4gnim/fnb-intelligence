from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


DATASETS = [
    "branches",
    "products",
    "customers",
    "ingredients",
    "product_ingredients",
    "holidays",
    "weather",
    "promotions",
    "daily_demand",
    "sales",
    "inventory_transactions",
]


def profile_dataset(name: str) -> None:

    path = DATA_DIR / f"{name}.csv"

    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(
        f"Memory     : "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    missing = df.isna().sum()

    print(
        missing[
            missing > 0
        ]
        if (missing > 0).any()
        else "None"
    )

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)


def main():

    for dataset in DATASETS:

        path = DATA_DIR / f"{dataset}.csv"

        if path.exists():
            profile_dataset(dataset)


if __name__ == "__main__":
    main()