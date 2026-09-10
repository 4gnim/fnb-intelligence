from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


def check_file(name: str):
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Duplicates : {df.duplicated().sum():,}")
    print(
        f"Missing    : "
        f"{df.isna().sum().sum():,}"
    )

    return df


def main():

    names = [
        "branches",
        "products",
        "customers",
        "ingredients",
        "product_ingredients",
        "holidays",
        "weather",
        "promotions",
        "sales",
        "inventory_transactions",
    ]

    for name in names:
        check_file(name)


if __name__ == "__main__":
    main()