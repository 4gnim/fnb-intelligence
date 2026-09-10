from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


def load(name: str) -> pd.DataFrame:

    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def validate_unique(
    df: pd.DataFrame,
    columns,
    table: str,
) -> None:
    """
    Validate that a column or combination of columns
    is unique.

    Examples:
        validate_unique(df, "customer_id", "customers")

        validate_unique(
            df,
            ["transaction_id", "product_id"],
            "sales",
        )
    """

    if isinstance(columns, str):
        columns = [columns]

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{table}: missing columns "
            f"{missing_columns}"
        )

    duplicates = df.duplicated(
        subset=columns
    ).sum()

    if duplicates > 0:
        key_name = " + ".join(columns)

        raise ValueError(
            f"{table}.{key_name} "
            f"contains {duplicates} duplicates"
        )


def validate_foreign_key(
    child: pd.DataFrame,
    child_column: str,
    parent: pd.DataFrame,
    parent_column: str,
    relation_name: str,
) -> None:

    invalid = (
        ~child[child_column]
        .isin(parent[parent_column])
    ).sum()

    if invalid > 0:
        raise ValueError(
            f"{relation_name}: "
            f"{invalid} invalid references"
        )


def validate_no_missing(
    df: pd.DataFrame,
    table: str,
) -> None:

    missing = (
        df.isna()
        .sum()
        .sum()
    )

    if missing > 0:
        raise ValueError(
            f"{table}: "
            f"{missing} missing values"
        )


def validate_all() -> None:

    branches = load("branches")
    products = load("products")
    customers = load("customers")
    ingredients = load("ingredients")

    recipes = load(
        "product_ingredients"
    )

    sales = load("sales")

    weather = load("weather")

    inventory = load(
        "inventory_transactions"
    )

    # ---------------------------------------------------------
    # Primary keys
    # ---------------------------------------------------------

    validate_unique(
        branches,
        "branch_id",
        "branches",
    )

    validate_unique(
        products,
        "product_id",
        "products",
    )

    validate_unique(
        customers,
        "customer_id",
        "customers",
    )

    validate_unique(
        ingredients,
        "ingredient_id",
        "ingredients",
    )

    # ---------------------------------------------------------
    # Sales composite key
    #
    # One transaction may contain multiple products.
    #
    # Example:
    #
    # T00000001 | P001
    # T00000001 | P002
    # T00000001 | P003
    #
    # Therefore transaction_id alone is NOT unique.
    # ---------------------------------------------------------

    validate_unique(
        sales,
        [
            "transaction_id",
            "product_id",
        ],
        "sales",
    )

    # ---------------------------------------------------------
    # Foreign keys
    # ---------------------------------------------------------

    validate_foreign_key(
        sales,
        "branch_id",
        branches,
        "branch_id",
        "sales → branches",
    )

    validate_foreign_key(
        sales,
        "product_id",
        products,
        "product_id",
        "sales → products",
    )

    validate_foreign_key(
        sales,
        "customer_id",
        customers,
        "customer_id",
        "sales → customers",
    )

    validate_foreign_key(
        recipes,
        "product_id",
        products,
        "product_id",
        "recipes → products",
    )

    validate_foreign_key(
        recipes,
        "ingredient_id",
        ingredients,
        "ingredient_id",
        "recipes → ingredients",
    )

    validate_foreign_key(
        inventory,
        "branch_id",
        branches,
        "branch_id",
        "inventory → branches",
    )

    validate_foreign_key(
        inventory,
        "ingredient_id",
        ingredients,
        "ingredient_id",
        "inventory → ingredients",
    )

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    for name, df in {
        "branches": branches,
        "products": products,
        "customers": customers,
        "ingredients": ingredients,
        "recipes": recipes,
        "sales": sales,
        "weather": weather,
        "inventory": inventory,
    }.items():

        validate_no_missing(
            df,
            name,
        )

    print(
        "\nDATA VALIDATION PASSED"
    )


if __name__ == "__main__":
    validate_all()