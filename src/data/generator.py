from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"

    n_branches: int = 10
    n_products: int = 50
    n_customers: int = 5_000

    seed: int = 42


class FNBDataGenerator:

    def __init__(self, config: GeneratorConfig):

        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.dates = pd.date_range(
            config.start_date,
            config.end_date,
            freq="D",
        )

    # =========================================================
    # MASTER DATA
    # =========================================================

    def generate_branches(self) -> pd.DataFrame:

        cities = [
            "Sidoarjo",
            "Surabaya",
            "Malang",
            "Gresik",
            "Mojokerto",
            "Kediri",
            "Pasuruan",
            "Jember",
            "Probolinggo",
            "Blitar",
        ]

        branch_types = [
            "Mall",
            "Street",
            "Campus",
            "Residential",
        ]

        return pd.DataFrame({
            "branch_id": [
                f"B{i + 1:03d}"
                for i in range(len(cities))
            ],

            "branch_name": [
                f"{city} Branch"
                for city in cities
            ],

            "city": cities,

            "region": [
                "East Java"
            ] * len(cities),

            "branch_type": self.rng.choice(
                branch_types,
                size=len(cities),
            ),

            "capacity": self.rng.integers(
                40,
                150,
                size=len(cities),
            ),

            "opening_date": [
                self.config.start_date
            ] * len(cities),
        })

    # ---------------------------------------------------------

    def generate_products(self) -> pd.DataFrame:

        categories = [
            "Food",
            "Beverage",
            "Snack",
            "Dessert",
        ]

        product_names = [
            "Ayam Geprek",
            "Chicken Burger",
            "Mie Goreng",
            "Nasi Goreng",
            "Chicken Rice",
            "Beef Burger",
            "Es Teh",
            "Iced Lemon Tea",
            "Kopi Susu",
            "Cappuccino",
            "Chocolate Ice",
            "Orange Juice",
            "French Fries",
            "Chicken Wings",
            "Onion Rings",
            "Tahu Crispy",
            "Ice Cream",
            "Pudding",
            "Brownies",
            "Banana Split",
        ]

        rows = []

        for i in range(self.config.n_products):

            category = categories[
                i % len(categories)
            ]

            name = product_names[
                i % len(product_names)
            ]

            if category == "Food":
                price = self.rng.integers(
                    15_000,
                    35_000,
                )

            elif category == "Beverage":
                price = self.rng.integers(
                    5_000,
                    20_000,
                )

            elif category == "Snack":
                price = self.rng.integers(
                    8_000,
                    22_000,
                )

            else:
                price = self.rng.integers(
                    10_000,
                    25_000,
                )

            cost_ratio = self.rng.uniform(
                0.35,
                0.65,
            )

            rows.append({
                "product_id": f"P{i + 1:03d}",
                "product_name": f"{name} {i + 1}",
                "category": category,
                "price": int(price),
                "cost": round(
                    price * cost_ratio,
                    2,
                ),
            })

        return pd.DataFrame(rows)

    # ---------------------------------------------------------

    def generate_customers(self) -> pd.DataFrame:

        age_groups = [
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55+",
        ]

        return pd.DataFrame({
            "customer_id": [
                f"C{i + 1:05d}"
                for i in range(
                    self.config.n_customers
                )
            ],

            "registration_date": self.rng.choice(
                self.dates,
                size=self.config.n_customers,
            ),

            "age_group": self.rng.choice(
                age_groups,
                size=self.config.n_customers,
            ),
        })

    # =========================================================
    # HOLIDAYS
    # =========================================================

    def generate_holidays(self) -> pd.DataFrame:

        holiday_rules = {
            (1, 1): "New Year",
            (5, 1): "Labour Day",
            (8, 17): "Independence Day",
            (12, 25): "Christmas",
        }

        rows = []

        for date in self.dates:

            name = holiday_rules.get(
                (date.month, date.day)
            )

            if name:
                rows.append({
                    "date": date,
                    "holiday_name": name,
                })

        return pd.DataFrame(rows)

    # =========================================================
    # WEATHER
    # =========================================================

    def generate_weather(
        self,
        branches: pd.DataFrame,
    ) -> pd.DataFrame:

        n_dates = len(self.dates)
        n_branches = len(branches)

        dates = np.repeat(
            self.dates.values,
            n_branches,
        )

        branch_ids = np.tile(
            branches["branch_id"].values,
            n_dates,
        )

        day_of_year = np.repeat(
            self.dates.dayofyear.values,
            n_branches,
        )

        temperature = (
            28
            + 1.5
            * np.sin(
                2
                * np.pi
                * day_of_year
                / 365
            )
            + self.rng.normal(
                0,
                1.5,
                len(dates),
            )
        )

        rain_probability = (
            self.rng.random(
                len(dates)
            )
        )

        rainfall = np.where(
            rain_probability < 0.35,
            self.rng.gamma(
                shape=1.5,
                scale=4,
                size=len(dates),
            ),
            0,
        )

        condition = np.select(
            [
                rainfall > 15,
                rainfall > 5,
            ],
            [
                "Heavy Rain",
                "Rain",
            ],
            default="Clear",
        )

        return pd.DataFrame({
            "date": dates,
            "branch_id": branch_ids,
            "temperature": np.round(
                temperature,
                2,
            ),
            "rainfall": np.round(
                rainfall,
                2,
            ),
            "weather_condition": condition,
        })

    # =========================================================
    # PROMOTIONS
    # =========================================================

    def generate_promotions(
        self,
        branches: pd.DataFrame,
        products: pd.DataFrame,
    ) -> pd.DataFrame:

        rows = []

        promotion_id = 1

        for date in self.dates:

            if self.rng.random() > 0.08:
                continue

            branch_id = self.rng.choice(
                branches["branch_id"]
            )

            product_id = self.rng.choice(
                products["product_id"]
            )

            discount = int(
                self.rng.choice(
                    [10, 15, 20, 25]
                )
            )

            promotion_type = self.rng.choice(
                [
                    "Discount",
                    "Buy 1 Get 1",
                    "Bundle",
                ]
            )

            rows.append({
                "promotion_id":
                    f"PROMO{promotion_id:05d}",

                "date": date,

                "branch_id":
                    branch_id,

                "product_id":
                    product_id,

                "promotion_type":
                    promotion_type,

                "discount_percentage":
                    discount,
            })

            promotion_id += 1

        return pd.DataFrame(rows)

    # =========================================================
    # INGREDIENTS
    # =========================================================

    def generate_ingredients(self) -> pd.DataFrame:

        names = [
            "Chicken",
            "Beef",
            "Rice",
            "Noodles",
            "Bread",
            "Potato",
            "Tea",
            "Coffee",
            "Milk",
            "Sugar",
            "Lemon",
            "Chocolate",
            "Ice",
            "Cooking Oil",
            "Flour",
            "Cheese",
            "Butter",
            "Onion",
            "Garlic",
            "Chili",
            "Tomato",
            "Lettuce",
            "Mayonnaise",
            "Egg",
            "Sauce",
            "Syrup",
            "Cocoa",
            "Salt",
            "Pepper",
            "Ice Cream",
        ]

        return pd.DataFrame({
            "ingredient_id": [
                f"I{i + 1:03d}"
                for i in range(len(names))
            ],

            "ingredient_name": names,

            "unit": self.rng.choice(
                [
                    "kg",
                    "liter",
                    "unit",
                ],
                size=len(names),
            ),

            "unit_cost": np.round(
                self.rng.uniform(
                    5_000,
                    50_000,
                    len(names),
                ),
                2,
            ),

            "supplier_id": [
                f"S{x:03d}"
                for x in self.rng.integers(
                    1,
                    11,
                    len(names),
                )
            ],

            "lead_time_days":
                self.rng.integers(
                    1,
                    5,
                    len(names),
                ),
        })

    # =========================================================
    # PRODUCT RECIPES
    # =========================================================

    def generate_product_ingredients(
        self,
        products: pd.DataFrame,
        ingredients: pd.DataFrame,
    ) -> pd.DataFrame:

        rows = []

        ingredient_ids = (
            ingredients["ingredient_id"]
            .tolist()
        )

        for product_id in products[
            "product_id"
        ]:

            n = int(
                self.rng.integers(
                    2,
                    6,
                )
            )

            selected = self.rng.choice(
                ingredient_ids,
                size=n,
                replace=False,
            )

            for ingredient_id in selected:

                rows.append({
                    "product_id":
                        product_id,

                    "ingredient_id":
                        ingredient_id,

                    "quantity_required":
                        round(
                            float(
                                self.rng.uniform(
                                    0.01,
                                    0.5,
                                )
                            ),
                            4,
                        ),
                })

        return pd.DataFrame(rows)

    # =========================================================
    # DAILY DEMAND
    # =========================================================

    def generate_daily_demand(
        self,
        branches: pd.DataFrame,
        products: pd.DataFrame,
        weather: pd.DataFrame,
        promotions: pd.DataFrame,
        holidays: pd.DataFrame,
    ) -> pd.DataFrame:

        dates = self.dates

        # -----------------------------------------------------
        # Create date × branch × product grid
        # -----------------------------------------------------

        grid = pd.MultiIndex.from_product(
            [
                dates,
                branches["branch_id"],
                products["product_id"],
            ],
            names=[
                "date",
                "branch_id",
                "product_id",
            ],
        ).to_frame(
            index=False
        )

        # -----------------------------------------------------
        # Date features
        # -----------------------------------------------------

        grid["day_of_week"] = (
            grid["date"].dt.dayofweek
        )

        grid["day_of_month"] = (
            grid["date"].dt.day
        )

        grid["month"] = (
            grid["date"].dt.month
        )

        grid["is_weekend"] = (
            grid["day_of_week"] >= 5
        )

        grid["is_payday"] = (
            grid["day_of_month"] >= 25
        )

        # -----------------------------------------------------
        # Holiday
        # -----------------------------------------------------

        holiday_dates = set(
            holidays["date"]
        )

        grid["is_holiday"] = (
            grid["date"]
            .isin(holiday_dates)
        )

        # -----------------------------------------------------
        # Product metadata
        # -----------------------------------------------------

        grid = grid.merge(
            products[
                [
                    "product_id",
                    "category",
                    "price",
                ]
            ],
            on="product_id",
            how="left",
        )

        # -----------------------------------------------------
        # Branch effect
        # -----------------------------------------------------

        branch_effect = pd.DataFrame({
            "branch_id":
                branches["branch_id"],

            "branch_factor":
                self.rng.uniform(
                    0.7,
                    1.4,
                    len(branches),
                ),
        })

        grid = grid.merge(
            branch_effect,
            on="branch_id",
            how="left",
        )

        # -----------------------------------------------------
        # Product baseline
        # -----------------------------------------------------

        product_effect = pd.DataFrame({
            "product_id":
                products["product_id"],

            "product_base":
                self.rng.uniform(
                    15,
                    80,
                    len(products),
                ),
        })

        grid = grid.merge(
            product_effect,
            on="product_id",
            how="left",
        )

        # -----------------------------------------------------
        # Weather
        # -----------------------------------------------------

        grid = grid.merge(
            weather,
            on=[
                "date",
                "branch_id",
            ],
            how="left",
        )

        # -----------------------------------------------------
        # Promotions
        # -----------------------------------------------------

        promo = promotions[
            [
                "date",
                "branch_id",
                "product_id",
                "discount_percentage",
            ]
        ].copy()

        grid = grid.merge(
            promo,
            on=[
                "date",
                "branch_id",
                "product_id",
            ],
            how="left",
        )

        grid[
            "discount_percentage"
        ] = grid[
            "discount_percentage"
        ].fillna(0)

        # -----------------------------------------------------
        # Demand formula
        # -----------------------------------------------------

        # Long-term trend
        day_index = (
            grid["date"]
            - grid["date"].min()
        ).dt.days

        trend = (
            1
            + 0.0004
            * day_index
        )

        demand = (
            grid["product_base"]
            * grid["branch_factor"]
            * trend
        )

        # Weekend
        demand *= np.where(
            grid["is_weekend"],
            1.25,
            1.0,
        )

        # Payday
        demand *= np.where(
            grid["is_payday"],
            1.15,
            1.0,
        )

        # Holiday
        demand *= np.where(
            grid["is_holiday"],
            1.20,
            1.0,
        )

        # Beverage weather effect
        beverage = (
            grid["category"]
            == "Beverage"
        )

        hot = (
            grid["temperature"]
            > 29
        )

        heavy_rain = (
            grid["rainfall"]
            > 10
        )

        demand *= np.where(
            beverage & hot,
            1.20,
            1.0,
        )

        demand *= np.where(
            beverage & heavy_rain,
            0.85,
            1.0,
        )

        # Food rain effect
        food = (
            grid["category"]
            == "Food"
        )

        demand *= np.where(
            food & heavy_rain,
            0.90,
            1.0,
        )

        # Promotion
        demand *= (
            1
            + (
                grid["discount_percentage"]
                / 100
            )
            * 1.5
        )

        # Random noise
        demand *= (
            self.rng.lognormal(
                mean=0,
                sigma=0.20,
                size=len(grid),
            )
        )

        # Final quantity
        grid["quantity"] = (
            self.rng.poisson(
                np.maximum(
                    demand,
                    0.1,
                )
            )
        )

        return grid[
            [
                "date",
                "branch_id",
                "product_id",
                "quantity",
                "temperature",
                "rainfall",
                "discount_percentage",
                "is_weekend",
                "is_payday",
                "is_holiday",
            ]
        ]

    # =========================================================
    # TRANSACTIONS
    # =========================================================

    def generate_transactions(
        self,
        daily_demand: pd.DataFrame,
        products: pd.DataFrame,
        customers: pd.DataFrame,
    ) -> pd.DataFrame:

        # =====================================================
        # PREPARE PRODUCT DATA
        # =====================================================

        product_info = products[
            [
                "product_id",
                "category",
                "price",
            ]
        ].copy()

        product_ids = (
            product_info["product_id"]
            .to_numpy()
        )

        product_categories = dict(
            zip(
                product_info["product_id"],
                product_info["category"],
            )
        )

        product_prices = dict(
            zip(
                product_info["product_id"],
                product_info["price"],
            )
        )

        # =====================================================
        # PRODUCT GROUPS
        # =====================================================

        category_products = {
            category: group[
                "product_id"
            ].to_numpy()
            for category, group
            in product_info.groupby("category")
        }

        food_products = category_products.get(
            "Food",
            np.array([], dtype=object),
        )

        beverage_products = category_products.get(
            "Beverage",
            np.array([], dtype=object),
        )

        snack_products = category_products.get(
            "Snack",
            np.array([], dtype=object),
        )

        dessert_products = category_products.get(
            "Dessert",
            np.array([], dtype=object),
        )

        # =====================================================
        # CUSTOMER BEHAVIOR
        # =====================================================

        customers = customers.copy()

        behavior_types = [
            "regular",
            "coffee_lover",
            "meal_customer",
            "snack_customer",
            "weekend_customer",
            "occasional",
        ]

        behavior_probabilities = [
            0.35,
            0.10,
            0.20,
            0.10,
            0.15,
            0.10,
        ]

        customers["behavior_type"] = (
            self.rng.choice(
                behavior_types,
                size=len(customers),
                p=behavior_probabilities,
            )
        )

        customer_ids = (
            customers["customer_id"]
            .to_numpy()
        )

        customer_behavior = dict(
            zip(
                customers["customer_id"],
                customers["behavior_type"],
            )
        )

        # =====================================================
        # DEMAND
        # =====================================================

        demand = daily_demand[
            daily_demand["quantity"] > 0
        ][
            [
                "date",
                "branch_id",
                "product_id",
                "quantity",
                "discount_percentage",
            ]
        ].copy()

        demand["date"] = pd.to_datetime(
            demand["date"]
        )

        records = []

        transaction_counter = 1

        # =====================================================
        # GENERATE TRANSACTIONS
        # =====================================================

        for row in demand.itertuples(
            index=False
        ):

            total_quantity = int(
                row.quantity
            )

            if total_quantity <= 0:
                continue

            # -------------------------------------------------
            # Number of transactions
            #
            # Instead of one transaction per unit,
            # assume average 2-4 units per transaction.
            # -------------------------------------------------

            average_units = self.rng.uniform(
                2.0,
                4.0,
            )

            transaction_count = max(
                1,
                int(
                    np.ceil(
                        total_quantity
                        / average_units
                    )
                ),
            )

            transaction_count = min(
                transaction_count,
                total_quantity,
            )

            # -------------------------------------------------
            # Distribute quantity
            # -------------------------------------------------

            quantities = np.ones(
                transaction_count,
                dtype=int,
            )

            remaining = (
                total_quantity
                - transaction_count
            )

            if remaining > 0:

                additions = (
                    self.rng.multinomial(
                        remaining,
                        np.ones(
                            transaction_count
                        )
                        / transaction_count,
                    )
                )

                quantities += additions

            # -------------------------------------------------
            # Individual transactions
            # -------------------------------------------------

            for quantity in quantities:

                customer_id = (
                    self.rng.choice(
                        customer_ids
                    )
                )

                behavior = (
                    customer_behavior[
                        customer_id
                    ]
                )

                transaction_id = (
                    f"T{transaction_counter:08d}"
                )

                transaction_counter += 1

                random_minutes = int(
                    self.rng.integers(
                        600,
                        1320,
                    )
                )

                timestamp = (
                    row.date
                    + pd.Timedelta(
                        minutes=random_minutes
                    )
                )

                product_price = float(
                    product_prices[
                        row.product_id
                    ]
                )

                discount = float(
                    row.discount_percentage
                )

                unit_price = (
                    product_price
                    * (
                        1
                        - discount / 100
                    )
                )

                payment_method = (
                    self.rng.choice(
                        [
                            "Cash",
                            "Debit",
                            "QRIS",
                            "E-Wallet",
                        ],
                        p=[
                            0.20,
                            0.20,
                            0.35,
                            0.25,
                        ],
                    )
                )

                total_amount = (
                    quantity
                    * unit_price
                )

                records.append(
                    {
                        "transaction_id":
                            transaction_id,

                        "timestamp":
                            timestamp,

                        "branch_id":
                            row.branch_id,

                        "customer_id":
                            customer_id,

                        "product_id":
                            row.product_id,

                        "quantity":
                            int(quantity),

                        "unit_price":
                            round(
                                unit_price,
                                2,
                            ),

                        "discount":
                            discount,

                        "total_amount":
                            round(
                                total_amount,
                                2,
                            ),

                        "payment_method":
                            payment_method,
                    }
                )

                # =============================================
                # COMPLEMENTARY PRODUCT
                # =============================================

                complementary_product = None

                probability = (
                    0.25
                )

                if behavior == "meal_customer":
                    probability = 0.55

                elif behavior == "coffee_lover":
                    probability = 0.50

                elif behavior == "regular":
                    probability = 0.35

                elif behavior == "snack_customer":
                    probability = 0.40

                elif behavior == "weekend_customer":
                    probability = 0.35

                elif behavior == "occasional":
                    probability = 0.15

                if (
                    self.rng.random()
                    < probability
                ):

                    # Meal → beverage
                    if (
                        behavior
                        == "meal_customer"
                        and len(
                            beverage_products
                        ) > 0
                    ):

                        complementary_product = (
                            self.rng.choice(
                                beverage_products
                            )
                        )

                    # Coffee lover → snack/dessert
                    elif (
                        behavior
                        == "coffee_lover"
                    ):

                        choices = []

                        if len(
                            snack_products
                        ) > 0:
                            choices.extend(
                                snack_products
                            )

                        if len(
                            dessert_products
                        ) > 0:
                            choices.extend(
                                dessert_products
                            )

                        if choices:
                            complementary_product = (
                                self.rng.choice(
                                    choices
                                )
                            )

                    # Snack customer → dessert
                    elif (
                        behavior
                        == "snack_customer"
                        and len(
                            dessert_products
                        ) > 0
                    ):

                        complementary_product = (
                            self.rng.choice(
                                dessert_products
                            )
                        )

                    # Regular → beverage
                    elif (
                        behavior
                        == "regular"
                        and len(
                            beverage_products
                        ) > 0
                    ):

                        complementary_product = (
                            self.rng.choice(
                                beverage_products
                            )
                        )

                    # Weekend customer → beverage
                    elif (
                        behavior
                        == "weekend_customer"
                        and len(
                            beverage_products
                        ) > 0
                    ):

                        complementary_product = (
                            self.rng.choice(
                                beverage_products
                            )
                        )

                # =============================================
                # ADD COMPLEMENTARY ITEM
                # =============================================

                if (
                    complementary_product
                    is not None
                    and complementary_product
                    != row.product_id
                ):

                    complementary_price = float(
                        product_prices[
                            complementary_product
                        ]
                    )

                    complementary_unit_price = (
                        complementary_price
                        * (
                            1
                            - discount / 100
                        )
                    )

                    complementary_amount = (
                        complementary_unit_price
                    )

                    records.append(
                        {
                            "transaction_id":
                                transaction_id,

                            "timestamp":
                                timestamp,

                            "branch_id":
                                row.branch_id,

                            "customer_id":
                                customer_id,

                            "product_id":
                                complementary_product,

                            "quantity":
                                1,

                            "unit_price":
                                round(
                                    complementary_unit_price,
                                    2,
                                ),

                            "discount":
                                discount,

                            "total_amount":
                                round(
                                    complementary_amount,
                                    2,
                                ),

                            "payment_method":
                                payment_method,
                        }
                    )

        transactions = pd.DataFrame(
            records
        )

        return transactions[
            [
                "transaction_id",
                "timestamp",
                "branch_id",
                "customer_id",
                "product_id",
                "quantity",
                "unit_price",
                "discount",
                "total_amount",
                "payment_method",
            ]
        ]
    # =========================================================
    # INVENTORY
    # =========================================================

    def generate_inventory(
        self,
        daily_demand: pd.DataFrame,
        product_ingredients: pd.DataFrame,
        ingredients: pd.DataFrame,
        branches: pd.DataFrame,
    ) -> pd.DataFrame:

        # ---------------------------------------------------------
        # Calculate ingredient consumption
        # ---------------------------------------------------------

        usage = daily_demand[
            [
                "date",
                "branch_id",
                "product_id",
                "quantity",
            ]
        ].merge(
            product_ingredients,
            on="product_id",
            how="inner",
        )

        usage["used_stock"] = (
            usage["quantity"]
            * usage["quantity_required"]
        )

        usage = (
            usage
            .groupby(
                [
                    "date",
                    "branch_id",
                    "ingredient_id",
                ],
                as_index=False,
            )["used_stock"]
            .sum()
        )

        # ---------------------------------------------------------
        # Create complete date × branch × ingredient grid
        # ---------------------------------------------------------

        grid = pd.MultiIndex.from_product(
            [
                self.dates,
                branches["branch_id"],
                ingredients["ingredient_id"],
            ],
            names=[
                "date",
                "branch_id",
                "ingredient_id",
            ],
        ).to_frame(index=False)

        grid = grid.merge(
            usage,
            on=[
                "date",
                "branch_id",
                "ingredient_id",
            ],
            how="left",
        )

        grid["used_stock"] = (
            grid["used_stock"]
            .fillna(0)
        )

        # ---------------------------------------------------------
        # Sort chronologically
        # ---------------------------------------------------------

        grid = grid.sort_values(
            [
                "branch_id",
                "ingredient_id",
                "date",
            ]
        ).reset_index(drop=True)

        # ---------------------------------------------------------
        # Initial stock for each branch × ingredient
        # ---------------------------------------------------------

        initial_stock = self.rng.uniform(
            100,
            500,
            size=(
                len(branches)
                * len(ingredients)
            ),
        )

        initial_map = pd.DataFrame({
            "branch_id": np.repeat(
                branches["branch_id"],
                len(ingredients),
            ),
            "ingredient_id": np.tile(
                ingredients["ingredient_id"],
                len(branches),
            ),
            "initial_stock": initial_stock,
        })

        grid = grid.merge(
            initial_map,
            on=[
                "branch_id",
                "ingredient_id",
            ],
            how="left",
        )

        # ---------------------------------------------------------
        # Sequential inventory simulation
        # ---------------------------------------------------------

        records = []

        for (
            branch_id,
            ingredient_id,
        ), group in grid.groupby(
            [
                "branch_id",
                "ingredient_id",
            ],
            sort=False,
        ):

            current_stock = float(
                group["initial_stock"].iloc[0]
            )

            for row in group.itertuples(
                index=False
            ):

                used_stock = float(
                    row.used_stock
                )

                # -------------------------------------------------
                # Replenishment
                #
                # Reorder when current stock is below
                # the daily requirement plus safety stock.
                # -------------------------------------------------

                safety_stock = max(
                    50.0,
                    used_stock * 2,
                )

                if (
                    current_stock
                    < used_stock + safety_stock
                ):
                    received_stock = max(
                        200.0,
                        used_stock * 4,
                    )
                else:
                    received_stock = 0.0

                opening_stock = current_stock

                current_stock = (
                    current_stock
                    + received_stock
                    - used_stock
                )

                # Numerical protection against
                # extremely small floating-point errors.
                current_stock = max(
                    0.0,
                    current_stock,
                )

                records.append({
                    "date": row.date,
                    "branch_id": branch_id,
                    "ingredient_id": ingredient_id,
                    "opening_stock": opening_stock,
                    "received_stock": received_stock,
                    "used_stock": used_stock,
                    "closing_stock": current_stock,
                })

        result = pd.DataFrame(
            records
        )

        # ---------------------------------------------------------
        # Round numeric columns only
        # ---------------------------------------------------------

        numeric_columns = [
            "opening_stock",
            "received_stock",
            "used_stock",
            "closing_stock",
        ]

        result[numeric_columns] = (
            result[numeric_columns]
            .round(2)
        )

        return result

    # =========================================================
    # MAIN PIPELINE
    # =========================================================

    def generate_all(
        self,
    ) -> dict[str, pd.DataFrame]:

        print("Generating branches...")
        branches = self.generate_branches()

        print("Generating products...")
        products = self.generate_products()

        print("Generating customers...")
        customers = self.generate_customers()

        print("Generating ingredients...")
        ingredients = self.generate_ingredients()

        print("Generating product recipes...")
        product_ingredients = (
            self.generate_product_ingredients(
                products,
                ingredients,
            )
        )

        print("Generating holidays...")
        holidays = self.generate_holidays()

        print("Generating weather...")
        weather = self.generate_weather(
            branches
        )

        print("Generating promotions...")
        promotions = self.generate_promotions(
            branches,
            products,
        )

        print("Generating daily demand...")
        daily_demand = (
            self.generate_daily_demand(
                branches,
                products,
                weather,
                promotions,
                holidays,
            )
        )

        print("Generating transactions...")
        sales = self.generate_transactions(
            daily_demand,
            products,
            customers,
        )

        print("Generating inventory...")
        inventory = self.generate_inventory(
            daily_demand,
            product_ingredients,
            ingredients,
            branches,
        )

        return {
            "branches": branches,
            "products": products,
            "customers": customers,
            "ingredients": ingredients,
            "product_ingredients":
                product_ingredients,
            "holidays": holidays,
            "weather": weather,
            "promotions": promotions,
            "daily_demand": daily_demand,
            "sales": sales,
            "inventory_transactions":
                inventory,
        }


def save_dataset(
    dataset: dict[str, pd.DataFrame],
    output_dir: str = "data/raw",
) -> None:

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, df in dataset.items():

        path = (
            output_path
            / f"{name}.csv"
        )

        df.to_csv(
            path,
            index=False,
        )

        print(
            f"[OK] {name}: "
            f"{len(df):,} rows"
        )