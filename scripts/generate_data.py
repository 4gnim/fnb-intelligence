from src.data.generator import (
    FNBDataGenerator,
    GeneratorConfig,
    save_dataset,
)


def main():
    config = GeneratorConfig(
        start_date="2024-01-01",
        end_date="2025-12-31",
        n_branches=10,
        n_products=50,
        n_customers=5_000,
        seed=42,
    )

    generator = FNBDataGenerator(config)

    dataset = generator.generate_all()

    save_dataset(dataset)


if __name__ == "__main__":
    main()