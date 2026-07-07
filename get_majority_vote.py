import polars as pl

DATA_PATH = "data/data.csv"


def majority_vote(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.group_by(["project", "job", "question"], maintain_order=True).agg(
        # Majority of binary labels via the mean; a tie (mean == 0.5) resolves
        # deterministically to 1.
        (pl.col("label").mean() >= 0.5).cast(pl.Int64).alias("majority_vote"),
        pl.col("label").std().alias("label_std"),
    )


def main() -> None:
    result = majority_vote(pl.scan_csv(DATA_PATH)).collect()
    print(result)


if __name__ == "__main__":
    main()
