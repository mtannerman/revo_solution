import polars as pl

DATA_PATH = "data/data.csv"
LOW_CONTRIBUTOR_PERCENTILE = 0.05


def contribution_counts(df: pl.LazyFrame) -> pl.LazyFrame:
    """Number of project/job/question triplets labelled by each assessor."""
    return df.group_by("assessor", maintain_order=True).agg(pl.len().alias("n_labels"))


def assessor_quality(df: pl.LazyFrame) -> pl.LazyFrame:
    counts = contribution_counts(df)

    # 5th percentile of the contribution counts; assessors below it have not
    # contributed enough for a reliable quality estimate.
    min_labels = (
        counts.select(
            pl.col("n_labels").quantile(LOW_CONTRIBUTOR_PERCENTILE, interpolation="lower")
        )
        .collect()
        .item()
    )

    agreements = (
        df.with_columns(
            pl.col("label").sum().over(["project", "job", "question"]).alias("label_sum"),
            pl.len().over(["project", "job", "question"]).alias("n_votes"),
        )
        # A triplet with a single assessor carries no agreement information.
        .filter(pl.col("n_votes") > 1)
        # Leave-one-out majority: the majority of the *other* assessors' labels,
        # so an assessor's own vote never counts towards its reference. Ties
        # resolve deterministically to 1, as in get_majority_vote.py.
        .with_columns(
            (
                (pl.col("label_sum") - pl.col("label")) / (pl.col("n_votes") - 1) >= 0.5
            )
            .cast(pl.Int64)
            .alias("loo_majority")
        )
        .with_columns((pl.col("label") == pl.col("loo_majority")).alias("agrees"))
    )

    return (
        agreements.group_by("assessor", maintain_order=True)
        .agg(pl.col("agrees"))
        .join(counts, on="assessor", how="left", maintain_order="left")
        .filter(pl.col("n_labels") >= min_labels)
        # Bayesian estimate of the agreement rate: Beta(1, 1) prior updated with
        # the per-assessor list of agreement booleans.
        .with_columns(
            (1 + pl.col("agrees").list.sum()).alias("alpha"),
            (1 + pl.col("agrees").list.len() - pl.col("agrees").list.sum()).alias("beta"),
        )
        .with_columns(
            (pl.col("alpha") / (pl.col("alpha") + pl.col("beta"))).alias("posterior_mean"),
            (
                (pl.col("alpha") * pl.col("beta"))
                / ((pl.col("alpha") + pl.col("beta")) ** 2 * (pl.col("alpha") + pl.col("beta") + 1))
            )
            .sqrt()
            .alias("posterior_std"),
        )
        .with_columns(
            (pl.col("posterior_mean") + pl.col("posterior_std")).alias("assessor_quality")
        )
        .select("assessor", "n_labels", "posterior_mean", "posterior_std", "assessor_quality")
    )


def label_quality(df: pl.LazyFrame, quality: pl.LazyFrame) -> pl.LazyFrame:
    return (
        # Inner join drops the votes of the filtered-out low contributors.
        df.join(
            quality.select("assessor", "assessor_quality"),
            on="assessor",
            how="inner",
            maintain_order="left",
        )
        .group_by(["project", "job", "question"], maintain_order=True)
        .agg(
            (
                (pl.col("label") * pl.col("assessor_quality")).sum()
                / pl.col("assessor_quality").sum()
            ).alias("weighted_vote")
        )
        .with_columns(
            (pl.col("weighted_vote") >= 0.5).cast(pl.Int64).alias("label_quality")
        )
    )


def main() -> None:
    df = pl.scan_csv(DATA_PATH)
    quality = assessor_quality(df)
    print(quality.collect())
    print(label_quality(df, quality).collect())


if __name__ == "__main__":
    main()
