"""
Builds a supervised learning dataset from the hourly aggregated trip counts.
Computes lag and rolling-mean features per zone, then writes
processed_data/features.parquet.
"""

from pathlib import Path

import polars as pl

INPUT_FILE = Path(__file__).parent.parent / "processed_data" / "aggregated.parquet"
OUTPUT_FILE = Path(__file__).parent.parent / "processed_data" / "features.parquet"

# Lag offsets in number of 1-hour buckets
LAG_OFFSETS = [1, 2, 3, 6, 12, 24]  # up to same hour yesterday
ROLLING_WINDOW = 6  # 6-hour rolling mean


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort(["zone_id", "time_bucket"])
        .with_columns(
            pl.col("time_bucket").dt.hour().alias("hour_of_day"),
            pl.col("time_bucket").dt.weekday().alias("day_of_week"),
            (pl.col("time_bucket").dt.weekday() >= 5).alias("is_weekend"),
            *[
                pl.col("trip_count")
                .shift(n)
                .over("zone_id")
                .alias(f"lag_{n}")
                for n in LAG_OFFSETS
            ],
            # shift(1) so the window covers [t-6, t-1], not [t-5, t] (which would leak the target)
            pl.col("trip_count")
            .shift(1)
            .rolling_mean(window_size=ROLLING_WINDOW)
            .over("zone_id")
            .alias("rolling_mean_6"),
        )
        .drop_nulls()
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}\n"
            "Run data/preprocessing.py first."
        )

    print(f"Loading {INPUT_FILE.name}...")
    df = pl.read_parquet(INPUT_FILE)
    print(f"  {len(df):,} rows loaded")

    features = build_features(df)
    print(f"  {len(features):,} rows after feature engineering (nulls dropped)")

    features.write_parquet(OUTPUT_FILE)
    print(f"  Saved -> {OUTPUT_FILE}")
    print(f"  Columns: {features.columns}")


if __name__ == "__main__":
    main()
