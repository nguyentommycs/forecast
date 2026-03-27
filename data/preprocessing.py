"""
Loads raw NYC Yellow Taxi parquet files, cleans and aggregates trip counts
into 1-hour buckets per zone, and writes processed_data/aggregated.parquet.
"""

from pathlib import Path

import polars as pl

RAW_DIR = Path(__file__).parent.parent / "raw_data"
OUTPUT_DIR = Path(__file__).parent.parent / "processed_data"
OUTPUT_FILE = OUTPUT_DIR / "aggregated.parquet"

VALID_ZONE_MIN = 1
VALID_ZONE_MAX = 263
BUCKET_HOURS = 1


def load_and_clean(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path, columns=["tpep_pickup_datetime", "PULocationID"])
        .drop_nulls()
        .filter(
            pl.col("PULocationID").is_between(VALID_ZONE_MIN, VALID_ZONE_MAX)
        )
        .rename({"PULocationID": "zone_id"})
    )


def aggregate(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            (
                pl.col("tpep_pickup_datetime").dt.truncate(f"{BUCKET_HOURS}h")
            ).alias("time_bucket")
        )
        .group_by(["zone_id", "time_bucket"])
        .agg(pl.len().alias("trip_count"))
        .sort(["zone_id", "time_bucket"])
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    parquet_files = sorted(RAW_DIR.glob("yellow_tripdata_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")

    print(f"Loading {len(parquet_files)} file(s)...")
    frames = [load_and_clean(p) for p in parquet_files]
    combined = pl.concat(frames)
    print(f"  Loaded {len(combined):,} trips")

    aggregated = aggregate(combined)
    print(f"  Aggregated to {len(aggregated):,} (zone, bucket) rows")

    aggregated.write_parquet(OUTPUT_FILE)
    print(f"  Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
