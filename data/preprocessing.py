"""
Loads raw NYC Yellow Taxi parquet files, cleans and aggregates trip counts
into 1-hour buckets per zone, and writes processed_data/aggregated.parquet.

Missing (zone, hour) combinations are filled with 0 so that lag features
computed downstream are always referencing the correct prior time step.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

RAW_DIR = Path(__file__).parent.parent / "raw_data"
OUTPUT_DIR = Path(__file__).parent.parent / "processed_data"
OUTPUT_FILE = OUTPUT_DIR / "aggregated.parquet"

VALID_ZONE_MIN = 1
VALID_ZONE_MAX = 263
BUCKET_HOURS = 1

# Filename pattern: yellow_tripdata_YYYY-MM.parquet
_FILENAME_RE = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet")


def date_range_from_files(paths: list[Path]) -> tuple[datetime, datetime]:
    """Derive expected start/end from the YYYY-MM filenames."""
    months = []
    for p in paths:
        m = _FILENAME_RE.match(p.name)
        if m:
            months.append((int(m.group(1)), int(m.group(2))))
    months.sort()
    start = datetime(months[0][0], months[0][1], 1, tzinfo=timezone.utc)
    # end = first hour of the month *after* the last file
    last_year, last_month = months[-1]
    if last_month == 12:
        end = datetime(last_year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(last_year, last_month + 1, 1, tzinfo=timezone.utc)
    return start, end


def load_and_clean(path: Path, start: datetime, end: datetime) -> pl.DataFrame:
    return (
        pl.read_parquet(path, columns=["tpep_pickup_datetime", "PULocationID"])
        .drop_nulls()
        .filter(
            pl.col("PULocationID").is_between(VALID_ZONE_MIN, VALID_ZONE_MAX)
            & pl.col("tpep_pickup_datetime").is_between(
                pl.lit(start).dt.replace_time_zone(None),
                pl.lit(end).dt.replace_time_zone(None),
            )
        )
        .rename({"PULocationID": "zone_id"})
    )


def aggregate(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("tpep_pickup_datetime").dt.truncate(f"{BUCKET_HOURS}h").alias("time_bucket")
        )
        .group_by(["zone_id", "time_bucket"])
        .agg(pl.len().alias("trip_count"))
    )


def fill_gaps(df: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    """Insert zero-count rows for every (zone, hour) missing from the data."""
    all_buckets = pl.datetime_range(
        start=start.replace(tzinfo=None),
        end=end.replace(tzinfo=None),
        interval=f"{BUCKET_HOURS}h",
        eager=True,
        closed="left",
    ).alias("time_bucket")

    grid = pl.DataFrame({"time_bucket": all_buckets}).join(
        df.select("zone_id").unique(), how="cross"
    )

    return (
        grid.join(df, on=["zone_id", "time_bucket"], how="left")
        .with_columns(pl.col("trip_count").fill_null(0))
        .sort(["zone_id", "time_bucket"])
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    parquet_files = sorted(RAW_DIR.glob("yellow_tripdata_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")

    start, end = date_range_from_files(parquet_files)
    print(f"Loading {len(parquet_files)} file(s)  [{start.date()} – {end.date()}]...")

    frames = [load_and_clean(p, start, end) for p in parquet_files]
    combined = pl.concat(frames)
    print(f"  Loaded {len(combined):,} trips")

    aggregated = aggregate(combined)
    print(f"  Aggregated to {len(aggregated):,} (zone, bucket) rows before gap-fill")

    filled = fill_gaps(aggregated, start, end)
    n_zeros = (filled["trip_count"] == 0).sum()
    print(f"  After gap-fill: {len(filled):,} rows ({n_zeros:,} zero-count buckets)")

    filled.write_parquet(OUTPUT_FILE)
    print(f"  Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
