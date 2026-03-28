"""
Kafka producer that replays NYC taxi trip events from streamed_data/.
Each message is a single trip event: {zone_id, timestamp}.

Throttles by hourly bucket: all events in a simulated hour are published
immediately, then the producer sleeps until `--seconds-per-hour` wall
seconds have elapsed. This gives consistent pacing regardless of how many
trips are in each hour.

Usage:
    python streaming/producer.py
    python streaming/producer.py --seconds-per-hour 1   # 1 simulated hour per real second
    python streaming/producer.py --seconds-per-hour 0   # no throttle (max speed)
"""

import argparse
import json
import time
from pathlib import Path

import polars as pl
from confluent_kafka import Producer

STREAMED_DIR = Path(__file__).parent.parent / "streamed_data"
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "taxi-trips"


def build_producer() -> Producer:
    return Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})


def load_events(streamed_dir: Path) -> pl.DataFrame:
    files = sorted(streamed_dir.glob("yellow_tripdata_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {streamed_dir}")

    frames = [
        pl.read_parquet(f, columns=["tpep_pickup_datetime", "PULocationID"])
        .drop_nulls()
        .filter(pl.col("PULocationID").is_between(1, 263))
        .rename({"PULocationID": "zone_id"})
        for f in files
    ]
    return (
        pl.concat(frames)
        .sort("tpep_pickup_datetime")
        .with_columns(
            pl.col("tpep_pickup_datetime").dt.truncate("1h").alias("hour_bucket")
        )
    )


def main(seconds_per_hour: float) -> None:
    producer = build_producer()
    events = load_events(STREAMED_DIR)

    total = len(events)
    hours = events["hour_bucket"].n_unique()
    print(f"Streaming {total:,} events across {hours:,} simulated hours from {STREAMED_DIR.name}/")
    print(f"  Pace: {'unlimited' if seconds_per_hour == 0 else f'{seconds_per_hour}s wall time per simulated hour'}")
    print(f"  Topic: {TOPIC}")
    print(f"  Broker: {KAFKA_BOOTSTRAP}\n")

    published = 0
    for (hour_bucket,), group in events.group_by(["hour_bucket"], maintain_order=True):
        bucket_start = time.monotonic()

        for row in group.iter_rows(named=True):
            msg = json.dumps({"zone_id": row["zone_id"], "timestamp": str(row["tpep_pickup_datetime"])})
            producer.produce(TOPIC, value=msg.encode())

        producer.poll(0)
        published += len(group)
        print(f"  [{published:>8,} / {total:,}]  bucket={hour_bucket}", end="\r")

        if seconds_per_hour > 0:
            elapsed = time.monotonic() - bucket_start
            sleep_for = seconds_per_hour - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    producer.flush()
    print(f"\nDone. {total:,} events published to '{TOPIC}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seconds-per-hour",
        type=float,
        default=30.0,
        help="Wall seconds to spend per simulated hour (default: 30). Use 0 for unlimited.",
    )
    args = parser.parse_args()
    main(args.seconds_per_hour)
