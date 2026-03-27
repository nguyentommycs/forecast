"""
Kafka producer that replays NYC taxi trip events from streamed_data/.
Each message is a single trip event: {zone_id, timestamp}.

Usage:
    python streaming/producer.py
    python streaming/producer.py --speed 100   # 100x real-time
    python streaming/producer.py --speed 0     # no throttle (max speed)
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
    return pl.concat(frames).sort("tpep_pickup_datetime")


def main(speed: float) -> None:
    producer = build_producer()
    events = load_events(STREAMED_DIR)

    total = len(events)
    print(f"Streaming {total:,} events from {STREAMED_DIR.name}/")
    print(f"  Speed: {'unlimited' if speed == 0 else f'{speed}x real-time'}")
    print(f"  Topic: {TOPIC}")
    print(f"  Broker: {KAFKA_BOOTSTRAP}\n")

    prev_event_ts = None
    prev_wall_ts = None

    for i, row in enumerate(events.iter_rows(named=True)):
        event_ts = row["tpep_pickup_datetime"]
        msg = json.dumps({"zone_id": row["zone_id"], "timestamp": str(event_ts)})

        # Throttle to simulate real-time replay
        if speed > 0 and prev_event_ts is not None:
            event_gap = (event_ts - prev_event_ts).total_seconds()
            wall_gap = time.monotonic() - prev_wall_ts
            sleep_for = (event_gap / speed) - wall_gap
            if sleep_for > 0:
                time.sleep(sleep_for)

        producer.produce(TOPIC, value=msg.encode())

        # Flush periodically to avoid buffering too many messages
        if i % 1000 == 0:
            producer.poll(0)
            print(f"  [{i:>8,} / {total:,}]  ts={event_ts}", end="\r")

        prev_event_ts = event_ts
        prev_wall_ts = time.monotonic()

    producer.flush()
    print(f"\nDone. {total:,} events published to '{TOPIC}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speed",
        type=float,
        default=3600.0,
        help="Replay speed multiplier (default: 3600 = 1 simulated hour per real second). Use 0 for unlimited.",
    )
    args = parser.parse_args()
    main(args.speed)
