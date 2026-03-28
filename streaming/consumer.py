"""
Kafka consumer that builds rolling hourly trip-count features per zone
and writes them to Redis for the prediction API to read.

Each incoming event is a single trip: {zone_id, timestamp}.
The consumer maintains a sliding window of completed hourly buckets per zone
and updates Redis after every event.

Usage:
    python streaming/consumer.py
"""

import json
import signal
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import redis
from confluent_kafka import Consumer, KafkaError

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "taxi-trips"
GROUP_ID = "feature-engine"

REDIS_HOST = "localhost"
REDIS_PORT = 6379

HISTORY_LEN = 24  # hours of completed buckets to keep (covers lag_24)
LAG_OFFSETS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOW = 6


@dataclass
class ZoneState:
    """Tracks completed hourly counts and the currently open bucket for one zone."""
    history: deque = field(default_factory=lambda: deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN))
    current_bucket: datetime | None = None
    current_count: int = 0


def compute_features(zone_id: int, state: ZoneState, bucket: datetime) -> dict:
    hist = list(state.history)  # oldest … newest completed hour
    lags = {f"lag_{n}": hist[-n] if n <= len(hist) else 0 for n in LAG_OFFSETS}
    rolling_mean_6 = sum(hist[-ROLLING_WINDOW:]) / ROLLING_WINDOW if len(hist) >= ROLLING_WINDOW else 0.0

    return {
        "zone_id": zone_id,
        "bucket": bucket.isoformat(),
        "hour_of_day": bucket.hour,
        "day_of_week": bucket.weekday(),
        "is_weekend": int(bucket.weekday() >= 5),
        "rolling_mean_6": rolling_mean_6,
        **lags,
    }


def advance_bucket(state: ZoneState, new_bucket: datetime) -> None:
    """Close the current bucket and open a new one, filling any skipped hours with 0."""
    if state.current_bucket is None:
        state.current_bucket = new_bucket
        return

    # Fill any gap between the last closed bucket and the new one with zeros
    hours_elapsed = int((new_bucket - state.current_bucket).total_seconds() // 3600)
    if hours_elapsed > 0:
        state.history.append(state.current_count)
        for _ in range(min(hours_elapsed - 1, HISTORY_LEN)):
            state.history.append(0)
        state.current_count = 0
        state.current_bucket = new_bucket


def main() -> None:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([TOPIC])
    print(f"Subscribed to '{TOPIC}' on {KAFKA_BOOTSTRAP}\n")

    zone_states: dict[int, ZoneState] = {}
    processed = 0

    def shutdown(sig, frame):
        print(f"\nShutting down. Processed {processed:,} events.")
        consumer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"Kafka error: {msg.error()}")
            continue

        event = json.loads(msg.value().decode())
        zone_id = int(event["zone_id"])
        ts = datetime.fromisoformat(event["timestamp"])
        bucket = ts.replace(minute=0, second=0, microsecond=0)

        if zone_id not in zone_states:
            zone_states[zone_id] = ZoneState()

        state = zone_states[zone_id]

        if state.current_bucket != bucket:
            advance_bucket(state, bucket)

        state.current_count += 1

        features = compute_features(zone_id, state, bucket)
        r.set(f"features:{zone_id}", json.dumps(features))

        processed += 1
        if processed % 10_000 == 0:
            print(f"  Processed {processed:,} events  |  {len(zone_states)} zones active")


if __name__ == "__main__":
    main()
